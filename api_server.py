"""
FastAPI 服务端点
提供 RESTful API 接口

对齐主干架构:
  chunk → term_extract → retrieve → parallel_trans → stats → END
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

from agent.graph import build_patent_agent
from agent.tools import init_tools
from agent.memory import get_memory_saver

from config import config
from terminology_extraction import TerminologyExtractor
from corpus.embeddings import EmbeddingService
from corpus.manager import CorpusManager

app = FastAPI(
    title="Document Translation API",
    description="专利翻译智能体",
    version="2.1.0",
)

# ==================== 初始化 ====================

embedding_service = EmbeddingService()
corpus_manager = CorpusManager(
    qdrant_url=config.QDRANT_URL,
    qdrant_api_key=config.QDRANT_API_KEY,
    collection_name=config.QDRANT_COLLECTION_NAME,
    embedding_service=embedding_service,
)

# Agent 工具层初始化(注入 corpus_manager 到全局单例)
init_tools(corpus_manager=corpus_manager)

# 使用 MemorySaver 初始化带 checkpointer 的 Agent
patent_agent = build_patent_agent(checkpointer=get_memory_saver())

# 术语提取器:用于独立的 /extract_terminology 端点
term_extractor = TerminologyExtractor()


# ==================== 数据模型 ====================

class TranslationRequest(BaseModel):
    """翻译请求模型"""
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str = "技术"
    glossary: Optional[Dict[str, str]] = None
    domain_prompt: Optional[str] = None
    use_corpus: bool = False
    corpus_id: Optional[str] = None
    corpus_threshold: float = 0.85

    # Agent 控制参数
    max_iterations: Optional[int] = 2
    consistency_threshold: Optional[float] = 0.85
    thread_id: Optional[str] = None
    use_memory: bool = False
    context_budget: Optional[Dict] = None


class TranslationResponse(BaseModel):
    """翻译响应模型"""
    translation: str
    term_dict: Dict[str, str]
    chunks_info: List[Dict]
    statistics: Dict
    terminology_stats: Optional[Dict] = None
    retrieval_stats: Optional[Dict] = None
    iteration_count: Optional[int] = None
    failed_chunks: Optional[List[int]] = None
    terminology_memory: Optional[Dict[str, str]] = None


class TerminologyExtractionRequest(BaseModel):
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str = "技术"
    window_size: Optional[int] = None
    overlap: Optional[int] = None
    max_terms: Optional[int] = None


class TerminologyExtractionResponse(BaseModel):
    terms: List[str]
    term_dict: Dict[str, str]
    statistics: Dict


class HealthResponse(BaseModel):
    status: str
    version: str
    config: Dict


class CorpusEntry(BaseModel):
    source: str
    target: str
    metadata: Optional[dict] = {}


class AddCorpusRequest(BaseModel):
    entries: List[CorpusEntry]
    corpus_id: str = "default"


class SearchRequest(BaseModel):
    query: str
    corpus_id: Optional[str] = None
    limit: int = 5
    threshold: float = 0.7


# ==================== 基础端点 ====================

@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "running",
        "version": "2.0.0",
        "config": {
            "llm_model": config.LLM_MODEL_NAME,
            "max_terms": config.MAX_TERMS,
            "window_size": config.WINDOW_SIZE,
            "corpus_enabled": corpus_manager is not None,
        },
    }


@app.get("/config")
async def get_config():
    return {
        "llm": {
            "base_url": config.LLM_BASE_URL,
            "model": config.LLM_MODEL_NAME,
        },
        "embedding": {
            "base_url": config.EMBED_BASE_URL,
        },
        "translation": {
            "max_chunk_length": config.MAX_CHUNK_LENGTH,
            "overlap_length": config.OVERLAP_LENGTH,
            "temperature": config.TRANSLATION_TEMPERATURE,
        },
        "terminology": {
            "max_terms": config.MAX_TERMS,
            "window_size": config.WINDOW_SIZE,
            "window_overlap": config.WINDOW_OVERLAP,
            "min_frequency": config.MIN_TERM_FREQUENCY,
        },
        "corpus": {
            "qdrant_url": config.QDRANT_URL,
            "collection_name": config.QDRANT_COLLECTION_NAME,
            "enabled": corpus_manager is not None,
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "config": {
            "llm_base_url": config.LLM_BASE_URL,
            "llm_model": config.LLM_MODEL_NAME,
            "corpus_enabled": corpus_manager is not None,
        },
    }


# ==================== 核心翻译端点 ====================

@app.post("/translate", response_model=TranslationResponse)
async def translate_document(request: TranslationRequest):
    """
    翻译长文档(Agent 自纠错 workflow)

    流程:fanout → chunk / term_extract → retrieve → parallel_trans → [repair]
         → stats → [prepare_retry loop] → format_output
    """
    try:
        # 构建初始 State(只传输入字段,节点产出字段由 graph 自动填充)
        initial_state = {
            "src_text":              request.src_text,
            "src_lang":              request.src_lang,
            "tgt_lang":              request.tgt_lang,
            "domain":                request.domain,
            "glossary":              request.glossary,
            "domain_prompt":         request.domain_prompt,
            "use_corpus":            request.use_corpus,
            "corpus_id":             request.corpus_id,
            "corpus_threshold":      request.corpus_threshold,
            "max_iterations":        request.max_iterations,
            "consistency_threshold": request.consistency_threshold,
            "use_memory":            request.use_memory,
            "context_budget":        request.context_budget,
            "iteration_count":       0,
            "failed_chunks":         [],
        }

        thread_id = request.thread_id or "default"

        # 运行 Agent(带 checkpointer)
        result_state = patent_agent.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # 组装响应
        term_stats = result_state.get("terminology_stats") or {}
        retrieval_results = result_state.get("retrieval_results") or {}
        statistics = result_state.get("statistics") or {}

        return {
            "translation":        result_state.get("translation", result_state.get("final_translation", "")),
            "term_dict":          result_state.get("term_dict", {}),
            "chunks_info": [
                {"chunk_id": c["chunk_id"], "length": len(c["text"])}
                for c in result_state.get("chunks", [])
            ],
            "statistics": {
                "source_length":               len(request.src_text),
                "translation_length":          len(result_state.get("final_translation", "")),
                "num_chunks":                  len(result_state.get("chunks", [])),
                "terminology_consistency_rate": term_stats.get("consistency_rate"),
                "terminology_hit":             term_stats.get("terminology_hit"),
                "terminology_total":           term_stats.get("terminology_total"),
                "iteration_count":             statistics.get("iteration_count", 0),
                "failed_chunks":               statistics.get("failed_chunks", []),
            },
            "terminology_stats":  term_stats or None,
            "retrieval_stats":    retrieval_results.get("stats") if retrieval_results else None,
            "iteration_count":    statistics.get("iteration_count", 0),
            "failed_chunks":      statistics.get("failed_chunks", []),
            "terminology_memory": result_state.get("terminology_memory") if request.use_memory else None,
        }

    except Exception as e:
        import traceback
        print(f"\n❌ Agent 执行失败: {str(e)}\n")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")


# ==================== 术语提取端点 ====================

@app.post("/extract_terminology", response_model=TerminologyExtractionResponse)
async def extract_terminology(request: TerminologyExtractionRequest):
    try:
        print(f"\n{'='*60}")
        print(f"开始提取术语...")
        print(f"- 源语言: {request.src_lang}")
        print(f"- 目标语言: {request.tgt_lang}")
        print(f"- 领域: {request.domain}")
        print(f"- 文本长度: {len(request.src_text)} 字符")
        print(f"{'='*60}\n")

        terms = term_extractor.sliding_window_extract(
            text=request.src_text,
            src_lang=request.src_lang,
            domain=request.domain,
            window_size=request.window_size,
            overlap=request.overlap,
            max_final_terms=request.max_terms,
        )
        print(f"\n✅ 提取到 {len(terms)} 个术语")

        if terms:
            term_dict = term_extractor.translate_terminology(
                terms=terms,
                src_lang=request.src_lang,
                tgt_lang=request.tgt_lang,
                domain=request.domain,
            )
            print(f"✅ 翻译完成,成功翻译 {len(term_dict)} 个术语")
        else:
            term_dict = {}
            print(f"⚠️  未提取到术语,跳过翻译步骤")

        statistics = {
            "text_length":      len(request.src_text),
            "terms_extracted":  len(terms),
            "terms_translated": len(term_dict),
            "src_lang":         request.src_lang,
            "tgt_lang":         request.tgt_lang,
            "domain":           request.domain,
            "window_size":      request.window_size or config.WINDOW_SIZE,
            "overlap":          request.overlap or config.WINDOW_OVERLAP,
        }

        return {
            "terms":      terms,
            "term_dict":  term_dict,
            "statistics": statistics,
        }

    except Exception as e:
        print(f"\n❌ 术语提取失败: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"术语提取失败: {str(e)}")


# ==================== 语料库管理端点 ====================

@app.post("/corpus/add")
async def add_corpus(request: AddCorpusRequest):
    try:
        entries = [entry.dict() for entry in request.entries]
        result = await corpus_manager.add_corpus_entries(
            entries=entries,
            corpus_id=request.corpus_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/corpus/search")
async def search_corpus(request: SearchRequest):
    try:
        results = await corpus_manager.search_similar(
            query_text=request.query,
            corpus_id=request.corpus_id,
            limit=request.limit,
            score_threshold=request.threshold,
        )
        return {
            "query":   request.query,
            "results": results,
            "count":   len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/corpus/{corpus_id}")
async def delete_corpus(corpus_id: str):
    try:
        result = corpus_manager.delete_corpus(corpus_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus/stats")
async def get_stats(corpus_id: Optional[str] = None):
    try:
        stats = corpus_manager.get_corpus_stats(corpus_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print(f"启动翻译服务...")
    print(f"监听地址: {config.API_HOST}:{config.API_PORT}")
    print(f"LLM 服务: {config.LLM_BASE_URL}")
    print(f"模型: {config.LLM_MODEL_NAME}")
    print(f"语料库: Qdrant @ {config.QDRANT_URL}\n")

    uvicorn.run(
        "api_server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info",
        reload=True
    )