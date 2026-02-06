"""
FastAPI服务端点
提供RESTful API接口
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

from config import config
from translation_core import DocumentTranslator
from terminology_extraction import TerminologyExtractor

app = FastAPI(
    title="Document Translation API",
    description="长文档翻译服务(带术语一致性处理)",
    version="1.0.0"
)

# 初始化翻译器和术语提取器
translator = DocumentTranslator()
term_extractor = TerminologyExtractor()


# ==================== 数据模型 ====================

class TranslationRequest(BaseModel):
    """翻译请求模型"""
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str = "技术"
    use_context: bool = True
    glossary: Optional[Dict[str, str]] = None


class TranslationResponse(BaseModel):
    """翻译响应模型"""
    translation: str
    term_dict: Dict[str, str]
    chunks_info: List[Dict]
    statistics: Dict


class TerminologyExtractionRequest(BaseModel):
    """术语提取请求模型"""
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str = "技术"
    window_size: Optional[int] = None
    overlap: Optional[int] = None
    max_terms: Optional[int] = None


class TerminologyExtractionResponse(BaseModel):
    """术语提取响应模型"""
    terms: List[str]
    term_dict: Dict[str, List[str]]
    statistics: Dict


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    config: Dict


# ==================== API端点 ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径,返回服务信息"""
    return {
        "status": "running",
        "version": "1.0.0",
        "config": {
            "llm_model": config.LLM_MODEL_NAME,
            "max_terms": config.MAX_TERMS,
            "window_size": config.WINDOW_SIZE
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "llm_base_url": config.LLM_BASE_URL,
            "llm_model": config.LLM_MODEL_NAME
        }
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_document(request: TranslationRequest):
    """翻译长文档"""
    try:
        result = translator.translate_document(
            src_text=request.src_text,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            domain=request.domain,
            use_context=request.use_context,
            glossary=request.glossary,
            parallel=True,      
            max_workers=3       
        )
        
        print(f"✅ 翻译完成\n")
        return result
        
    except Exception as e:
        print(f"\n❌ 翻译失败: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")


@app.post("/extract_terminology", response_model=TerminologyExtractionResponse)
async def extract_terminology(request: TerminologyExtractionRequest):
    """
    从文档中提取专业术语并翻译
    
    Args:
        request: 术语提取请求
            - src_text: 源文本
            - src_lang: 源语言代码 (如 'zh', 'en')
            - tgt_lang: 目标语言代码
            - domain: 领域信息 (默认"技术")
            - window_size: 窗口大小 (可选)
            - overlap: 重叠区域大小 (可选)
            - max_terms: 最多提取术语数量 (可选)
        
    Returns:
        TerminologyExtractionResponse: 提取结果
            - terms: 源语言术语列表
            - term_dict: 术语对照字典 {源术语: 目标术语}
            - statistics: 统计信息
    """
    try:
        print(f"\n{'='*60}")
        print(f"开始提取术语...")
        print(f"- 源语言: {request.src_lang}")
        print(f"- 目标语言: {request.tgt_lang}")
        print(f"- 领域: {request.domain}")
        print(f"- 文本长度: {len(request.src_text)} 字符")
        print(f"{'='*60}\n")
        
        # 1. 提取源语言术语
        terms = term_extractor.sliding_window_extract(
            text=request.src_text,
            src_lang=request.src_lang,
            domain=request.domain,
            window_size=request.window_size,
            overlap=request.overlap,
            max_final_terms=request.max_terms
        )
        
        print(f"\n✅ 提取到 {len(terms)} 个术语")
        
        # 2. 翻译术语
        if terms:
            print(f"\n开始翻译术语...")
            term_dict = term_extractor.translate_terminology(
                terms=terms,
                src_lang=request.src_lang,
                tgt_lang=request.tgt_lang,
                domain=request.domain
            )
            print(f"✅ 翻译完成，成功翻译 {len(term_dict)} 个术语")
        else:
            term_dict = {}
            print(f"⚠️  未提取到术语，跳过翻译步骤")
        
        # 3. 构建统计信息
        statistics = {
            "text_length": len(request.src_text),
            "terms_extracted": len(terms),
            "terms_translated": len(term_dict),
            "src_lang": request.src_lang,
            "tgt_lang": request.tgt_lang,
            "domain": request.domain,
            "window_size": request.window_size or config.WINDOW_SIZE,
            "overlap": request.overlap or config.WINDOW_OVERLAP
        }
        
        print(f"\n{'='*60}")
        print(f"术语提取完成！")
        print(f"{'='*60}\n")
        
        return {
            "terms": terms,
            "term_dict": term_dict,
            "statistics": statistics
        }
        
    except Exception as e:
        print(f"\n❌ 术语提取失败: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"术语提取失败: {str(e)}")


@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "llm": {
            "base_url": config.LLM_BASE_URL,
            "model": config.LLM_MODEL_NAME
        },
        "translation": {
            "max_chunk_length": config.MAX_CHUNK_LENGTH,
            "overlap_length": config.OVERLAP_LENGTH,
            "temperature": config.TRANSLATION_TEMPERATURE
        },
        "terminology": {
            "max_terms": config.MAX_TERMS,
            "window_size": config.WINDOW_SIZE,
            "window_overlap": config.WINDOW_OVERLAP,
            "min_frequency": config.MIN_TERM_FREQUENCY
        }
    }


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print(f"启动翻译服务...")
    print(f"监听地址: {config.API_HOST}:{config.API_PORT}")
    print(f"LLM服务: {config.LLM_BASE_URL}")
    print(f"模型: {config.LLM_MODEL_NAME}\n")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )