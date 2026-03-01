# agent/tools.py
"""
三个核心 Agent Tool
所有实际逻辑均代理给现有模块，Tool 只负责接口适配
"""
from langchain_core.tools import tool
from typing import Dict, List, Optional, Tuple
import concurrent.futures

# 复用现有模块
from terminology_extraction import TerminologyExtractor
from translation_core import DocumentTranslator
from utils import split_text_by_paragraph
from config import config

# 全局单例（避免重复初始化）
_term_extractor: Optional[TerminologyExtractor] = None
_translator: Optional[DocumentTranslator] = None


def get_term_extractor() -> TerminologyExtractor:
    global _term_extractor
    if _term_extractor is None:
        _term_extractor = TerminologyExtractor()
    return _term_extractor


def get_translator(corpus_manager=None) -> DocumentTranslator:
    """
    注意：corpus_manager 不能作为 Tool 参数传入（不可序列化）
    通过此工厂函数在模块级注入
    """
    global _translator
    if _translator is None:
        _translator = DocumentTranslator(corpus_manager=corpus_manager)
    return _translator


def init_tools(corpus_manager=None):
    """在 FastAPI 启动时调用，注入 corpus_manager"""
    get_translator(corpus_manager=corpus_manager)


# ============================================================
# Tool 1: 术语提取 + 文档分块
# ============================================================

@tool
def term_extract_tool(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    【术语提取工具】
    对输入文档执行：
    1. 按段落分块（8000字符/2000重叠）
    2. 滑动窗口提取专业术语，跨窗口频率统计
    3. 外部术语库注入优先（glossary 中的术语直接覆盖）
    
    Returns:
        {
            "chunks": List[Dict],          # 分块结果
            "term_dict": Dict[str, str],   # {源术语: 目标术语}
            "term_count": int
        }
    """
    extractor = get_term_extractor()
    
    # Step 1: 文档分块（复用 utils.py 现有函数）
    chunks = split_text_by_paragraph(src_text, config.MAX_CHUNK_LENGTH)
    
    # Step 2: 术语处理
    if glossary:
        # 外部术语表直接使用，无需 LLM 提取
        print(f"[Tool1] 使用外部术语表，共 {len(glossary)} 个术语")
        term_dict = glossary
    else:
        # 滑动窗口提取
        print(f"[Tool1] 滑动窗口提取术语，文档长度 {len(src_text)} 字符")
        terms = extractor.sliding_window_extract(
            text=src_text,
            src_lang=src_lang,
            domain=domain
        )
        term_dict = extractor.translate_terminology(
            terms=terms,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            domain=domain
        )
    
    print(f"[Tool1] 完成：{len(chunks)} 个分块，{len(term_dict)} 个术语")
    
    return {
        "chunks": chunks,
        "term_dict": term_dict,
        "term_count": len(term_dict)
    }


# ============================================================
# Tool 2: 并行翻译（含 Qdrant RAG 加速）
# ============================================================

@tool
def parallel_trans_tool(
    chunks: List[Dict],
    term_dict: Dict[str, str],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    domain_prompt: Optional[str] = None,
    use_corpus: bool = False,
    corpus_id: Optional[str] = None,
    corpus_threshold: float = 0.85,
    retry_hint: Optional[List[str]] = None,   # 重试时只翻译违规块
) -> Dict:
    """
    【并行翻译工具】
    3 线程并发翻译所有分块；
    命中 Qdrant 历史译文（阈值 0.85）时直接复用，跳过 LLM 调用。
    
    retry_hint: 传入违规术语列表时，仅重新翻译含违规术语的 chunks（节省成本）
    
    Returns:
        {
            "translated_chunks": List[str],   # 按 chunk_id 顺序
            "corpus_stats": Dict
        }
    """
    translator = get_translator()
    translations = [None] * len(chunks)
    corpus_stats = {
        "enabled": use_corpus,
        "total_sentences": 0,
        "total_hits": 0,
        "total_misses": 0
    }
    
    # 确定需要翻译的 chunk 索引
    if retry_hint:
        # 重试模式：只翻译包含违规术语的 chunk
        def contains_violation(chunk_text: str) -> bool:
            return any(v.split("->")[0].strip() in chunk_text 
                      for v in retry_hint)
        target_indices = [
            i for i, c in enumerate(chunks) 
            if contains_violation(c["text"])
        ]
        print(f"[Tool2] 重试模式：仅重翻 {len(target_indices)}/{len(chunks)} 个违规块")
    else:
        target_indices = list(range(len(chunks)))
    
    def translate_task(idx: int) -> Tuple[int, str, Optional[Dict]]:
        chunk = chunks[idx]
        translation, stats = translator.translate_chunk(
            chunk_text=chunk["text"],
            chunk_id=idx,
            total_chunks=len(chunks),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            domain=domain,
            term_dict=term_dict,
            domain_prompt=domain_prompt,
            context=None,                # 并行模式不用上下文
            use_corpus=use_corpus,
            corpus_id=corpus_id,
            corpus_threshold=corpus_threshold
        )
        return idx, translation, stats
    
    # 3 线程并行
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(translate_task, i): i 
                   for i in target_indices}
        
        for future in concurrent.futures.as_completed(futures):
            idx, translation, stats = future.result()
            translations[idx] = translation
            
            if stats:
                corpus_stats["total_sentences"] += stats.get("total_sentences", 0)
                corpus_stats["total_hits"] += stats.get("hits", 0)
                corpus_stats["total_misses"] += stats.get("misses", 0)
            
            print(f"[Tool2] 完成 chunk {idx+1}/{len(chunks)}")
    
    # 命中率计算
    if corpus_stats["total_sentences"] > 0:
        corpus_stats["hit_rate"] = (
            corpus_stats["total_hits"] / corpus_stats["total_sentences"]
        )
    
    return {
        "translated_chunks": translations,
        "corpus_stats": corpus_stats
    }


# ============================================================
# Tool 3: 术语一致性验证
# ============================================================

@tool
def validate_tool(
    translated_chunks: List[str],
    term_dict: Dict[str, str],
    src_text: str,
    tgt_lang: str
) -> Dict:
    """
    【一致性验证工具】
    全文扫描术语使用情况；中英文语言自适应去重。
    返回验证报告，Orchestrator 据此决定是否重试。
    
    Returns:
        {
            "passed": bool,
            "inconsistencies": List[str],   # ["术语A -> 译法A", ...]
            "consistency_rate": float,
            "full_translation": str          # 拼接好的全文
        }
    """
    translator = get_translator()
    full_text = "\n\n".join(t for t in translated_chunks if t)
    
    is_consistent, inconsistencies = translator.validate_terminology_consistency(
        translation=full_text,
        term_dict=term_dict,
        src_text=src_text,
        tgt_lang=tgt_lang
    )
    
    consistency_rate = 1.0 - len(inconsistencies) / max(len(term_dict), 1)
    
    print(f"[Tool3] 一致性验证：{'✅ 通过' if is_consistent else f'❌ {len(inconsistencies)} 个违规'}")
    
    return {
        "passed": is_consistent,
        "inconsistencies": inconsistencies,
        "consistency_rate": consistency_rate,
        "full_translation": full_text
    }