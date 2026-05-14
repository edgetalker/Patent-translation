# agent/state.py
"""
LangGraph Agent State 定义
主干:chunk → term_extract → retrieve → parallel_trans → stats → END
纯线性,无重试,无降级
"""
from typing import TypedDict, List, Optional, Dict


class TranslationState(TypedDict):
    # ── 输入(来自用户请求)────────────────────────────
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str
    glossary: Optional[Dict[str, str]]        # 外部术语表(注入优先)
    domain_prompt: Optional[str]
    use_corpus: bool
    corpus_id: Optional[str]
    corpus_threshold: float

    # ── 节点产出 ────────────────────────────────────
    chunks: List[Dict]                        # chunk_tool 输出
    term_dict: Dict[str, str]                 # term_extract_tool 输出
    retrieval_results: Dict                   # retrieve_tool 输出 (含 enabled/per_chunk/stats)
    translated_chunks: List[str]              # parallel_trans_tool 输出
    terminology_stats: Dict                   # stats_tool 输出

    # ── 最终结果 ────────────────────────────────────
    final_translation: str