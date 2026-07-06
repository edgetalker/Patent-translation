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

    # ── Agent 控制参数 ──────────────────────────────
    max_iterations: int                       # 自纠错最大迭代次数
    consistency_threshold: float              # 术语一致性阈值
    thread_id: Optional[str]                  # 跨调用记忆线程 ID
    use_memory: bool                          # 是否启用术语记忆
    context_budget: Optional[Dict]            # 动态上下文预算

    # ── 节点产出 / 循环状态 ───────────────────────────
    chunks: List[Dict]                        # chunk_tool 输出
    term_dict: Dict[str, str]                 # term_extract_tool 输出
    retrieval_results: Dict                   # retrieve_tool 输出 (含 enabled/per_chunk/stats)
    translated_chunks: List[str]              # parallel_trans_tool 输出
    failed_chunks: List[int]                  # parallel_trans_tool 输出的失败 chunk_id
    terminology_stats: Dict                   # stats_tool 输出
    terminology_memory: Dict[str, str]        # 跨调用术语记忆
    iteration_count: int                      # 当前自纠错迭代次数
    feedback_prompt: Optional[str]            # 自纠错反馈提示
    retry_chunk_ids: List[int]                # 需要重译的 chunk_id 列表

    # ── 最终结果 ────────────────────────────────────
    final_translation: str
    translation: str                          # Streamlit 兼容别名
    statistics: Optional[Dict]                # Streamlit 兼容别名