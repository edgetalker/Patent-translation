# agent/state.py
"""
LangGraph Agent State 定义
复用现有模块的数据结构，仅添加Agent执行状态字段
"""
from typing import TypedDict, List, Optional, Dict, Annotated
import operator


class TranslationState(TypedDict):
    # ── 输入（来自用户请求）──────────────────────────────
    src_text: str
    src_lang: str
    tgt_lang: str
    domain: str
    glossary: Optional[Dict[str, str]]       # 外部术语表（注入优先）
    domain_prompt: Optional[str]
    use_corpus: bool
    corpus_id: Optional[str]
    corpus_threshold: float

    # ── Tool 1 输出：术语提取结果 ─────────────────────────
    chunks: List[Dict]                        # split_text_by_paragraph 结果
    term_dict: Dict[str, str]                 # {源术语: 目标术语}

    # ── Tool 2 输出：并行翻译结果 ─────────────────────────
    translated_chunks: List[str]              # 按 chunk_id 顺序

    # ── Tool 3 输出：一致性验证结果 ───────────────────────
    validation_passed: bool
    inconsistencies: List[str]                # 违规术语列表
    retry_count: int                          # 防死循环

    # ── 最终结果 ──────────────────────────────────────────
    final_translation: str
    corpus_stats: Optional[Dict]

    # ── Agent 控制流 ──────────────────────────────────────
    messages: Annotated[List, operator.add]   # LangGraph 消息历史
    use_pipeline_fallback: bool               # 降级标志
    error: Optional[str]