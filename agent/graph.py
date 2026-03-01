# agent/graph.py
"""
LangGraph Orchestrator
将三个 Tool 串联为完整的 Tool-Use 闭环
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import TranslationState
from agent.tools import term_extract_tool, parallel_trans_tool, validate_tool

MAX_RETRY = 3   # 最大重试次数，防死循环


# ============================================================
# 节点函数
# ============================================================

def run_term_extract(state: TranslationState) -> TranslationState:
    """节点1：调用 Tool1，写入 chunks 和 term_dict"""
    result = term_extract_tool.invoke({
        "src_text":  state["src_text"],
        "src_lang":  state["src_lang"],
        "tgt_lang":  state["tgt_lang"],
        "domain":    state["domain"],
        "glossary":  state.get("glossary"),
    })
    return {
        **state,
        "chunks":    result["chunks"],
        "term_dict": result["term_dict"],
    }


def run_parallel_trans(state: TranslationState) -> TranslationState:
    """节点2：调用 Tool2，写入 translated_chunks"""
    # 重试时把违规术语作为 hint 传入，减少 API 调用
    retry_hint = state.get("inconsistencies") if state.get("retry_count", 0) > 0 else None
    
    result = parallel_trans_tool.invoke({
        "chunks":           state["chunks"],
        "term_dict":        state["term_dict"],
        "src_lang":         state["src_lang"],
        "tgt_lang":         state["tgt_lang"],
        "domain":           state["domain"],
        "domain_prompt":    state.get("domain_prompt"),
        "use_corpus":       state.get("use_corpus", False),
        "corpus_id":        state.get("corpus_id"),
        "corpus_threshold": state.get("corpus_threshold", 0.85),
        "retry_hint":       retry_hint,
    })
    
    # 重试时只替换违规块，其余保留原译文
    if retry_hint and state.get("translated_chunks"):
        old = list(state["translated_chunks"])
        new = result["translated_chunks"]
        merged = [new[i] if new[i] is not None else old[i] 
                  for i in range(len(old))]
        translated = merged
    else:
        translated = result["translated_chunks"]
    
    return {
        **state,
        "translated_chunks": translated,
        "corpus_stats":      result["corpus_stats"],
    }


def run_validate(state: TranslationState) -> TranslationState:
    """节点3：调用 Tool3，写入验证结果"""
    result = validate_tool.invoke({
        "translated_chunks": state["translated_chunks"],
        "term_dict":         state["term_dict"],
        "src_text":          state["src_text"],
        "tgt_lang":          state["tgt_lang"],
    })
    return {
        **state,
        "validation_passed":  result["passed"],
        "inconsistencies":    result["inconsistencies"],
        "final_translation":  result["full_translation"],   # 提前写入，失败也有值
    }


def run_pipeline_fallback(state: TranslationState) -> TranslationState:
    """
    降级节点：异常路径下调用原 Pipeline，保证系统稳定运行。
    完全复用 translation_core.DocumentTranslator.translate_document()
    """
    print("[Fallback] Agent 异常，切换至 Pipeline 模式...")
    
    from agent.tools import get_translator
    translator = get_translator()
    
    result = translator.translate_document(
        src_text=state["src_text"],
        src_lang=state["src_lang"],
        tgt_lang=state["tgt_lang"],
        domain=state.get("domain", "技术"),
        glossary=state.get("glossary"),
        domain_prompt=state.get("domain_prompt"),
        parallel=True,
        max_workers=3,
        use_corpus=state.get("use_corpus", False),
        corpus_id=state.get("corpus_id"),
        corpus_threshold=state.get("corpus_threshold", 0.85),
    )
    
    return {
        **state,
        "final_translation": result["translation"],
        "term_dict":         result["term_dict"],
        "corpus_stats":      result["corpus_stats"],
    }


# ============================================================
# 路由函数
# ============================================================

def route_after_validate(state: TranslationState) -> str:
    """
    验证后的路由决策：
    - 通过 → 结束
    - 失败 + 未超重试上限 → 重新翻译
    - 失败 + 超限 → Pipeline 降级
    """
    if state["validation_passed"]:
        return "end"
    
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRY:
        print(f"[Router] 验证失败，第 {retry_count+1} 次重试...")
        return "retry"
    else:
        print(f"[Router] 超过最大重试次数 ({MAX_RETRY})，降级至 Pipeline")
        return "fallback"


def increment_retry(state: TranslationState) -> TranslationState:
    """重试前递增计数器"""
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ============================================================
# 构建 Graph
# ============================================================

def build_patent_agent():
    graph = StateGraph(TranslationState)
    
    # 注册节点
    graph.add_node("term_extract",       run_term_extract)
    graph.add_node("parallel_trans",     run_parallel_trans)
    graph.add_node("validate",           run_validate)
    graph.add_node("increment_retry",    increment_retry)
    graph.add_node("pipeline_fallback",  run_pipeline_fallback)
    
    # 主流程边
    graph.set_entry_point("term_extract")
    graph.add_edge("term_extract",   "parallel_trans")
    graph.add_edge("parallel_trans", "validate")
    
    # 验证后路由
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "end":      END,
            "retry":    "increment_retry",
            "fallback": "pipeline_fallback",
        }
    )
    
    # 重试回环
    graph.add_edge("increment_retry", "parallel_trans")
    graph.add_edge("pipeline_fallback", END)
    
    return graph.compile()


# 全局 Agent 实例
patent_agent = build_patent_agent()