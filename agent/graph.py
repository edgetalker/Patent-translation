# agent/graph.py
"""
LangGraph Orchestrator
将 5 个 Tool 编排为 Tool-Use 闭环:
  fanout → chunk / term_extract → retrieve → parallel_trans → [repair] → stats → END

设计原则:
  - chunk 与 term_extract 可并行(均只依赖 src_text)
  - parallel_trans 失败后自动进入 repair_trans 修复
  - 每个节点职责单一,只操作自己那一段 state
"""
from langgraph.graph import StateGraph, END
from agent.state import TranslationState
from agent.tools import (
    chunk_tool,
    term_extract_tool,
    retrieve_tool,
    parallel_trans_tool,
    repair_trans_tool,
    stats_tool,
)


# ============================================================
# 节点函数(每个节点只写自己产出的字段)
# ============================================================

def fanout(state: TranslationState) -> dict:
    """入口 fan-out: 触发可并行的独立分支"""
    return {}


def run_chunk(state: TranslationState) -> dict:
    """分支 1:分块"""
    result = chunk_tool.invoke({
        "src_text": state["src_text"],
        "context_budget": state.get("context_budget"),
    })
    return {"chunks": result["chunks"]}


def run_term_extract(state: TranslationState) -> dict:
    """分支 2:术语表获取(glossary or 自动抽取)"""
    result = term_extract_tool.invoke({
        "src_text": state["src_text"],
        "src_lang": state["src_lang"],
        "tgt_lang": state["tgt_lang"],
        "domain":   state["domain"],
        "glossary": state.get("glossary"),
    })
    return {"term_dict": result["term_dict"]}


def run_retrieve(state: TranslationState) -> dict:
    """节点 3:RAG 预检索(为所有 chunk 批量查询 few_shots)"""
    result = retrieve_tool.invoke({
        "chunks":           state["chunks"],
        "corpus_id":        state.get("corpus_id"),
        "corpus_threshold": state.get("corpus_threshold", 0.85),
        "use_corpus":       state.get("use_corpus", False),
    })
    return {"retrieval_results": result}


def run_parallel_trans(state: TranslationState) -> dict:
    """节点 4:并行翻译(消费预检索结果)"""
    result = parallel_trans_tool.invoke({
        "chunks":               state["chunks"],
        "retrieval_per_chunk":  state["retrieval_results"]["per_chunk"],
        "term_dict":            state["term_dict"],
        "src_lang":             state["src_lang"],
        "tgt_lang":             state["tgt_lang"],
        "domain":               state["domain"],
        "domain_prompt":        state.get("domain_prompt"),
        "context_budget":       state.get("context_budget"),
    })
    return {
        "translated_chunks": result["translated_chunks"],
        "failed_chunks":     result["failed_chunks"],
    }


def run_repair_trans(state: TranslationState) -> dict:
    """节点 4.5:修复并行翻译中失败的 chunk"""
    if not state.get("failed_chunks"):
        return {}

    result = repair_trans_tool.invoke({
        "chunks":              state["chunks"],
        "translated_chunks":   state["translated_chunks"],
        "failed_chunks":       state["failed_chunks"],
        "retrieval_per_chunk": state["retrieval_results"]["per_chunk"],
        "term_dict":           state["term_dict"],
        "src_lang":            state["src_lang"],
        "tgt_lang":            state["tgt_lang"],
        "domain":              state["domain"],
        "domain_prompt":       state.get("domain_prompt"),
    })
    return {
        "translated_chunks": result["translated_chunks"],
        "failed_chunks":     result["failed_chunks"],
    }


def route_after_parallel_trans(state: TranslationState) -> str:
    """若存在失败 chunk,进入 repair_trans;否则直接统计"""
    if state.get("failed_chunks"):
        return "repair_trans"
    return "stats"


def run_stats(state: TranslationState) -> dict:
    """节点 5:术语一致性统计 + 拼接全文"""
    result = stats_tool.invoke({
        "translated_chunks": state["translated_chunks"],
        "term_dict":         state["term_dict"],
        "src_text":          state["src_text"],
    })
    return {
        "terminology_stats": result["terminology_stats"],
        "final_translation": result["full_translation"],
    }


# ============================================================
# 构建 Graph
# ============================================================

def build_patent_agent():
    graph = StateGraph(TranslationState)

    # 注册节点
    graph.add_node("fanout",         fanout)
    graph.add_node("chunk",          run_chunk)
    graph.add_node("term_extract",   run_term_extract)
    graph.add_node("retrieve",       run_retrieve)
    graph.add_node("parallel_trans", run_parallel_trans)
    graph.add_node("repair_trans",   run_repair_trans)
    graph.add_node("stats",          run_stats)

    # fan-out: chunk 与 term_extract 并行,完成后汇入 retrieve
    graph.set_entry_point("fanout")
    graph.add_edge("fanout",         "chunk")
    graph.add_edge("fanout",         "term_extract")
    graph.add_edge("chunk",          "retrieve")
    graph.add_edge("term_extract",   "retrieve")
    graph.add_edge("retrieve",       "parallel_trans")

    # 翻译失败时进入修复节点,修复后进入统计
    graph.add_conditional_edges(
        "parallel_trans",
        route_after_parallel_trans,
        {"repair_trans": "repair_trans", "stats": "stats"}
    )
    graph.add_edge("repair_trans", "stats")

    graph.add_edge("stats", END)

    return graph.compile()


# 全局 Agent 实例
patent_agent = build_patent_agent()
