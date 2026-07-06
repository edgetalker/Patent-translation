# agent/graph.py
"""
LangGraph Orchestrator
将 5 个 Tool 串联为纯线性 Tool-Use 闭环:
  chunk → term_extract → retrieve → parallel_trans → stats → END

设计原则:
  - 无分支(no conditional edges)
  - 无重试(no retry loop)  
  - 无降级(no fallback)
  - 每个节点职责单一,只操作自己那一段 state
"""
from langgraph.graph import StateGraph, END
from agent.state import TranslationState
from agent.tools import (
    chunk_tool,
    term_extract_tool,
    retrieve_tool,
    parallel_trans_tool,
    stats_tool,
)


# ============================================================
# 节点函数(每个节点只写自己产出的字段)
# ============================================================

def run_chunk(state: TranslationState) -> dict:
    """节点 1:分块"""
    result = chunk_tool.invoke({
        "src_text": state["src_text"],
        "context_budget": state.get("context_budget"),
    })
    return {"chunks": result["chunks"]}


def run_term_extract(state: TranslationState) -> dict:
    """节点 2:术语表获取(glossary or 自动抽取)"""
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
    return {"translated_chunks": result["translated_chunks"]}


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
    graph.add_node("chunk",          run_chunk)
    graph.add_node("term_extract",   run_term_extract)
    graph.add_node("retrieve",       run_retrieve)
    graph.add_node("parallel_trans", run_parallel_trans)
    graph.add_node("stats",          run_stats)
    
    # 纯线性边
    graph.set_entry_point("chunk")
    graph.add_edge("chunk",          "term_extract")
    graph.add_edge("term_extract",   "retrieve")
    graph.add_edge("retrieve",       "parallel_trans")
    graph.add_edge("parallel_trans", "stats")
    graph.add_edge("stats",          END)
    
    return graph.compile()


# 全局 Agent 实例
patent_agent = build_patent_agent()