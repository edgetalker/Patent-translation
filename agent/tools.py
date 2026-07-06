import concurrent.futures
import asyncio
import time
from langchain_core.tools import tool
from typing import Dict, List, Optional

# 复用现有模块
from terminology_extraction import TerminologyExtractor
from translation_core import DocumentTranslator
from retrieval_service import RetrievalService, BatchRetrievalResult
from utils import split_text_by_paragraph, calculate_context_budget
from config import config

# 全局单例
_term_extractor: Optional[TerminologyExtractor] = None
_translator: Optional[DocumentTranslator] = None
_retrieval_service: Optional[RetrievalService] = None


def get_term_extractor() -> TerminologyExtractor:
    global _term_extractor
    if _term_extractor is None:
        _term_extractor = TerminologyExtractor()
    return _term_extractor


def get_translator() -> DocumentTranslator:
    global _translator
    if _translator is None:
        _translator = DocumentTranslator()  # 不再传 corpus_manager
    return _translator


def get_retrieval_service() -> Optional[RetrievalService]:
    """可能返回 None(未初始化 corpus_manager 时)"""
    return _retrieval_service


def init_tools(corpus_manager=None, src_lang: str = "zh", tgt_lang: str = "en"):
    """
    在 FastAPI 启动时调用
    
    Args:
        corpus_manager: 可选的语料库管理器(None 表示系统不启用 RAG)
        src_lang / tgt_lang: 初始语言对(可在运行时覆盖)
    """
    global _retrieval_service
    get_translator()  # 预初始化
    get_term_extractor()
    
    if corpus_manager:
        _retrieval_service = RetrievalService(
            corpus_manager=corpus_manager,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        print(f"[init_tools] RetrievalService 已初始化 (src={src_lang}, tgt={tgt_lang})")
    else:
        _retrieval_service = None
        print("[init_tools] 未提供 corpus_manager,RAG 功能将被跳过")


@tool
def chunk_tool(src_text: str, context_budget: Optional[Dict] = None) -> Dict:
    """
    【分块工具】
    将长文本按段落切分成 chunk,作为后续所有处理的基本单位。
    支持通过 context_budget 动态调整最大 chunk 字符数。

    Returns:
        {
            "chunks": List[Dict],  # [{"text": str, "chunk_id": int, "start_pos": int}, ...]
            "total_chunks": int
        }
    """
    max_length = config.MAX_CHUNK_LENGTH
    if context_budget:
        dynamic_max = context_budget.get("max_chunk_chars")
        if dynamic_max:
            max_length = max(dynamic_max, config.MIN_CHUNK_LENGTH)

    chunks = split_text_by_paragraph(src_text, max_length)
    print(f"[chunk_tool] 文档分为 {len(chunks)} 个 chunk (max_length={max_length})")
    return {
        "chunks": chunks,
        "total_chunks": len(chunks),
    }

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
    术语表获取策略:
    - 有 glossary → 直接使用(绕过 LLM 调用,节省成本)
    - 无 glossary → 滑动窗口提取 + LLM 翻译
    
    Returns:
        {
            "term_dict": Dict[str, str],  # {源术语: 目标术语}
            "term_count": int,
            "source": str  # "glossary" | "extracted"
        }
    """
    if glossary:
        print(f"[term_extract_tool] 使用外部术语表,共 {len(glossary)} 个")
        return {
            "term_dict": glossary,
            "term_count": len(glossary),
            "source": "glossary",
        }
    
    extractor = get_term_extractor()
    terms = extractor.sliding_window_extract(
        text=src_text,
        src_lang=src_lang,
        domain=domain,
    )
    term_dict = extractor.translate_terminology(
        terms=terms,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        domain=domain,
    )
    
    print(f"[term_extract_tool] 抽取并翻译 {len(term_dict)} 个术语")
    return {
        "term_dict": term_dict,
        "term_count": len(term_dict),
        "source": "extracted",
    }

@tool
def retrieve_tool(
    chunks: List[Dict],
    corpus_id: Optional[str] = None,
    corpus_threshold: float = 0.85,
    use_corpus: bool = True,
) -> Dict:
    """
    【RAG 预检索工具】
    对所有 chunks 统一做句子级 RAG 检索,为每个 chunk 预备 Few-Shot 参考。
    
    调度模式:独立节点,在翻译之前完成。
    失败策略:降级为全量空 few_shots,翻译继续。
    
    Args:
        chunks: [{"chunk_id": int, "text": str}, ...]
        corpus_id: 语料库 ID(None 或 use_corpus=False 时跳过检索)
        corpus_threshold: 相似度阈值
        use_corpus: 显式开关
    
    Returns:
        {
            "enabled": bool,                # 本次是否真实执行了检索
            "per_chunk": List[Dict],        # 每个 chunk 的 few_shots 和明细
            "stats": {                      # 全局统计
                "total_sentences": int,
                "total_hits": int,
                "total_misses": int,
                "hit_rate": float,
                "embedding_api_calls": int,
                "success": bool,
                "error_message": Optional[str]
            }
        }
    """
    # ---------- 跳过路径 ----------
    service = get_retrieval_service()
    if not use_corpus or not corpus_id or service is None:
        reason = (
            "use_corpus=False" if not use_corpus else
            "corpus_id=None" if not corpus_id else
            "RetrievalService 未初始化"
        )
        print(f"[retrieve_tool] 跳过检索: {reason}")
        return {
            "enabled": False,
            "per_chunk": [
                {"chunk_id": c["chunk_id"], "few_shots": [], "hit_details": []}
                for c in chunks
            ],
            "stats": {
                "total_sentences": 0,
                "total_hits": 0,
                "total_misses": 0,
                "hit_rate": 0.0,
                "embedding_api_calls": 0,
                "success": True,
                "error_message": None,
            }
        }
    
    # ---------- 检索路径 ----------
    async def _do():
        return await service.batch_retrieve(
            chunks=chunks,
            corpus_id=corpus_id,
            threshold=corpus_threshold,
        )
    
    # 在任意上下文下运行(FastAPI / Jupyter 环境的 event loop 兼容)
    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False
    
    if has_loop:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            batch_result: BatchRetrievalResult = pool.submit(
                asyncio.run, _do()
            ).result()
    else:
        batch_result: BatchRetrievalResult = asyncio.run(_do())
    
    # ---------- 组装输出 ----------
    per_chunk_output = []
    for r in batch_result.per_chunk:
        hit_details = []
        if r.retrieval_result is not None:
            for sent in r.retrieval_result.sentences:
                if sent.matched:
                    hit_details.append({
                        "sentence_index": sent.index,
                        "query_source": sent.source,
                        "matched_corpus_source": sent.corpus_source,
                        "matched_translation": sent.translation,
                        "similarity": sent.similarity,
                    })
        per_chunk_output.append({
            "chunk_id": r.chunk_id,
            "few_shots": r.few_shots,
            "hit_details": hit_details,
        })
    
    return {
        "enabled": True,
        "per_chunk": per_chunk_output,
        "stats": {
            "total_sentences": batch_result.total_sentences,
            "total_hits": batch_result.total_hits,
            "total_misses": batch_result.total_misses,
            "hit_rate": batch_result.hit_rate,
            "embedding_api_calls": batch_result.embedding_api_calls,
            "success": batch_result.success,
            "error_message": batch_result.error_message,
        }
    }

@tool
def parallel_trans_tool(
    chunks: List[Dict],
    retrieval_per_chunk: List[Dict],  # retrieve_tool 返回的 per_chunk
    term_dict: Dict[str, str],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    domain_prompt: Optional[str] = None,
    max_workers: int = 3,
    max_few_shots: int = 5,
    context_budget: Optional[Dict] = None,
) -> Dict:
    """
    【并行翻译工具】
    消费预检索结果,并发翻译所有 chunks。
    每个 worker 使用对应 chunk 的 few_shots,内部不再触发检索。
    支持通过 context_budget 动态调整术语注入和 few-shot 数量。

    Args:
        chunks: 分块列表
        retrieval_per_chunk: retrieve_tool 输出的 per_chunk 字段
        term_dict: 术语表(全文级,translate_chunk 内部会再做 chunk 级过滤)
        max_workers: 并发数
        max_few_shots: Top-K 截断(可被 context_budget 覆盖)
        context_budget: 动态上下文预算

    Returns:
        {
            "translated_chunks": List[str],  # 按 chunk_id 顺序
            "translation_time": float,       # 总耗时
            "failed_chunks": List[int]       # 失败的 chunk_id 列表
        }
    """
    translator = get_translator()

    # 建立 chunk_id → few_shots 映射,保证对齐
    few_shots_map = {
        item["chunk_id"]: item["few_shots"]
        for item in retrieval_per_chunk
    }

    translations: List[Optional[str]] = [None] * len(chunks)
    failed_chunks: List[int] = []

    def translate_task(idx: int) -> tuple:
        chunk = chunks[idx]
        cid = chunk["chunk_id"]
        few_shots = few_shots_map.get(cid, [])

        # 按当前 chunk 的实际文本计算动态预算
        per_chunk_budget = None
        if context_budget is None:
            per_chunk_budget = calculate_context_budget(
                chunk_text=chunk["text"],
                system_overhead=config.SYSTEM_PROMPT_OVERHEAD,
                term_dict=term_dict,
                model_context_window=config.MODEL_CONTEXT_WINDOW,
                max_output_tokens=config.MAX_TOKENS,
                safe_margin=config.SAFE_MARGIN,
            )
        else:
            per_chunk_budget = dict(context_budget)

        try:
            translation = translator.translate_chunk(
                chunk_text=chunk["text"],
                chunk_id=cid,
                total_chunks=len(chunks),
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                term_dict=term_dict,
                domain_prompt=domain_prompt,
                few_shots=few_shots,
                max_few_shots=max_few_shots,
                context_budget=per_chunk_budget,
            )
            return idx, translation, None
        except Exception as e:
            return idx, f"[TRANSLATION FAILED: chunk {cid}]", str(e)

    # 并发执行
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_task, i) for i in range(len(chunks))]
        for future in concurrent.futures.as_completed(futures):
            idx, translation, err = future.result()
            translations[idx] = translation
            if err:
                failed_chunks.append(chunks[idx]["chunk_id"])
                print(f"[parallel_trans_tool] chunk {idx} 失败: {err}")
            else:
                print(f"[parallel_trans_tool] 完成 chunk {idx+1}/{len(chunks)}")

    elapsed = time.time() - start
    print(f"[parallel_trans_tool] 总翻译耗时: {elapsed:.2f}s")

    return {
        "translated_chunks": translations,
        "translation_time": elapsed,
        "failed_chunks": failed_chunks,
    }


@tool
def repair_trans_tool(
    chunks: List[Dict],
    translated_chunks: List[str],
    failed_chunks: List[int],
    retrieval_per_chunk: List[Dict],
    term_dict: Dict[str, str],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    domain_prompt: Optional[str] = None,
) -> Dict:
    """
    【翻译修复工具】
    针对 parallel_trans_tool 返回的失败 chunk 进行单线程修复重译。

    修复策略:
    - 关闭 few_shots,避免有问题的参考示例干扰
    - 附加修复提示,要求严格遵循术语表
    - 单线程顺序执行,降低并发带来的不稳定因素

    Args:
        chunks: 全部分块列表
        translated_chunks: 当前译文列表(会被原地更新)
        failed_chunks: 失败的 chunk_id 列表
        retrieval_per_chunk: retrieve_tool 输出的 per_chunk 字段
        term_dict: 术语表
        src_lang / tgt_lang / domain: 翻译参数
        domain_prompt: 领域级额外指令

    Returns:
        {
            "translated_chunks": List[str],  # 更新后的译文列表
            "failed_chunks": List[int],      # 修复后仍失败的 chunk_id
        }
    """
    if not failed_chunks:
        return {"translated_chunks": translated_chunks, "failed_chunks": []}

    translator = get_translator()
    few_shots_map = {
        item["chunk_id"]: item["few_shots"]
        for item in retrieval_per_chunk
    }

    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    still_failed: List[int] = []

    print(f"[repair_trans_tool] 开始修复 {len(failed_chunks)} 个失败 chunk: {failed_chunks}")

    for cid in failed_chunks:
        chunk = chunk_map.get(cid)
        if chunk is None:
            still_failed.append(cid)
            continue

        per_chunk_budget = calculate_context_budget(
            chunk_text=chunk["text"],
            system_overhead=config.SYSTEM_PROMPT_OVERHEAD,
            term_dict=term_dict,
            model_context_window=config.MODEL_CONTEXT_WINDOW,
            max_output_tokens=config.MAX_TOKENS,
            safe_margin=config.SAFE_MARGIN,
        )

        try:
            translation = translator.translate_chunk(
                chunk_text=chunk["text"],
                chunk_id=cid,
                total_chunks=len(chunks),
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                term_dict=term_dict,
                domain_prompt=domain_prompt,
                few_shots=[],
                max_few_shots=0,
                context_budget=per_chunk_budget,
                repair_mode=True,
            )

            # 找到 translated_chunks 中对应的位置并更新
            for idx, c in enumerate(chunks):
                if c["chunk_id"] == cid:
                    translated_chunks[idx] = translation
                    break

            if translation.startswith("[TRANSLATION FAILED"):
                still_failed.append(cid)
                print(f"[repair_trans_tool] chunk {cid} 修复后仍失败")
            else:
                print(f"[repair_trans_tool] chunk {cid} 修复成功")

        except Exception as e:
            still_failed.append(cid)
            print(f"[repair_trans_tool] chunk {cid} 修复异常: {e}")

    return {
        "translated_chunks": translated_chunks,
        "failed_chunks": still_failed,
    }

@tool
def stats_tool(
    translated_chunks: List[str],
    term_dict: Dict[str, str],
    src_text: str,
) -> Dict:
    """
    【统计工具】
    输出术语一致性统计(纯描述,不做重试决策)。
    
    注:本工具仅产出指标,是否需要根据指标调整行为由上层(人类/业务规则)决定。
    
    Returns:
        {
            "full_translation": str,         # 拼接好的全文
            "terminology_stats": Dict,       # 来自 compute_terminology_stats 的完整输出
        }
    """
    translator = get_translator()
    full_text = "\n\n".join(t for t in translated_chunks if t)
    
    term_stats = translator.compute_terminology_stats(
        translation=full_text,
        term_dict=term_dict,
        src_text=src_text,
    )
    
    rate = term_stats["consistency_rate"]
    print(
        f"[stats_tool] 术语一致性: {rate*100:.1f}% "
        f"(命中 {term_stats['terminology_hit']}/{term_stats['terminology_total']}, "
        f"不一致 {term_stats['terminology_miss']})"
    )
    
    return {
        "full_translation": full_text,
        "terminology_stats": term_stats,
    }