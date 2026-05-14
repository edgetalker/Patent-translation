"""
预检索服务
负责把 chunks 的所有句子合并后批量检索,按 chunk 归还结果
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from corpus.manager import CorpusManager
from corpus_retrieval import CorpusRetriever, RetrievalResult


@dataclass
class ChunkRetrievalResult:
    """单个 chunk 的检索结果(供 parallel_trans_tool 消费)"""
    chunk_id: int
    retrieval_result: RetrievalResult  # 句子级原始结果
    few_shots: List[tuple]              # [(corpus_src, corpus_tgt, sim), ...]


@dataclass
class BatchRetrievalResult:
    """整批 chunks 的检索结果"""
    per_chunk: List[ChunkRetrievalResult]
    # 全局统计
    total_sentences: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    # 调试信息
    embedding_api_calls: int = 0  # 理论上永远是 1
    success: bool = True
    error_message: Optional[str] = None


class RetrievalService:
    """
    预检索服务:把多个 chunk 的检索合并成一次 API 调用
    
    设计要点:
    - 无状态服务,一个实例可复用
    - 检索失败时优雅降级(返回空 few_shots,不阻塞翻译)
    - 跨 chunk 批量是其核心价值,保留 chunk 边界
    """
    
    def __init__(
        self,
        corpus_manager: CorpusManager,
        src_lang: str = "zh",
        tgt_lang: str = "en",
    ):
        self.corpus_manager = corpus_manager
        self.retriever = CorpusRetriever(
            corpus_manager=corpus_manager,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
    
    async def batch_retrieve(
        self,
        chunks: List[Dict],
        corpus_id: str,
        threshold: float = 0.85,
    ) -> BatchRetrievalResult:
        """
        批量预检索
        
        Args:
            chunks: [{"chunk_id": int, "text": str}, ...]
            corpus_id: 语料库 ID
            threshold: 相似度阈值
        
        Returns:
            BatchRetrievalResult:包含每个 chunk 的检索结果 + 全局统计
        """
        if not chunks:
            return BatchRetrievalResult(per_chunk=[])
        
        per_chunk_results: List[ChunkRetrievalResult] = []
        total_sentences = 0
        total_hits = 0
        
        try:
            # 当前实现:每个 chunk 独立调用 retrieve_for_chunk
            # 原因:底层 batch_search_similar 已经是单次 embedding 调用
            #      跨 chunk 合并的进一步优化可放到 Future Work
            # 
            # 但! 关键不同在于:现在它在"预检索阶段"统一做,
            #     翻译 workers 里不再发生检索,这是架构收益的真正来源
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                retrieval_result = await self.retriever.retrieve_for_chunk(
                    chunk=chunk["text"],
                    corpus_id=corpus_id,
                    threshold=threshold,
                )
                few_shots = retrieval_result.get_few_shots()
                
                per_chunk_results.append(ChunkRetrievalResult(
                    chunk_id=chunk_id,
                    retrieval_result=retrieval_result,
                    few_shots=few_shots,
                ))
                
                total_sentences += len(retrieval_result.sentences)
                total_hits += retrieval_result.hit_count
            
            total_misses = total_sentences - total_hits
            hit_rate = total_hits / total_sentences if total_sentences > 0 else 0.0
            
            print(
                f"[RetrievalService] 预检索完成: "
                f"{len(chunks)} chunks, {total_sentences} 句, "
                f"命中 {total_hits} ({hit_rate*100:.1f}%)"
            )
            
            return BatchRetrievalResult(
                per_chunk=per_chunk_results,
                total_sentences=total_sentences,
                total_hits=total_hits,
                total_misses=total_misses,
                hit_rate=hit_rate,
                embedding_api_calls=len(chunks),  # 实际值,论文可写
                success=True,
            )
        
        except Exception as e:
            # 降级:任何检索失败 → 全部 few_shots 空,翻译继续
            print(f"[RetrievalService] 预检索失败,全量降级为零参考: {e}")
            return BatchRetrievalResult(
                per_chunk=[
                    ChunkRetrievalResult(
                        chunk_id=chunk["chunk_id"],
                        retrieval_result=None,
                        few_shots=[],
                    )
                    for chunk in chunks
                ],
                success=False,
                error_message=str(e),
            )