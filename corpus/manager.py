"""
语料库管理模块
负责 Qdrant 向量数据库的 CRUD 操作
"""
import asyncio
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, 
)

from corpus.embeddings import EmbeddingService


class CorpusManager:
    """语料库管理器"""
    
    def __init__(
        self,
        qdrant_url: str = "localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: str = "patent_corpus"
    ):
        """
        初始化语料库管理器
        
        Args:
            qdrant_host: Qdrant 服务器地址
            qdrant_port: Qdrant 端口
            embedding_service: Embedding 服务实例
            collection_name: Collection 名称
        """
        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url,api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=qdrant_url)
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        
        # 确保 collection 存在
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保 collection 存在"""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection 不存在，创建
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            print(f"创建 collection: {self.collection_name}")
    
    async def add_corpus_entries(
        self,
        entries: List[Dict],
        corpus_id: str = "default"
    ) -> Dict:
        """
        添加语料到语料库
        
        Args:
            entries: 语料列表 [{"source": "...", "target": "...", "metadata": {...}}, ...]
            corpus_id: 语料库ID
        
        Returns:
            {"success": True, "count": N, "corpus_id": "..."}
        """
        if not entries:
            return {"success": False, "error": "No entries provided"}
        
        # 提取所有源文本
        source_texts = [entry["source"] for entry in entries]
        
        # 批量获取 embeddings
        try:
            embeddings = await self.embedding_service.get_embeddings(
                texts=source_texts,
                is_query=False
            )
        except Exception as e:
            return {"success": False, "error": f"Failed to get embeddings: {str(e)}"}
        
        # 构建 points
        points = []
        for i, (entry, embedding) in enumerate(zip(entries, embeddings)):
            payload = {
                "source": entry["source"],
                "target": entry["target"],
                "corpus_id": corpus_id,
                **entry.get("metadata", {})
            }
            
            points.append(
                PointStruct(
                    id=self._generate_point_id(),
                    vector=embedding,
                    payload=payload
                )
            )
        
        # 批量插入
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return {
                "success": True,
                "count": len(points),
                "corpus_id": corpus_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search_similar(
        self,
        query_text: str,
        corpus_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        检索相似语料（单个文本）
        
        Args:
            query_text: 查询文本
            corpus_id: 语料库ID（可选）
            limit: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            [{"source": "...", "target": "...", "score": 0.95, "corpus_id": "..."}, ...]
        """
        # 获取查询向量
        try:
            embeddings = await self.embedding_service.get_embeddings(
                texts=[query_text],
                is_query=True
            )
            query_embedding = embeddings[0]
        except Exception as e:
            print(f"❌ Embedding失败: {str(e)}")
            return []
        
        # 调用内部检索方法
        return await self._search_by_embedding(
            query_embedding=query_embedding,
            corpus_id=corpus_id,
            limit=limit,
            score_threshold=score_threshold
        )
    
    async def batch_search_similar(
        self,
        query_texts: List[str],
        corpus_id: Optional[str] = None,
        limit: int = 1,
        score_threshold: float = 0.7
    ) -> List[List[Dict]]:
        """
        批量检索相似语料（核心优化：一次embedding调用）
        
        Args:
            query_texts: 查询文本列表 ["句子1", "句子2", ...]
            corpus_id: 语料库ID
            limit: 每个查询返回结果数
            score_threshold: 分数阈值
        
        Returns:
            [[结果1], [结果2], ...]  每个文本的检索结果
        """
        if not query_texts:
            return []
        
        # 🔑 关键：批量获取 embeddings（一次API调用）
        try:
            embeddings = await self.embedding_service.get_embeddings(
                texts=query_texts,
                is_query=True
            )
        except Exception as e:
            print(f"❌ 批量embedding失败: {str(e)}")
            return [[] for _ in query_texts]
        
        # 验证数量
        if len(embeddings) != len(query_texts):
            print(f"⚠️  Embedding数量不匹配: 期望{len(query_texts)}, 实际{len(embeddings)}")
            return [[] for _ in query_texts]
        
        # 🔑 并发检索所有向量（在Qdrant中快速查询）
        tasks = [
            self._search_by_embedding(
                query_embedding=emb,
                corpus_id=corpus_id,
                limit=limit,
                score_threshold=score_threshold
            )
            for emb in embeddings
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️  句子{i}检索失败: {str(result)}")
                final_results.append([])
            else:
                final_results.append(result)
        
        return final_results
    
    async def _search_by_embedding(
        self,
        query_embedding: List[float],
        corpus_id: Optional[str],
        limit: int,
        score_threshold: float
    ) -> List[Dict]:
        """
        使用已有的向量检索（内部方法）
        
        Args:
            query_embedding: 查询向量
            corpus_id: 语料库ID
            limit: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            检索结果列表
        """
        try:
            # 构建过滤器
            query_filter = None
            if corpus_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="corpus_id",
                            match=MatchValue(value=corpus_id)
                        )
                    ]
                )
            
            # 🔑 Qdrant 检索 - 使用新版 query_points API
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,  # 新API使用 query 参数
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 🔑 新API返回的结构可能不同，需要适配
            # 检查返回结构
            if hasattr(results, 'points'):
                # 新版本返回 ScoredPoint 对象列表
                return [
                    {
                        "source": point.payload["source"],
                        "target": point.payload["target"],
                        "score": point.score,
                        "corpus_id": point.payload.get("corpus_id", "unknown")
                    }
                    for point in results.points
                ]
            else:
                # 旧版本或直接返回列表
                return [
                    {
                        "source": result.payload["source"],
                        "target": result.payload["target"],
                        "score": result.score,
                        "corpus_id": result.payload.get("corpus_id", "unknown")
                    }
                    for result in results
                ]
            
        except Exception as e:
            raise Exception(f"Qdrant检索失败: {str(e)}")
    
    def delete_corpus(self, corpus_id: str) -> Dict:
        """
        删除指定语料库
        
        Args:
            corpus_id: 语料库ID
        
        Returns:
            {"success": True, "corpus_id": "..."}
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="corpus_id",
                            match=MatchValue(value=corpus_id)
                        )
                    ]
                )
            )
            return {"success": True, "corpus_id": corpus_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_corpus_stats(self, corpus_id: Optional[str] = None) -> Dict:
        """
        获取语料库统计信息
        
        Args:
            corpus_id: 语料库ID（可选）
        
        Returns:
            {"total_entries": N, "vector_size": 1024}
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_entries": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_point_id(self) -> int:
        """生成唯一的 point ID"""
        import time
        import random
        return int(time.time() * 1000000) + random.randint(0, 999999)