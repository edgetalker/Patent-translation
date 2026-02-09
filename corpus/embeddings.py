"""
Embedding 服务封装
负责文本向量化
"""
import httpx
from typing import List, Union
from config import config


class EmbeddingService:
    """封装Embeddings API调用"""
    
    def __init__(self, base_url: str = config.EMBED_BASE_URL):
        """
        初始化 Embedding 服务
        
        Args:
            base_url: Embedding 服务地址
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        is_query: bool = False
    ) -> List[List[float]]:
        """
        获取文本的embedding向量
        
        Args:
            texts: 单个文本或文本列表
            is_query: 是否为查询模式
        
        Returns:
            embedding向量列表
        """
        # 🔑 添加：空值检查
        if not texts:
            return []
        
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 🔑 添加：过滤空字符串
        texts = [t.strip() for t in texts if t and t.strip()]
        
        # 🔑 添加：再次检查过滤后是否为空
        if not texts:
            return []
        
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "texts": texts,
            "is_query": is_query
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                raise Exception(f"Embedding service error: {error_msg}")
            
            embeddings = data["embeddings"]
            
            # 确保返回列表格式
            if embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
            
            return embeddings
            
        except httpx.HTTPStatusError as e:
            # 🔑 增强：HTTP错误处理
            error_detail = e.response.text[:200] if e.response.text else "No details"
            raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        
        except httpx.RequestError as e:
            # 🔑 增强：网络错误处理
            raise Exception(f"Network error: {type(e).__name__}: {str(e)}")
        
        except KeyError as e:
            # 🔑 增强：响应解析错误
            raise Exception(f"Response parsing error, missing key: {e}")
        
        except Exception as e:
            # 🔑 增强：其他错误
            raise Exception(f"Failed to get embeddings: {type(e).__name__}: {str(e)}")
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()