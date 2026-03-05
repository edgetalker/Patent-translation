"""
Embedding 服务封装
负责文本向量化
兼容部署调用以及OpenAI API格式调用
"""
import httpx
from typing import List, Union
from urllib.parse import urlparse
from openai import OpenAI

class EmbeddingService:
    """封装Embeddings API调用"""
    
    def __init__(self, base_url: str | None = None):
        """
        初始化 Embedding 服务
        
        Args:
            base_url: Embedding 服务地址
        """
        if not base_url:
            from config import config
            self.base_url = getattr(config, 'EMBED_BASE_URL')
            self.api_key = getattr(config, 'EMBED_API_KEY', None)
        else:
            self.base_url = base_url
            self.api_key = None

        self.service_type = None  # 'openai' 或 'private'
        self.model = None
        self.embeddings_url = None

        # 解析embedding模型
        self._parse_model()    
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: str | None = None,
    ) -> List[List[float]]:
        """
        获取文本的embedding向量
        """
        # 空值检查
        if not texts:
            return []
        
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 过滤空字符串
        texts = [t.strip() for t in texts if t and t.strip()]
        
        if self.service_type == 'openai':
            client = OpenAI(
                base_url=self.embeddings_url,
                api_key=self.api_key,
            )
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                embeddings: List[List[float]] = [
                    item.embedding for item in sorted(response.data, key=lambda x: x.index)
                ]
                return embeddings
            except Exception as e:
                raise Exception(f"API调用失败: {str(e)}")
        else:
            self.get_local_embeddings(texts)
        
    async def _get_local_embeddings(
        self, 
        texts: Union[str, List[str]], 
        is_query: bool = False
    ) -> List[List[float]]:
        """
        私有服务端口调用
        
        Args:
            texts: 单个文本或文本列表
            is_query: 是否为查询模式
        
        Returns:
        {
            "embeddings": embedding向量列表
            "model": embedding 模型
            "success": 响应标识
        }            
        """        
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
        
    def _parse_model(self):
        """
        解析base_url，判断服务类型并设置model
        """
        if not self.base_url:
            raise ValueError("base_url不能为空")
        
        # 解析URL
        parsed = urlparse(self.base_url)
        domain = parsed.netloc.lower()
        
        # 判断服务类型
        if 'openai.com' in domain:
            self.service_type = 'openai'
            # OpenAI官方API默认模型
            self.model = 'text-embedding-3-small'
            # 构建embeddings地址
            if self.base_url.endswith('/v1'):
                self.embeddings_url = f"{self.base_url}/embeddings"
            else:
                self.embeddings_url = f"{self.base_url}/v1/embeddings"
        
        elif 'modelscope' in domain:
            self.service_type = 'openai'
            # 魔搭平台 
            self.model = 'Qwen/Qwen3-Embedding-0.6B'
            self.embeddings_url = self.base_url
        
        else:
            # 默认为私有服务
            self.service_type = 'private'
            self.model = 'default-model'
            self.embeddings_url = self.base_url
