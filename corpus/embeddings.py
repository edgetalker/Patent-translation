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

        self.service_type = None 
        self.model = None
        self.embeddings_url = None

        self.client = httpx.AsyncClient(timeout=60.0)

        # 解析embedding模型
        self._parse_model()    
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: str | None = None,
        is_query: bool = False, 
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
            return await self._get_local_embeddings(texts, is_query=is_query)
        
    async def _get_local_embeddings(
        self,
        texts: Union[str, List[str]],
        is_query: bool = False
    ) -> List[List[float]]:
        """私有服务端口调用 (含分批/节流/重试)"""
        import asyncio
        
        if not texts:
            return []
        
        # ─── 自动分批 + 节流 + 重试 ─────────────────────────
        BATCH_SIZE = 32
        INTER_BATCH_SLEEP = 0.3   # 每个 sub-batch 间隔
        MAX_RETRIES = 3
        
        if len(texts) > BATCH_SIZE:
            n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[embed] 批量 {len(texts)} 条 > {BATCH_SIZE},分 {n_batches} 批 (节流 {INTER_BATCH_SLEEP}s)")
            
            all_embeddings = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                
                # 重试逻辑
                last_err = None
                for attempt in range(MAX_RETRIES):
                    try:
                        sub = await self._get_local_embeddings(batch, is_query=is_query)
                        all_embeddings.extend(sub)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < MAX_RETRIES - 1:
                            wait = 1.5 ** attempt   # 1.0s, 1.5s
                            print(f"[embed] batch {i//BATCH_SIZE+1}/{n_batches} 第 {attempt+1} 次失败 ({type(e).__name__}),{wait:.1f}s 后重试")
                            await asyncio.sleep(wait)
                
                if last_err is not None:
                    print(f"[embed] ❌ batch {i//BATCH_SIZE+1}/{n_batches} 重试 {MAX_RETRIES} 次后仍失败")
                    raise last_err
                
                # 节流
                await asyncio.sleep(INTER_BATCH_SLEEP)
            
            return all_embeddings
        # ──────────────────────────────────────────────────
        
        # 单批走原逻辑
        url = f"{self.base_url}/embeddings"
        payload = {"texts": texts, "is_query": is_query}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                print(f"⚠️ 服务端软失败 raw response: {data}")
                raise Exception(f"Embedding service error: {error_msg} | raw={data}")
            
            embeddings = data["embeddings"]
            if embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
            return embeddings
        
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:200] if e.response.text else "No details"
            raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
        except httpx.RequestError as e:
            raise Exception(f"Network error: {type(e).__name__}: {str(e)}")
        except KeyError as e:
            raise Exception(f"Response parsing error, missing key: {e}")
        except Exception as e:
            # 透传 Embedding service error,不再二次包装
            if "Embedding service error" in str(e):
                raise
            raise Exception(f"Failed to get embeddings: {type(e).__name__}: {repr(e)}")
        
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
            self.model = 'Qwen3-Embedding-0.6B'
            self.embeddings_url = self.base_url
