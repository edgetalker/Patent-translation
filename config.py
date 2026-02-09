"""
配置管理模块 - 优化版（基于30784 tokens上下文）
"""
import os
from typing import Dict
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class Config:
    """全局配置类"""
    
    # ==================== API服务配置 ====================
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))
    
    # ==================== LLM服务配置 ====================
    LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://117.50.221.70:8000/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5-14b")

    # ==================== Embedding服务配置 ==================
    EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://103.234.21.156:11001/api")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
    
    # ==================== 模型上下文配置 ====================
    MODEL_CONTEXT_WINDOW = 64000    # 实际可用上下文
    SYSTEM_PROMPT_OVERHEAD = 3000   # 系统prompt + 术语表开销
    SAFE_MARGIN = 0.85              # 安全余量（留15%缓冲）
    
    # ==================== 术语提取配置（平衡精度和速度）====================
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "8000"))           
    WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2000"))     
    MAX_TERMS = int(os.getenv("MAX_TERMS", "60"))                 
    MIN_TERM_FREQUENCY = int(os.getenv("MIN_TERM_FREQUENCY", "1")) # 最低出现1次
    
    # ==================== 翻译分块配置（精度优先）====================
    MAX_CHUNK_LENGTH = int(os.getenv("MAX_CHUNK_LENGTH", "10000"))  
    OVERLAP_LENGTH = int(os.getenv("OVERLAP_LENGTH", "500"))        
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))    

    # ==================== Qdrant配置 ==================
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = "patent_translations"          

    # ==================== 性能配置 ====================
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
    TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
    TERMINOLOGY_TEMPERATURE = float(os.getenv("TERMINOLOGY_TEMPERATURE", "0.3"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))  # 10分钟超时
    
    # ==================== 性能优化配置 ====================
    AUTO_CHUNK_SIZING = True        # 启用动态chunk计算
    MIN_CHUNK_LENGTH = 5000         # 最小chunk（避免过度分块）
    MAX_OUTPUT_RATIO = 2.5          
    
    # ==================== 语言映射 ====================
    LANGUAGE_NAMES = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韩文",
        "de": "德文",
        "fr": "法文",
        "es": "西班牙文"
    }
    
    @classmethod
    def get_language_name(cls, lang_code: str) -> str:
        """获取语言名称"""
        return cls.LANGUAGE_NAMES.get(lang_code, lang_code)
    
# 创建全局配置实例
config = Config()