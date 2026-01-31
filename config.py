"""
配置管理模块    
"""

import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class Config:
    """全局配置类"""

    # API服务配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = os.getenv("API_PORT", "8080")

    # LLM服务配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://117.50.221.70:8080/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5-14b")

    # 模型上下文配置
    MODEL_CONTEXT_WINDOW = 64000
    SYSTEM_PROMPT_OVERHEAD = 3000
    SAFE_MARGIN = 0.85

    # 术语提取配置
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "8000"))
    WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2000"))
    MAX_TERMS = int(os.getenv("MAX_TERMS", "60"))
    MIN_TERM_FREQUENCY = int(os.getenv("MIN_TERM_FREQUENCY", 1))

    # 翻译分块设置
    MAX_CHUNK_LENGTH = int(os.getenv("MAX_CHUNK_LENGTH", "10000"))
    OVERLAP_LENGTH = int(os.getenv("OVERLAP_LENGTH", "500"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "20000"))

    # 语言映射
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
    
    
Config = Config()