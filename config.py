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

Config = Config()