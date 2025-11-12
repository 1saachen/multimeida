"""
配置文件
"""

import os


class Config:
    """配置类"""

    # 通义千问API配置
    QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY", "your_api_key_here")
    QIANWEN_BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    QIANWEN_MODEL = "qwen-turbo"

    # 生成参数
    GENERATION_TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    TIMEOUT = 60

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    # 质量检查阈值
    MIN_QUALITY_SCORE = 70
    MIN_ENGLISH_WORDS = 100
    MIN_SUGGESTIONS = 3

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"