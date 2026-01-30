"""
术语提取模块
使用滑动窗口从长文档中提取关键术语并翻译
"""

import time
from typing import List, Dict
from collections import Counter
from openai import OpenAI
from config import config

class TerminologyExtractor:
    """术语提取器"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model_name = config.LLM_MODEL_NAME

    def extract_from_window(
            self,
            text: str,
            src_lang: str,
            domain: str,
            window_id: int,
            max_terms: int = 20 
    ) -> List[str]:
        """从单个窗口中抽取术语

        Args:
            text (str): 窗口文本
            src_lang (str): 源语言代码
            domain (str): 领域信息
            window_id (int): 窗口ID
            max_terms(int, optional): 最多抽取术语数量. Defaults to 20.

        Returns:
            List[str]: 术语列表
        """
        src_lang = config.get_language_name(src_lang)

        prompt = f"""你是一位专业的{domain}领域术语专家。请从以下{src_lang}文本中提取关键的专业术语.

要求：
1. 只提取专业技术术语，不要提取通用词汇
2. 优先提取在文中多次出现的术语
3. 提取的术语应该是{domain}领域的标准表达
4. 每个术语长度在2-50个字之间（允许多词短语）
5. 最多提取{max_terms}个术语
6. 按照重要性排序（最重要的在前）
7. 直接输出术语列表，每行一个术语，不要添加编号或者其他说明

文本内容：
{text}

请提取术语："""
        
        for attempt in range(config.MAX_RETRIES):
            try:
                responese = self.client.chat.completions.create(
                    model=self.model_name,
                    message=[
                        {"role": "system", "content": f""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TERMINOLOGY_TEMPERATURE,
                    max_tokens=1000
                )
                result = responese.choices[0].message.content.strip()

                # 解析术语列表
                import re
                terms = []
                for line in result.split('\n'):
                    line = line.strip()
                    line = re.sub(r'^\d+[\.\)、 ]\s', '', line)
                    line = re.sub()

                    if line and 2 <= len(line) <= 50:
                        terms.append(line)
            
                return terms[:max_terms]
        
            except Exception as e:
                print(f"窗口{window_id}提取失败（尝试{attempt+1}/{config.MAX_RETRIES}): {str(e)} ")
                if attempt < config.MAX_RETRIES -1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"窗口{window_id}术语抽取失败")
                    return []
