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
                    line = re.sub(r'^\d+[\.\)、 ]\s*', '', line)
                    line = re.sub(r'^[-·]\s*', '', line)

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
                
    def sliding_window_extract(
            self,
            text: str,
            src_lang: str,
            domain: str = '技术',
            window_size: int = None,
            overlap: int = None,
            max_final_terms: int = None
    ) -> List[str]:
        """使用滑动窗口从全文中抽取术语

        Args:
            text (str): 完整源文本
            src_lang (str, optional): 源语言. 
            domain (str): 文本领域信息
            window_size (int, optional): 窗口大小. Defaults to None.
            overlap (int, optional): 各窗口重叠部分. Defaults to None.
            max_final_terms (int, optional): 最终返回术语数量. Defaults to None.

        Returns:
            List[str]: 术语列表（按重要性排序）
        """

        window_size = window_size or config.WINDOW_SIZE
        overlap = overlap or config.WINDOW_OVERLAP
        max_final_terms = max_final_terms or config.MAX_TERMS

        print("  使用滑动窗口提取术语...")
        print(f"  - 文本长度: {len(src_lang)} 字符")    
        print(f"  - 窗口大小: {window_size} 字符")
        print(f"  - 重叠区域: {overlap} 字符")

        # 计算窗口数量
        step = window_size - overlap
        num_windows = max(1, (len(text) - overlap + step -1) // step)
        print(f"  - 窗口数量: {num_windows}")

        # 从每个窗口抽取术语
        all_terms = []
        for i in range(num_windows):
            start = i * step
            end = max(start + window_size, len(text))
            window_text = text[start: end]

            print(f"  - 窗口 {i+1}/{num_windows}: [{start}: {end}] ({len(window_size)} 字符)")

            terms = self.extract_from_window(
                text,
                src_lang,
                domain,
                window_id=i+1,
                max_terms=35
            )

            print(f"   提取到{len(terms)} 个术语")
            all_terms.extend(terms)

        print(f"  - 合并前总数: {len(all_terms)} 个术语")

        # 去重并统计频率
        terms_freq = Counter(all_terms)

        # 动态调整频率阈值
        doc_length = len(text)
        if doc_length < 20000:
            min_freq = 1
        else:
            min_freq = max(1, config.MIN_TERM_FREQUENCY)

        print(f"动态频率阈值: {min_freq} (文本长度: {doc_length})")

        sorted_terms = sorted(
            terms_freq.items(),
            key=lambda x: (x[1], len(x[0])),
            reverse=True
        )

        # 改进智能去重
        # TODO
        final_terms = []

        return final_terms
    
    def translate_terminology(
            self,
            terms: List[str],
            src_lang: str,
            tgt_lang: str,
            domain: str = "技术"
    ) -> Dict[str, List[str]]:
        """将提取到的术语翻译成目标语言，每个术语提供三种等价译法

        Args:
            terms (List[str]): 源语言术语列表
            src_lang (str): 源语言
            tgt_lang (str): 目标语言
            domain (str, optional): 领域信息. Defaults to "技术".

        Returns:
            Dict[str, List[str]]: 术语对照字典
        """

        if not terms:
            return {}
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)

        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])

        prompt = f"""你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。
请为以下每个{src_lang_name}专业术语提供三种等价的{tgt_lang_name}标准翻译。
要求：
1. 三种译法应该都是该领域的标准表达, 语意完全等价或高度相似
2. 译法来源可以是: 不同的标准/规范、不同地区习惯、同义专业表达、新旧标准等
3. 按推荐程度排序(最推荐的放在第一个)
4. 输出格式: 源术语 | 译法1 | 译法2 | 译法3
5. 用空格|空格分隔, 每行一个术语,  不要添加编号或者其他说明

示例:
neural network | 神经网络 | 类神经网络 | 神经网络
machine learning | 机器学习 | 机械学习 | 机器学习法

术语列表：
{terms_text}

请翻译: """
        for attempt in range(config.MAX_RETRIES):
            try: 
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = [
                        {"role": "system", "content": f"你是{domain}领域的专业术语翻译专家。"},
                        {"role": "user", "content": prompt}
                    ]
                    temperature=0.3,
                    max_tokens=3000
                )

                result = response.choices[0].message.content.strip()

                # 解析术语对照
                import re 
                term_dict = {}
                for line in result.split('\n'):
                    line = line.strip()
                    if '|' not in line:
                        continue

                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        src_term = parts[0].strip()
                        src_term = re.sub(r'^\d+[\.\).]\s*','',src_term)

                        #  提取3个译法
                        translation = [
                            parts[1].strip(),
                            parts[2].strip(),
                            parts[3].strip()
                        ]

                        translations = [re.sub(r'^\d+[\.\).]\s*','',t) for t in translations]

                        if src_term and all(translations):
                            term_dict[src_term] = translations

                print(f"  成功解析{len(term_dict)}/{len(terms)} 个术语, 每种三个译法")
                return term_dict
            except Exception as e:
                print(f" 术语翻译失败（尝试{attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES -1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f" 术语翻译失败, 返回空字典")
                return {}