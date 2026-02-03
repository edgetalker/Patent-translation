"""
翻译核心模块
翻译长文本段和术语一致性验证 
"""
import time
from openai import OpenAI
from typing import Dict, Tuple, List

from config import config
from terminology_extraction import TerminologyExtractor

class DocumentTranslator:
    """文档翻译器"""

    def __init__(self):
        self.client = OpenAI(
            api_key = config.API_KEY,
            base_url = config.LLM_BASE_URL
        )
        self.model_name = config.LLM_MODEL_NAME
        self.term_extractor = TerminologyExtractor()
    
    def translate_chunck(
            self,
            chunk_text: str,
            chunk_id: int,
            total_chunks: int,
            src_lang: str,
            tgt_lang: str,
            domain: str,
            term_dict: Dict[str, str] = None,
            context: str = None
    ) -> str:
        """翻译单个文本块

        Args:
            chunk_text (str): 待翻译本块
            chunk_id (int): 当前块id
            total_chunks (int): 总块数
            src_lang (str): 源语言
            tgt_lang (str): 目标语言
            domain (str): 领域信息
            term_dict (Dict[str, str], optional): 术语对照字典. Defaults to None.
            context (str, optional): 上下文信息. Defaults to None.

        Returns:
            str: 翻译结果
        """
        start_time = time.time()

        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)

        prompt_parts = [
            f"你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。",
            f"\n当前正在翻译第{chunk_id}段, 共{total_chunks}段。"
        ]

        if term_dict:
            # 快速精确匹配
            chunk_lower = chunk_text.lower()
            quick_matches = sum(1 for term in term_dict.keys() if term.lower() in chunk_lower)

            # 提供完整的术语表给LLM
            term_list = "\n".join([f"   - {src} -> {tgt}" for src, tgt in term_dict.items()])

            prompt_parts.append(
                f"\n 【术语表】以下是{domain}领域的专业术语对照表（共{len(term_dict)}个）: \n{term_list}\n"
                f"\n 【重要翻译要求】"
                f"\n1. 当待翻译文本中出现术语表中的术语（包括其变形、复数、时态、词组等任何形式）时, 必须严格使用指定的{tgt_lang_name}翻译"
                f"\n2. 注意识别术语的各种变体形式: "
                f"\n   - 英文: 单复数变化、动词时态、派生词等（如 optimize/optimizes/optimization 都对应同一术语）"
                f"\n   - 中文: 词组包含关系（如'机器学习算法' 包含 '机器学习' 术语）"
                f"\n   - 其他语言: 根据该语言的语法特点灵活匹配"
                f"\n3. 即使术语在原文中只以部分形式出现, 也要保持译文术语的一致性"
                f"\n4. 对于多词术语, 确保整体翻译的准确性"
            )

            print(f" Chunk {chunk_id+1}: 术语表{len(term_dict)}个，精确匹配{quick_matches}个 -> LLM将灵活匹配全部")

        if context:
            prompt_parts.append(f"\n 【前文参考】\n{context[:200]}...\n")
        
        # 添加翻译要求
        prompt_parts.append(
            f"\n请将以下{src_lang_name}内容翻译成{tgt_lang_name}："
            "\n1. 严格遵守上述术语表的翻译规范"
            "\n2. 保持文档的专业性和准确性"
            "\n3. 保持原文的段落结构"
            "\n4. **禁止使用任何格式标记**：不要输出markdown格式（如 ** __ * _ #等）、HTML标签或其他任何格式字符"
            "\n5. 输入纯文本译文，不要添加任何解释、说明或格式修饰"
            f"\n\n 【待翻译内容】\n{chunk_text}"
        )

        prompt = "".join(prompt_parts)
        prompt_time = time.time() - start_time

        for attempt in range(config.MAX_RETRIES):
            try:
                api_start = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位专业的{domain}领域翻译专家，擅长{src_lang_name}到{tgt_lang_name}的翻译。你对术语的各种变形形式有深刻理解, 能够灵活匹配并保持术语一致性。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TRANSLATION_TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )

                api_time = time.time() - api_start
                translation = response.choices[0].message.content.strip()
                total_time = time.time() - start_time

                print(f"    Chunk {chunk_id+1} 耗时: prompt构建{prompt_time:.2f}s + API调用{api_time:.2f}s = {total_time:.2f}s")
                print(f"    输入{len(chunk_text)}字 -> 输出{len(translation)}字 ({len(translation) / len(chunk_text):.2f}x)")

                return translation
            
            except Exception as e:
                print(f" 术语翻译失败（尝试{attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES -1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"    翻译chunk {chunk_id+1} 最终失败，返回原文")
                    return f"[TRANSLATION FAILED: {chunk_text}]"
            
    def validate_terminology_consistency(
            self,
            translation: str,
            term_dict: Dict[str, str],
            src_text: str,
            tgt_lang: str
    ) -> Tuple[bool, List[str]]:
        """验证翻译结果中的术语一致性"""
        inconsistencies = []

        for src_term, tgt_term in term_dict.items():
            if src_term not in src_text:
                continue

    def translate_document(
            self,
            src_text: str,
            src_lang: str,
            tgt_lang: str,
            domain: str = "技术",
            use_content: bool = True,
            glossary: Dict[str, str] = None,
            parallel: bool = True,
            max_workers: int = 3
    ) -> Dict:
        """翻译长文本, 术语一致性检查

        Args:
            src_text (str): 源文本
            src_lang (str): 源文本语言
            tgt_lang (str): 目标语言
            domain (str, optional): 领域信息. Defaults to "技术".
            use_content (bool, optional): 是否使用上下文管理. Defaults to True.
            glossary (Dict[str, str], optional): 术语表. Defaults to None.
            parallel (bool, optional): 是否启用并行翻译. Defaults to True.
            max_workers (int, optional): 并行翻译的最大工作线程数. Defaults to 3.

        Returns:
            Dict: 翻译结果
        """
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)

        start_time = time.time()

        # Step 1 & 2: 术语处理


        # Step 3: 文档分块


        # Step 4: 翻译


        # Step 5 & 6: 组装和验证
          