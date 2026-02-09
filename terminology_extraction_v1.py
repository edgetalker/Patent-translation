"""
术语提取模块
使用滑动窗口从长文档中提取关键术语并翻译
"""
import time
from typing import List, Dict
from collections import Counter
from openai import OpenAI
import re
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
        """
        从单个窗口中抽取术语
        
        Args:
            text: 窗口文本
            src_lang: 源语言代码
            domain: 领域信息
            window_id: 窗口ID
            max_terms: 最多抽取术语数量
            
        Returns:
            术语列表
        """
        src_lang_name = config.get_language_name(src_lang)
        
        prompt = f"""你是一位专业的{domain}领域术语专家。请从以下{src_lang_name}文本中提取关键的专业术语。

要求：
1. 只提取专业技术术语，不要提取通用词汇
2. 优先提取在文中多次出现的术语
3. 提取的术语应该是{domain}领域的标准表达
4. 每个术语长度在2-50个字之间（允许多词短语）
5. 最多提取{max_terms}个术语
6. 按照重要性排序（最重要的在前）
7. 直接输出术语列表，每行一个术语，不要添加编号或其他说明

文本内容：
{text}

请提取术语："""

        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位{domain}领域的术语提取专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TERMINOLOGY_TEMPERATURE,
                    max_tokens=1000
                )
                
                result = response.choices[0].message.content.strip()
                
                # 解析术语列表
                import re
                terms = []
                for line in result.split('\n'):
                    line = line.strip()
                    line = re.sub(r'^\d+[\.\)、]\s*', '', line)
                    line = re.sub(r'^[-•]\s*', '', line)
                    
                    # 放宽长度限制到50字符
                    if line and 2 <= len(line) <= 50:
                        terms.append(line)
                
                return terms[:max_terms]
                
            except Exception as e:
                print(f"  ⚠️  窗口{window_id}术语抽取失败 (尝试 {attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"  ❌ 窗口{window_id}术语抽取最终失败")
                    return []
    
    def sliding_window_extract(
        self,
        text: str,
        src_lang: str,
        domain: str = "技术",
        window_size: int = None,
        overlap: int = None,
        max_final_terms: int = None
    ) -> List[str]:
        """
        使用滑动窗口从全文抽取术语
        
        Args:
            text: 完整源文本
            src_lang: 源语言代码
            domain: 领域信息
            window_size: 窗口大小（默认使用配置）
            overlap: 重叠区域大小（默认使用配置）
            max_final_terms: 最终返回术语数量（默认使用配置）
            
        Returns:
            术语列表（按重要性排序）
        """
        window_size = window_size or config.WINDOW_SIZE
        overlap = overlap or config.WINDOW_OVERLAP
        max_final_terms = max_final_terms or config.MAX_TERMS
        
        print(f"   使用滑动窗口提取术语...")
        print(f"   - 文本长度: {len(text)} 字符")
        print(f"   - 窗口大小: {window_size} 字符")
        print(f"   - 重叠区域: {overlap} 字符")
        
        # 计算窗口数量
        step = window_size - overlap
        num_windows = max(1, (len(text) - overlap + step - 1) // step)
        print(f"   - 窗口数量: {num_windows}")
        
        # 从每个窗口抽取术语
        all_terms = []
        for i in range(num_windows):
            start = i * step
            end = min(start + window_size, len(text))
            window_text = text[start:end]
            
            print(f"   - 窗口 {i+1}/{num_windows}: [{start}:{end}] ({len(window_text)} 字符)")
            
            terms = self.extract_from_window(
                window_text,
                src_lang,
                domain,
                window_id=i+1,
                max_terms=35
            )
            
            print(f"     提取到 {len(terms)} 个术语")
            all_terms.extend(terms)
        
        print(f"   - 合并前总数: {len(all_terms)} 个术语")
        
        # 去重并统计频率
        term_freq = Counter(all_terms)
        
        # 动态调整频率阈值
        doc_length = len(text)
        if doc_length < 5000:
            min_freq = 1  # 短文档：出现1次即可
        elif doc_length < 20000:
            min_freq = 1  # 中等文档：至少1次
        else:
            min_freq = max(1, getattr(config, 'MIN_TERM_FREQUENCY', 2) - 1)  # 长文档：使用配置值
        
        print(f"   - 动态频率阈值: {min_freq} (文档长度: {doc_length})")
        
        # 按频率排序
        sorted_terms = sorted(
            term_freq.items(),
            key=lambda x: (x[1], len(x[0])),
            reverse=True
        )
        
        #  改进智能去重，避免误删独立术语
        final_terms = []
        seen_terms = set()
        
        for term, freq in sorted_terms:
            if freq < min_freq:
                continue
            
            is_redundant = False
            
            # 检查是否为纯子串（避免误删独立概念）
            for seen_term in list(seen_terms):
                # 只有当term是seen_term的真子串，且不是独立词时才认为冗余
                if term in seen_term and term != seen_term:
                    # 检查是否为独立词（用空格或标点分隔）
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if not re.search(pattern, seen_term, re.IGNORECASE):
                        is_redundant = True
                        break
                
                # 反向：如果seen_term是term的真子串，移除seen_term
                elif seen_term in term and seen_term != term:
                    pattern = r'\b' + re.escape(seen_term) + r'\b'
                    if not re.search(pattern, term, re.IGNORECASE):
                        seen_terms.discard(seen_term)
                        if seen_term in final_terms:
                            final_terms.remove(seen_term)
                        break
            
            if not is_redundant and len(final_terms) < max_final_terms:
                final_terms.append(term)
                seen_terms.add(term)
        
        print(f"   - 去重后术语数: {len(final_terms)}")
        if sorted_terms:
            avg_freq = sum(f for _, f in sorted_terms[:len(final_terms)]) / len(final_terms) if final_terms else 0
            print(f"   - 术语频率分布: 最高{sorted_terms[0][1]}次, 平均{avg_freq:.1f}次")
        
        return final_terms
    
    def translate_terminology(
        self,
        terms: List[str],
        src_lang: str,
        tgt_lang: str,
        domain: str = "技术"
    ) -> Dict[str, str]:
        """
        将提取的术语翻译成目标语言（一对一翻译）
        
        Args:
            terms: 源语言术语列表
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            domain: 领域信息
            
        Returns:
            术语对照字典 {源术语: 译法}
        """
        if not terms:
            return {}
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
        
        # 为每个术语编号，便于解析
        terms_text = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        
        # ✅ 增强提示词，明确格式要求
        prompt = f"""你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。

【任务说明】
请将以下{src_lang_name}专业术语翻译成{tgt_lang_name}。

【重要要求】
1. 每个术语提供一个最准确的{tgt_lang_name}翻译
2. 优先使用国际标准、行业规范中的官方译法
3. 严格按照格式输出，每行一个术语
4. **输出格式**（非常重要）：
   {src_lang_name}术语 | {tgt_lang_name}译法
   
5. 用空格|空格分隔（前后各有一个空格）
6. 不要添加编号、解释或其他内容
7. 保持术语的完整性，不要遗漏或改变

【输出示例】
假设翻译中文术语到英文：
神经网络 | neural network
机器学习 | machine learning
优化算法 | optimization algorithm

假设翻译英文术语到中文：
neural network | 神经网络
machine learning | 机器学习
optimization algorithm | 优化算法

【待翻译的{src_lang_name}术语列表】
{terms_text}

【开始翻译】
请严格按照"{src_lang_name}术语 | {tgt_lang_name}译法"的格式输出："""

        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": f"你是{domain}领域的专业术语翻译专家。你精通{src_lang_name}和{tgt_lang_name}的标准术语翻译，严格按照用户要求的格式输出。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # 低温度确保格式一致
                    max_tokens=3000
                )
            
                result = response.choices[0].message.content.strip()
            
                term_dict = {}
                failed_lines = []
                
                for line_num, line in enumerate(result.split('\n'), 1):
                    line = line.strip()
                    
                    # 跳过空行
                    if not line:
                        continue
                    
                    # 跳过明显的标题或说明行
                    if line.startswith(('【', '#', '注意', 'Note', '说明')):
                        continue
                    
                    # 必须包含分隔符
                    if '|' not in line:
                        failed_lines.append(f"第{line_num}行缺少分隔符: {line}")
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    
                    if len(parts) < 2:
                        failed_lines.append(f"第{line_num}行格式错误: {line}")
                        continue
                    
                    # 提取源术语和译法
                    src_term = parts[0].strip()
                    tgt_term = parts[1].strip()
                    
                    # 清理可能的编号
                    src_term = re.sub(r'^\d+[\.\)、]\s*', '', src_term)
                    tgt_term = re.sub(r'^\d+[\.\)、]\s*', '', tgt_term)
                    
                    # ✅ 验证：源术语必须在原始术语列表中
                    if src_term not in terms:
                        # 尝试模糊匹配
                        matched = False
                        for original_term in terms:
                            if src_term == original_term or src_term in original_term or original_term in src_term:
                                src_term = original_term
                                matched = True
                                break
                        
                        if not matched:
                            failed_lines.append(f"第{line_num}行源术语不在列表中: {src_term}")
                            continue
                    
                    # ✅ 验证：译法不能为空，不能和源术语相同
                    if not tgt_term:
                        failed_lines.append(f"第{line_num}行译法为空: {src_term}")
                        continue
                    
                    if src_term == tgt_term:
                        failed_lines.append(f"第{line_num}行源术语和译法相同: {src_term}")
                        continue
                    
                    # ✅ 验证：语言方向正确
                    # 简单检查：中文应该有汉字，英文应该主要是字母
                    if src_lang == 'zh':
                        # 中文源术语应该包含汉字
                        if not re.search(r'[\u4e00-\u9fff]', src_term):
                            failed_lines.append(f"第{line_num}行源术语应为中文: {src_term}")
                            continue
                    elif src_lang == 'en':
                        # 英文源术语应该主要是字母
                        if not re.search(r'[a-zA-Z]', src_term):
                            failed_lines.append(f"第{line_num}行源术语应为英文: {src_term}")
                            continue
                    
                    # 通过所有验证，添加到字典
                    term_dict[src_term] = tgt_term
                
                # 输出解析结果
                print(f"   ✅ 成功解析 {len(term_dict)}/{len(terms)} 个术语")
                
                if failed_lines:
                    print(f"   ⚠️  解析失败 {len(failed_lines)} 行:")
                    for fail_msg in failed_lines[:5]:  # 只显示前5个
                        print(f"      - {fail_msg}")
                    if len(failed_lines) > 5:
                        print(f"      ... 还有 {len(failed_lines)-5} 个错误")
                
                # ✅ 如果解析成功率太低，重试
                success_rate = len(term_dict) / len(terms) if terms else 0
                if success_rate < 0.5:  # 成功率低于50%
                    print(f"   ⚠️  成功率仅 {success_rate*100:.1f}%，重试...")
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(config.RETRY_DELAY)
                        continue
                
                return term_dict
            
            except Exception as e:
                print(f"  ⚠️  术语翻译失败 (尝试 {attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"  ❌ 术语翻译最终失败，返回空字典")
                    return {}