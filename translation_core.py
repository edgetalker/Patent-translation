"""
翻译核心模块
处理长文档翻译和术语一致性验证
"""
import time
import concurrent.futures
from typing import Dict, List, Tuple
from openai import OpenAI

from config import config
from utils import split_text_by_paragraph
from terminology_extraction import TerminologyExtractor


class DocumentTranslator:
    """文档翻译器"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model_name = config.LLM_MODEL_NAME
        self.term_extractor = TerminologyExtractor()
    
    def translate_chunk(
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
        """
        翻译单个文本块
        
        Args:
            chunk_text: 待翻译文本
            chunk_id: 当前块ID
            total_chunks: 总块数
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            domain: 领域信息
            term_dict: 术语对照字典
            context: 前文上下文
            
        Returns:
            翻译结果
        """
        chunk_start = time.time()
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
        
        prompt_parts = [
            f"你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。",
            f"\n当前正在翻译第 {chunk_id + 1} 段，共 {total_chunks} 段。"
        ]
        
        #  简化匹配 + 提示词驱动
        if term_dict:
            # 快速精确匹配（仅用于统计）
            chunk_lower = chunk_text.lower()
            quick_matches = sum(1 for term in term_dict.keys() if term.lower() in chunk_lower)
            
            # 提供完整术语表给LLM
            terms_list = "\n".join([f"  - {src} → {tgt}" for src, tgt in term_dict.items()])
            
            prompt_parts.append(
                f"\n【术语表】以下是{domain}领域的专业术语对照表（共{len(term_dict)}个）：\n{terms_list}\n"
                f"\n【重要翻译要求】"
                f"\n1. 当待翻译文本中出现术语表中的术语（包括其变形、复数、时态、词组等任何形式）时，必须严格使用指定的{tgt_lang_name}翻译"
                f"\n2. 注意识别术语的各种变体形式："
                f"\n   - 英文：单复数变化、动词时态、派生词等（如 optimize/optimizes/optimization 都对应同一术语）"
                f"\n   - 中文：词组包含关系（如 '机器学习算法' 包含 '机器学习' 术语）"
                f"\n   - 其他语言：根据该语言的语法特点灵活匹配"
                f"\n3. 即使术语在原文中只以部分形式出现，也要保持译文的术语一致性"
                f"\n4. 对于多词术语，确保整体翻译的准确性"
            )
            
            # 🔍 简洁诊断日志
            print(f" Chunk {chunk_id+1}: 术语表{len(term_dict)}个, 精确匹配{quick_matches}个 → LLM将灵活匹配全部")
        
        # 添加上下文
        if context:
            prompt_parts.append(f"\n【前文参考】\n{context[:200]}...\n")
        
        # 添加翻译要求
        prompt_parts.append(
            f"\n请将以下{src_lang_name}内容翻译成{tgt_lang_name}："
            "\n1. 严格遵守上述术语表的翻译规范"
            "\n2. 保持文档的专业性和准确性"
            "\n3. 保持原文的段落结构"
            "\n4. **禁止使用任何格式标记**：不要输出markdown格式（如 ** __ * _ # 等）、HTML标签或其他任何格式符号"
            "\n5. 输出纯文本译文，不要添加任何解释、说明或格式修饰"
            f"\n\n【待翻译内容】\n{chunk_text}"
        )
        
        prompt = "".join(prompt_parts)
        prompt_time = time.time() - chunk_start
        
        for attempt in range(config.MAX_RETRIES):
            try:
                api_start = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位专业的{domain}领域翻译专家，擅长{src_lang_name}到{tgt_lang_name}的翻译。你对术语的各种变形形式有深刻理解，能够灵活匹配并保持翻译一致性。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TRANSLATION_TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
                api_time = time.time() - api_start
                translation = response.choices[0].message.content.strip()
                total_time = time.time() - chunk_start
                
                print(f"   ⏱️  Chunk {chunk_id+1} 耗时: Prompt构建{prompt_time:.2f}s + API调用{api_time:.2f}s = {total_time:.2f}s")
                print(f"      输入{len(chunk_text)}字 → 输出{len(translation)}字 ({len(translation)/len(chunk_text):.2f}x)")
                
                return translation
                
            except Exception as e:
                print(f"  ⚠️  翻译chunk {chunk_id + 1} 失败 (尝试 {attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"  ❌ 翻译chunk {chunk_id + 1} 最终失败，返回原文")
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
            
            tgt_variants = [
                tgt_term,
                tgt_term.lower(),
                tgt_term.capitalize(),
            ]
            
            if tgt_lang == 'en':
                if tgt_term.endswith('y'):
                    tgt_variants.append(tgt_term[:-1] + 'ies')
                else:
                    tgt_variants.append(tgt_term + 's')
            
            found = any(variant in translation for variant in tgt_variants)
            
            if not found:
                words = tgt_term.lower().split()
                if len(words) > 1:
                    all_words_present = all(word in translation.lower() for word in words)
                    if not all_words_present:
                        inconsistencies.append(f"{src_term} -> {tgt_term}")
                else:
                    inconsistencies.append(f"{src_term} -> {tgt_term}")
        
        is_consistent = len(inconsistencies) == 0
        return is_consistent, inconsistencies
    
    def translate_document(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        domain: str = "技术",
        use_context: bool = True,
        glossary: Dict[str, str] = None,
        parallel: bool = True,
        max_workers: int = 3
    ) -> Dict:
        """
        翻译长文档（带术语一致性处理）
    
        Args:
            src_text: 源文本
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            domain: 领域信息
            use_context: 是否使用上下文管理（并行模式下自动禁用）
            glossary: 术语对照字典
            parallel: 是否启用并行翻译
            max_workers: 并行翻译的最大工作线程数
        
        Returns:
            result: 包含translation, term_dict, chunks_info, statistics
        """
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
    
        print(f"\n{'='*60}")
        print(f"开始翻译长文档（{src_lang_name} → {tgt_lang_name}，域：{domain}）")
        print(f"{'='*60}\n")
    
        start_time = time.time()
    
        # Step 1 & 2: 术语处理
        if glossary:
            print("📌 步骤1-2: 使用传入的术语表...")
            term_dict = glossary
            terms = list(glossary.keys())
            print(f"\n   ✅ 使用 {len(term_dict)} 个预定义术语")
        else:
            print("📌 步骤1: 使用滑动窗口抽取关键术语...")
            terms = self.term_extractor.sliding_window_extract(src_text, src_lang, domain)
            print(f"\n   ✅ 最终提取 {len(terms)} 个关键术语")
            
            print(f"\n📌 步骤2: 将术语翻译成{tgt_lang_name}...")
            term_dict = self.term_extractor.translate_terminology(terms, src_lang, tgt_lang, domain)
            print(f"   成功翻译 {len(term_dict)} 个术语")
    
        # Step 3: 文档分块
        print("\n📌 步骤3: 文档分块...")
        chunks = split_text_by_paragraph(src_text, config.MAX_CHUNK_LENGTH)
        print(f"   文档已分为 {len(chunks)} 个块")
    
        # Step 4: 翻译
        if parallel and len(chunks) > 2:
            print(f"\n📌 步骤4: 并行翻译（{max_workers}线程并发，LLM智能术语匹配）...")
            print(f"   ⚡ 预计耗时约为顺序翻译的 {1/min(max_workers, len(chunks)):.0%}\n")
            
            translations = [None] * len(chunks)
            
            def translate_task(idx):
                chunk = chunks[idx]
                return self.translate_chunk(
                    chunk_text=chunk["text"],
                    chunk_id=idx,
                    total_chunks=len(chunks),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    term_dict=term_dict,
                    context=None
                )
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(translate_task, i): i
                    for i in range(len(chunks))
                }
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        translations[idx] = future.result()
                        completed += 1
                        print(f"\n   ✓ 完成 {completed}/{len(chunks)} 个chunks")
                    except Exception as e:
                        print(f"   ❌ Chunk {idx+1} 失败: {e}")
                        translations[idx] = f"[TRANSLATION FAILED]"
                        completed += 1
        else:
            print(f"\n📌 步骤4: 顺序翻译（LLM智能术语匹配）...")
            translations = []
            context = None
            
            for i, chunk in enumerate(chunks):
                print(f"\n   翻译 Chunk {i+1}/{len(chunks)}...")
                
                if use_context and i > 0:
                    prev_translation = translations[-1]
                    context = prev_translation[-config.OVERLAP_LENGTH:] if len(prev_translation) > config.OVERLAP_LENGTH else prev_translation
                else:
                    context = None
                
                translation = self.translate_chunk(
                    chunk_text=chunk["text"],
                    chunk_id=i,
                    total_chunks=len(chunks),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    term_dict=term_dict,
                    context=context
                )
                
                translations.append(translation)
                print(f"   ✓ 完成，输出长度: {len(translation)} 字符")
    
        # Step 5 & 6: 组装和验证
        print("\n📌 步骤5: 组装翻译结果...")
        full_translation = "\n\n".join(translations)
    
        print("\n📌 步骤6: 验证术语一致性...")
        is_consistent, inconsistencies = self.validate_terminology_consistency(
            full_translation, term_dict, src_text, tgt_lang
        )
    
        if is_consistent:
            print("   ✅ 所有术语翻译一致")
        else:
            print(f"   ⚠️  发现 {len(inconsistencies)} 个术语可能未正确使用")
    
        # 统计信息
        end_time = time.time()
        statistics = {
            "source_length": len(src_text),
            "translation_length": len(full_translation),
            "num_chunks": len(chunks),
            "num_terms_extracted": len(terms),
            "num_terms_translated": len(term_dict),
            "terminology_consistent": is_consistent,
            "num_inconsistencies": len(inconsistencies),
            "time_elapsed": round(end_time - start_time, 2),
            "avg_time_per_chunk": round((end_time - start_time) / len(chunks), 2),
            "glossary_provided": glossary is not None,
            "parallel_enabled": parallel
        }
    
        print(f"\n{'='*60}")
        print(f"翻译完成！")
        print(f"{'='*60}")
        print(f"翻译模式: {'⚡ 并行翻译' if parallel else '🐌 顺序翻译'}")
        print(f"总耗时: {statistics['time_elapsed']} 秒")
        print(f"平均每块耗时: {statistics['avg_time_per_chunk']} 秒\n")
    
        return {
            "translation": full_translation,
            "term_dict": term_dict,
            "chunks_info": [{"chunk_id": c["chunk_id"], "length": len(c["text"])} for c in chunks],
            "statistics": statistics
        }