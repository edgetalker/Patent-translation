"""
翻译核心模块
处理长文档翻译和术语一致性验证
集成语料库检索功能
"""
import time
import asyncio
import concurrent.futures
import threading
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from config import config
from utils import split_text_by_paragraph
from terminology_extraction import TerminologyExtractor

class DocumentTranslator:
    """文档翻译器（支持语料库加速）"""
    
    def __init__(self, corpus_manager=None):
        """
        初始化翻译器
        
        Args:
            corpus_manager: 可选的语料库管理器
        """
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model_name = config.LLM_MODEL_NAME
        self.term_extractor = TerminologyExtractor()
        self.log_lock = threading.Lock()
        
        # 🆕 语料库支持
        self.corpus_manager = corpus_manager
        self.corpus_retriever = None  # 延迟初始化
    
    def _safe_print(self, *args, **kwargs):
        """线程安全的打印"""
        with self.log_lock:
            print(*args, **kwargs)
            
    def _translate_sentences(
        self,
        sentences: List[Tuple[int, str]],
        src_lang: str,
        tgt_lang: str,
        domain: str,
        term_dict: Dict[str, str] = None,
        domain_prompt: str = None,
        chunk_id: int = 0
    ) -> Dict[int, str]:
        """
        翻译句子列表（批量）
        
        Args:
            sentences: [(index, sentence), ...]
            src_lang: 源语言
            tgt_lang: 目标语言
            domain: 领域
            term_dict: 术语字典
            domain_prompt: 领域提示词
            chunk_id: 块ID（用于日志）
        
        Returns:
            {index: translation}
        """
        if not sentences:
            return {}
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
        
        # 拼接所有句子
        sentence_texts = [sent for _, sent in sentences]
        batch_text = "\n".join(sentence_texts)
        
        # 构建提示词
        prompt_parts = [
            f"你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。"
        ]
        
        if term_dict:
            terms_list = "\n".join([f"  - {src} → {tgt}" for src, tgt in term_dict.items()])
            prompt_parts.append(
                f"\n【术语表】以下是专业术语对照表：\n{terms_list}\n"
                f"\n【翻译要求】"
                f"\n1. 严格遵守术语表的翻译规范"
                f"\n2. 保持句子的独立性，每句一行"
                f"\n3. 禁止使用任何格式标记"
                f"\n4. 按原顺序输出译文"
            )
        
        if domain_prompt:
            prompt_parts.append(domain_prompt)
            
        prompt_parts.append(
            f"\n请将以下{len(sentence_texts)}句{src_lang_name}翻译成{tgt_lang_name}，"
            f"每句一行，按顺序输出：\n\n{batch_text}"
        )
        
        prompt = "".join(prompt_parts)
        
        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"你是专业的{domain}领域翻译专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TRANSLATION_TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 分割结果（按行）
            result_lines = [line.strip() for line in result_text.split('\n') if line.strip()]
            
            # 映射回索引
            translations = {}
            for i, (idx, _) in enumerate(sentences):
                if i < len(result_lines):
                    translations[idx] = result_lines[i]
                else:
                    # 翻译结果不足，使用原文
                    translations[idx] = sentences[i][1]
            
            return translations
            
        except Exception as e:
            print(f"  ⚠️  批量翻译失败: {str(e)}")
            # 返回原文
            return {idx: sent for idx, sent in sentences}
    
    def translate_chunk(
        self,
        chunk_text: str,
        chunk_id: int,
        total_chunks: int,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        term_dict: Dict[str, str] = None,
        domain_prompt: str = None,
        context: str = None,
        # 语料库参数
        use_corpus: bool = False,
        corpus_id: str = None,
        corpus_threshold: float = 0.85
    ) -> Tuple[str, Optional[Dict]]:
        """
        翻译单个文本块（支持语料库检索）
        
        Args:
            chunk_text: 待翻译文本
            chunk_id: 当前块ID
            total_chunks: 总块数
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            domain: 领域信息
            term_dict: 术语对照字典
            domain_prompt: 领域提示词
            context: 前文上下文
            use_corpus: 是否使用语料库检索
            corpus_id: 语料库ID
            corpus_threshold: 相似度阈值
            
        Returns:
            (翻译结果, 语料库统计信息 or None)
        """
        chunk_start = time.time()
        
        # 语料库检索分支
        if use_corpus and self.corpus_retriever and corpus_id:
            return self._translate_chunk_with_corpus(
                chunk_text=chunk_text,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                term_dict=term_dict,
                domain_prompt=domain_prompt,
                corpus_id=corpus_id,
                corpus_threshold=corpus_threshold,
                chunk_start=chunk_start
            )
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
        
        prompt_parts = [
            f"你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。",
            f"\n当前正在翻译第 {chunk_id + 1} 段，共 {total_chunks} 段。"
        ]
        
        if term_dict:
            relevant_terms = self._get_relevant_terms(
                chunk_text=chunk_text,
                term_dict=term_dict,
                max_inject=config.MAX_INJECT_TERMS
            )

            chunk_lower = chunk_text.lower()
            exact_matches = sum(1 for src in relevant_terms if src.lower() in chunk_lower)
    
            terms_list = "\n".join([f"  - {src} → {tgt}" for src, tgt in relevant_terms.items()])
    
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
            
            print(f" Chunk {chunk_id+1}: 术语表{len(term_dict)}个, 精确匹配{exact_matches}个 → LLM将灵活匹配全部")
        
        if domain_prompt:
            prompt_parts.append(domain_prompt)

        if context:
            prompt_parts.append(f"\n【前文参考】\n{context[:200]}...\n")
        
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
                
                return translation, None  
                
            except Exception as e:
                print(f"  ⚠️  翻译chunk {chunk_id + 1} 失败 (尝试 {attempt + 1}/{config.MAX_RETRIES}): {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"  ❌ 翻译chunk {chunk_id + 1} 最终失败，返回原文")
                    return f"[TRANSLATION FAILED: {chunk_text}]", None
    
    def _translate_chunk_with_corpus(
        self,
        chunk_text: str,
        chunk_id: int,
        total_chunks: int,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        term_dict: Dict[str, str],
        domain_prompt: str,
        corpus_id: str,
        corpus_threshold: float,
        chunk_start: float
    ) -> Tuple[str, Dict]:
        """
        使用语料库检索的chunk翻译
        
        Returns:
            (翻译结果, 统计信息)
        """
        self._safe_print(f"\n   🔍 Chunk {chunk_id+1}: 检索语料库...")
        
        try:
            retrieval_result = asyncio.run(
                self.corpus_retriever.retrieve_for_chunk(
                    chunk=chunk_text,
                    corpus_id=corpus_id,
                    threshold=corpus_threshold
                )
            )    
        except RuntimeError as e:
            if "cannot be called when another event loop is running" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.corpus_retriever.retrieve_for_chunk(
                            chunk=chunk_text,
                            corpus_id=corpus_id, 
                            threshold=corpus_threshold
                        )
                    )
                    retrieval_result = future.result()
            else:
                raise
        except Exception as e:
            self._safe_print(f"   ⚠️  检索失败: {str(e)}，回退到直接翻译")
            translation, _ = self.translate_chunk(
                chunk_text=chunk_text,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                domain_prompt=domain_prompt,
                term_dict=term_dict,
                use_corpus=False
            )
            return translation, None
        
        # 2. 统计
        self._safe_print(f"      ✓ 分句: {len(retrieval_result.sentences)}句")
        self._safe_print(f"      ✓ 命中: {retrieval_result.hit_count}句 ({retrieval_result.hit_rate*100:.1f}%)")
        self._safe_print(f"      ✓ 未命中: {retrieval_result.miss_count}句")
        
        # 3. 获取未命中的句子
        unmatched = retrieval_result.get_unmatched_sentences()
        
        # 4. 翻译未命中的句子
        llm_translations = {}
        if unmatched:
            self._safe_print(f"      🤖 LLM翻译 {len(unmatched)} 句...")
            llm_start = time.time()

            unmatched_text = "\n".join(sent for _, sent in unmatched)

            relevant_terms = self._get_relevant_terms(
                chunk_text=unmatched_text,
                term_dict=term_dict,
                max_inject=config.MAX_INJECT_TERMS
            ) if term_dict else None

            llm_translations = self._translate_sentences(
                sentences=unmatched,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                term_dict=relevant_terms,
                domain_prompt=domain_prompt,
                chunk_id=chunk_id
            )
            
            llm_time = time.time() - llm_start
            self._safe_print(f"      ✓ LLM翻译耗时: {llm_time:.2f}s")
        else:
            self._safe_print(f"      ⚡ 全部命中，跳过LLM翻译")
        
        # 5. 合并结果
        translation = self.corpus_retriever.merge_translation(
            retrieval_result=retrieval_result,
            llm_translations=llm_translations
        )
        
        total_time = time.time() - chunk_start
        self._safe_print(f"   ⏱️  Chunk {chunk_id+1} 总耗时: {total_time:.2f}s")
        self._safe_print(f"      输入{len(chunk_text)}字 → 输出{len(translation)}字")
        
        # 6. 返回统计信息
        stats = {
            "total_sentences": len(retrieval_result.sentences),
            "hits": retrieval_result.hit_count,
            "misses": retrieval_result.miss_count,
            "hit_rate": retrieval_result.hit_rate,
            "time": total_time
        }
        
        return translation, stats
    
    def _get_relevant_terms(
        self,
        chunk_text: str,
        term_dict: Dict[str, str],
        max_inject: int = 25
    ) -> Dict[str, str]:
        """
        语言无关的术语过滤：精确匹配优先 + 频率保底
        
        Args:
            chunk_text: 当前 chunk 文本
            term_dict: 完整术语字典（已按提取频率排序）
            max_inject: 最大注入数量
        
        Returns:
            过滤后的术语字典
        """
        chunk_lower = chunk_text.lower()
        
        # 第一层：精确子串匹配，语言无关
        matched = {
            src: tgt for src, tgt in term_dict.items()
            if src.lower() in chunk_lower
        }
        
        # 第二层：精确匹配不足上限时，按频率顺序补足
        if len(matched) < max_inject:
            for src, tgt in term_dict.items():
                if src not in matched:
                    matched[src] = tgt
                if len(matched) >= max_inject:
                    break
        
        return matched
    
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
        domain_prompt: str = None,
        parallel: bool = True,
        max_workers: int = 3,
        # 🆕 语料库参数
        corpus_id: Optional[str] = None,
        use_corpus: bool = False,
        corpus_threshold: float = 0.85
    ) -> Dict:
        """
        翻译长文档（带术语一致性处理 + 语料库加速）
    
        Args:
            src_text: 源文本
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            domain: 领域信息
            use_context: 是否使用上下文管理（并行模式下自动禁用）
            glossary: 术语对照字典
            domain_prompt: 领域提示词
            parallel: 是否启用并行翻译
            max_workers: 并行翻译的最大工作线程数
            corpus_id: 语料库ID，不传则自动生成
            use_corpus: 是否使用语料库检索加速
            corpus_threshold: 语料库相似度阈值
        
        Returns:
            result: 包含translation, term_dict, chunks_info, statistics, corpus_stats
        """
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
    
        print(f"\n{'='*60}")
        print(f"开始翻译长文档（{src_lang_name} → {tgt_lang_name}，域：{domain}）")
        if use_corpus and self.corpus_manager:
            print(f"⚡ 语料库加速已启用（阈值: {corpus_threshold}）")
        print(f"{'='*60}\n")
    
        start_time = time.time()
        
        # 初始化语料库检索器
        if use_corpus and self.corpus_manager:
            from corpus_retrieval import CorpusRetriever
            self.corpus_retriever = CorpusRetriever(
                corpus_manager=self.corpus_manager,
                src_lang=src_lang,
                tgt_lang=tgt_lang
            )
            if not corpus_id:
                corpus_id = f"{src_lang}_{tgt_lang}_{domain}"
        else:
            corpus_id = None
    
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
        
        # 语料库统计
        corpus_stats = {
            "enabled": use_corpus and self.corpus_manager is not None,
            "total_sentences": 0,
            "total_hits": 0,
            "total_misses": 0,
            "overall_hit_rate": 0.0
        }
    
        # Step 4: 翻译
        if parallel and len(chunks) > 2:
            mode_desc = "⚡ 并行翻译"
            if use_corpus and corpus_id:
                mode_desc += " + 🔍 语料库检索"
            
            print(f"\n📌 步骤4: {mode_desc}（{max_workers}线程并发）...")
            print(f"   ⚡ 预计耗时约为顺序翻译的 {1/min(max_workers, len(chunks)):.0%}\n")
            
            translations = [None] * len(chunks)
            chunk_stats = [None] * len(chunks)
            
            def translate_task(idx):
                chunk = chunks[idx]
                translation, stats = self.translate_chunk(
                    chunk_text=chunk["text"],
                    chunk_id=idx,
                    total_chunks=len(chunks),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    term_dict=term_dict,
                    domain_prompt=domain_prompt,
                    context=None,
                    use_corpus=use_corpus,
                    corpus_id=corpus_id,
                    corpus_threshold=corpus_threshold
                )
                return translation, stats
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(translate_task, i): i
                    for i in range(len(chunks))
                }
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        translation, stats = future.result()
                        translations[idx] = translation
                        chunk_stats[idx] = stats
                        
                        # 累计语料库统计
                        if stats:
                            corpus_stats["total_sentences"] += stats["total_sentences"]
                            corpus_stats["total_hits"] += stats["hits"]
                            corpus_stats["total_misses"] += stats["misses"]
                        
                        completed += 1
                        print(f"\n   ✓ 完成 {completed}/{len(chunks)} 个chunks")
                    except Exception as e:
                        print(f"   ❌ Chunk {idx+1} 失败: {e}")
                        translations[idx] = f"[TRANSLATION FAILED]"
                        completed += 1
        else:
            mode_desc = "🐌 顺序翻译"
            if use_corpus and corpus_id:
                mode_desc += " + 🔍 语料库检索"
            
            print(f"\n📌 步骤4: {mode_desc}...")
            translations = []
            context = None
            
            for i, chunk in enumerate(chunks):
                print(f"\n   翻译 Chunk {i+1}/{len(chunks)}...")
                
                if use_context and i > 0 and not use_corpus:
                    prev_translation = translations[-1]
                    context = prev_translation[-config.OVERLAP_LENGTH:] if len(prev_translation) > config.OVERLAP_LENGTH else prev_translation
                else:
                    context = None
                
                translation, stats = self.translate_chunk(
                    chunk_text=chunk["text"],
                    chunk_id=i,
                    total_chunks=len(chunks),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    term_dict=term_dict,
                    domain_prompt=domain_prompt,
                    context=context,
                    use_corpus=use_corpus,
                    corpus_id=corpus_id,
                    corpus_threshold=corpus_threshold
                )
                
                translations.append(translation)
                
                # 累计语料库统计
                if stats:
                    corpus_stats["total_sentences"] += stats["total_sentences"]
                    corpus_stats["total_hits"] += stats["hits"]
                    corpus_stats["total_misses"] += stats["misses"]
                
                if not use_corpus:
                    print(f"   ✓ 完成，输出长度: {len(translation)} 字符")
    
        # 计算总命中率
        if corpus_stats["total_sentences"] > 0:
            corpus_stats["overall_hit_rate"] = corpus_stats["total_hits"] / corpus_stats["total_sentences"]
    
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
        
        # 语料库统计输出
        if corpus_stats["enabled"]:
            print(f"语料库加速: 🔍 已启用")
            print(f"  - 总句子数: {corpus_stats['total_sentences']}")
            print(f"  - 命中数: {corpus_stats['total_hits']}")
            print(f"  - LLM翻译数: {corpus_stats['total_misses']}")
            print(f"  - 命中率: {corpus_stats['overall_hit_rate']*100:.1f}%")
            if corpus_stats['total_sentences'] > 0:
                time_saved = statistics['time_elapsed'] * corpus_stats['overall_hit_rate']
                print(f"  - 预计节省时间: ~{time_saved:.1f}秒")
        
        print(f"总耗时: {statistics['time_elapsed']} 秒")
        print(f"平均每块耗时: {statistics['avg_time_per_chunk']} 秒\n")
    
        return {
            "translation": full_translation,
            "term_dict": term_dict,
            "chunks_info": [{"chunk_id": c["chunk_id"], "length": len(c["text"])} for c in chunks],
            "statistics": statistics,
            "corpus_stats": corpus_stats
        }