"""
语料库检索模块
负责句子级别的相似语料检索和结果组织
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
from corpus.manager import CorpusManager


# ==================== 数据结构 ====================

@dataclass
class SentenceMatch:
    """单句匹配结果"""
    index: int              # 句子在chunk中的位置
    source: str             # 源句子
    matched: bool           # 是否命中语料库
    translation: str        # 译文（命中时有值，未命中为空）
    similarity: float       # 相似度分数（0.0-1.0）
    corpus_source: str      # 命中的语料库原文（用于调试）


@dataclass
class RetrievalResult:
    """检索结果"""
    sentences: List[SentenceMatch]
    hit_count: int
    miss_count: int
    hit_rate: float
    
    def get_unmatched_sentences(self) -> List[Tuple[int, str]]:
        """
        获取未命中的句子
        
        Returns:
            List[(index, sentence)]
        """
        return [
            (sent.index, sent.source)
            for sent in self.sentences
            if not sent.matched
        ]
    
    def get_matched_translations(self) -> Dict[int, str]:
        """
        获取已命中句子的翻译
        
        Returns:
            {index: translation}
        """
        return {
            sent.index: sent.translation
            for sent in self.sentences
            if sent.matched
        }


# ==================== 语言配置 ====================

class LanguageConfig:
    """多语言分句配置"""
    
    # 不同语言的句子边界符
    SENTENCE_DELIMITERS = {
        'zh': ['。', '！', '？', '；'],      # 中文
        'en': ['. ', '! ', '? ', '; '],     # 英文（注意空格）
        'ja': ['。', '！', '？', '；'],      # 日文
        'de': ['. ', '! ', '? ', '; '],     # 德语
        'fr': ['. ', '! ', '? ', '; '],     # 法语
    }
    
    # 固定句式模式（用于提高阈值）
    FIXED_PATTERNS = {
        'zh': [
            r'^本发明涉及',
            r'^所述\w+包括',
            r'^根据本发明',
            r'^如图\d+',
            r'^与现有技术相比',
            r'^具体实施方式',
            r'^在.*实施例中',
        ],
        'en': [
            r'^The present invention relates to',
            r'^The \w+ comprises',
            r'^According to the present invention',
            r'^As shown in (FIG\.|Figure)',
            r'^Compared with the prior art',
            r'^In an embodiment',
        ],
        'ja': [
            r'^本発明は',
            r'^前記\w+は',
            r'^図\d+に示す',
        ]
    }
    
    @classmethod
    def get_delimiters(cls, lang: str) -> List[str]:
        """获取语言的分句符"""
        return cls.SENTENCE_DELIMITERS.get(lang, cls.SENTENCE_DELIMITERS['en'])
    
    @classmethod
    def get_fixed_patterns(cls, lang: str) -> List[str]:
        """获取固定句式模式"""
        return cls.FIXED_PATTERNS.get(lang, [])


# ==================== 分句器 ====================

class SentenceSplitter:
    """多语言分句器"""
    
    def __init__(self, lang: str = 'zh'):
        """
        Args:
            lang: 语言代码 ('zh', 'en', 'ja' 等)
        """
        self.lang = lang
        self.delimiters = LanguageConfig.get_delimiters(lang)
    
    def split(self, text: str, min_length: int = 5, max_length: int = 500) -> List[str]:
        """
        将文本分句
        
        Args:
            text: 输入文本
            min_length: 最小句子长度（过滤太短的句子）
            max_length: 最大句子长度（超长句子会被强制分割）
        
        Returns:
            句子列表
        """
        # 按分句符分割
        sentences = self._split_by_delimiters(text)
        
        # 过滤和清理
        sentences = self._clean_sentences(sentences, min_length, max_length)
        
        return sentences
    
    def _split_by_delimiters(self, text: str) -> List[str]:
        """按分句符分割"""
        sentences = [text]
        
        for delimiter in self.delimiters:
            new_sentences = []
            for sent in sentences:
                # 分割并保留分隔符
                parts = sent.split(delimiter)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        # 非最后一部分，加回分隔符
                        new_sentences.append(part + delimiter.strip())
                    else:
                        # 最后一部分
                        if part.strip():
                            new_sentences.append(part)
            sentences = new_sentences
        
        return sentences
    
    def _clean_sentences(
        self, 
        sentences: List[str], 
        min_length: int, 
        max_length: int
    ) -> List[str]:
        """清理句子：去空格、过滤太短、拆分超长"""
        cleaned = []
        
        for sent in sentences:
            # 去除首尾空格
            sent = sent.strip()
            
            # 过滤空句子
            if not sent:
                continue
            
            # 过滤太短的句子
            if len(sent) < min_length:
                continue
            
            # 处理超长句子（强制分割）
            if len(sent) > max_length:
                # 尝试在逗号处分割
                if '，' in sent or ',' in sent:
                    sub_sents = re.split(r'[，,]', sent)
                    cleaned.extend([s.strip() for s in sub_sents if len(s.strip()) >= min_length])
                else:
                    # 无法分割，保留原句
                    cleaned.append(sent)
            else:
                cleaned.append(sent)
        
        return cleaned


# ==================== 语料库检索器 ====================

class CorpusRetriever:
    """语料库检索器（句子级）"""
    
    def __init__(
        self, 
        corpus_manager: Optional[CorpusManager] = None,
        src_lang: str = 'zh',
        tgt_lang: str = 'en'
    ):
        """
        Args:
            corpus_manager: 语料库管理器
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
        """
        self.corpus_manager = corpus_manager
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.splitter = SentenceSplitter(lang=src_lang)
        self.fixed_patterns = LanguageConfig.get_fixed_patterns(src_lang)
    
    async def retrieve_for_chunk(
        self,
        chunk: str,
        corpus_id: str,
        threshold: float = 0.85,
        min_sentence_length: int = 5,
        max_sentence_length: int = 500
    ) -> RetrievalResult:
        """
        对单个chunk进行句子级检索
        
        Args:
            chunk: 文本块
            corpus_id: 语料库ID
            threshold: 相似度阈值
            min_sentence_length: 最小句子长度
            max_sentence_length: 最大句子长度
        
        Returns:
            RetrievalResult
        """
        # 1. 分句
        sentences = self.splitter.split(
            chunk, 
            min_length=min_sentence_length,
            max_length=max_sentence_length
        )
        
        # 过滤空句子并添加索引
        sentences = [
            (idx, sent) 
            for idx, sent in enumerate(sentences) 
            if sent and sent.strip()
        ]
        
        if not sentences:
            return RetrievalResult(
                sentences=[],
                hit_count=0,
                miss_count=0,
                hit_rate=0.0
            )
        
        # 2. 批量检索
        if self.corpus_manager:
            sentence_matches = await self._batch_retrieve(
                sentences=sentences,
                corpus_id=corpus_id,
                threshold=threshold
            )
        else:
            # 无语料库管理器，全部标记为未命中
            sentence_matches = [
                SentenceMatch(
                    index=idx,
                    source=sent,
                    matched=False,
                    translation="",
                    similarity=0.0,
                    corpus_source=""
                )
                for idx, sent in sentences
            ]
        
        # 3. 统计
        hit_count = sum(1 for m in sentence_matches if m.matched)
        miss_count = len(sentence_matches) - hit_count
        hit_rate = hit_count / len(sentence_matches) if sentence_matches else 0.0
        
        return RetrievalResult(
            sentences=sentence_matches,
            hit_count=hit_count,
            miss_count=miss_count,
            hit_rate=hit_rate
        )
    
    async def _batch_retrieve(
        self,
        sentences: List[Tuple[int, str]],
        corpus_id: str,
        threshold: float
    ) -> List[SentenceMatch]:
        """
        批量检索句子（优化版：批量embedding）
        
        性能提升：100个句子从100次API调用 → 1次API调用
        """
        # 提取所有句子文本
        sentence_texts = [sent for idx, sent in sentences]
        
        # 🔑 批量检索（一次embedding API调用）
        try:
            batch_results = await self.corpus_manager.batch_search_similar(
                query_texts=sentence_texts,
                corpus_id=corpus_id,
                limit=1,
                score_threshold=threshold
            )
        except Exception as e:
            print(f"❌ 批量检索失败: {str(e)}")
            # 全部标记为未命中
            return [
                SentenceMatch(
                    index=idx,
                    source=sent,
                    matched=False,
                    translation="",
                    similarity=0.0,
                    corpus_source=""
                )
                for idx, sent in sentences
            ]
        
        # 组织结果
        sentence_matches = []
        for (idx, sent), results in zip(sentences, batch_results):
            if results and len(results) > 0:
                # 命中
                best_match = results[0]
                sentence_matches.append(
                    SentenceMatch(
                        index=idx,
                        source=sent,
                        matched=True,
                        translation=best_match["target"],
                        similarity=best_match["score"],
                        corpus_source=best_match["source"]
                    )
                )
            else:
                # 未命中
                sentence_matches.append(
                    SentenceMatch(
                        index=idx,
                        source=sent,
                        matched=False,
                        translation="",
                        similarity=0.0,
                        corpus_source=""
                    )
                )
        
        return sentence_matches
    
    def _get_adaptive_threshold(self, sentence: str, base_threshold: float) -> float:
        """
        自适应阈值：固定句式使用更高阈值
        
        Args:
            sentence: 源句子
            base_threshold: 基础阈值
        
        Returns:
            调整后的阈值
        """
        # 检查是否为固定句式
        for pattern in self.fixed_patterns:
            if re.match(pattern, sentence):
                # 固定句式，提高阈值到 0.90+
                return max(base_threshold, 0.90)
        
        # 普通句子，使用基础阈值
        return base_threshold
    
    def merge_translation(
        self,
        retrieval_result: RetrievalResult,
        llm_translations: Dict[int, str]
    ) -> str:
        """
        合并检索结果和LLM翻译
        
        Args:
            retrieval_result: 检索结果
            llm_translations: {句子索引: LLM翻译}
        
        Returns:
            完整翻译文本
        """
        merged_sentences = []
        
        for sent_match in retrieval_result.sentences:
            if sent_match.matched:
                # 使用语料库翻译
                merged_sentences.append(sent_match.translation)
            else:
                # 使用LLM翻译
                if sent_match.index in llm_translations:
                    merged_sentences.append(llm_translations[sent_match.index])
                else:
                    # 未提供翻译，保留原文（错误情况）
                    print(f"⚠️  警告: 句子{sent_match.index}既未命中也未翻译")
                    merged_sentences.append(sent_match.source)
        
        # 拼接句子
        # 根据目标语言决定分隔符
        if self.tgt_lang == 'zh':
            # 中文不需要额外空格
            return "".join(merged_sentences)
        else:
            # 英文等语言，句子间加空格
            return " ".join(merged_sentences)
    
    def get_statistics(self, retrieval_result: RetrievalResult) -> Dict:
        """
        获取检索统计信息
        
        Returns:
            {
                "total_sentences": int,
                "hit_count": int,
                "miss_count": int,
                "hit_rate": str,
                "avg_similarity": float
            }
        """
        matched_scores = [
            sent.similarity 
            for sent in retrieval_result.sentences 
            if sent.matched
        ]
        
        avg_similarity = (
            sum(matched_scores) / len(matched_scores) 
            if matched_scores else 0.0
        )
        
        return {
            "total_sentences": len(retrieval_result.sentences),
            "hit_count": retrieval_result.hit_count,
            "miss_count": retrieval_result.miss_count,
            "hit_rate": f"{retrieval_result.hit_rate * 100:.1f}%",
            "avg_similarity": round(avg_similarity, 4)
        }