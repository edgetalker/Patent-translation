"""
翻译核心模块
处理长文档翻译和术语一致性验证
集成语料库检索功能
"""
import time
import re
import unicodedata
import threading
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from config import config

def build_prompt_with_few_shot(
    chunk_text: str,
    few_shots: List[Tuple[str, str, float]],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    term_dict: Dict[str, str],
    domain_prompt: Optional[str] = None,
    chunk_id: int = 0,
    total_chunks: int = 1,
    max_few_shots: int = 5,
) -> str:
    """
    构造带 Few-Shot 参考的翻译 Prompt
    
    核心设计:
    1. 参考示例区(Few-Shot)与待翻译区结构化分离,避免 LLM 混淆
    2. 示例按相似度降序,最相关的在前,取 Top-K
    3. 显式标注相似度,让 LLM 自判参考可信度
    4. 严格"忠实原文"指令,防止参考内容污染译文
    
    Args:
        chunk_text: 当前要翻译的文本块
        few_shots: 从 RAG 检索到的命中句对,格式:
                   [(corpus_source, corpus_target, similarity), ...]
                   - corpus_source: 语料库里的原文(与 chunk 中某句相似)
                   - corpus_target: 该原文对应的标准译文
                   - similarity: 语料库 source 与 chunk 中句子的相似度分数
        src_lang: 源语言代码(如 'zh')
        tgt_lang: 目标语言代码(如 'en')
        domain: 领域名称(如 '医学')
        term_dict: 术语表 {源术语: 目标术语},可选
        chunk_id: 当前 chunk 序号(从 0 开始)
        total_chunks: 总 chunk 数
        max_few_shots: 最多注入的参考示例数量(Top-K)
    
    Returns:
        完整的 Prompt 字符串
    
    Raises:
        ValueError: 当 few_shots 元组结构不合法时
    """
    # ---------- 输入校验 ----------
    src_lang_name = config.get_language_name(src_lang)
    tgt_lang_name = config.get_language_name(tgt_lang)
    
    # ---------- Few-Shot Top-K 筛选 ----------
    # 按相似度降序,取前 K 条,防止 Prompt 过长
    if few_shots:
        # 防御性:防止 few_shots 元素结构异常
        try:
            sorted_shots = sorted(
                few_shots,
                key=lambda x: x[2],  # 按 similarity 排序
                reverse=True
            )[:max_few_shots]
        except (IndexError, TypeError) as e:
            raise ValueError(
                f"few_shots 格式不合法,期望 [(src, tgt, similarity), ...],"
                f"实际收到: {few_shots[:1]}... 错误: {e}"
            )
    else:
        sorted_shots = []
    
    # ---------- 分段构造 Prompt ----------
    parts: List[str] = []
    
    # 1. 角色声明 + 任务定位
    parts.append(
        f"你是一位专业的{domain}领域{src_lang_name}-{tgt_lang_name}翻译专家。\n"
        f"当前正在翻译第 {chunk_id + 1} 段,共 {total_chunks} 段。"
    )
    
    # 2. Few-Shot 参考区(仅在有命中时出现)
    if sorted_shots:
        parts.append(
            f"\n\n【翻译参考】以下是语料库中与【待翻译内容】语义相似的平行句对,"
            f"请参考其术语使用和翻译风格,但必须严格忠实于【待翻译内容】的原意,"
            f"不得引入示例中未出现在待翻译内容里的信息:\n"
        )
        for i, (corpus_src, corpus_tgt, sim) in enumerate(sorted_shots, 1):
            parts.append(
                f"\n参考示例 {i}:\n"
                f"  原文: {corpus_src}\n"
                f"  译文: {corpus_tgt}\n"
                f"  相似度: {sim:.2f}"
            )
    
    # 3. 术语表
    if term_dict:
        terms_list = "\n".join([f"  - {src} → {tgt}" for src, tgt in term_dict.items()])
        parts.append(
            f"\n\n【术语表】以下是{domain}领域的专业术语对照表(共{len(term_dict)}个):\n"
            f"{terms_list}"
        )

    # 3.5. 领域补充指令
    if domain_prompt:
        parts.append(
            f"\n\n【领域提示】\n{domain_prompt.strip()}"
        )
        
    # 4. 待翻译内容(主任务)
    parts.append(
        f"\n\n【待翻译内容】\n{chunk_text}"
    )
    
    # 5. 翻译要求(关键指令区)
    requirements: List[str] = [
        "严格忠实于【待翻译内容】的原意,即使示例信息量更大也不得补充",
    ]
    if sorted_shots:
        requirements.append(
            "在术语选择和句式风格上优先参考上述示例,但不照搬示例的具体内容"
        )
    if term_dict:
        requirements.append(
            "严格遵守术语表的翻译规范;若术语表与示例冲突,以术语表为准"
        )
    requirements.extend([
        "保持文档的专业性和原文的段落结构",
        "禁止使用任何格式标记(如 markdown、HTML 标签等)",
        "输出纯文本译文,不要添加任何解释或说明",
    ])
    
    req_text = "\n".join([f"{i}. {r}" for i, r in enumerate(requirements, 1)])
    parts.append(f"\n\n【翻译要求】\n{req_text}")
    
    return "".join(parts)

class DocumentTranslator:
    """文档翻译器（支持语料库加速）"""
    
    def __init__(self):
        """初始化翻译器(纯翻译层,不感知语料库)"""
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
        self.model_name = config.LLM_MODEL_NAME
        self.log_lock = threading.Lock()
    
    def _safe_print(self, *args, **kwargs):
        """线程安全的打印"""
        with self.log_lock:
            print(*args, **kwargs)
         
    def translate_chunk(
        self,
        chunk_text: str,
        chunk_id: int,
        total_chunks: int,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        term_dict: Optional[Dict[str, str]] = None,
        domain_prompt: Optional[str] = None,
        few_shots: Optional[List[Tuple[str, str, float]]] = None,
        max_few_shots: int = 5,
        context_budget: Optional[Dict] = None,
    ) -> str:
        """
        翻译单个文本块(纯翻译函数)

        不感知 RAG 检索的存在——few_shots 由上游 retrieve_tool 预先准备好后传入。
        few_shots 为 None 或空列表时,自动退化为零参考翻译。

        流程:
        1. 术语表过滤(chunk 内相关术语)
        2. 构造 Few-Shot Prompt
        3. 单次 LLM 调用翻译整个 chunk

        Args:
            chunk_text: 待翻译文本块
            chunk_id: 当前 chunk 序号(从 0 开始)
            total_chunks: 总 chunk 数
            src_lang / tgt_lang / domain: 翻译参数
            term_dict: 完整术语表(本函数会自动过滤出与 chunk 相关的)
            domain_prompt: 领域级额外指令
            few_shots: 预检索好的 Few-Shot 参考 [(corpus_src, corpus_tgt, sim), ...]
            max_few_shots: Top-K 截断数量
            context_budget: 动态上下文预算,覆盖 max_inject_terms / max_few_shots

        Returns:
            译文字符串(失败时返回 "[TRANSLATION FAILED: ...]" 占位)
        """
        chunk_start = time.time()

        # 动态上下文预算
        budget = context_budget or {}
        max_inject_terms = budget.get("max_inject_terms", config.MAX_INJECT_TERMS)
        effective_max_few_shots = budget.get("max_few_shots", max_few_shots)

        # ---------- Step 1: 术语表过滤 ----------
        relevant_terms = None
        if term_dict:
            relevant_terms = self._get_relevant_terms(
                chunk_text=chunk_text,
                term_dict=term_dict,
                max_inject=max_inject_terms
            )
        
        # ---------- Step 2: 构造 Prompt ----------
        prompt = build_prompt_with_few_shot(
            chunk_text=chunk_text,
            few_shots=few_shots or [],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            domain=domain,
            term_dict=relevant_terms,
            domain_prompt=domain_prompt,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            max_few_shots=effective_max_few_shots,
        )
        prompt_time = time.time() - chunk_start
        
        src_lang_name = config.get_language_name(src_lang)
        tgt_lang_name = config.get_language_name(tgt_lang)
        
        # ---------- Step 3: LLM 翻译(带重试)----------
        translation = None
        for attempt in range(config.MAX_RETRIES):
            try:
                api_start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"你是一位专业的{domain}领域翻译专家,"
                                f"擅长{src_lang_name}到{tgt_lang_name}的翻译。"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TRANSLATION_TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                api_time = time.time() - api_start
                translation = response.choices[0].message.content.strip()
                
                total_time = time.time() - chunk_start
                few_shot_info = f"few_shots={len(few_shots) if few_shots else 0}"
                self._safe_print(
                    f"   ⏱️  Chunk {chunk_id+1} 耗时: "
                    f"Prompt {prompt_time:.2f}s + API {api_time:.2f}s = {total_time:.2f}s "
                    f"({few_shot_info})"
                )
                self._safe_print(
                    f"      输入 {len(chunk_text)} 字 → 输出 {len(translation)} 字 "
                    f"({len(translation)/max(len(chunk_text),1):.2f}x)"
                )
                break
                
            except Exception as e:
                self._safe_print(
                    f"   ⚠️  Chunk {chunk_id+1} 翻译失败 "
                    f"(尝试 {attempt+1}/{config.MAX_RETRIES}): {str(e)}"
                )
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    self._safe_print(f"   ❌ Chunk {chunk_id+1} 最终失败,返回占位")
                    translation = f"[TRANSLATION FAILED: {chunk_text}]"
        
        return translation
    
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
    
    def compute_terminology_stats(
        self,
        translation: str,
        term_dict: Dict[str, str],
        src_text: str,
    ) -> Dict:
        """
        计算术语一致性统计(纯描述性,不做判定)

        策略:
        - NFKC + lower 归一化,抹平全半角/重音/大小写
        - 拉丁术语:词边界 + 允许词尾 0-2 字母(覆盖单复数/过去式,拒绝词性变化)
        - 非拉丁术语(CJK 等):纯子串匹配
        """
        def normalize(text: str) -> str:
            return unicodedata.normalize("NFKC", text).lower().strip()
        
        def matches(tgt_norm: str, translation_norm: str) -> bool:
            if re.match(r"^[a-z0-9\s\-]+$", tgt_norm):
                return bool(re.search(
                    r"\b" + re.escape(tgt_norm) + r"[a-z]{0,2}\b",
                    translation_norm
                ))
            return tgt_norm in translation_norm
        
        translation_norm = normalize(translation)
        
        relevant = [
            (src, tgt) for src, tgt in term_dict.items()
            if src in src_text and tgt.strip()
        ]
        inconsistencies = [
            f"{src} -> {tgt}"
            for src, tgt in relevant
            if not matches(normalize(tgt), translation_norm)
        ]
        
        total = len(relevant)
        return {
            "terminology_total": total,
            "terminology_hit": total - len(inconsistencies),
            "terminology_miss": len(inconsistencies),
            "consistency_rate": 1 - len(inconsistencies) / max(total, 1),
            "inconsistencies": inconsistencies,
        }