"""
通用工具函数模块
包含文本处理、token估算等基础功能
"""
import re
from typing import List, Dict, Optional

from config import config


def estimate_tokens(text: str) -> int:
    """估算文本的token数量"""
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other_chars = len(text) - chinese_chars - english_words
    return int(chinese_chars * 1.5 + english_words + other_chars * 0.5)


def calculate_context_budget(
    chunk_text: str,
    system_overhead: int,
    term_dict: Dict[str, str],
    model_context_window: int = 64000,
    max_output_tokens: int = 8000,
    safe_margin: float = 0.85,
) -> Dict[str, int]:
    """
    根据模型上下文窗口动态分配 prompt 各模块预算。

    策略:
    1. 先扣除 system prompt、输出 token 和安全余量,得到可用 prompt token。
    2. 根据 chunk 文本长度、术语表规模估算可注入术语数与 few-shot 数。
    3. 同时给出建议的 chunk 最大字符数,避免 chunk 本身超出上下文。

    Returns:
        {
            "max_chunk_chars": int,      # 建议单 chunk 最大字符数
            "max_inject_terms": int,     # 建议注入术语数上限
            "max_few_shots": int,        # 建议 few-shot 数上限
            "available_prompt_tokens": int,
        }
    """
    available_tokens = int(
        (model_context_window - system_overhead - max_output_tokens) * safe_margin
    )
    available_tokens = max(available_tokens, 0)

    # chunk 文本按中文字符估算,保守按 1.5 tokens/char
    chunk_tokens = estimate_tokens(chunk_text)

    # 剩余 token 用于术语表 + few-shots
    remaining_tokens = available_tokens - chunk_tokens
    remaining_tokens = max(remaining_tokens, 0)

    # 每条术语平均约 15 tokens,每条 few-shot 约 80 tokens
    avg_term_tokens = 15
    avg_few_shot_tokens = 80

    max_inject_terms = min(
        len(term_dict),
        max(1, remaining_tokens // avg_term_tokens),
    ) if term_dict else 0

    # 留出术语预算后,再计算 few_shots
    term_budget_tokens = max_inject_terms * avg_term_tokens
    remaining_after_terms = max(0, remaining_tokens - term_budget_tokens)
    max_few_shots = max(0, remaining_after_terms // avg_few_shot_tokens)
    max_few_shots = min(max_few_shots, config.MAX_DYNAMIC_FEW_SHOTS)

    # chunk 字符数: 在剩余可用 prompt token 内,按 1.5 tokens/char 反推
    text_budget_tokens = available_tokens // 2  # 给原文留一半可用 token
    max_chunk_chars = max(500, int(text_budget_tokens / 1.5))

    return {
        "max_chunk_chars": max_chunk_chars,
        "max_inject_terms": max_inject_terms,
        "max_few_shots": max_few_shots,
        "available_prompt_tokens": available_tokens,
    }


def split_text_by_paragraph(text: str, max_length: int = 1500) -> List[Dict]:
    """
    按段落边界分块，保持语义完整性

    Args:
        text: 输入文本
        max_length: 最大块长度

    Returns:
        分块列表，每个块包含 text, chunk_id, start_pos
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = re.split(r'\n\s*\n|\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_id = 0
    global_pos = 0
    chunk_start_pos = 0

    for para in paragraphs:
        if len(para) > max_length:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": chunk_id,
                    "start_pos": chunk_start_pos
                })
                chunk_id += 1
                global_pos += len(current_chunk)
                current_chunk = ""
                chunk_start_pos = global_pos

            sentences = re.split(r'([。！？\.!?])', para)
            pairs = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1 , 2)]
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                pairs.append(sentences[-1])

            for sent in pairs:
                if len(current_chunk) + len(sent) < max_length:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "chunk_id": chunk_id,
                            "start_pos": chunk_start_pos
                        })
                        chunk_id += 1
                        global_pos += len(current_chunk)
                        chunk_start_pos = global_pos
                    current_chunk = sent

        elif len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": chunk_id,
                "start_pos": chunk_start_pos
            })
            chunk_id += 1
            global_pos += len(current_chunk)
            current_chunk = para + "\n\n"
            chunk_start_pos += global_pos

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_id": chunk_id,
            "start_pos": chunk_start_pos
        })

    return chunks
