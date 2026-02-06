"""
通用工具函数模块
包含文本处理、token估算等基础功能
"""
import re
from typing import List, Dict


def estimate_tokens(text: str) -> int:
    """估算文本的token数量"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other_chars = len(text) - chinese_chars - english_words
    return int(chinese_chars * 1.5 + english_words + other_chars * 0.5)


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
    current_start = 0
    chunk_id = 0
    
    for para in paragraphs:
        if len(para) > max_length:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": chunk_id,
                    "start_pos": current_start
                })
                chunk_id += 1
                current_chunk = ""
            
            sentences = re.split(r'([。！？\.!?])', para)
            sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
            
            for sent in sentences:
                if len(current_chunk) + len(sent) < max_length:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "chunk_id": chunk_id,
                            "start_pos": current_start
                        })
                        chunk_id += 1
                    current_chunk = sent
                    current_start += len(current_chunk)
        
        elif len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": chunk_id,
                "start_pos": current_start
            })
            chunk_id += 1
            current_chunk = para + "\n\n"
            current_start += len(current_chunk)
    
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_id": chunk_id,
            "start_pos": current_start
        })
    
    return chunks