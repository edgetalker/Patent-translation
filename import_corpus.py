# 脚本：import_corpus.py
import json
import requests
from tqdm import tqdm

def import_to_qdrant(corpus_file: str, corpus_id: str, batch_size: int = 100):
    """批量导入语料到Qdrant"""
    # 读取语料
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    print(f"📦 准备导入 {len(corpus_data)} 条语料...")
    
    # 分批导入
    for i in tqdm(range(0, len(corpus_data), batch_size)):
        batch = corpus_data[i:i+batch_size]
        
        response = requests.post(
            "http://localhost:8080/corpus/add",
            json={
                "corpus_id": corpus_id,
                "entries": batch
            }
        )
        
        if response.status_code != 200:
            print(f"❌ 批次 {i//batch_size} 失败: {response.text}")
            break
    
    print(f"✅ 导入完成！")

if __name__ == "__main__":
    import_to_qdrant(
        corpus_file="./corpus_data.json",
        corpus_id="patent_40k",
        batch_size=100
    )