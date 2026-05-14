"""
语料库灌库脚本
==================
- 复用项目现有 CorpusManager + EmbeddingService
- 读取 data/raw/*.json (字段: src/tgt/srcLang/tgtLang/domain)
- 按 \n 段落对齐,双向入库(zh→en 检索 + en→zh 检索)
- corpus_id 标记: thesis_demo
"""

import asyncio
import json
import sys
from pathlib import Path

# 把项目根目录加入 sys.path,以便 import 项目模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from corpus.embeddings import EmbeddingService
from corpus.manager import CorpusManager
# ==================== 配置 ====================
RAW_DIR = PROJECT_ROOT / "data" / "raw"
COLLECTION_NAME = "patent_corpus"   # 复用项目默认 collection
CORPUS_ID = "thesis_demo"            # 通过 corpus_id 区分本次入库
MIN_PARA_LEN = 20                    # 段落最小字符数
ENABLE_BIDIRECTIONAL = True          # 双向入库(zh源 + en源 各一份)
# =============================================


def load_documents():
    """加载所有 JSON 文档"""
    docs = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = fp.name
        docs.append(data)
        print(f"  📄 {fp.name}: src={len(data['src'])}字, tgt={len(data['tgt'])}字")
    return docs


def build_entries(docs):
    """段落对齐 + 构造 entries(支持双向)"""
    entries = []
    skipped = 0

    for doc_id, doc in enumerate(docs):
        src_paras = [p.strip() for p in doc["src"].split("\n") if p.strip()]
        tgt_paras = [p.strip() for p in doc["tgt"].split("\n") if p.strip()]

        if len(src_paras) != len(tgt_paras):
            print(f"  ⚠️  [{doc['_file']}] 段落数不一致: "
                  f"src={len(src_paras)}, tgt={len(tgt_paras)} -> 截断")
        n = min(len(src_paras), len(tgt_paras))

        doc_pairs = 0
        for i in range(n):
            src, tgt = src_paras[i], tgt_paras[i]
            if len(src) < MIN_PARA_LEN or len(tgt) < MIN_PARA_LEN:
                skipped += 1
                continue

            base_meta = {
                "doc_id": doc_id,
                "para_id": i,
                "source_file": doc["_file"],
                "domain": doc.get("domain", "patent"),
            }

            # zh→en 方向(中文做向量,英文为译文参考)
            entries.append({
                "source": src,
                "target": tgt,
                "metadata": {**base_meta, "direction": "zh2en"},
            })

            # en→zh 方向(英文做向量,中文为译文参考)
            if ENABLE_BIDIRECTIONAL:
                entries.append({
                    "source": tgt,
                    "target": src,
                    "metadata": {**base_meta, "direction": "en2zh"},
                })

            doc_pairs += 1

        print(f"  ✅ [{doc['_file']}] 入库段落对: {doc_pairs} (双向后: {doc_pairs * 2 if ENABLE_BIDIRECTIONAL else doc_pairs})")

    if skipped > 0:
        print(f"  ⏭️  跳过过短段落: {skipped}")

    return entries


async def verify_search(manager: CorpusManager):
    """灌库后做一次检索验证"""
    print("\n" + "=" * 60)
    print("🔍 检索验证")
    print("=" * 60)

    # 中→英测试
    test_zh = "本发明涉及一种玻璃熔窑"
    print(f"\n[查询·中文] {test_zh}")
    results = await manager.search_similar(
        query_text=test_zh,
        corpus_id=CORPUS_ID,
        limit=2,
        score_threshold=0.3,
    )
    if not results:
        print("  ❌ 无结果(检查 score_threshold 是否过高)")
    for r in results:
        print(f"  [score={r['score']:.3f}] src: {r['source'][:50]}...")
        print(f"                       tgt: {r['target'][:50]}...")

    # 英→中测试
    test_en = "This invention relates to a glass melting tank"
    print(f"\n[查询·英文] {test_en}")
    results = await manager.search_similar(
        query_text=test_en,
        corpus_id=CORPUS_ID,
        limit=2,
        score_threshold=0.3,
    )
    if not results:
        print("  ❌ 无结果")
    for r in results:
        print(f"  [score={r['score']:.3f}] src: {r['source'][:50]}...")
        print(f"                       tgt: {r['target'][:50]}...")


async def main():
    print("=" * 60)
    print("📦 毕设语料库灌库")
    print("=" * 60)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Corpus ID  : {CORPUS_ID}")
    print(f"双向入库   : {ENABLE_BIDIRECTIONAL}")
    print(f"Qdrant     : {config.QDRANT_URL}")
    print(f"Embedding  : {config.EMBED_BASE_URL}")
    
    # Step 1: 加载数据
    print("\n[1/4] 加载原始 JSON")
    docs = load_documents()
    if not docs:
        print(f"❌ {RAW_DIR} 下没有 JSON 文件")
        return

    # Step 2: 构造 entries
    print("\n[2/4] 段落对齐 + 构造 entries")
    entries = build_entries(docs)
    print(f"\n📊 总计待入库: {len(entries)} 条")

    if not entries:
        print("❌ 没有可入库的数据")
        return

    # Step 3: 初始化服务并入库
    embed_svc = EmbeddingService()
    manager = CorpusManager(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=getattr(config, "QDRANT_API_KEY", None),
        embedding_service=embed_svc,
        collection_name=COLLECTION_NAME,
    )

    BATCH_SIZE = 16  # 保守值,避免触发服务端限制

    print(f"\n[3/4] 分批入库(batch_size={BATCH_SIZE})")
    total_success = 0
    total_fail = 0

    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i: i + BATCH_SIZE]
        result = await manager.add_corpus_entries(batch, corpus_id=CORPUS_ID)
        if result.get("success"):
            total_success += result["count"]
            print(f"  ✅ batch {i//BATCH_SIZE + 1}: +{result['count']} 条 (累计 {total_success})")
        else:
            total_fail += len(batch)
            print(f"  ❌ batch {i//BATCH_SIZE + 1} 失败: {result.get('error')}")

    print(f"\n📊 入库结果: 成功 {total_success} | 失败 {total_fail}")
    if total_fail > 0:
        print("⚠️  有批次失败,检查上面的错误信息")

    # 显示 collection 总体统计
    stats = manager.get_corpus_stats()
    print(f"\n📊 Collection 统计: {stats}")

    # Step 4: 检索验证
    await verify_search(manager)

    # 关闭 httpx 连接
    await embed_svc.client.aclose()
    print("\n" + "=" * 60)
    print("✅ 全部完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())