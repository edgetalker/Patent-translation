### 健康检查

```python
curl -X GET http://localhost:8080/health
```

+ 响应示例

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "config": {
    "llm_base_url": "http://localhost:8080",
    "llm_model": "deepseek-chat",
    "corpus_enabled": true
  }
}
```

### 文档翻译

+ 翻译端点参数说明

| 参数               | 类型    | 说明               | 默认值   |
| ------------------ | ------- | ------------------ | -------- |
| `src_text`         | str     | 待翻译文本         | 必填     |
| `src_lang`         | str     | 源语言             | 例：zh   |
| `tgt_lang`         | str     | 目标语言           | 例：en   |
| `domain`           | str     | 文档领域           | 例：技术 |
| `use_context`      | boolean | 上下文重叠         | True     |
| `glossary`         | json    | 术语表             | 非必须   |
| `use_corpus`       | boolean | 是否启用语料库加速 | 非必须   |
| `corpus_id`        | str     | 指定的语料库ID     | 非必须   |
| `corpus_threshold` | float   | 语料库匹配阈值     | 非必须   |

+ 默认翻译 - 不使用语料库加速

```python
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{
    "src_text": "本发明提出一种新型注意力机制...",
    "src_lang": "zh",
    "tgt_lang": "en",
    "domain": "机器学习",
    "use_context": true,
    "glossary": {
      "注意力机制": "attention mechanism",
      "卷积神经网络": "convolutional neural network"
    }
```

+ 使用语料库加速（建议格式：`{src_lang}_{tgt_lang}_{domain}）`

```python
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{
    "src_text": "本发明涉及一种图像处理方法...",
    "src_lang": "zh",
    "tgt_lang": "en",
    "domain": "技术",
    "use_context": true,
    "use_corpus": true,
    "corpus_id": "zh-en-技术"
    "corpus_threshold": 0.85
  }'
```

+ 响应示例

```json
{
  "translation": "The present invention relates to an image recognition method based on deep learning...",
  "term_dict": {
    "深度学习": "deep learning",
    "图像识别": "image recognition",
    "卷积神经网络": "convolutional neural network"
  },
  "chunks_info": [
    {
      "chunk_id": 0,
      "src_text": "本发明涉及一种基于深度学习的图像识别方法...",
      "translation": "The present invention relates to...",
      "start_pos": 0,
      "end_pos": 150
    }
  ],
  "statistics": {
    "total_chunks": 1,
    "total_chars": 150,
    "translation_time": 2.34
  },
  "corpus_stats": {
    "enabled": true,
    "total_sentences": 10,
    "total_hits": 6,
    "total_misses": 4,
    "overall_hit_rate": 0.6
  }
}
```

### 基础术语提取

```python
curl -Method POST -Uri http://localhost:8080/extract_terminology -Body (@{
    src_text = "本发明涉及一种基于深度学习的图像识别方法..."
    src_lang = "zh"
    tgt_lang = "en"
    domain = "人工智能"
} | ConvertTo-Json) -ContentType 'application/json'
```

**自动参数提取**

```python
curl -Method POST -Uri http://localhost:8080/extract_terminology -Body (@{
    src_text = "一种专利文献自动翻译系统，..."
    src_lang = "zh"
    tgt_lang = "en"
    domain = "专利"
    window_size = 2000
    overlap = 300
    max_terms = 30
} | ConvertTo-Json) -ContentType 'application/json'
```

### 语料库管理

+ 添加语料（meta为可选项，按需添加）

```bash
curl -X POST http://localhost:8080/corpus/add \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {
        "source": "本发明涉及一种图像处理方法。",
        "target": "The present invention relates to an image processing method.",
        "metadata": {"type": "patent_opening", "category": "method"}
      },
      {
        "source": "该装置包括处理器、存储器和通信模块。",
        "target": "The device comprises a processor, a memory, and a communication module.",
        "metadata": {"type": "structure_description", "category": "device"}
      },
      {
        "source": "所述方法包括以下步骤。",
        "target": "The method includes the following steps.",
        "metadata": {"type": "method_intro", "category": "method"}
      },
      ...
    ],
    "corpus_id": "zh-en-技术"
  }'
```

  + 响应示例

  ```json
{
  "corpus_id": "ch-zn-技术",
  "added_count": 3,
  "total_count": 103,
  "status": "success"
}
  ```

  + 检索相似语料

  ```bash
curl -X POST http://localhost:8080/corpus/search \
-H "Content-Type: application/json" \
-d '{
  "query": "本发明涉及一种新型图像处理装置",
  "corpus_id": "ch-zh-技术",
  "limit": 5,
  "threshold": 0.7
}'
  ```

  + 响应示例

  ```bash
{
  "query": "本发明涉及一种新型图像处理装置",
  "results": [
    {
      "source": "本发明涉及",
      "target": "The present invention relates to",
      "score": 0.92,
      "metadata": {"type": "patent_opening"}
    },
    {
      "source": "一种图像处理方法",
      "target": "An image processing method",
      "score": 0.85,
      "metadata": {"domain": "image_processing"}
    }
  ],
  "count": 2
}
  ```

+ 获取所有语料库统计

```bash
curl -X GET http://localhost:8080/corpus/stats
```

+ 获取指定语料库统计

```bash
curl -X GET http://localhost:8080/corpus/stats?corpus_id=patent_corpus_001
```

+ 删除语料库
```bash
curl -X DELETE http://localhost:8080/corpus/patent_corpus_001
```