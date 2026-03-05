# API 接口文档 v1.0
## 健康检查
`GET /health` - 检查服务运行状态

**请求示例：**
```bash
curl "http://localhost:8080/health"
```

**响应示例：**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "config": {
    "llm_base_url": "hhttps://api.deepseek.com",
    "llm_model": "deepseek-chat",
    "corpus_enabled": true
  }
}
```
## 查看当前配置
`GET /config` - 获取当前配置

**请求示例：**

```bash
curl "http://localhost:8080/health"
```

**响应示例：**
```json
{
  "llm": {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  },
  "embedding": {
    "base_url": "YOUR_EMBED_BASE_URL"
  },
  "translation": {
    "max_chunk_length": 6000,
    "overlap_length": 1000,
    "temperature": 0.3
  },
  "terminology": {
    "max_terms": 60,
    "window_size": 8000,
    "window_overlap": 2000,
    "min_frequency": 1
  },
  "corpus": {  
    "qdrant_host": "localhost",
    "qdrant_port": "6333",
    "collection_name": "patent-translations",
    "enabled": true
  }
}
```

## 基础术语提取
`POST /extract_terminology`

**参数说明**

| 参数               | 类型    | 说明               | 默认值   |
| ------------------ | ------- | ----------------| -------- |
| `src_text`         | str     |  待翻译文本 | 必填 |
| `src_lang`         | str     |  源语言    | 例：zh |
| `tgt_lang`         | str     |  目标语言   | 例：en |
| `domain`           | str     |   文档领域 | 技术 |
| `window_size`      | int     | 窗口大小   | 非必填 |
| `overlap`          | int     | 重叠区域大小 | 非必填 |
| `max_terms`        | int     | 最多提取术语数量 | 非必填 |

**请求示例**
```bash
curl "http://localhost:8080/extract_terminology" \
    -H "Content-Type: application/json" \
    -d '{
      src_text = "本发明涉及一种基于深度学习的图像识别方法...",
      src_lang = "zh",
      tgt_lang = "en",
      domain = "人工智能"
    }'
```
**响应示例**
```json
{
  "terms": ["深度学习", "图像识别", "卷积神经网络"],
  "term_dict": {
    "深度学习": "deep learning",
    "图像识别": "image recognition",
    "卷积神经网络": "convolutional neural network"
  },
  "statistics": {
    "text_length": 8469,
    "terms_extracted": 52,
    "terms_translated": 52,
    "src_lang": "zh",
    "tgt_lang": "en",
    "domain": "技术",
    "window_size": 8000,
    "overlap": 2000
  }
}
```

## 文档翻译
`POST /translate` 

**参数说明**

| 参数               | 类型    | 说明               | 默认值   |
| ------------------ | ------- | ------------------ | -------- |
| `src_text`         | str     | 待翻译文本         | 必填     |
| `src_lang`         | str     | 源语言             | 例：zh   |
| `tgt_lang`         | str     | 目标语言           | 例：en   |
| `domain`           | str     | 文档领域           | 技术 |
| `use_context`      | boolean | 上下文重叠         | True     |
| `glossary`         | dict    | 术语表             | 非必填   |
| `domain_prompt`    | str     | 领域提示词        | 非必填  |
| `use_corpus`       | boolean | 是否启用语料库加速 | False   |
| `corpus_threshold` | float   | 语料库匹配阈值     | 0.85   |

**请求示例** 
+ 基础请求
```bash
curl "http://localhost:8080/translate" \
    -H "Content-Type: application/json" \
    -d '{
      "src_text": "本发明提出一种新型注意力机制...",
      "src_lang": "zh",
      "tgt_lang": "en",
      "domain": "机器学习"
    }'
```

+ 使用`外部术语库`+`领域提示词`
```bash
curl "http://localhost:8080/translate" \
    -H "Content-Type: application/json" \
    -d '{
      "src_text": "本发明提出一种新型注意力机制...",
      "src_lang": "zh",
      "tgt_lang": "ko",
      "domain": "机器学习",
      "glossary": {
        "注意力机制": "어텐션 메커니즘",
        "卷积神经网络": "합성곱 신경망"
      },
      "prompt": "使用합쇼체敬语体；专利惯用句式以「~하는 것을 특징으로 한다」 结尾；技术动词优先使用「수행하다/처리하다/구성하다」；量词使用韩语固有量词而非汉字量词"
    }'
```
+ 使用`语料库`加速

```bash
curl "http://localhost:8080/translate" \
    -H "Content-Type: application/json" \
    -d '{
      "src_text": "本发明涉及一种图像处理方法...",
      "src_lang": "zh",
      "tgt_lang": "en",
      "domain": "技术",
      "use_corpus": true,
      "corpus_threshold": 0.85
    }'
```

**响应示例**

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
      "length": 5922
    },
    {
      "chunk_id": 1,
      "length": 2586
    }
  ],
  "statistics": {
    "source_length": 8469,
    "translation_length": 26694,
    "num_chunks": 2,
    "num_terms_extracted": 52,
    "num_terms_translated": 52,
    "terminology_consistent": true,
    "num_inconsistencies": 0,
    "time_elapsed": 350.87,
    "avg_time_per_chunk": 175.44,
    "glossary_provided": false,
    "parallel_enabled": true
  },
  "corpus_stats": {
    "enabled": false,
    "total_sentences": 0,
    "total_hits": 0,
    "total_misses": 0,
    "overall_hit_rate": 0.0
  }
}
```

## 语料库管理

### 1. 获取所有语料库统计
`GET /corpus/stats`
+ 所有语料库统计
```bash
curl "http://localhost:8080/corpus/stats"
```
+ 指定语料库统计
```bash
curl "http://localhost:8080/corpus/stats?corpus_id=patent_corpus_001"
```

### 2. 删除语料库
`DELETE /corpus/'corpus_id'`
```bash
curl -X DELETE "http://localhost:8080/corpus/patent_corpus_001"
```

### 3. 添加语料
`POST /corpus/add`

**参数说明**

| 参数        | 类型     | 说明               | 默认值 |
| ---------- | ------- | ------------------ | -------- |
| `entries`  | list    | 添加的条目信息       | 必填 |
| `corpus_id` | str    | 集合id              | 必填 |


**请求示例**

```bash
curl "http://localhost:8080/corpus/add" \
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

**响应示例**

```json
{
  "corpus_id": "ch-zn-技术",
  "added_count": 3,
  "total_count": 103,
  "status": "success"
}
```

### 4. 检索相似语料
`GET /corpus/search`

**参数说明**

| 参数        | 类型     | 说明               | 默认值 |
| ---------- | ------- | ------------------ | -------- |
| `query`  | str    | 查询字段       | 必填 |
| `corpus_id` | str    | 集合id              | 非必填 |
| `limit` | int    | 答案限制条目             | 必填 |
| `threshold` | float    | 相似度阈值         | 必填 |

**请求示例**
```bash
curl "http://localhost:8080/corpus/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "本发明涉及一种新型图像处理装置",
    "corpus_id": "ch-zh-技术",
    "limit": 5,
    "threshold": 0.7
  }'
```

**响应示例**

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