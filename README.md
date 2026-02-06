# 专利文档翻译系统


## 1. 项目结构

```
patent-translation/
├── .env                          # 环境变量配置
├── config.py                     # 系统配置参数
├── requirements.txt              # Python依赖包列表
├── terminology_extraction.py    # 术语提取模块
├── translation_core.py          # 核心翻译引擎
├── utils.py                     # 工具函数模块
├── api_server.py                # FastAPI服务接口
│   ├── POST /translate          # 翻译API端点
│   └── GET /health              # 健康检查端点
└── README.md                    
```

### 核心模块功能

| 模块 | 功能说明 |
|------|---------|
| **智能分段器** | 按段落语义边界分块，避免句子截断 |
| **滑动窗口术语提取** | 3000字符窗口+500字符重叠  |
| **术语智能去重** | 处理完全相同、包含关系等冗余术语 |
| **术语翻译** | 提取src术语对应的tgt翻译 |
| **术语约束翻译** | 强制使用术语映射表，保证一致性 |
| **一致性验证** | 检测译文中术语使用的统一性 |

---

## 2. API 调用

### curl调用

**文档翻译接口**

```python
curl -Method POST -Uri http://localhost:8080/translate -Body (@{
    src_text = "本发明涉及一种基于深度学习的图像识别方法..."
    src_lang = "zh"
    tgt_lang = "en"
    domain = "人工智能"
    use_context = $true
    glossary = json
} | ConvertTo-Json) -ContentType 'application/json'
```

**基础术语提取**

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
curl -Method POST -Uri http://localhost:8000/extract_terminology -Body (@{
    src_text = "一种专利文献自动翻译系统，..."
    src_lang = "zh"
    tgt_lang = "en"
    domain = "专利"
    window_size = 2000
    overlap = 300
    max_terms = 30
} | ConvertTo-Json) -ContentType 'application/json'
```

### 翻译端点参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `src_text` | str | 待翻译文本 | 必填 |
| `src_lang` | str | 源语言 | 例：zh |
| `tgt_lang` | str | 目标语言 | 例：en |
| `domain` | str | 文档领域 | 例：技术 |
| `use_context` | boolean | 上下文重叠 | True |

### 返回结果结构

```python
[
  {
    "translation": "The invention relates to a group of compounds ......",
    "term_dict": {
      "乙醚": "ether",
      "二氯苯乙烯基": "dichlorobenzenevinyl",
      "碳酸氢钾": "potassium hydrogen carbonate",
      "元素分析": "elemental analysis",
      ......
    },
    "chunks_info": [
      {
        "chunk_id": 0,
        "length": 1397
      },
      ......
    ],
    "statistics": {
      "source_length": 40564,
      "translation_length": 94849,
      "num_chunks": 32,
      "num_terms_extracted": 30,
      "num_terms_translated": 30,
      "terminology_consistent": true,
      "num_inconsistencies": 0,
      "time_elapsed": 4785.87,
      "avg_time_per_chunk": 149.56
    }
  }
]
```

---

