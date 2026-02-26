# Patent Translation Agent | 专利文档翻译智能体

> 面向知识产权垂直领域的长文档翻译系统。
> 通过**滑动窗口术语提取 + RAG 语料库加速 + 并行分块翻译**，
> 在保证全文术语 100% 一致性的前提下，实现 8K 字符文档 6 分钟内处理完成。
> 
> 系统当前为 Pipeline 架构（Fallback 层），正在向 Agent 架构演进。

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

##  项目结构

```
patent-translation/
├── api_server.py              # FastAPI 服务层，7 个 RESTful 端点
├── translation_core.py        # 翻译核心：并行分块 + 语料库加速双分支
├── terminology_extraction.py  # 滑动窗口术语提取 + 双语术语翻译
├── corpus_retrieval.py        # Qdrant 语义检索 + 命中/未命中句子合并
├── utils.py                   # 段落感知分块、Token 估算
├── config.py                  # 统一配置管理（LLM / Qdrant / 翻译参数）
├── corpus/
│   ├── __init__.py
│   ├── embeddings.py          # 文本向量化服务（支持多种 Embedding 模型）
│   └── manager.py             # Qdrant Collection CRUD 管理
├── .env.example               # 环境变量模板
└── requirements.txt
```
## 系统架构
```
输入：文档 + 目标语言 [+ 术语表]
         │
         ▼
┌─────────────────────────────────┐
│  术语处理层                      │
│  ① 滑动窗口提取（若无传入术语表）  │
│  ② 跨窗口频率统计 + 语言自适应去重 │
│  ③ LLM 术语翻译 → 双语术语表      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  分块层                          │
│  段落感知分块（保持语义完整性）     │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌─────────────────────┐
│ 直接翻译  │  │  RAG 语料库加速      │
│ ThreadPool│  │  Qdrant 语义检索     │
│ 3线程并行  │  │  命中 → 复用译文     │
│           │  │  未命中 → LLM 翻译   │
└─────┬─────┘  └──────────┬──────────┘
      └──────┬────────────┘
             ▼
┌─────────────────────────────────┐
│  后处理层                        │
│  合并 + 术语一致性验证            │
└─────────────────────────────────┘
```

> **注**：当前为 Pipeline 架构，各处理阶段已按职责拆分，
> 为后续 Agent 化改造（各阶段封装为独立 Tool）预留接口边界。
## 核心模块功能

| **滑动窗口术语提取** | 窗口 8000 字符 / 重叠 2000 字符，跨窗口频率统计解决长文档术语覆盖不全问题 |

| **语言自适应术语去重** | 中文使用字符边界匹配，英文使用 \b 单词边界，避免正则对中文失效 |

| **精确匹配优先的术语注入** | 每 chunk 注入上限 25 条，精确命中优先 + 频率保底，避免全量注入稀释 LLM 注意力 |

| **RAG 语料库加速** | 命中句子（阈值 0.85）直接复用历史译文，未命中送 LLM，减少重复 API 调用 |


## 翻译引擎

兼容 OpenAI 格式 API，支持 DeepSeek / GPT-4o / Claude 等云端服务，
以及通过 vLLM 本地部署的开源模型。

实测配置：**Qwen2.5-14B-AWQ**（vLLM + RTX 4090），AWQ 量化较 FP16 显存降低 50%，
20K 字符文档端到端处理时间 **6 分钟**。
## 🚀 Quick Start

**环境要求**：Python 3.9+ | Qdrant | vLLM 兼容推理服务
```bash
# 1. 克隆仓库
git clone https://github.com/edgetalker/Patent-translation.git
cd Patent-translation

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 LLM_BASE_URL / API_KEY / QDRANT_HOST

# 4. 启动服务
python api_server.py

# 5. 验证服务
curl http://localhost:8080/health
```

## 📊 性能指标

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| 文档处理速度 | 8K 字符 / 6 分钟 | Qwen2.5-14B-AWQ，3 线程并行 |
| 术语一致性 | **100%** | 专利技术类文档 |
| 专业词汇率 | **83%** | 人工评估 |
| 语料库命中加速 | 命中句子 0 LLM 调用 | 阈值 0.85 |
| 支持语言 | 7 种 | 中 / 英 / 日 / 韩 / 法 / 德 / 西 |

## 🗺️ Roadmap

**已完成**
- [x] 滑动窗口术语提取 + 语言自适应去重
- [x] 精确匹配优先的术语注入策略（上限 25 条）
- [x] RAG 语料库加速（Qdrant + 并行翻译）
- [x] FastAPI 7 模块 RESTful 服务

**进行中**
- [ ] Agent 化重构：现有 Pipeline 各阶段封装为独立 Tool
- [ ] Orchestrator 层：引入调度策略，支持动态选择翻译路径

**计划中**
- [ ] MoE 微调：基于专利领域数据进行监督微调
- [ ] 前端界面：Vue 3 + Element Plus 可视化操作

## 完整 API 文档

启动服务后访问 `http://localhost:8080/docs` 查看交互式文档（FastAPI 自动生成）

详细参数说明见 [docs/API.md](./docs/API.md)