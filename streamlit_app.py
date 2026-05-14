"""
专利翻译智能体 · Streamlit Demo
"""

import streamlit as st
import time
import io, contextlib
import html
import logging
from typing import Optional, Dict

# 消除子线程 ScriptRunContext 警告噪音
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


class LiveStream(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self._placeholder = placeholder
        self._lock = __import__("threading").Lock()

    def write(self, s):
        with self._lock:
            super().write(s)
            try:
                self._placeholder.code(self.getvalue(), language="text")
            except Exception:
                pass  # 子线程无 session context,静默忽略
        return len(s)


# ─── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="专利翻译智能体",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 全局样式 ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Serif SC', sans-serif;
}
.stApp {
    background: #0e0e12;
    color: #e8e6df;
}

.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #2a2a35;
    margin-bottom: 2rem;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #b5a97a;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Noto Serif SC', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0ede4;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.hero-title span { color: #c9a84c; }
.hero-sub {
    font-size: 0.88rem;
    color: #706e68;
    margin-top: 0.5rem;
    font-weight: 300;
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #b5a97a;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
}
.badge-ok   { background:#1a2e1e; color:#6fcf97; border:1px solid #2d4a33; }
.badge-warn { background:#2e2718; color:#f2c94c; border:1px solid #4a3d20; }
.badge-err  { background:#2e1818; color:#eb5757; border:1px solid #4a2020; }

.term-table { width:100%; border-collapse:collapse; font-size:0.83rem; }
.term-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #b5a97a;
    text-transform: uppercase;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #2a2a35;
    text-align: left;
}
.term-table td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid #1e1e26;
    color: #d4d0c8;
    vertical-align: top;
}
.term-table tr:hover td { background: #1c1c26; }
.term-src { color: #e8e6df; font-weight: 500; }
.term-tgt { color: #c9a84c; }
.term-user {
    font-family:'JetBrains Mono',monospace;
    font-size:0.6rem;
    color:#6fcf97;
    margin-left:0.3rem;
}

.translation-box {
    background: #12121a;
    border: 1px solid #252530;
    border-left: 3px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.9;
    color: #dddad0;
    white-space: pre-wrap;
    max-height: 420px;
    overflow-y: auto;
    font-family: 'Noto Serif SC', serif;
}

.stat-row {
    display: flex; gap: 1.2rem; flex-wrap: wrap; margin-top: 0.6rem;
}
.stat-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #706e68;
}
.stat-item b { color: #b5a97a; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: #c9a84c !important;
    color: #0e0e12 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 2rem !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #e0bc5a !important;
    transform: translateY(-1px) !important;
}

.stTextArea textarea {
    background: #12121a !important;
    border: 1px solid #252530 !important;
    color: #e8e6df !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c33 !important;
}
.stSelectbox > div > div {
    background: #12121a !important;
    border: 1px solid #252530 !important;
    color: #e8e6df !important;
}
.stFileUploader {
    background: #12121a !important;
    border: 1px dashed #252530 !important;
    border-radius: 8px !important;
}
.stExpander {
    border: 1px solid #252530 !important;
    border-radius: 8px !important;
    background: #16161e !important;
}

/* 原文译文对照 */
.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
}
.comparison-col {
    background: #12121a;
    border: 1px solid #252530;
    border-radius: 8px;
    overflow: hidden;
}
.comparison-col.src { border-left: 3px solid #706e68; }
.comparison-col.tgt { border-left: 3px solid #c9a84c; }
.comparison-header {
    background: #16161e;
    padding: 0.55rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: #b5a97a;
    text-transform: uppercase;
    border-bottom: 1px solid #252530;
}
.comparison-body {
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    line-height: 1.85;
    color: #dddad0;
    font-family: 'Noto Serif SC', serif;
    max-height: 520px;
    overflow-y: auto;
    white-space: pre-wrap;
}

hr { border-color: #252530 !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── 预制示例语料 ───────────────────────────────────────────────────────────────
PRESETS = {
    "⚙️ 技术文档(中→英)": {
        "text": """<<< 这里替换为你原文件中的中文专利全文 >>>""",
        "src_lang": "中文", "tgt_lang": "英文", "domain": "技术文档"
    },
    "💊 医学文档(韩→中)": {
        "text": """<<< 这里替换为你原文件中的韩文专利全文 >>>""",
        "src_lang": "韩文", "tgt_lang": "中文", "domain": "医疗文档"
    },
    "⚙️ 技术文档(西→英)": {
        "text": """<<< 这里替换为你原文件中的西班牙文专利全文 >>>""",
        "src_lang": "西班牙文", "tgt_lang": "英文", "domain": "技术文档"
    },
}

LANG_OPTIONS   = ["中文", "英文", "韩文", "日文", "德文", "法文", "西班牙文", "俄文"]
DOMAIN_OPTIONS = ["技术文档", "机械工程", "电化学", "医疗文档", "半导体", "化学", "计算机"]


# ─── 缓存初始化核心服务 ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_services():
    """懒加载并缓存核心服务"""
    from config import config
    from terminology_extraction import TerminologyExtractor
    from corpus.embeddings import EmbeddingService
    from corpus.manager import CorpusManager
    from agent.graph import patent_agent
    from agent.tools import init_tools

    embedding_service = EmbeddingService()
    corpus_manager = CorpusManager(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding_service=embedding_service,
    )
    term_extractor = TerminologyExtractor()
    init_tools(corpus_manager=corpus_manager)

    return {
        "patent_agent":   patent_agent,
        "term_extractor": term_extractor,
        "config":         config,
    }


# ─── 翻译调用 ──────────────────────────────────────────────────────────────────
def run_translation(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary: Optional[Dict[str, str]] = None,
    use_corpus: bool = False,
    corpus_id: Optional[str] = None,
) -> dict:
    services = load_services()
    agent    = services["patent_agent"]

    initial_state = {
        "src_text":              src_text,
        "src_lang":              src_lang,
        "tgt_lang":              tgt_lang,
        "domain":                domain,
        "glossary":              glossary or {},
        "domain_prompt":         None,
        "use_corpus":            use_corpus,
        "corpus_id":             corpus_id,
        "corpus_threshold":      0.85,
        "messages":              [],
        "use_pipeline_fallback": False,
        "retry_count":           0,
        "validation_passed":     False,
        "inconsistencies":       [],
        "translated_chunks":     [],
        "final_translation":     "",
    }

    try:
        result = agent.invoke(initial_state)
        return {
            "ok":          True,
            "translation": result.get("translation", result.get("final_translation", "")),
            "term_dict":   result.get("term_dict", {}),
            "stats": {
                "source_length":          len(src_text),
                "translation_length":     len(result.get("translation", "")),
                "num_chunks":             result.get("statistics", {}).get("num_chunks", 0),
                "terminology_consistent": result.get("validation_passed", False),
                "retry_count":            result.get("retry_count", 0),
            },
        }
    except Exception as e:
        import traceback
        with open("agent_error.log", "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(traceback.format_exc())
        return {"ok": False, "error": str(e)}


def parse_glossary(raw: str) -> Dict[str, str]:
    glossary = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("：", ":").split(":", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            glossary[parts[0]] = parts[1]
    return glossary


# ══════════════════════════════════════════════════════════════════════════════
#  预热核心服务(必须在任何 with 块/spinner/redirect_stdout 之前)
#  原因: run_translation() 中使用了 contextlib.redirect_stdout(LiveStream(...)),
#  若 load_services() 首次执行发生在此上下文内,EmbeddingService/CorpusManager
#  初始化时的 print() 会经 LiveStream.write() 调到 st.code(),被 @st.cache_resource
#  录制为副作用,导致后续 cache hit replay 时抛 CacheReplayClosureError。
#  在此处提前预热,首次填充发生在干净环境下,后续永远是空 replay。
# ══════════════════════════════════════════════════════════════════════════════
try:
    _ = load_services()
except Exception as _e:
    st.error(f"核心服务初始化失败: {_e}")
    st.caption("请检查 Qdrant / DeepSeek API 配置和网络连通性")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  页面主体
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-tag">Patent Translation Agent · LangGraph + RAG</div>
  <div class="hero-title">专利文档<span>智能翻译</span>系统</div>
  <div class="hero-sub">基于 LangGraph Agent 架构 · 术语一致性验证 · 专利语料库增强</div>
</div>
""", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

# ══════════════ 左栏:输入 ═══════════════════════════════════════════════════
with col_in:
    input_mode = st.radio(
        "输入方式",
        ["📝 文本输入", "📄 上传文件"],
        horizontal=True,
        label_visibility="collapsed",
    )

    src_text = ""

    if input_mode == "📝 文本输入":
        st.markdown('<div class="card-label">快速示例</div>', unsafe_allow_html=True)
        preset_cols = st.columns(len(PRESETS))
        selected_preset = None
        for idx, (label, data) in enumerate(PRESETS.items()):
            with preset_cols[idx]:
                if st.button(label, key=f"preset_{idx}", use_container_width=True):
                    selected_preset = data

        if selected_preset:
            st.session_state["preset_text"]   = selected_preset["text"]
            st.session_state["preset_src"]    = selected_preset["src_lang"]
            st.session_state["preset_tgt"]    = selected_preset["tgt_lang"]
            st.session_state["preset_domain"] = selected_preset["domain"]

        src_text = st.text_area(
            "待翻译文本",
            value=st.session_state.get("preset_text", ""),
            height=240,
            placeholder="粘贴专利文本,或点击上方示例快速填入…",
            label_visibility="collapsed",
        )

    else:
        uploaded = st.file_uploader(
            "上传专利文档",
            type=["txt", "pdf"],
            label_visibility="collapsed",
        )
        if uploaded:
            if uploaded.name.endswith(".pdf"):
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded) as pdf:
                        src_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                except Exception:
                    st.error("PDF 解析失败,请尝试上传 txt 格式。")
            else:
                src_text = uploaded.read().decode("utf-8", errors="ignore")
            st.caption(f"已读取 {len(src_text):,} 字符")

    st.divider()

    # ── 翻译参数 ──
    st.markdown('<div class="card-label">翻译参数</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    src_lang = p1.selectbox(
        "源语言", LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.get("preset_src", "中文")),
    )
    tgt_lang = p2.selectbox(
        "目标语言", LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.get("preset_tgt", "英文")),
    )
    domain = p3.selectbox(
        "领域", DOMAIN_OPTIONS,
        index=DOMAIN_OPTIONS.index(st.session_state.get("preset_domain", "技术文档"))
        if st.session_state.get("preset_domain", "技术文档") in DOMAIN_OPTIONS else 0,
    )

    # ── 语料库 RAG ──
    st.markdown('<div class="card-label">语料库 RAG</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 2])
    use_corpus = rc1.toggle("启用语料库", value=False)
    corpus_id = rc2.text_input(
        "Corpus ID",
        value="thesis_demo",
        disabled=not use_corpus,
        label_visibility="collapsed",
        placeholder="corpus_id",
    )

    # ── 高级选项 ──
    with st.expander("⚙️ 高级选项 · 自定义术语表"):
        st.caption("每行一条,格式:`原文:译文`,例如 `液压泵:hydraulic pump`")
        glossary_raw = st.text_area(
            "术语表", height=130, label_visibility="collapsed",
            placeholder="液压泵:hydraulic pump\n柱塞:plunger\n斜盘:swashplate",
        )

    st.markdown("")
    run_btn = st.button("▶ 开始翻译", type="primary", use_container_width=True)

# ══════════════ 右栏:输出 ═══════════════════════════════════════════════════
with col_out:
    result_placeholder = st.empty()

    if not run_btn:
        result_placeholder.markdown("""
<div style="height:400px;display:flex;align-items:center;justify-content:center;
            border:1px dashed #252530;border-radius:10px;">
  <div style="text-align:center;color:#3a3a45;">
    <div style="font-size:2rem;margin-bottom:0.5rem;">⚗️</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                letter-spacing:0.15em;">AWAITING INPUT</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════ 翻译执行 ═══════════════════════════════════════════════════
if run_btn:
    if not src_text.strip():
        with col_out:
            result_placeholder.warning("请输入待翻译文本或上传文件。")
        st.stop()
    if src_lang == tgt_lang:
        with col_out:
            result_placeholder.warning("源语言与目标语言相同,请重新选择。")
        st.stop()

    glossary = parse_glossary(glossary_raw) if glossary_raw.strip() else {}

    with col_out:
        with result_placeholder.container():
            log_placeholder = st.empty()
            with st.spinner("Agent 推理中…"):
                t0 = time.time()
                live = LiveStream(log_placeholder)
                with contextlib.redirect_stdout(live):
                    result = run_translation(
                        src_text, src_lang, tgt_lang, domain, glossary,
                        use_corpus=use_corpus,
                        corpus_id=corpus_id if use_corpus else None,
                    )
                elapsed = time.time() - t0

        with result_placeholder.container():
            if not result["ok"]:
                st.markdown(
                    f'<span class="badge badge-err">ERROR</span> {html.escape(result["error"])}',
                    unsafe_allow_html=True,
                )
                st.caption("详细堆栈已写入 `agent_error.log`")
            else:
                consistent = result["stats"].get("terminology_consistent", False)
                consist_badge = (
                    '<span class="badge badge-ok">术语一致 ✓</span>'
                    if consistent
                    else '<span class="badge badge-warn">术语待校验</span>'
                )
                rag_badge = (
                    '<span class="badge badge-ok">RAG ON</span>'
                    if use_corpus
                    else '<span class="badge badge-warn">RAG OFF</span>'
                )
                st.markdown(
                    f'<span class="badge badge-ok">AGENT</span> &nbsp; {rag_badge} &nbsp; {consist_badge}',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                st.markdown('<div class="card-label">译文输出</div>', unsafe_allow_html=True)
                translation_text = result.get("translation", "")
                st.markdown(
                    f'<div class="translation-box">{html.escape(translation_text)}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇ 下载译文",
                    data=translation_text,
                    file_name="translation.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                st.divider()

                # ── 术语对照表 ──
                term_dict = result.get("term_dict", {})
                user_terms = set(glossary.keys()) if glossary else set()

                st.markdown('<div class="card-label">术语对照表</div>', unsafe_allow_html=True)

                if term_dict:
                    rows = ""
                    for src_t, tgt_t in term_dict.items():
                        tag = (
                            '<span class="term-user">用户</span>'
                            if src_t in user_terms else ""
                        )
                        rows += (
                            f'<tr><td class="term-src">{html.escape(src_t)}{tag}</td>'
                            f'<td class="term-tgt">{html.escape(tgt_t)}</td></tr>'
                        )
                    st.markdown(f"""
<div style="max-height:220px;overflow-y:auto;">
<table class="term-table">
  <thead><tr><th>原文术语</th><th>译文术语</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>""", unsafe_allow_html=True)
                else:
                    st.caption("未提取到术语")

                with st.expander("📊 技术统计"):
                    stats = result["stats"]
                    st.markdown(f"""
<div class="stat-row">
  <span class="stat-item">源文 <b>{stats.get('source_length',0):,}</b> 字符</span>
  <span class="stat-item">译文 <b>{stats.get('translation_length',0):,}</b> 字符</span>
  <span class="stat-item">分块 <b>{stats.get('num_chunks',0)}</b> 块</span>
  <span class="stat-item">重试 <b>{stats.get('retry_count',0)}</b> 次</span>
  <span class="stat-item">耗时 <b>{elapsed:.1f}s</b></span>
</div>""", unsafe_allow_html=True)

    # ══════════════ 全宽对照视图 ═══════════════
    if result.get("ok"):
        st.divider()
        st.markdown('<div class="card-label">原文译文对照</div>', unsafe_allow_html=True)
        translation_text = result.get("translation", "")
        st.markdown(f"""
<div class="comparison-grid">
  <div class="comparison-col src">
    <div class="comparison-header">SRC · {src_lang} · {len(src_text):,} chars</div>
    <div class="comparison-body">{html.escape(src_text)}</div>
  </div>
  <div class="comparison-col tgt">
    <div class="comparison-header">TGT · {tgt_lang} · {len(translation_text):,} chars</div>
    <div class="comparison-body">{html.escape(translation_text)}</div>
  </div>
</div>
""", unsafe_allow_html=True)