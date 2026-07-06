"""
自纠错循环单元测试
验证: 路由决策、不一致术语提取、增量重译 chunk 选择、迭代上限
"""
import sys
import os

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import (
    route_after_stats,
    prepare_retry,
    _extract_source_terms,
    _find_chunks_with_terms,
)
from agent.tools import parallel_trans_tool
from config import config


# ============================================================
# route_after_stats
# ============================================================

def test_route_after_stats_retries_when_consistency_low():
    state = {
        "terminology_stats": {"consistency_rate": 0.5},
        "iteration_count": 0,
        "max_iterations": 2,
        "consistency_threshold": 0.85,
    }
    assert route_after_stats(state) == "retry"


def test_route_after_stats_ends_when_consistency_high():
    state = {
        "terminology_stats": {"consistency_rate": 0.95},
        "iteration_count": 0,
        "max_iterations": 2,
        "consistency_threshold": 0.85,
    }
    assert route_after_stats(state) == "end"


def test_route_after_stats_ends_when_max_iterations_reached():
    state = {
        "terminology_stats": {"consistency_rate": 0.5},
        "iteration_count": 2,
        "max_iterations": 2,
        "consistency_threshold": 0.85,
    }
    assert route_after_stats(state) == "end"


# ============================================================
# 术语 / chunk 匹配
# ============================================================

def test_extract_source_terms():
    inconsistencies = [
        "机器翻译方法 -> machine translation method",
        "  源语言文本  ->  source language text  ",
        "no arrow here",
    ]
    terms = _extract_source_terms(inconsistencies)
    assert terms == ["机器翻译方法", "源语言文本"]


def test_find_chunks_with_terms():
    chunks = [
        {"chunk_id": 0, "text": "本发明涉及一种机器翻译方法。"},
        {"chunk_id": 1, "text": "该方法接收源语言文本。"},
        {"chunk_id": 2, "text": "输出目标语言文本。"},
    ]
    matched = _find_chunks_with_terms(chunks, ["机器翻译方法", "源语言文本"])
    assert set(matched) == {0, 1}


def test_find_chunks_with_terms_empty_terms():
    chunks = [
        {"chunk_id": 0, "text": "本发明涉及一种机器翻译方法。"},
    ]
    assert _find_chunks_with_terms(chunks, []) == []


# ============================================================
# prepare_retry
# ============================================================

def test_prepare_retry_computes_retry_chunk_ids():
    state = {
        "terminology_stats": {
            "inconsistencies": [
                "机器翻译方法 -> machine translation method",
                "源语言文本 -> source language text",
            ]
        },
        "chunks": [
            {"chunk_id": 0, "text": "本发明涉及一种机器翻译方法。"},
            {"chunk_id": 1, "text": "该方法接收源语言文本。"},
            {"chunk_id": 2, "text": "输出目标语言文本。"},
        ],
        "iteration_count": 0,
    }
    result = prepare_retry(state)
    assert result["iteration_count"] == 1
    assert "机器翻译方法" in result["feedback_prompt"]
    assert set(result["retry_chunk_ids"]) == {0, 1}
    assert result["failed_chunks"] == []


def test_prepare_retry_without_inconsistencies():
    state = {
        "terminology_stats": {"inconsistencies": []},
        "chunks": [
            {"chunk_id": 0, "text": "本发明涉及一种机器翻译方法。"},
        ],
        "iteration_count": 1,
    }
    result = prepare_retry(state)
    assert result["iteration_count"] == 2
    assert result["retry_chunk_ids"] == []


# ============================================================
# parallel_trans_tool 增量重译
# ============================================================

class FakeTranslator:
    """模拟翻译器,记录被调用过的 chunk_id"""

    def __init__(self):
        self.called_chunk_ids = []

    def translate_chunk(self, chunk_text, chunk_id, total_chunks, **kwargs):
        self.called_chunk_ids.append(chunk_id)
        return f"[translated] {chunk_text[:10]}"


def test_parallel_trans_tool_full_translation_on_first_run():
    """首次运行: translated_chunks 为空,翻译全部 chunk"""
    fake = FakeTranslator()

    # 替换全局 translator 单例
    from agent import tools as tools_module
    original = tools_module._translator
    tools_module._translator = fake
    tools_module.get_translator = lambda: fake

    try:
        result = parallel_trans_tool.invoke({
            "chunks": [
                {"chunk_id": 0, "text": "第一段文本。"},
                {"chunk_id": 1, "text": "第二段文本。"},
            ],
            "retrieval_per_chunk": [
                {"chunk_id": 0, "few_shots": []},
                {"chunk_id": 1, "few_shots": []},
            ],
            "term_dict": {},
            "src_lang": "zh",
            "tgt_lang": "en",
            "domain": "技术",
        })

        assert len(result["translated_chunks"]) == 2
        assert set(fake.called_chunk_ids) == {0, 1}
        assert result["failed_chunks"] == []
    finally:
        tools_module._translator = original
        tools_module.get_translator = tools_module.get_translator


def test_parallel_trans_tool_incremental_retry():
    """增量重试: 只翻译 retry_chunk_ids 中的 chunk"""
    fake = FakeTranslator()

    from agent import tools as tools_module
    original = tools_module._translator
    original_get_translator = tools_module.get_translator
    tools_module._translator = fake
    tools_module.get_translator = lambda: fake

    try:
        result = parallel_trans_tool.invoke({
            "chunks": [
                {"chunk_id": 0, "text": "第一段文本。"},
                {"chunk_id": 1, "text": "第二段文本。"},
                {"chunk_id": 2, "text": "第三段文本。"},
            ],
            "retrieval_per_chunk": [
                {"chunk_id": 0, "few_shots": []},
                {"chunk_id": 1, "few_shots": []},
                {"chunk_id": 2, "few_shots": []},
            ],
            "term_dict": {},
            "src_lang": "zh",
            "tgt_lang": "en",
            "domain": "技术",
            "translated_chunks": ["old_0", "old_1", "old_2"],
            "retry_chunk_ids": [1],
        })

        assert result["translated_chunks"] == ["old_0", "[translated] 第二段文本。", "old_2"]
        assert fake.called_chunk_ids == [1]
        assert result["failed_chunks"] == []
    finally:
        tools_module._translator = original
        tools_module.get_translator = original_get_translator


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    import traceback

    tests = [
        test_route_after_stats_retries_when_consistency_low,
        test_route_after_stats_ends_when_consistency_high,
        test_route_after_stats_ends_when_max_iterations_reached,
        test_extract_source_terms,
        test_find_chunks_with_terms,
        test_find_chunks_with_terms_empty_terms,
        test_prepare_retry_computes_retry_chunk_ids,
        test_prepare_retry_without_inconsistencies,
        test_parallel_trans_tool_full_translation_on_first_run,
        test_parallel_trans_tool_incremental_retry,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
