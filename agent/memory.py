"""
LangGraph Checkpointer 封装

当前使用内存版 InMemorySaver(MemorySaver),用于演示跨调用的 Agent State 持久化。
如需跨进程/跨重启持久化,可替换为 langgraph-checkpoint-sqlite 的 SqliteSaver。
"""
from langgraph.checkpoint.memory import MemorySaver


_memory_saver: MemorySaver = None


def get_memory_saver() -> MemorySaver:
    """获取全局单例 MemorySaver"""
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = MemorySaver()
    return _memory_saver
