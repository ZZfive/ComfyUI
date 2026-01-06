import contextvars
from typing import Optional, NamedTuple

class ExecutionContext(NamedTuple):
    """
    Context information about the currently executing node.

    Attributes:
        node_id: The ID of the currently executing node
        list_index: The index in a list being processed (for operations on batches/lists)
    """
    prompt_id: str
    node_id: str
    list_index: Optional[int]

current_executing_context: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar("current_executing_context", default=None)  # 异步安全的上下文存储方式，在多线程环境中保持隔离

def get_executing_context() -> Optional[ExecutionContext]:
    return current_executing_context.get(None)

class CurrentNodeContext:
    """
    Context manager for setting the current executing node context. 上下文管理器，用于设置当前执行节点上下文

    Sets the current_executing_context on enter and resets it on exit. 在进入和退出时设置和重置当前执行节点上下文

    Example:
        with CurrentNodeContext(node_id="123", list_index=0):
            # Code that should run with the current node context set
            process_image()
    """
    def __init__(self, prompt_id: str, node_id: str, list_index: Optional[int] = None):
        self.context = ExecutionContext(
            prompt_id= prompt_id,
            node_id= node_id,
            list_index= list_index
        )
        self.token = None

    def __enter__(self):
        self.token = current_executing_context.set(self.context)  # 设置当前执行节点上下文
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            current_executing_context.reset(self.token)  # 重置当前执行节点上下文
