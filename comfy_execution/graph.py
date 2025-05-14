from __future__ import annotations
from typing import Type, Literal

import nodes
from comfy_execution.graph_utils import is_link
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, InputTypeOptions

class DependencyCycleError(Exception):
    pass

class NodeInputError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

class DynamicPrompt:
    def __init__(self, original_prompt):
        # The original prompt provided by the user
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution
        self.ephemeral_prompt = {}  # 存储执行过程中动态创建的临时节点
        self.ephemeral_parents = {}  # 存储执行过程中动态创建的临时节点到其父节点的映射关系
        self.ephemeral_display = {}  # 存储执行过程中动态创建的临时节点到显示节点的映射关系

    def get_node(self, node_id):
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id

    def get_real_node_id(self, node_id):
        while node_id in self.ephemeral_parents:  # 如果节点ID在ephemeral_parents中，则不断更新节点ID为父节点ID
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))  # 返回所有节点ID的集合

    def get_original_prompt(self):
        return self.original_prompt  # 返回原始prompt

def get_input_info(
    class_def: Type[ComfyNodeABC],
    input_name: str,
    valid_inputs: InputTypeDict | None = None
) -> tuple[str, Literal["required", "optional", "hidden"], InputTypeOptions] | tuple[None, None, None]:
    """Get the input type, category, and extra info for a given input name.

    Arguments:
        class_def: The class definition of the node.
        input_name: The name of the input to get info for.
        valid_inputs: The valid inputs for the node, or None to use the class_def.INPUT_TYPES().

    Returns:
        tuple[str, str, dict] | tuple[None, None, None]: The input type, category, and extra info for the input name.
    """

    valid_inputs = valid_inputs or class_def.INPUT_TYPES()
    input_info = None
    input_category = None
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    if input_info is None:
        return None, None, None
    input_type = input_info[0]
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    return input_type, input_category, extra_info

class TopologicalSort:
    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self.pendingNodes = {}  # 存储需要执行的节点，key是节点id，value是布尔值
        self.blockCount = {} # Number of nodes this node is directly blocked by；key是节点id，value是int对象，表示阻塞该节点的其他节点总数
        self.blocking = {} # Which nodes are blocked by this node；key是节点id，value是一个字典，表示被该节点阻塞的所有节点

    def get_input_info(self, unique_id, input_name):
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    def make_input_strong_link(self, to_node_id, to_input):
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        if to_input not in inputs:
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but there is no input to that node at all")
        value = inputs[to_input]
        if not is_link(value):
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but that value is a constant")
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):  # 建立节点之间的强连接，from_node_id节点必须在to_node_id节点之前执行
        if not self.is_cached(from_node_id):  # from_node_id节点没有缓存
            self.add_node(from_node_id)  # 当前的link中，from_node_id就是一个还没执行的节点，需要向前构建拓扑执行关系，与最初从外部将输出类节点id传入add_node中一样
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}  # 初始化阻塞关系，from_node_id节点阻塞to_node_id节点
                self.blockCount[to_node_id] += 1  # 阻塞计数加1，表示to_node_id节点被阻塞的次数
            self.blocking[from_node_id][to_node_id][from_socket] = True  # 记录from_node_id节点阻塞to_node_id节点的具体socket

    def add_node(self, node_unique_id, include_lazy=False, subgraph_nodes=None):  # 从给定节点开始，向前遍历节点，统计工作流的一个可执行分支中的每个节点阻塞的具体节点和阻塞节点的数量
        node_ids = [node_unique_id]
        links = []

        while len(node_ids) > 0:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue

            self.pendingNodes[unique_id] = True
            self.blockCount[unique_id] = 0
            self.blocking[unique_id] = {}

            inputs = self.dynprompt.get_node(unique_id)["inputs"]  # 从工作流中获取当前节点的输入
            for input_name in inputs:
                value = inputs[input_name]
                if is_link(value):  # 如果value是link类型
                    from_node_id, from_socket = value  # from_node_id表示当前节点来源于节点的id，from_socket表示具体是来源节点中的哪个输出
                    if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                        continue
                    _, _, input_info = self.get_input_info(unique_id, input_name)  # 获取当前节点的输入信息
                    is_lazy = input_info is not None and "lazy" in input_info and input_info["lazy"]  # 判断当前输入是否为lazy
                    if (include_lazy or not is_lazy) and not self.is_cached(from_node_id):  # 当from_node_id节点不在output缓存中才将其加到node_ids
                        node_ids.append(from_node_id)
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    def is_cached(self, node_id):
        return False

    def get_ready_nodes(self):  # 获取没有被其他节点阻塞的节点，即不被其他节点阻塞的节点
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    def pop_node(self, unique_id):  # 弹出处理完的节点
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:  # 遍历被unique_id阻塞的所有节点
            self.blockCount[blocked_node_id] -= 1  # 因为unique_id已经处理完，所以被unique_id阻塞的节点不再被阻塞，阻塞计数减1
        del self.blocking[unique_id]  # 删除被unique_id阻塞的节点的记录信息

    def is_empty(self):
        return len(self.pendingNodes) == 0

class ExecutionList(TopologicalSort):
    """
    ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
    it can still be returned to the graph after having further dependencies added.
    """
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None  # 表示工作流中正在执行的节点

    def is_cached(self, node_id):
        return self.output_cache.get(node_id) is not None

    def stage_node_execution(self):  # 逐步采样每次需要执行的节点
        assert self.staged_node_id is None
        if self.is_empty():
            return None, None, None
        available = self.get_ready_nodes()  # 获取所有已经准备好的节点
        if len(available) == 0:  # 没有节点不被阻塞，即所有节点都被阻塞，表示工作流中存在环，处理循环依赖问题
            cycled_nodes = self.get_nodes_in_cycle()  #  找出整个环中的所有节点
            # Because cycles composed entirely of static nodes are caught during initial validation,  因为工作流中静态节点组成的环在初始校验时被捕获，
            # we will 'blame' the first node in the cycle that is not a static node.  将环中第一个不是静态节点的节点作为错误节点
            blamed_node = cycled_nodes[0]
            for node_id in cycled_nodes:
                display_node_id = self.dynprompt.get_display_node_id(node_id)
                if display_node_id != node_id:
                    blamed_node = display_node_id
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            error_details = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            return None, error_details, ex

        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None

    def ux_friendly_pick_node(self, node_list):  # 尽量选择可以让用户更早看到视觉结果的节点，提升交互体验
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        def is_output(node_id):
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                return True
            return False
        # 如果输出节点可用，请首先执行它。从技术上讲，这对整体执行时间没有影响，但作为用户，能够尽快看到 PreviewImage 显示结果会感觉更好。这里还可以使用一些其他启发式方法来进一步改善用户体验。
        for node_id in node_list:
            if is_output(node_id):
                return node_id

        #This should handle the VAEDecode -> preview case；优先返回被阻塞的节点中存在输出节点的节点
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                if is_output(blocked_node_id):
                    return node_id

        #This should handle the VAELoader -> VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                for blocked_node_id1 in self.blocking[blocked_node_id]:
                    if is_output(blocked_node_id1):
                        return node_id

        #TODO: this function should be improved
        return node_list[0]  # 如果以上条件都不满足，则返回node_list中的第一个节点

    def unstage_node_execution(self):
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None

    def get_nodes_in_cycle(self):  # 通过反向拓扑排序，找出环中的所有节点
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        blocked_by = { node_id: {} for node_id in self.pendingNodes }
        for from_node_id in self.blocking:
            for to_node_id in self.blocking[from_node_id]:
                if True in self.blocking[from_node_id][to_node_id].values():
                    blocked_by[to_node_id][from_node_id] = True
        to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        while len(to_remove) > 0:
            for node_id in to_remove:
                for to_node_id in blocked_by:
                    if node_id in blocked_by[to_node_id]:
                        del blocked_by[to_node_id][node_id]
                del blocked_by[node_id]
            to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        return list(blocked_by.keys())

class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
    """
    def __init__(self, message):
        self.message = message

