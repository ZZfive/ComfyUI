import sys
import copy
import logging
import threading
import heapq
import time
import traceback
from enum import Enum
import inspect
from typing import List, Literal, NamedTuple, Optional

import torch
import nodes

import comfy.model_management
from comfy_execution.graph import get_input_info, ExecutionList, DynamicPrompt, ExecutionBlocker
from comfy_execution.graph_utils import is_link, GraphBuilder
from comfy_execution.caching import HierarchicalCache, LRUCache, DependencyAwareCache, CacheKeySetInputSignature, CacheKeySetID
from comfy_execution.validation import validate_node_input

class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2

class DuplicateNodeError(Exception):
    pass

class IsChangedCache:
    def __init__(self, dynprompt, outputs_cache):
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    def get(self, node_id):
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        if not hasattr(class_def, "IS_CHANGED"):
            self.is_changed[node_id] = False
            return self.is_changed[node_id]

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return self.is_changed[node_id]

        # Intentionally do not use cached outputs here. We only want constants in IS_CHANGED
        input_data_all, _ = get_input_data(node["inputs"], class_def, node_id, None)
        try:
            is_changed = _map_node_over_list(class_def, input_data_all, "IS_CHANGED")  # 调用节点的IS_CHANGED方法检测入参是否有变化
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except Exception as e:
            logging.warning("WARNING: {}".format(e))
            node["is_changed"] = float("NaN")  # 如果IS_CHANGED方法执行失败，则将is_changed设置为NaN，因为两个NaN不相等
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]


class CacheType(Enum):
    CLASSIC = 0
    LRU = 1
    DEPENDENCY_AWARE = 2


class CacheSet:
    def __init__(self, cache_type=None, cache_size=None):
        if cache_type == CacheType.DEPENDENCY_AWARE:
            self.init_dependency_aware_cache()
            logging.info("Disabling intermediate node cache.")
        elif cache_type == CacheType.LRU:
            if cache_size is None:
                cache_size = 0
            self.init_lru_cache(cache_size)
            logging.info("Using LRU cache")
        else:
            self.init_classic_cache()

        self.all = [self.outputs, self.ui, self.objects]

    # Performs like the old cache -- dump data ASAP
    def init_classic_cache(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.ui = HierarchicalCache(CacheKeySetInputSignature)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_lru_cache(self, cache_size):
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.ui = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.objects = HierarchicalCache(CacheKeySetID)

    # only hold cached items while the decendents have not executed
    def init_dependency_aware_cache(self):
        self.outputs = DependencyAwareCache(CacheKeySetInputSignature)
        self.ui = DependencyAwareCache(CacheKeySetInputSignature)
        self.objects = DependencyAwareCache(CacheKeySetID)

    def recursive_debug_dump(self):
        result = {
            "outputs": self.outputs.recursive_debug_dump(),
            "ui": self.ui.recursive_debug_dump(),
        }
        return result

def get_input_data(inputs, class_def, unique_id, outputs=None, dynprompt=None, extra_data={}):
    valid_inputs = class_def.INPUT_TYPES()  # 节点定义中的对输入的定义
    input_data_all = {}
    missing_keys = {}
    for x in inputs:
        input_data = inputs[x]
        _, input_category, input_info = get_input_info(class_def, x, valid_inputs)
        def mark_missing():
            missing_keys[x] = True
            input_data_all[x] = (None,)  # 将确实的输入设置为None，而不是抛出异常
        if is_link(input_data) and (not input_info or not input_info.get("rawLink", False)):
            input_unique_id = input_data[0]  # 输入来源的源节点ID
            output_index = input_data[1]  # 输入来源的源节点输出索引
            if outputs is None:  # 如果outputs为None，表示输入来源的源节点没有输出，直接标记为缺失
                mark_missing()
                continue # This might be a lazily-evaluated input
            cached_output = outputs.get(input_unique_id)  # 从outputs中获取输入来源的源节点的输出
            if cached_output is None:  # 如果输入来源的源节点没有输出，直接标记为缺失
                mark_missing()
                continue
            if output_index >= len(cached_output):  # 如果输入来源的源节点输出索引大于等于源节点的输出数量，直接标记为缺失
                mark_missing()
                continue
            obj = cached_output[output_index]  # 从输出缓存中获取实际数据
            input_data_all[x] = obj
        elif input_category is not None:  # 如果输入不是link，而是直接的值
            input_data_all[x] = [input_data]  # 将直接值包装成列表添加到结果中

    if "hidden" in valid_inputs:  # 处理hidden类输入
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [dynprompt.get_original_prompt() if dynprompt is not None else {}]  # 如果dynprompt不为None，则获取原始prompt，否则获取一个空字典
            if h[x] == "DYNPROMPT":
                input_data_all[x] = [dynprompt]
            if h[x] == "EXTRA_PNGINFO":
                input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
            if h[x] == "AUTH_TOKEN_COMFY_ORG":
                input_data_all[x] = [extra_data.get("auth_token_comfy_org", None)]
            if h[x] == "API_KEY_COMFY_ORG":
                input_data_all[x] = [extra_data.get("api_key_comfy_org", None)]
    return input_data_all, missing_keys

map_node_over_list = None #Don't hook this please

def _map_node_over_list(obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
    # check if node wants the lists
    input_is_list = getattr(obj, "INPUT_IS_LIST", False)  # 首先通过节点类中是否有INPUT_IS_LIST属性来判断是否将整个输入列表作为一个整体处理
    # 确定输入的最大程度
    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max(len(x) for x in input_data_all.values())

    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        return {k: v[i if len(v) > i else -1] for k, v in d.items()}

    results = []
    def process_inputs(inputs, index=None, input_is_list=False):
        if allow_interrupt:  # 如果允许中断
            nodes.before_node_execution()  # 在执行前调用
        execution_block = None  # 初始化执行阻塞对象为None
        for k, v in inputs.items():  # 检查所有输入是否存在阻塞对象
            if input_is_list:
                for e in v:
                    if isinstance(e, ExecutionBlocker):  # 如果v中存在阻塞对象，则将阻塞对象赋值给v
                        v = e
                        break
            if isinstance(v, ExecutionBlocker):  # 如果v是阻塞对象，则执行阻塞对象的回调函数
                execution_block = execution_block_cb(v) if execution_block_cb else v
                break
        if execution_block is None:  # 如果执行阻塞对象为None，则执行节点函数
            if pre_execute_cb is not None and index is not None:  # 如果执行前的回调函数不为None，并且index不为None，则执行执行前的回调函数
                pre_execute_cb(index)
            results.append(getattr(obj, func)(**inputs))  # 执行传入的函数并记录结果
        else:
            results.append(execution_block)  # 如果执行阻塞对象不为None，则将执行阻塞对象记录到结果中

    if input_is_list:  # 如果输入是列表
        process_inputs(input_data_all, 0, input_is_list=input_is_list)  # 处理整个输入列表
    elif max_len_input == 0:  # 如果输入为空
        process_inputs({})  # 处理空输入
    else:
        for i in range(max_len_input):  # 遍历输入列表
            input_dict = slice_dict(input_data_all, i)  # 获取输入列表的第i个元素
            process_inputs(input_dict, i)  # 处理第i个元素
    return results

def merge_result_data(results, obj):
    # check which outputs need concatenating
    output = []
    output_is_list = [False] * len(results[0])
    if hasattr(obj, "OUTPUT_IS_LIST"):  # 如果节点类中存在OUTPUT_IS_LIST属性，则将该属性赋值给output_is_list
        output_is_list = obj.OUTPUT_IS_LIST

    # merge node execution results
    for i, is_list in zip(range(len(results[0])), output_is_list):  # 将节点的输出与对应输出是否为list的列表打包一起遍历
        if is_list:
            value = []
            for o in results:
                if isinstance(o[i], ExecutionBlocker):
                    value.append(o[i])
                else:
                    value.extend(o[i])
            output.append(value)
        else:
            output.append([o[i] for o in results])
    return output

def get_output_data(obj, input_data_all, execution_block_cb=None, pre_execute_cb=None):
    results = []
    uis = []
    subgraph_results = []
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)  # 执行节点功能函数函数返回结果
    has_subgraph = False
    for i in range(len(return_values)):  # 此处的return_values是一个列表，元素是节点功能函数返回的元素，可能是字典或元组
        r = return_values[i]
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'expand' in r:
                # Perform an expansion, but do not append results
                has_subgraph = True
                new_graph = r['expand']
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif 'result' in r:
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        else:
            if isinstance(r, ExecutionBlocker):
                r = tuple([r] * len(obj.RETURN_TYPES))
            results.append(r)
            subgraph_results.append((None, r))

    if has_subgraph:
        output = subgraph_results
    elif len(results) > 0:
        output = merge_result_data(results, obj)
    else:
        output = []
    ui = dict()
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    parent_node_id = dynprompt.get_parent_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']  # 从输入的工作流中获取节点对应的输入
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]  # 获取节点对应的节点类
    if caches.outputs.get(unique_id) is not None:  # 如果给定的节点ID有outputs类型的缓存
        if server.client_id is not None:
            cached_output = caches.ui.get(unique_id) or {}
            server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": cached_output.get("output",None), "prompt_id": prompt_id }, server.client_id)
        return (ExecutionResult.SUCCESS, None, None)

    input_data_all = None
    try:
        if unique_id in pending_subgraph_results:
            cached_results = pending_subgraph_results[unique_id]
            resolved_outputs = []
            for is_subgraph, result in cached_results:
                if not is_subgraph:
                    resolved_outputs.append(result)
                else:
                    resolved_output = []
                    for r in result:
                        if is_link(r):
                            source_node, source_output = r[0], r[1]
                            node_output = caches.outputs.get(source_node)[source_output]
                            for o in node_output:
                                resolved_output.append(o)

                        else:
                            resolved_output.append(r)
                    resolved_outputs.append(tuple(resolved_output))
            output_data = merge_result_data(resolved_outputs, class_def)
            output_ui = []
            has_subgraph = False
        else:
            input_data_all, missing_keys = get_input_data(inputs, class_def, unique_id, caches.outputs, dynprompt, extra_data)  # 将从工作流中提取的节点输入与其定义中要求的输入对比，检测输入是否正常
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": unique_id, "display_node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            obj = caches.objects.get(unique_id)  # 从缓存中获取该节点id对应节点类型的实例对象缓存
            if obj is None:  # 如果缓存中没有该节点id对应节点类型的实例对象缓存，则创建一个
                obj = class_def()  # 创建一个节点实例对象
                caches.objects.set(unique_id, obj)  # 将创建的节点实例对象缓存到objects类型缓存中

            if hasattr(obj, "check_lazy_status"):
                required_inputs = _map_node_over_list(obj, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and (
                    x not in input_data_all or x in missing_keys
                )]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return (ExecutionResult.PENDING, None, None)

            def execution_block_cb(block):  # 执行阻塞时的回调函数
                if block.message is not None:
                    mes = {
                        "prompt_id": prompt_id,
                        "node_id": unique_id,
                        "node_type": class_type,
                        "executed": list(executed),

                        "exception_message": f"Execution Blocked: {block.message}",
                        "exception_type": "ExecutionBlocked",
                        "traceback": [],
                        "current_inputs": [],
                        "current_outputs": [],
                    }
                    server.send_sync("execution_error", mes, server.client_id)
                    return ExecutionBlocker(None)  # 返回一个阻塞对象，表示执行被阻塞
                else:
                    return block
            def pre_execute_cb(call_index):  # 执行前的回调函数
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)
            output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)  # 节点运行
        if len(output_ui) > 0:  # 如果输出UI不为空
            caches.ui.set(unique_id, {
                "meta": {
                    "node_id": unique_id,
                    "display_node": display_node_id,
                    "parent_node": parent_node_id,
                    "real_node_id": real_node_id,
                },
                "output": output_ui
            })  # 将输出UI缓存到ui类型缓存中
            if server.client_id is not None:
                server.send_sync("executed", { "node": unique_id, "display_node": display_node_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)
        if has_subgraph:
            cached_outputs = []
            new_node_ids = []
            new_output_ids = []
            new_output_links = []
            for i in range(len(output_data)):
                new_graph, node_outputs = output_data[i]
                if new_graph is None:
                    cached_outputs.append((False, node_outputs))
                else:
                    # Check for conflicts
                    for node_id in new_graph.keys():
                        if dynprompt.has_node(node_id):
                            raise DuplicateNodeError(f"Attempt to add duplicate node {node_id}. Ensure node ids are unique and deterministic or use graph_utils.GraphBuilder.")
                    for node_id, node_info in new_graph.items():
                        new_node_ids.append(node_id)
                        display_id = node_info.get("override_display_id", unique_id)
                        dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
                        # Figure out if the newly created node is an output node
                        class_type = node_info["class_type"]
                        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            new_output_ids.append(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            new_output_links.append((from_node_id, from_socket))
                    cached_outputs.append((True, node_outputs))
            new_node_ids = set(new_node_ids)
            for cache in caches.all:
                cache.ensure_subcache_for(unique_id, new_node_ids).clean_unused()
            for node_id in new_output_ids:
                execution_list.add_node(node_id)
            for link in new_output_links:
                execution_list.add_strong_link(link[0], link[1], unique_id)
            pending_subgraph_results[unique_id] = cached_outputs
            return (ExecutionResult.PENDING, None, None)
        caches.outputs.set(unique_id, output_data)  # 将输出数据缓存到outputs类型缓存中
    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": real_node_id,
        }

        return (ExecutionResult.FAILURE, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        logging.error(f"!!! Exception during processing !!! {ex}")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": real_node_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted
        }
        if isinstance(ex, comfy.model_management.OOM_EXCEPTION):
            logging.error("Got an OOM, unloading all loaded models.")
            comfy.model_management.unload_all_models()

        return (ExecutionResult.FAILURE, error_details, ex)

    executed.add(unique_id)  # 记录已执行的节点

    return (ExecutionResult.SUCCESS, None, None)  # 返回成功

class PromptExecutor:
    def __init__(self, server, cache_type=False, cache_size=None):
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.server = server
        self.reset()

    def reset(self):
        self.caches = CacheSet(cache_type=self.cache_type, cache_size=self.cache_size)
        self.status_messages = []
        self.success = True

    def add_message(self, event, data: dict, broadcast: bool):
        data = {
            **data,
            "timestamp": int(time.time() * 1000),
        }
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:  # 如果当前处理任务对应的客户端ID不为None，或者广播为True
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": list(current_outputs),
            }
            self.add_message("execution_error", mes, broadcast=False)

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)  # 设置中断处理为False

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]  # 更新当前处理任务对应的客户端ID
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)  # 发送开始执行的消息

        with torch.inference_mode():
            dynamic_prompt = DynamicPrompt(prompt)
            is_changed_cache = IsChangedCache(dynamic_prompt, self.caches.outputs)
            for cache in self.caches.all:  # 遍历所有缓存，更新缓存；顺序是outputs -> ui -> objects
                cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)  # 设置缓存；主要是给各个节点更新缓存key
                cache.clean_unused()  # 清理未使用的缓存key及对应的缓存值

            cached_nodes = []  # 记录可以复用的缓存节点ID
            for node_id in prompt:
                if self.caches.outputs.get(node_id) is not None:  # 如果给定的节点ID有outputs类型的缓存
                    cached_nodes.append(node_id)

            comfy.model_management.cleanup_models_gc()  # 对未使用的模型进行回收
            self.add_message("execution_cached",
                          { "nodes": cached_nodes, "prompt_id": prompt_id},
                          broadcast=False)  # 发送当前工作流中可以复用的缓存节点
            pending_subgraph_results = {}
            executed = set()
            execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)
            current_outputs = self.caches.outputs.all_node_ids()  # 获取当前工作流的所有节点id
            for node_id in list(execute_outputs):  # 此处的execute_outputs中是任务提交前工作流校验后的输出类节点列表
                execution_list.add_node(node_id)  # 会从输出类节点开始，向前遍历节点，统计工作流的一个可执行分支中的每个节点阻塞的具体节点和阻塞节点的数量；只针对需要重新执行的节点

            while not execution_list.is_empty():  # 如果执行列表不为空，则继续执行
                node_id, error, ex = execution_list.stage_node_execution()  # 从执行列表中获取一个节点，作为执行的初始节点
                if error is not None:  # 如果执行过程中有错误，则处理错误
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break

                result, error, ex = execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results)
                self.success = result != ExecutionResult.FAILURE
                if result == ExecutionResult.FAILURE:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break
                elif result == ExecutionResult.PENDING:
                    execution_list.unstage_node_execution()
                else: # result == ExecutionResult.SUCCESS:
                    execution_list.complete_node_execution()  # staged_node_id节点执行结束，更新执行节点队列中的信息
            else:
                # Only execute when the while-loop ends without break
                self.add_message("execution_success", { "prompt_id": prompt_id }, broadcast=False)

            ui_outputs = {}
            meta_outputs = {}
            all_node_ids = self.caches.ui.all_node_ids()
            for node_id in all_node_ids:
                ui_info = self.caches.ui.get(node_id)
                if ui_info is not None:
                    ui_outputs[node_id] = ui_info["output"]
                    meta_outputs[node_id] = ui_info["meta"]
            self.history_result = {
                "outputs": ui_outputs,
                "meta": meta_outputs,
            }
            self.server.last_node_id = None
            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()


def validate_inputs(prompt, item, validated):
    unique_id = item  # 获取节点编码
    if unique_id in validated:  # 如果节点编码已经在validated中，表示该节点已经验证过，直接返回验证结果
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']  # 获取prompt中设置的节点输入参数，是一个字典，key是输入参数名称，value是一个有两个元素的列表，第一个元素表示该输入来自于哪个节点，即该节点的id，第二个元素表示具体是该节点的哪个输出
    class_type = prompt[unique_id]['class_type']  # 获取节点类型名
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]  # 获取节点类型对应的具体节点类

    class_inputs = obj_class.INPUT_TYPES()  # 获取节点类中定义的输入类型信息，一个字典
    valid_inputs = set(class_inputs.get('required',{})).union(set(class_inputs.get('optional',{})))  # 获取该节点类定义中的required输入和optional输入

    errors = []  # 存放错误信息
    valid = True  # 用于存储验证结果

    validate_function_inputs = []
    validate_has_kwargs = False
    if hasattr(obj_class, "VALIDATE_INPUTS"):
        argspec = inspect.getfullargspec(obj_class.VALIDATE_INPUTS)  # 获取节点中定义的validate_inputs函数的参数信息
        validate_function_inputs = argspec.args  # 获取validate_inputs函数的参数名列表
        validate_has_kwargs = argspec.varkw is not None  # 验证validate_inputs函数是否有可变关键字参数
    received_types = {}  # 用于存储输入参数的来源节点的输出类型

    for x in valid_inputs:  # 校验prompt中节点的inputs是否与定义的required_inputs对齐
        input_type, input_category, extra_info = get_input_info(obj_class, x, class_inputs)  # 从节点类的定义中获取输入参数的类型、类别和额外信息
        assert extra_info is not None
        if x not in inputs:  # 如果节点定义中的要求的输入参数不在当前prompt工作流中定义的节点输入参数中
            if input_category == "required":  # 如果输入参数是required输入
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)  # 将错误信息添加到errors列表中
            continue

        val = inputs[x]
        info = (input_type, extra_info)
        if isinstance(val, list):  # 对来自于其他节点的输出进行校验，即节点中的link节点
            if len(val) != 2:  # 如果输入参数是一个列表，并且列表长度不为2，则报错
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]  # 获取输入参数的来源节点id
            o_class_type = prompt[o_id]['class_type']  # 获取输入参数来源节点的类型名
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES  # 获取输入参数来源节点的输出类型
            received_type = r[val[1]]  # 获取输入参数来源节点的输出类型中的具体类型
            received_types[x] = received_type  # 将输入参数的来源节点的输出类型中的具体类型添加到received_types字典中
            if 'input_types' not in validate_function_inputs and not validate_node_input(received_type, input_type):  # 如果当前节点要求的输入类型与上一节点的输出类型不一致
                details = f"{x}, received_type({received_type}) mismatch input_type({input_type})"  # 报错信息
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)  # 将错误信息添加到errors列表中
                continue
            try:
                r = validate_inputs(prompt, o_id, validated)  # 递归验证节点输入参数
                if r[0] is False:  # 如果验证失败，则将valid设置为False
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = (False, reasons, o_id)
                continue
        else:  # 不是来自于其他节点的输入进行校验，即节点中的非link节点
            try:
                # Unwraps values wrapped in __value__ key. This is used to pass
                # list widget value to execution, as by default list value is
                # reserved to represent the connection between nodes.
                if isinstance(val, dict) and "__value__" in val:
                    val = val["__value__"]
                    inputs[x] = val

                if input_type == "INT":
                    val = int(val)
                    inputs[x] = val
                if input_type == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if input_type == "STRING":
                    val = str(val)
                    inputs[x] = val
                if input_type == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {input_type} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if x not in validate_function_inputs and not validate_has_kwargs:
                if "min" in extra_info and val < extra_info["min"]:  # 如果设置了最小是，val不能小于最小值
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, extra_info["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in extra_info and val > extra_info["max"]:  # 如果设置了最大值，val不能大于最大值
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, extra_info["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

                if isinstance(input_type, list):
                    combo_options = input_type  # 如果输入类型是列表，则就是combo类型，存放所有可选的具体可选值
                    if val not in combo_options:  # val必须在设置的可选值范围内
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(combo_options) > 20:
                            list_info = f"(list of length {len(combo_options)})"
                            input_config = None
                        else:
                            list_info = str(combo_options)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: '{val}' not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(validate_function_inputs) > 0 or validate_has_kwargs:  # 如果节点类中定义了validate_inputs函数，或者节点类中定义了validate_inputs函数有可变关键字参数
        input_data_all, _ = get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs or validate_has_kwargs:
                input_filtered[x] = input_data_all[x]
        if 'input_types' in validate_function_inputs:
            input_filtered['input_types'] = [received_types]

        #ret = obj_class.VALIDATE_INPUTS(**input_filtered)
        ret = _map_node_over_list(obj_class, input_filtered, "VALIDATE_INPUTS")
        for x in input_filtered:
            for i, r in enumerate(ret):
                if r is not True and not isinstance(r, ExecutionBlocker):
                    details = f"{x}"
                    if r is not False:
                        details += f" - {str(r)}"

                    error = {
                        "type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                        }
                    }
                    errors.append(error)
                    continue

    if len(errors) > 0 or valid is not True:
        ret = (False, errors, unique_id)  # 如果验证失败，则返回False和错误信息
    else:
        ret = (True, [], unique_id)  # 如果验证成功，则返回True和空列表

    validated[unique_id] = ret  # 将验证结果添加到validated字典中
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt):
    outputs = set()  # 存放输出类节点编码
    for x in prompt:  # prompt是一个包含具体节点信息的字典，key是节点编码，value就是节点具体信息
        if 'class_type' not in prompt[x]:  # prompt中的每个节点信息必须具有class_type属性，该属性表征节点类型
            error = {
                "type": "invalid_prompt",
                "message": "Cannot execute because a node is missing the class_type property.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return (False, error, [], {})

        class_type = prompt[x]['class_type']
        class_ = nodes.NODE_CLASS_MAPPINGS.get(class_type, None)  # 获取节点类型对应的节点类
        if class_ is None:  # 如果节点类型不存在，则返回错误
            error = {
                "type": "invalid_prompt",
                "message": f"Cannot execute because node {class_type} does not exist.",
                "details": f"Node ID '#{x}'",
                "extra_info": {}
            }
            return (False, error, [], {})

        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE is True:  # 如果节点类具有OUTPUT_NODE属性，并且该属性为True，则将节点编码添加到outputs集合中
            outputs.add(x)

    if len(outputs) == 0:  # 整个workflow中必须有一个输出属性的节点如SaveImage，将生成的结果向外输出，不然直接报错
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], {})
    # 上述是对工作流中的节点是否存在进行验证，下面是对工作流节点前后连接是否正确，即各节点输入参数是否有效进行验证
    good_outputs = set()  # 存放验证成功的输出类节点编码
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:  # 从输出类型节点向前递归遍历
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)  # 会从该输出节点向前递归遍历所有相连节点并进行输入验证，各个节点的具体信息会存储在validated中，此处返回的是o这一输出类节点的验证信息
            valid = m[0]  # 验证结果
            reasons = m[1]  # 错误信息
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:  # 如果验证成功，则将节点编码添加到good_outputs集合中
            good_outputs.add(o)
        else:
            logging.error(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")  # 可能是多跳之前的上游节点连接导致的当前输出节点验证失败，可能会有多条错误信息
            errors += [(o, reasons)]  # 记录验证失败的输出类节点和对应的详细信息
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:  # 如果节点上游出现问题，那对应的下游节点也会标志为invalid，但是这些节点不带有error reasons；基于此规则将此类节点排除，不返回
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        logging.error(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            logging.error(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logging.error("Output will be ignored")  # 没有校验成功的输出类节点会被忽视

    if len(good_outputs) == 0:  # 一个工作流中必须包含验证成功的输出类接节点，如果数量为0，记录具体错误信息并返回
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)

MAXIMUM_HISTORY_SIZE = 10000

class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()  # 一个可重入锁（RLock），用于实现线程间的互斥访问；可重入锁允许同一线程多次获取锁，而不会造成死锁
        self.not_empty = threading.Condition(self.mutex)  # 创建了一个条件变量，条件变量通常用于线程间的通信和同步，它可以让一个线程等待某个条件的发生，另一个线程在满足条件时通知等待的线程
        self.task_counter = 0
        self.queue = []  # 存放任务的队列；内部存放的元素为(number, prompt_id, prompt, extra_data, outputs_to_execute)的元组，任务放入和弹出时通过heapq实现最小堆，number越小，优先级越高
        self.currently_running = {}  # 存放正在执行的任务
        self.history = {}  # 存放历史任务
        self.flags = {}  # 存放标志
        # 将自身又赋值给了传入server对象的prompt_queue属性，属于双向引用设计模式，适合需要双向通信但又要保持模块独立的场景
        # 即此处的server在接收外部请求后会将任务添加到self.queue中，而在队列内部又可通过调用self.server.queue_updated()向调用方发送消息
        server.prompt_queue = self

    def put(self, item):
        with self.mutex:  # 获取互斥锁
            heapq.heappush(self.queue, item)  # 将任务放入堆中；可进行优先级控制
            self.server.queue_updated()  # 调用server的queue_updated方法，通知客户端队列更新
            self.not_empty.notify()  # 唤醒等待该条件的线程；实现生产者-消费者模式中的信号机制

    def get(self, timeout=None):
        with self.not_empty:  # 获取条件变量
            while len(self.queue) == 0:  # 如果队列为空，则等待
                self.not_empty.wait(timeout=timeout)  # 等待put中的self.not_empty.notify信号；如果一直没有收到“有任务的信号”，最多等待timeout时间
                if timeout is not None and len(self.queue) == 0:  # 如果timeout不为None，并且队列为空，则返回None
                    return None
            item = heapq.heappop(self.queue)  # 从堆中弹出优先级最高的任务
            i = self.task_counter  # 获取任务计数器
            self.currently_running[i] = copy.deepcopy(item)  # 将任务添加到正在执行的任务列表中
            self.task_counter += 1  # 任务计数器加1
            self.server.queue_updated()  # 调用server的queue_updated方法，通知客户端队列更新
            return (item, i)  # 返回任务和任务计数器

    class ExecutionStatus(NamedTuple):
        status_str: Literal['success', 'error']
        completed: bool
        messages: List[str]

    def task_done(self, item_id, history_result,
                  status: Optional['PromptQueue.ExecutionStatus']):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())

            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": {},
                'status': status_dict,
            }
            self.history[prompt[1]].update(history_result)
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        out[k] = self.history[k]
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret = self.flags
                self.flags = {}
                return ret
            else:
                return self.flags.copy()
