import bisect
import gc
import itertools
import psutil
import time
import torch
from typing import Sequence, Mapping, Dict
from comfy_execution.graph import DynamicPrompt
from abc import ABC, abstractmethod

import nodes

from comfy_execution.graph_utils import is_link

NODE_CLASS_CONTAINS_UNIQUE_ID: Dict[str, bool] = {}


def include_unique_id_in_input(class_type: str) -> bool:  # 判断节点类定义中的INPUT_TYPES是否包含UNIQUE_ID
    if class_type in NODE_CLASS_CONTAINS_UNIQUE_ID:
        return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    NODE_CLASS_CONTAINS_UNIQUE_ID[class_type] = "UNIQUE_ID" in class_def.INPUT_TYPES().get("hidden", {}).values()  # 如果节点类定义中的INPUT_TYPES包含UNIQUE_ID，则返回True
    return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]

class CacheKeySet(ABC):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        self.keys = {}
        self.subcache_keys = {}

    @abstractmethod
    async def add_keys(self, node_ids):
        raise NotImplementedError()

    def all_node_ids(self):
        return set(self.keys.keys())

    def get_used_keys(self):
        return self.keys.values()

    def get_used_subcache_keys(self):
        return self.subcache_keys.values()

    def get_data_key(self, node_id):
        return self.keys.get(node_id, None)

    def get_subcache_key(self, node_id):
        return self.subcache_keys.get(node_id, None)

class Unhashable:
    def __init__(self):
        self.value = float("NaN")

def to_hashable(obj):  # 将任意python对象递归地转换为可哈希的形式
    # So that we don't infinitely recurse since frozenset and tuples
    # are Sequences.
    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return obj
    elif isinstance(obj, Mapping):  # 如果obj是字典类型
        return frozenset([(to_hashable(k), to_hashable(v)) for k, v in sorted(obj.items())])  # frozenset是一种不可变的、可哈希的集合类型
    elif isinstance(obj, Sequence):  # 如果obj是列表、元组等序列类型
        return frozenset(zip(itertools.count(), [to_hashable(i) for i in obj]))  # zip函数将两个可迭代对象合并成一个，itertools.count()生成一个无限递增的计数器
    else:
        # TODO - Support other objects like tensors?
        return Unhashable()

class CacheKeySetID(CacheKeySet):
    def __init__(self, dynprompt: DynamicPrompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:  # 如果节点ID已经在keys中，则跳过
                continue
            if not self.dynprompt.has_node(node_id):  # 如果节点ID不存在，则跳过
                continue
            node = self.dynprompt.get_node(node_id)  # 获取节点信息
            self.keys[node_id] = (node_id, node["class_type"])  # 将节点ID和节点类型作为键值对存储在keys中
            self.subcache_keys[node_id] = (node_id, node["class_type"])  # 将节点ID和节点类型作为键值对存储在subcache_keys中

class CacheKeySetInputSignature(CacheKeySet):  # 创建基于节点输入结构和内容的缓存key
    def __init__(self, dynprompt: DynamicPrompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)  # 此处指定父类的初始化函数是为了将之前的缓存key全部清空
        self.dynprompt = dynprompt
        self.is_changed_cache = is_changed_cache

    def include_node_id_in_input(self) -> bool:  # 决定生成缓存键时是否默认模板节点ID
        return False  # 表示默认不考虑节点ID，只依赖于节点的输入内容；相同输入的节点可以共享缓存

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:  # 如果当前节点ID已经在keys中，则跳过
                continue
            if not self.dynprompt.has_node(node_id):  # 如果当前节点ID工作流中不存在，则跳过
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = await self.get_node_signature(self.dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    async def get_node_signature(self, dynprompt, node_id):
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        signature.append(await self.get_immediate_node_signature(dynprompt, node_id, order_mapping))
        for ancestor_id in ancestors:
            signature.append(await self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))
        return to_hashable(signature)

    async def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        if not dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return [float("NaN")]
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, await self.is_changed_cache.get(node_id)]
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or include_unique_id_in_input(class_type):
            signature.append(node_id)
        inputs = node["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                signature.append((key, inputs[key]))  # 将输入的键和输入的值添加到缓存键中
        return signature

    # This function returns a list of all ancestors of the given node. The order of the list is
    # deterministic based on which specific inputs the ancestor is connected by.
    def get_ordered_ancestry(self, dynprompt, node_id):  # 给定一个节点ID，返回该节点所有祖先节点的列表，并按照它们在节点输入中的顺序排列
        ancestors = []
        order_mapping = {}
        self.get_ordered_ancestry_internal(dynprompt, node_id, ancestors, order_mapping)
        return ancestors, order_mapping

    def get_ordered_ancestry_internal(self, dynprompt, node_id, ancestors, order_mapping):  # 递归地获取所有祖先节点，并按照它们在节点输入中的顺序排列
        if not dynprompt.has_node(node_id):
            return
        inputs = dynprompt.get_node(node_id)["inputs"]
        input_keys = sorted(inputs.keys())
        for key in input_keys:
            if is_link(inputs[key]):
                ancestor_id = inputs[key][0]
                if ancestor_id not in order_mapping:  # 如果祖先节点ID不在order_mapping中，则将其添加到ancestors列表中，并记录其在列表中的位置
                    ancestors.append(ancestor_id)  # 将祖先节点ID添加到ancestors列表中
                    order_mapping[ancestor_id] = len(ancestors) - 1  # 记录祖先节点ID在列表中的位置
                    self.get_ordered_ancestry_internal(dynprompt, ancestor_id, ancestors, order_mapping)  # 递归地获取祖先节点的所有祖先节点

class BasicCache:
    def __init__(self, key_class):
        self.key_class = key_class  # 具体使用的缓存key的类
        self.initialized = False
        self.dynprompt: DynamicPrompt
        self.cache_key_set: CacheKeySet
        self.cache = {}
        self.subcaches = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed_cache)
        await self.cache_key_set.add_keys(node_ids)
        self.is_changed_cache = is_changed_cache
        self.initialized = True

    def all_node_ids(self):
        assert self.initialized
        node_ids = self.cache_key_set.all_node_ids()
        for subcache in self.subcaches.values():  # 遍历所有子缓存中的值
            node_ids = node_ids.union(subcache.all_node_ids())
        return node_ids

    def _clean_cache(self):
        preserve_keys = set(self.cache_key_set.get_used_keys())  # 获取工作流更新后的缓存key，是需要保留的
        to_remove = []
        for key in self.cache:  # 遍历原始缓存中的所有缓存key
            if key not in preserve_keys:  # 如果缓存key不在需要保留的缓存key中，则删除
                to_remove.append(key)
        for key in to_remove:  # 也会将不能重复的节点对应的缓存删除，即使该节点在更新后的工作流中仍然存在，但其需要重新执行
            del self.cache[key]

    def _clean_subcaches(self):
        preserve_subcaches = set(self.cache_key_set.get_used_subcache_keys())  # 获取工作流更新后的缓存key，是需要保留的

        to_remove = []
        for key in self.subcaches:
            if key not in preserve_subcaches:
                to_remove.append(key)
        for key in to_remove:
            del self.subcaches[key]

    def clean_unused(self):
        assert self.initialized
        self._clean_cache()
        self._clean_subcaches()

    def poll(self, **kwargs):
        pass

    def _set_immediate(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value  # 将给定节点的缓存值设置为value

    def _get_immediate(self, node_id):
        if not self.initialized:  # 如果缓存未初始化
            return None  # 返回None
        cache_key = self.cache_key_set.get_data_key(node_id)  # 获取给定节点的缓存key
        if cache_key in self.cache:  # 如果缓存key在缓存中
            return self.cache[cache_key]
        else:
            return None

    async def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        await subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
        return subcache

    def _get_subcache(self, node_id):  # 获取给定节点的子缓存
        assert self.initialized
        subcache_key = self.cache_key_set.get_subcache_key(node_id)  # 获取给定节点的子缓存key
        if subcache_key in self.subcaches:  # 如果子缓存key在子缓存中
            return self.subcaches[subcache_key]  # 返回子缓存
        else:
            return None  # 如果子缓存key不在子缓存中，则返回None

    def recursive_debug_dump(self):
        result = []
        for key in self.cache:
            result.append({"key": key, "value": self.cache[key]})
        for key in self.subcaches:
            result.append({"subcache_key": key, "subcache": self.subcaches[key].recursive_debug_dump()})
        return result

class HierarchicalCache(BasicCache):
    def __init__(self, key_class):
        super().__init__(key_class)

    def _get_cache_for(self, node_id):  # 递归向前查找给定节点的缓存
        assert self.dynprompt is not None
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is None:  # 如果给定节点没有父节点
            return self

        hierarchy = []
        while parent_id is not None:  # 一直向前找父节点
            hierarchy.append(parent_id)
            parent_id = self.dynprompt.get_parent_node_id(parent_id)

        cache = self
        for parent_id in reversed(hierarchy):  # 将最早的父节点向后遍历
            cache = cache._get_subcache(parent_id)  # 获取给定节点的子缓存
            if cache is None:
                return None  # 如果子缓存为空，则返回None
        return cache  # 返回给定节点的缓存

    def get(self, node_id):
        cache = self._get_cache_for(node_id)  # 获取给定节点的缓存
        if cache is None:
            return None  # 如果缓存为空，则返回None
        return cache._get_immediate(node_id)  # 返回给定节点当前的缓存值

    def set(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        cache._set_immediate(node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return await cache._ensure_subcache(node_id, children_ids)

class NullCache:

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        pass

    def all_node_ids(self):
        return []

    def clean_unused(self):
        pass

    def poll(self, **kwargs):
        pass

    def get(self, node_id):
        return None

    def set(self, node_id, value):
        pass

    async def ensure_subcache_for(self, node_id, children_ids):
        return self

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100):
        super().__init__(key_class)
        self.max_size = max_size
        self.min_generation = 0
        self.generation = 0
        self.used_generation = {}
        self.children = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)
        self.generation += 1
        for node_id in node_ids:
            self._mark_used(node_id)

    def clean_unused(self):
        while len(self.cache) > self.max_size and self.min_generation < self.generation:
            self.min_generation += 1
            to_remove = [key for key in self.cache if self.used_generation[key] < self.min_generation]
            for key in to_remove:
                del self.cache[key]
                del self.used_generation[key]
                if key in self.children:
                    del self.children[key]
        self._clean_subcaches()

    def get(self, node_id):
        self._mark_used(node_id)
        return self._get_immediate(node_id)

    def _mark_used(self, node_id):
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.used_generation[cache_key] = self.generation

    def set(self, node_id, value):
        self._mark_used(node_id)
        return self._set_immediate(node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        # Just uses subcaches for tracking 'live' nodes
        await super()._ensure_subcache(node_id, children_ids)

        await self.cache_key_set.add_keys(children_ids)
        self._mark_used(node_id)
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.children[cache_key] = []
        for child_id in children_ids:
            self._mark_used(child_id)
            self.children[cache_key].append(self.cache_key_set.get_data_key(child_id))
        return self


#Iterating the cache for usage analysis might be expensive, so if we trigger make sure
#to take a chunk out to give breathing space on high-node / low-ram-per-node flows.

RAM_CACHE_HYSTERESIS = 1.1

#This is kinda in GB but not really. It needs to be non-zero for the below heuristic
#and as long as Multi GB models dwarf this it will approximate OOM scoring OK

RAM_CACHE_DEFAULT_RAM_USAGE = 0.1

#Exponential bias towards evicting older workflows so garbage will be taken out
#in constantly changing setups.

RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER = 1.3

class RAMPressureCache(LRUCache):

    def __init__(self, key_class):
        super().__init__(key_class, 0)
        self.timestamps = {}

    def clean_unused(self):
        self._clean_subcaches()

    def set(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        super().set(node_id, value)

    def get(self, node_id):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        return super().get(node_id)

    def poll(self, ram_headroom):
        def _ram_gb():
            return psutil.virtual_memory().available / (1024**3)

        if _ram_gb() > ram_headroom:
            return
        gc.collect()
        if _ram_gb() > ram_headroom:
            return

        clean_list = []

        for key, (outputs, _), in self.cache.items():
            oom_score =  RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER ** (self.generation - self.used_generation[key])

            ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
            def scan_list_for_ram_usage(outputs):
                nonlocal ram_usage
                if outputs is None:
                    return
                for output in outputs:
                    if isinstance(output, list):
                        scan_list_for_ram_usage(output)
                    elif isinstance(output, torch.Tensor) and output.device.type == 'cpu':
                        #score Tensors at a 50% discount for RAM usage as they are likely to
                        #be high value intermediates
                        ram_usage += (output.numel() * output.element_size()) * 0.5
                    elif hasattr(output, "get_ram_usage"):
                        ram_usage += output.get_ram_usage()
            scan_list_for_ram_usage(outputs)

            oom_score *= ram_usage
            #In the case where we have no information on the node ram usage at all,
            #break OOM score ties on the last touch timestamp (pure LRU)
            bisect.insort(clean_list, (oom_score, self.timestamps[key], key))

        while _ram_gb() < ram_headroom * RAM_CACHE_HYSTERESIS and clean_list:
            _, _, key = clean_list.pop()
            del self.cache[key]
            gc.collect()
