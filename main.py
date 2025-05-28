import comfy.options
comfy.options.enable_args_parsing()  # 将comfy.options.args_parsing设置为True

import os
import importlib.util
import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys

if __name__ == "__main__":
    #NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'  # 禁用 Hugging Face Hub 的遥测数据收集
    os.environ['DO_NOT_TRACK'] = '1'  # 一个常用的环境变量，许多开源库和工具会检查此变量，以确定是否收集用户数据


setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

def apply_custom_paths():
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)  # 加载extra_model_paths.yaml文件中的配置

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):  # extra_model_paths_config中可能包含多个路径，所以需要遍历
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)  # 设置输出目录

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)  # 从文件路径创建一个模块规范
            module = importlib.util.module_from_spec(spec)  # 从规范创建一个模块
            spec.loader.exec_module(module)  # 动态加载并执行script_path对应的脚本
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    if args.disable_all_custom_nodes:  # 如果禁用了所有自定义节点，则直接返回
        return

    node_paths = folder_paths.get_folder_paths("custom_nodes")  # 获取自定义节点的路径
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)  # 获取自定义节点路径下的所有文件和文件夹
        node_prestartup_times = []  # 用于记录自定义节点预启动时间

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue  # 如果module_path是文件或以.disabled结尾或是__pycache__，则跳过

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()  # 记录开始时间
                success = execute_script(script_path)  # 执行脚本
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))  # 记录执行时间
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

apply_custom_paths()
execute_prestartup_script()


# Main code
import asyncio
import shutil
import threading
import gc


if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils

import execution
import server
from server import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:  # 如果设置了lru缓存大小，则使用LRU缓存
        cache_type = execution.CacheType.LRU
    elif args.cache_none:  # 如果设置了不使用缓存，则使用依赖感知的缓存
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=args.cache_lru)
    last_gc_collect = 0  # 上次垃圾回收时间
    need_gc = False  # 是否需要垃圾回收
    gc_collect_interval = 10.0  # 垃圾回收间隔

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)  # 如果需要垃圾回收，则将timeout设置为当前时间减去上次垃圾回收时间，如果不需要垃圾回收，则timeout为0

        queue_item = q.get(timeout=timeout)  # 从任务队列中获取任务
        if queue_item is not None:  # 如果获取到任务
            item, item_id = queue_item  # 解包任务
            execution_start_time = time.perf_counter()  # 记录开始时间
            prompt_id = item[1]  # 获取prompt_id；item的格式为(number, prompt_id, prompt, extra_data, outputs_to_execute)
            server_instance.last_prompt_id = prompt_id  # 更新last_prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])  # 执行任务
            need_gc = True
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))  # 更新处理完的任务状态，记录到历史队列中
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:  # 如果当前时间减去上次垃圾回收时间大于垃圾回收间隔，则进行垃圾回收
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()  # publish_loop会持续地从messages队列中获取消息，并将消息发送给客户端
    )  # 异步同时运行server.start_multi_address和server.publish_loop两个函数


def hijack_progress(server_instance):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()  # 采样过程若中断会排除异常
        progress = {"value": value, "max": total, "prompt_id": server_instance.last_prompt_id, "node": server_instance.last_node_id}

        server_instance.send_sync("progress", progress, server_instance.client_id)  # 将信息信息发送给客户端，其可以更新进度
        if preview_image is not None:
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server_instance.client_id)  # 如果有预览图像可用，通过服务器的 send_sync 方法将其发送给客户端

    comfy.utils.set_progress_bar_global_hook(hook)  # 设置全局进度条钩子，将其设置为内部定义的 hook 函数。这意味着每当进度更新时，都会调用 hook 函数来处理更新事件


def cleanup_temp():  # 清理临时文件夹
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()  # 创建一个新的事件循环
        asyncio.set_event_loop(asyncio_loop)  # 设置事件循环
    prompt_server = server.PromptServer(asyncio_loop)  # 创建PromptServer实例，处理HTTP请求的服务器

    hook_breaker_ac10a0.save_functions()
    nodes.init_extra_nodes(init_custom_nodes=not args.disable_all_custom_nodes, init_api_nodes=not args.disable_api_nodes)  # 动态加载自定义节点
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()  # 检查目前使用的显卡是否支持cuda-malloc

    prompt_server.add_routes()  # prompt_server对象添加hTTP路由
    hijack_progress(prompt_server)  # 设置进度条钩子

    threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server,)).start()  # 以守护子线程的方式启动任务处理函数prompt_worker，其中有while True，此子线程会一直存在，只有当主线程结束时才会终止

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():  # 异步起动函数
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)  # 运行事件循环，直到所有任务完成
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
