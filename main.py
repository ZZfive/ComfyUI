import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time

def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")  # 获取自定义节点路径
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue  # 能加在的节点需要时路径，module_path不是路径直接跳过

            script_path = os.path.join(module_path, "prestartup_script.py")  # 找到自定义节点目录下的prestartup_script.py文件
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)  # 执行自定义节点中的prestartup_script.py文件
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

execute_prestartup_script()


# Main code
import asyncio
import itertools
import shutil
import threading
import gc

from comfy.cli_args import args
import logging

if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils
import yaml

import execution  # 推理执行
import server  # web服务
from server import BinaryEventTypes
from nodes import init_custom_nodes
import comfy.model_management

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()  # 获取device对象
    device_name = comfy.model_management.get_torch_device_name(device)  # 获取device名称
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")

def prompt_worker(q, server):  # 复杂从队列中获取任务并执行
    e = execution.PromptExecutor(server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)  # 循环等待从队列中获取任务，超时时间为 1000 秒
        if queue_item is not None:  # 执行获取到的任务，其中包括设置执行开始时间、执行任务、处理执行结果等操作
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])  # 开始执行任务
            need_gc = True
            q.task_done(item_id,
                        e.outputs_ui,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))  # 将处理完的任务存储，并通知服务器队列已更新
            if server.client_id is not None:
                server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):  # 根据队列中的标志执行相应的操作，如释放内存、清空模型、执行垃圾回收等
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:  # 定期执行垃圾回收，间隔时间为 gc_collect_interval
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False

async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    # asyncio.gather同时运行两个异步任务start和publish_loop
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())


def hijack_progress(server):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server.last_prompt_id, "node": server.last_node_id}

        server.send_sync("progress", progress, server.client_id)  # 将进度信息发送给客户端，便于其更新进度
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)  # 如果有预览图像可用，通过服务器的 send_sync 方法将其发送给客户端
    comfy.utils.set_progress_bar_global_hook(hook)  # 设置全局进度条钩子，将其设置为内部定义的 hook 函数。这意味着每当进度更新时，都会调用 hook 函数来处理更新事件


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)  # 递归删除路径temp_dir


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                logging.info("Adding extra search path {} {}".format(x, full_path))
                folder_paths.add_model_folder_path(x, full_path)


if __name__ == "__main__":
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater() #  更新 Windows 平台的更新器脚本和批处理文件
        except:
            pass

    loop = asyncio.new_event_loop()  # 创建新的事件循环对象
    asyncio.set_event_loop(loop)  # 将新创建的事件循环对象设置为当前的事件循环
    server = server.PromptServer(loop)  # 整个服务对象
    q = execution.PromptQueue(server)  # 维护生成任务的队列

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")  # 获取额外模型配置文件路径
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)  # 加载额外模型路径

    if args.extra_model_paths_config:  # 如果命令行参数中指定了extra_model_paths_config，加载对应模型路径
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()  # 加载Comfyui自带和自定义节点

    cuda_malloc_warning()          

    server.add_routes()  # 添加接口路由
    hijack_progress(server)  # 劫持server的进度，将进度信息发送给服务器

    threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()  # 创建、启动守护线程后台运行prompt_worker

    if args.output_directory:  # 设置图片输出路径
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    if args.auto_launch:  # 自动启动ui界面
        def startup_server(address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            webbrowser.open(f"http://{address}:{port}")
        call_on_start = startup_server

    try:
        loop.run_until_complete(run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start))
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
