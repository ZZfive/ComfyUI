import os
import sys
import asyncio
import traceback

import nodes
import folder_paths
import execution
import uuid
import urllib
import json
import glob
import struct
import ssl
import socket
import ipaddress
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

import aiohttp
from aiohttp import web
import logging

import mimetypes
from comfy.cli_args import args
import comfy.utils
import comfy.model_management
import node_helpers
from comfyui_version import __version__
from app.frontend_management import FrontendManager
from app.user_manager import UserManager
from app.model_manager import ModelFileManager
from app.custom_node_manager import CustomNodeManager
from typing import Optional, Union
from api_server.routes.internal.internal_routes import InternalRoutes

class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3

async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError, BrokenPipeError, ConnectionError) as err:
        logging.warning("send error: {}".format(err))

@web.middleware
async def cache_control(request: web.Request, handler):
    response: web.Response = await handler(request)
    if request.path.endswith('.js') or request.path.endswith('.css') or request.path.endswith('index.json'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response

@web.middleware
async def compress_body(request: web.Request, handler):  # 压缩HTTP响应体的中间件
    accept_encoding = request.headers.get("Accept-Encoding", "")  # 获取客户端支持的压缩方式
    response: web.Response = await handler(request)
    if not isinstance(response, web.Response):  # 如果响应不是web.Response，直接返回
        return response
    if response.content_type not in ["application/json", "text/plain"]:  # 只压缩JSON和纯文本
        return response
    if response.body and "gzip" in accept_encoding:
        response.enable_compression()  # 启动压缩
    return response


def create_cors_middleware(allowed_origin: str):  # 返回一个跨资源共享中间件
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin  # 允许的源
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'  # 允许的HTTP方法
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'  # 允许的请求头
        response.headers['Access-Control-Allow-Credentials'] = 'true'  # 允许携带凭证
        return response

    return cors_middleware

def is_loopback(host):
    if host is None:
        return False
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
        else:
            return False
    except:
        pass

    loopback = False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in r:
                if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                    return loopback
                else:
                    loopback = True
        except socket.gaierror:
            pass

    return loopback


def create_origin_only_middleware():
    @web.middleware
    async def origin_only_middleware(request: web.Request, handler):
        #this code is used to prevent the case where a random website can queue comfy workflows by making a POST to 127.0.0.1 which browsers don't prevent for some dumb reason.
        #in that case the Host and Origin hostnames won't match
        #I know the proper fix would be to add a cookie but this should take care of the problem in the meantime
        if 'Host' in request.headers and 'Origin' in request.headers:
            host = request.headers['Host']
            origin = request.headers['Origin']
            host_domain = host.lower()
            parsed = urllib.parse.urlparse(origin)
            origin_domain = parsed.netloc.lower()
            host_domain_parsed = urllib.parse.urlsplit('//' + host_domain)

            #limit the check to when the host domain is localhost, this makes it slightly less safe but should still prevent the exploit
            loopback = is_loopback(host_domain_parsed.hostname)  # 检查主机域名是否为本地回环地址

            if parsed.port is None: #if origin doesn't have a port strip it from the host to handle weird browsers, same for host
                host_domain = host_domain_parsed.hostname
            if host_domain_parsed.port is None:
                origin_domain = parsed.hostname

            if loopback and host_domain is not None and origin_domain is not None and len(host_domain) > 0 and len(origin_domain) > 0:
                if host_domain != origin_domain:  # 防止CSRF攻击，限制只允许同源访问
                    logging.warning("WARNING: request with non matching host and origin {} != {}, returning 403".format(host_domain, origin_domain))
                    return web.Response(status=403)

        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        return response

    return origin_only_middleware

class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self  # 将自身赋值给PromptServer.instance，不严格的单例模式，可以创建多个实例，只是后续的实例会覆盖instance

        mimetypes.init()  # 正确配置文件的MIME类型
        mimetypes.add_type('application/javascript; charset=utf-8', '.js')
        mimetypes.add_type('image/webp', '.webp')

        self.user_manager = UserManager()  # 用户管理
        self.model_file_manager = ModelFileManager()  # 模型管理
        self.custom_node_manager = CustomNodeManager()  # 自定义节点管理
        self.internal_routes = InternalRoutes(self)  # 内部路由
        self.supports = ["custom_nodes_from_web"]
        self.prompt_queue = None
        self.loop = loop
        self.messages = asyncio.Queue()  # 从任务执行端获取信息，再将信息发送出去
        self.client_session:Optional[aiohttp.ClientSession] = None
        self.number = 0

        middlewares = [cache_control]
        if args.enable_compress_response_body:
            middlewares.append(compress_body)

        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))  # 允许不同源共享
        else:
            middlewares.append(create_origin_only_middleware())  # 只允许同源访问

        max_upload_size = round(args.max_upload_size * 1024 * 1024)  # 设置最大上传大小
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)  # 初始化服务对象，同时配置中间件
        self.sockets = dict()  # 存储客户端ID与WebSocketResponse的映射
        self.web_root = (
            FrontendManager.init_frontend(args.front_end_version)  # 返回comfyui-frontend-package这个库的static文件夹路径
            if args.front_end_root is None
            else args.front_end_root
        )  # 前端的根目录
        logging.info(f"[Prompt Server] web root: {self.web_root}")
        routes = web.RouteTableDef()  # 初始化路由
        self.routes = routes
        self.last_node_id = None  # 用于记录当前处理任务执行节点信息
        self.client_id = None  # 用于记录当前处理任务对应的客户端ID 

        self.on_prompt_handlers = []  # 用于存储on_prompt事件的回调函数

        @routes.get('/ws')
        async def websocket_handler(request):  # WebSocket连接处理接口
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')  # 获取客户端ID
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)  # 移除旧的客户端ID，因为后面会重新关联，所以相当于复用
            else:
                sid = uuid.uuid4().hex  # 生成新的客户端ID

            self.sockets[sid] = ws  # 将客户端ID与WebSocketResponse关联

            try:
                # Send initial state to the new client
                await self.send("status", { "status": self.get_queue_info(), 'sid': sid }, sid)  # 发送初始状态给新客户端
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:  # 如果当前客户端的请求正在处理时，将处理的节点信息返回给客户端
                    await self.send("executing", { "node": self.last_node_id }, sid)  # 给客户端发送中间状态信息

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid, None)  # 移除客户端ID
            return ws

        @routes.get("/")
        async def get_root(request):
            response = web.FileResponse(os.path.join(self.web_root, "index.html"))
            response.headers['Cache-Control'] = 'no-cache'
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @routes.get("/embeddings")
        def get_embeddings(self):  # embedding文件查询接口
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/models")
        def list_model_types(request):  # 获取所有可用的模型类型
            model_types = list(folder_paths.folder_names_and_paths.keys())

            return web.json_response(model_types)

        @routes.get("/models/{folder}")
        async def get_models(request):  # 模型文件查询接口
            folder = request.match_info.get("folder", None)
            if not folder in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = folder_paths.get_filename_list(folder)
            return web.json_response(files)

        @routes.get("/extensions")
        async def get_extensions(request):  # 扩展文件查询接口
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)

            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type):  # 根据类型获取文件夹路径
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def compare_image_hash(filepath, image):
            hasher = node_helpers.hasher()  # 获取哈希函数

            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hasher()
                b = hasher()
                with open(filepath, "rb") as f:
                    a.update(f.read())  # 读取从filepath路径读取的文件内容，并更新到a中
                    b.update(image.file.read())  # 读取从image.file路径读取的文件内容，并更新到b中
                    image.file.seek(0)
                    f.close()
                return a.hexdigest() == b.hexdigest()  # 比较两个哈希值是否相等
            return False

        def image_upload(post, image_save_function=None):  # 此处的post不是一个字典，而是http请求中的一个请求体
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)  # 获取上传文件的目录和类型

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):  # 如果文件已存在
                        if compare_image_hash(filepath, image): #compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True  # 目前传入的图片路径名已存在，并且与当前图片的哈希值相同，则认为图片是重复的
                            break
                        filename = f"{split[0]} ({i}){split[1]}"  # 如果文件已存在，则将文件名加上（i），并更新filepath
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:  # 如果图片不是重复的
                    if image_save_function is not None:
                        image_save_function(image, post, filepath)  # 如果image_save_function不为空，则调用image_save_function函数
                    else:
                        with open(filepath, "wb") as f:  # 如果image_save_function为空，则将图片保存到filepath路径
                            f.write(image.file.read())

                return web.json_response({"name" : filename, "subfolder": subfolder, "type": image_upload_type})
            else:
                return web.Response(status=400)

        @routes.post("/upload/image")
        async def upload_image(request):  # 上传图片接口；前端运行如图生图此类工作流时，要先调用此接口将图片上传到服务端
            post = await request.post()
            return image_upload(post)


        @routes.post("/upload/mask")
        async def upload_mask(request):  # 上传掩码图片接口；在LoadImage节点中给图片绘制了mask信息后保存时前端会调用此接口
            post = await request.post()

            def image_save_function(image, post, filepath):  # 相对于上传图片接口，就是传入此处定义的图片保存函数image_save_function
                original_ref = json.loads(post.get("original_ref"))  # 当前上传的mask的对应的原始图片上传时设置的filename、type、subfolder信息
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil,'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')  # 将原始图片转换为RGBA模式
                        mask_pil = Image.open(image.file).convert('RGBA')  # 将上传的mask图片转换为RGBA模式

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')  # 获取mask图片的alpha通道
                        original_pil.putalpha(new_alpha)  # 将mask图片的alpha通道复制到原始图片的alpha通道
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)  # 将最终合并后的RGBA图片保存到filepath路径

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                filename,output_dir = folder_paths.annotated_filepath(filename)

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = request.rel_url.query.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if "subfolder" in request.rel_url.query:
                    full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                filename = os.path.basename(filename)
                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    if 'preview' in request.rel_url.query:
                        with Image.open(file) as img:
                            preview_info = request.rel_url.query['preview'].split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        # Get content type from mimetype, defaulting to 'application/octet-stream'
                        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

                        # For security, force certain extensions to download instead of display
                        file_extension = os.path.splitext(filename)[1].lower()
                        if file_extension in {'.html', '.htm', '.js', '.css'}:
                            content_type = 'application/octet-stream'  # Forces download

                        return web.FileResponse(
                            file,
                            headers={
                                "Content-Disposition": f"filename=\"{filename}\"",
                                "Content-Type": content_type
                            }
                        )

            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):  # 查看模型文件的metadata信息接口
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if not "filename" in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = comfy.utils.safetensors_header(safetensors_path, max_size=1024*1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if not "__metadata__" in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def system_stats(request):
            device = comfy.model_management.get_torch_device()
            device_name = comfy.model_management.get_torch_device_name(device)
            cpu_device = comfy.model_management.torch.device("cpu")
            ram_total = comfy.model_management.get_total_memory(cpu_device)
            ram_free = comfy.model_management.get_free_memory(cpu_device)
            vram_total, torch_vram_total = comfy.model_management.get_total_memory(device, torch_total_too=True)
            vram_free, torch_vram_free = comfy.model_management.get_free_memory(device, torch_free_too=True)

            system_stats = {
                "system": {
                    "os": os.name,
                    "ram_total": ram_total,
                    "ram_free": ram_free,
                    "comfyui_version": __version__,
                    "python_version": sys.version,
                    "pytorch_version": comfy.model_management.torch_version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded",
                    "argv": sys.argv
                },
                "devices": [
                    {
                        "name": device_name,
                        "type": device.type,
                        "index": device.index,
                        "vram_total": vram_total,
                        "vram_free": vram_free,
                        "torch_vram_total": torch_vram_total,
                        "torch_vram_free": torch_vram_free,
                    }
                ]
            }
            return web.json_response(system_stats)

        @routes.get("/prompt")
        async def get_prompt(request):  # 获取当前队列信息接口
            return web.json_response(self.get_queue_info())

        def node_info(node_class):  # 获取节点信息接口
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
            info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")  # 获取节点所在的python模块
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY

            if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
                info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

            if getattr(obj_class, "DEPRECATED", False):
                info['deprecated'] = True
            if getattr(obj_class, "EXPERIMENTAL", False):
                info['experimental'] = True

            if hasattr(obj_class, 'API_NODE'):
                info['api_node'] = obj_class.API_NODE
            return info

        @routes.get("/object_info")
        async def get_object_info(request):  # 获取所有节点信息接口
            with folder_paths.cache_helper:
                out = {}
                for x in nodes.NODE_CLASS_MAPPINGS:
                    try:
                        out[x] = node_info(x)
                    except Exception:
                        logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                        logging.error(traceback.format_exc())
                return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):  # 获取指定节点信息接口
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):  # 获取历史记录接口
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history_prompt_id(request):  # 获取指定prompt_id的历史记录接口
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):  # 获取当前队列信息接口
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue()
            queue_info['queue_running'] = current_queue[0]
            queue_info['queue_pending'] = current_queue[1]
            return web.json_response(queue_info)

        @routes.post("/prompt")
        async def post_prompt(request):  # 提交prompt接口
            logging.info("got prompt")
            json_data =  await request.json()  # 获取请求体中的JSON数据，其中会包含api形式的工作流json对象
            json_data = self.trigger_on_prompt(json_data)

            if "number" in json_data:
                number = float(json_data['number'])  # 如果json_data中存在number，则使用json_data中的number
            else:
                number = self.number  # 如果json_data中没有number，则使用self.number
                if "front" in json_data:
                    if json_data['front']:  # 如果json_data中存在front，并且为True，则将number设置为-number
                        number = -number  # 取反后会在任务队列中有最高的优先级，最先被执行

                self.number += 1  # 任务数+1

            if "prompt" in json_data:
                prompt = json_data["prompt"]  # 如果json_data中存在prompt，则使用json_data中的prompt
                valid = execution.validate_prompt(prompt)  # 验证prompt是否有效
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if valid[0]:  # 工作流校验成功
                    prompt_id = str(uuid.uuid4())  # 生成当前工作流的唯一id
                    outputs_to_execute = valid[2]  # 获取校验成功的输出类节点
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))  # 将校验成功的任务放入任务队列中
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    logging.warning("invalid prompt: {}".format(valid[1]))
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                error = {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {}
                }
                return web.json_response({"error": error, "node_errors": {}}, status=400)

        @routes.post("/queue")
        async def post_queue(request):  # 队列操作接口；可清空任务队列和删除指定任务
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()  # 清空任务队列
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)  # 从任务队列中删除指定任务

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):  # 中断处理接口
            nodes.interrupt_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):  # 释放内存接口
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):  # 历史记录操作接口；可清空历史记录和删除指定历史记录
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)

    async def setup(self):
        timeout = aiohttp.ClientTimeout(total=None) # no timeout
        self.client_session = aiohttp.ClientSession(timeout=timeout)  # 创建HTTP客户端

    def add_routes(self):
        self.user_manager.add_routes(self.routes)  # 添加用户管理路由
        self.model_file_manager.add_routes(self.routes)  # 添加模型文件管理路由
        self.custom_node_manager.add_routes(self.routes, self.app, nodes.LOADED_MODULE_DIRS.items())  # 添加自定义节点管理路由
        self.app.add_subapp('/internal', self.internal_routes.get_app())  # 将内部路由的子应用挂在到app的指定路径前缀下

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        # 在每个路由前添加 /api 前缀，以便于委托匹配。这对于需要转发除静态文件服务之外所有信息的前端开发服务器非常有用。
        # 目前，旧的无前缀端点和新的带前缀端点均受支持。
        api_routes = web.RouteTableDef()
        for route in self.routes:  # 遍历所有已定义的路由
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):  # 只处理非静态资源的路由
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)  # 为每个路由创建一个带 /api 前缀的副本
        self.app.add_routes(api_routes)  # 添加带 /api 前缀的路由
        self.app.add_routes(self.routes)  # 添加原始路由，即同时兼容了两类路由

        # Add routes from web extensions.
        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([web.static('/extensions/' + name, dir)])  # 为每个扩展添加静态文件路由

        workflow_templates_path = FrontendManager.templates_path()
        if workflow_templates_path:
            self.app.add_routes([
                web.static('/templates', workflow_templates_path)
            ])  # 添加comfyui_workflow_templates这个库中提供的官方工作流模板

        self.app.add_routes([
            web.static('/', self.web_root),
        ])  # 将应用程序的根目录设置为静态文件服务，这意味着可以通过服务应用的根目录访问应用程序的静态文件，就是前端页面

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()  # 获取队列中剩余的任务数量
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)  # 发送图像
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)  # 发送字节数据
        else:
            await self.send_json(event, data, sid)  # 发送JSON数据

    def encode_bytes(self, event, data):  # 将事件和数据编码为字节数据
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)  # 将事件转换为字节数据
        message = bytearray(packed)  # 将字节数据转换为字节数组
        message.extend(data)  # 将数据添加到字节数组中
        return message

    async def send_image(self, image_data, sid=None):  # 发送图像
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)  # 将预览图像发送给客户端

    async def send_bytes(self, event, data, sid=None):  # 向 WebSocket 客户端发送字节消息
        message = self.encode_bytes(event, data)  # 将事件和数据编码为字节数据

        if sid is None:  # 如果sid不存在，会将data广播给所有ws连接
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)  # 将消息发送给客户端
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)  # 将消息发送sid对应的客户端

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)  # 将信息广播给所有的客户端
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)  # 将消息发送sid对应的客户端

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(  # 确保消息队列的线程安全性
            self.messages.put_nowait, (event, data, sid))  # 将消息添加到消息队列中

    def queue_updated(self):  # 在队列更新时，调用 send_sync 方法发送队列状态信息
        self.send_sync("status", { "status": self.get_queue_info() })  # 将队列状态发送给客户端

    async def publish_loop(self):  # 持续地从messages队列中获取消息，并将消息发送给客户端
        while True:
            msg = await self.messages.get()  # 从消息队列中获取消息
            await self.send(*msg)  # 将消息发送给客户端

    async def start(self, address, port, verbose=True, call_on_start=None):
        await self.start_multi_address([(address, port)], call_on_start=call_on_start)

    async def start_multi_address(self, addresses, call_on_start=None, verbose=True):  # 将web应用与多个地址绑定，允许服务同时在多个网络接口上监听连接请求
        runner = web.AppRunner(self.app, access_log=None)  # 创建aiohttp.web.AppRunner实例来管理web应用
        await runner.setup()  # app应用运行时初始化；应用生命周期时间触发、中间件初始化等
        ssl_ctx = None
        scheme = "http"  # 默认使用http协议
        if args.tls_keyfile and args.tls_certfile:  # 如果设置了TLS密钥和证书文件
                ssl_ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER, verify_mode=ssl.CERT_NONE)
                ssl_ctx.load_cert_chain(certfile=args.tls_certfile,
                                keyfile=args.tls_keyfile)  # 加载TLS证书和密钥
                scheme = "https"  # 设置scheme为https协议

        if verbose:
            logging.info("Starting server\n")
        for addr in addresses:  # 遍历地址列表，多地址监听
            address = addr[0]
            port = addr[1]
            site = web.TCPSite(runner, address, port, ssl_context=ssl_ctx)  # 创建一个TCPSite对象，用于管理TCP连接
            await site.start()  # 启动TCPSite，开始监听TCP连接；网络服务初始化，启动网络监听等

            if not hasattr(self, 'address'):  # 保存第一个地址作为默认地址
                self.address = address #TODO: remove this
                self.port = port
            # 格式化地址显示
            if ':' in address:
                address_print = "[{}]".format(address)
            else:
                address_print = address

            if verbose:
                logging.info("To see the GUI go to: {}://{}:{}".format(scheme, address_print, port))

        if call_on_start is not None:
            call_on_start(scheme, self.address, self.port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception:
                logging.warning("[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

        return json_data

    def send_progress_text(
        self, text: Union[bytes, bytearray, str], node_id: str, sid=None
    ):
        if isinstance(text, str):
            text = text.encode("utf-8")
        node_id_bytes = str(node_id).encode("utf-8")

        # Pack the node_id length as a 4-byte unsigned integer, followed by the node_id bytes
        message = struct.pack(">I", len(node_id_bytes)) + node_id_bytes + text

        self.send_sync(BinaryEventTypes.TEXT, message, sid)
