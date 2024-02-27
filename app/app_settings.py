import os
import json
from aiohttp import web


class AppSettings():
    def __init__(self, user_manager):
        self.user_manager = user_manager

    def get_settings(self, request):  # 获取用户设置
        file = self.user_manager.get_request_user_filepath(
            request, "comfy.settings.json")  # 从请求中获取对应用户的设置文件路径
        if os.path.isfile(file):
            with open(file) as f:
                return json.load(f)
        else:
            return {}

    def save_settings(self, request, settings):  # 保存用户设置
        file = self.user_manager.get_request_user_filepath(
            request, "comfy.settings.json")
        with open(file, "w") as f:
            f.write(json.dumps(settings, indent=4))

    def add_routes(self, routes):
        @routes.get("/settings")  # 获取用户全部设置信息
        async def get_settings(request):
            return web.json_response(self.get_settings(request))

        @routes.get("/settings/{id}")  # 获取单个设置的值
        async def get_setting(request):
            value = None
            settings = self.get_settings(request)
            setting_id = request.match_info.get("id", None)  # 获取传入的路由参数id
            if setting_id and setting_id in settings:
                value = settings[setting_id]
            return web.json_response(value)

        @routes.post("/settings")  # 追加设置信息
        async def post_settings(request):
            settings = self.get_settings(request)
            new_settings = await request.json()
            self.save_settings(request, {**settings, **new_settings})
            return web.Response(status=200)

        @routes.post("/settings/{id}")  # 修改单个设置的值
        async def post_setting(request):
            setting_id = request.match_info.get("id", None)
            if not setting_id:
                return web.Response(status=400)
            settings = self.get_settings(request)
            settings[setting_id] = await request.json()
            self.save_settings(request, settings)
            return web.Response(status=200)