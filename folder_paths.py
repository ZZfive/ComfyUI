from __future__ import annotations

import os
import time
import mimetypes
import logging
from typing import Literal, List
from collections.abc import Collection

from comfy.cli_args import args

supported_pt_extensions: set[str] = {'.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft'}  # 支持的模型文件扩展名

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}  # value是一个元组，第一个元素是存放一种数据类型的路径列表，一种数据类型可以存储在多个路径下，第二个元素也是一个集合，存放对应该数据类型支持的文件扩展名

# --base-directory - Resets all default pahts configured in folder_paths with a new base path
if args.base_directory:  # 如果指定了--base-directory参数，则使用指定的路径作为基础路径
    base_path = os.path.abspath(args.base_directory)
else:
    base_path = os.path.dirname(os.path.realpath(__file__))  # 如果没有指定--base-directory参数，则使用当前文件的目录作为基础路径

models_dir = os.path.join(base_path, "models")
folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["text_encoders"] = ([os.path.join(models_dir, "text_encoders"), os.path.join(models_dir, "clip")], supported_pt_extensions)
folder_names_and_paths["diffusion_models"] = ([os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")], supported_pt_extensions)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], ["folder"])
folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)

folder_names_and_paths["controlnet"] = ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
folder_names_and_paths["gligen"] = ([os.path.join(models_dir, "gligen")], supported_pt_extensions)

folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)

folder_names_and_paths["latent_upscale_models"] = ([os.path.join(models_dir, "latent_upscale_models")], supported_pt_extensions)

folder_names_and_paths["custom_nodes"] = ([os.path.join(base_path, "custom_nodes")], set())

folder_names_and_paths["hypernetworks"] = ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions)

folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)

folder_names_and_paths["classifiers"] = ([os.path.join(models_dir, "classifiers")], {""})

folder_names_and_paths["model_patches"] = ([os.path.join(models_dir, "model_patches")], supported_pt_extensions)

folder_names_and_paths["audio_encoders"] = ([os.path.join(models_dir, "audio_encoders")], supported_pt_extensions)

output_directory = os.path.join(base_path, "output")
temp_directory = os.path.join(base_path, "temp")
input_directory = os.path.join(base_path, "input")
user_directory = os.path.join(base_path, "user")

filename_list_cache: dict[str, tuple[list[str], dict[str, float], float]] = {}

class CacheHelper:
    """
    Helper class for managing file list cache data.
    """
    def __init__(self):
        self.cache: dict[str, tuple[list[str], dict[str, float], float]] = {}
        self.active = False

    def get(self, key: str, default=None) -> tuple[list[str], dict[str, float], float]:
        if not self.active:  # 如果缓存未激活，则返回默认值
            return default
        return self.cache.get(key, default)

    def set(self, key: str, value: tuple[list[str], dict[str, float], float]) -> None:
        if self.active:  # 如果缓存激活，则将value设置为key对应的值
            self.cache[key] = value

    def clear(self):
        self.cache.clear()  # 清空缓存

    def __enter__(self): # 进入上下文管理器
        self.active = True  # 激活缓存
        return self

    def __exit__(self, exc_type, exc_value, traceback): # 退出上下文管理器
        self.active = False
        self.clear()  # 清空缓存

cache_helper = CacheHelper()  # 创建一个缓存助手对象

extension_mimetypes_cache = {  # 创建一个扩展名与MIME类型映射的缓存
    "webp" : "image",
    "fbx" : "model",
}

def map_legacy(folder_name: str) -> str:  # 将旧的文件夹名映射为新的文件夹名
    legacy = {"unet": "diffusion_models",
              "clip": "text_encoders"}
    return legacy.get(folder_name, folder_name)  # 如果folder_name在legacy中，则返回legacy[folder_name]，否则返回folder_name

if not os.path.exists(input_directory):
    try:
        os.makedirs(input_directory)
    except:
        logging.error("Failed to create input directory")

def set_output_directory(output_dir: str) -> None:
    global output_directory
    output_directory = output_dir

def set_temp_directory(temp_dir: str) -> None:
    global temp_directory
    temp_directory = temp_dir

def set_input_directory(input_dir: str) -> None:
    global input_directory
    input_directory = input_dir

def get_output_directory() -> str:
    global output_directory
    return output_directory

def get_temp_directory() -> str:
    global temp_directory
    return temp_directory

def get_input_directory() -> str:
    global input_directory
    return input_directory

def get_user_directory() -> str:
    return user_directory

def set_user_directory(user_dir: str) -> None:
    global user_directory
    user_directory = user_dir


# System User Protection - Protects system directories from HTTP endpoint access
# System Users are internal-only users that cannot be accessed via HTTP endpoints.
# They use the '__' prefix convention (similar to Python's private member convention).
SYSTEM_USER_PREFIX = "__"  # 系统用户前缀


def get_system_user_directory(name: str = "system") -> str: # 获取系统用户目录
    """
    Get the path to a System User directory.

    System User directories (prefixed with '__') are only accessible via internal API,
    not through HTTP endpoints. Use this for storing system-internal data that
    should not be exposed to users.

    Args:
        name: System user name (e.g., "system", "cache"). Must be alphanumeric
              with underscores allowed, but cannot start with underscore.

    Returns:
        Absolute path to the system user directory.

    Raises:
        ValueError: If name is empty, invalid, or starts with underscore.

    Example:
        >>> get_system_user_directory("cache")
        '/path/to/user/__cache'
    """
    if not name or not isinstance(name, str):
        raise ValueError("System user name cannot be empty")
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid system user name: '{name}'")
    if name.startswith("_"):
        raise ValueError("System user name should not start with underscore")
    return os.path.join(get_user_directory(), f"{SYSTEM_USER_PREFIX}{name}")


def get_public_user_directory(user_id: str) -> str | None:
    """
    Get the path to a Public User directory for HTTP endpoint access.

    This function provides structural security by returning None for any
    System User (prefixed with '__'). All HTTP endpoints should use this
    function instead of directly constructing user paths.

    Args:
        user_id: User identifier from HTTP request.

    Returns:
        Absolute path to the user directory, or None if user_id is invalid
        or refers to a System User.

    Example:
        >>> get_public_user_directory("default")
        '/path/to/user/default'
        >>> get_public_user_directory("__system")
        None
    """
    if not user_id or not isinstance(user_id, str):
        return None
    if user_id.startswith(SYSTEM_USER_PREFIX):
        return None
    return os.path.join(get_user_directory(), user_id)


#NOTE: used in http server so don't put folders that should not be accessed remotely
def get_directory_by_type(type_name: str) -> str | None:
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None

def filter_files_content_types(files: list[str], content_types: List[Literal["image", "video", "audio", "model"]]) -> list[str]:  # 过滤文件，只保留符合指定数据类型的文件
    """
    Example:
        files = os.listdir(folder_paths.get_input_directory())
        videos = filter_files_content_types(files, ["video"])

    Note:
        - 'model' in MIME context refers to 3D models, not files containing trained weights and parameters 对应3D模型，不是包含训练权重和参数的模型文件
    """
    global extension_mimetypes_cache
    result = []
    for file in files:
        extension = file.split('.')[-1]
        if extension not in extension_mimetypes_cache:
            mime_type, _ = mimetypes.guess_type(file, strict=False)  # 猜测文件的MIME类型
            if not mime_type:
                continue
            content_type = mime_type.split('/')[0]
            extension_mimetypes_cache[extension] = content_type  # key是文件后缀，value是文件的MIME类型“/”前面的部分；覆盖或添加键值对
        else:
            content_type = extension_mimetypes_cache[extension]

        if content_type in content_types:
            result.append(file)
    return result  # 返回符合条件的文件列表

# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format 根据文件名确定基础路径，如果文件名以"[output]"、"[input]"或"[temp]"结尾，则分别使用输出目录、输入目录或临时目录作为基础路径，否则使用默认路径作为基础路径
# otherwise use default_path as base_dir
def annotated_filepath(name: str) -> tuple[str, str | None]:  # 根据文件名确定基础路径
    if name.endswith("[output]"):
        base_dir = get_output_directory()  # 使用输出目录作为基础路径
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()  # 使用输入目录作为基础路径
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()  # 使用临时目录作为基础路径
        name = name[:-7]
    else:
        return name, None  # 如果文件名不以"[output]"、"[input]"或"[temp]"结尾，则返回文件名和None

    return name, base_dir  # 返回文件名和基础路径


def get_annotated_filepath(name: str, default_dir: str | None=None) -> str:  # 根据文件名确定基础路径
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path 如果默认路径为None，则使用输入目录作为基础路径

    return os.path.join(base_dir, name)  # 返回文件名和基础路径的组合


def exists_annotated_filepath(name) -> bool:  # 判断文件名中对应的标注路径是否存在
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        base_dir = get_input_directory()  # fallback path

    filepath = os.path.join(base_dir, name)
    return os.path.exists(filepath)


def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name in folder_names_and_paths:
        paths, _exts = folder_names_and_paths[folder_name]
        if full_folder_path in paths:
            if is_default and paths[0] != full_folder_path:
                # If the path to the folder is not the first in the list, move it to the beginning.  将默认路径移动到列表的第一个位置
                paths.remove(full_folder_path)
                paths.insert(0, full_folder_path)
        else:
            if is_default:
                paths.insert(0, full_folder_path)
            else:
                paths.append(full_folder_path)
    else:
        folder_names_and_paths[folder_name] = ([full_folder_path], set())  # 将新的数据类型路径添加到文件夹名称和路径的映射中

def get_folder_paths(folder_name: str) -> list[str]:
    folder_name = map_legacy(folder_name)
    return folder_names_and_paths[folder_name][0][:]

def recursive_search(directory: str, excluded_dir_names: list[str] | None=None) -> tuple[list[str], dict[str, float]]:  # 递归搜索指定目录下的所有文件，以相对路径的方式返回；同时返回所有子路径机器对应的修改时间，以绝对路径的方式返回
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs = {}

    # Attempt to add the initial directory to dirs with error handling
    try:
        dirs[directory] = os.path.getmtime(directory)  # 将目录的修改时间添加到dirs中；key是目录路径，value是目录的修改时间
    except FileNotFoundError:
        logging.warning(f"Warning: Unable to access {directory}. Skipping this path.")

    logging.debug("recursive file list on directory {}".format(directory))
    dirpath: str
    subdirs: list[str]
    filenames: list[str]

    for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):  # 遍历目录，followlinks=True表示跟随链接，topdown=True表示从上到下遍历
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]  # 过滤掉 excluded_dir_names 中的目录
        for file_name in filenames:
            try:
                relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)  # 将文件名转换为相对于directory的相对路径
                result.append(relative_path)
            except:
                logging.warning(f"Warning: Unable to access {file_name}. Skipping this file.")
                continue

        for d in subdirs:
            path: str = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                logging.warning(f"Warning: Unable to access {path}. Skipping this path.")
                continue
    logging.debug("found {} files".format(len(result)))
    return result, dirs

def filter_files_extensions(files: Collection[str], extensions: Collection[str]) -> list[str]:  # 过滤文件，只保留符合指定扩展名的文件
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions or len(extensions) == 0, files)))  # 返回符合条件的文件列表



def get_full_path(folder_name: str, filename: str) -> str | None:
    """
    Get the full path of a file in a folder, has to be a file
    """
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")  # 将文件名转换为相对于根目录的相对路径
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path
        elif os.path.islink(full_path):  # 如果文件是一个链接
            logging.warning("WARNING path {} exists but doesn't link anywhere, skipping.".format(full_path))

    return None


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    """
    Get the full path of a file in a folder, has to be a file
    """
    full_path = get_full_path(folder_name, filename)
    if full_path is None:
        raise FileNotFoundError(f"Model in folder '{folder_name}' with filename '{filename}' not found.")
    return full_path


def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]  # 获取folder_name此类数据类型所有的路径
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])  # 递归搜索
        output_list.update(filter_files_extensions(files, folders[1]))  # 过滤文件，只保留符合指定扩展名的文件
        output_folders = {**output_folders, **folders_all}  # 将folders_all合并到output_folders中

    return sorted(list(output_list)), output_folders, time.perf_counter()  # 返回符合条件的文件列表、所有子路径及其对应的修改时间、当前时间

def cached_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float] | None:
    strong_cache = cache_helper.get(folder_name)
    if strong_cache is not None:
        return strong_cache

    global filename_list_cache
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:  # 如果目录的修改时间与time_modified不同，表明该目录下的文件发生了变化，需要重新获取文件列表，不能直接返回cache
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:  # 如果folder_name在folder_names_and_paths包含的路径不在cache中，表明folder_name对应的数据类型路径发生了变化，需要重新获取文件列表，不能直接返回cache
                return None

    return out

def get_filename_list(folder_name: str) -> list[str]:
    folder_name = map_legacy(folder_name)
    out = cached_filename_list_(folder_name)  # 尝试从缓存中获取文件列表
    if out is None:
        out = get_filename_list_(folder_name)  # 如果缓存中没有文件列表，则重新获取文件列表
        global filename_list_cache
        filename_list_cache[folder_name] = out  # 更新缓存
    cache_helper.set(folder_name, out)
    return list(out[0])  # 返回所有的文件名列表

def get_save_image_path(filename_prefix: str, output_dir: str, image_width=0, image_height=0) -> tuple[str, str, int, str, str]:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))  # 获取文件名前缀的长度
        prefix = filename[:prefix_len + 1]  # 获取文件名前缀
        try:
            digits = int(filename[prefix_len + 1:].split('_')[0])  # 获取文件名中的数字
        except:
            digits = 0  # 如果文件名中没有数字，则返回0
        return digits, prefix

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))  # 替换文件名中的%width%为image_width
        input = input.replace("%height%", str(image_height))  # 替换文件名中的%height%为image_height
        now = time.localtime()  # 获取当前时间
        input = input.replace("%year%", str(now.tm_year))  # 替换文件名中的%year%为当前年份
        input = input.replace("%month%", str(now.tm_mon).zfill(2))  # 替换文件名中的%month%为当前月份
        input = input.replace("%day%", str(now.tm_mday).zfill(2))  # 替换文件名中的%day%为当前日期
        input = input.replace("%hour%", str(now.tm_hour).zfill(2))  # 替换文件名中的%hour%为当前小时
        input = input.replace("%minute%", str(now.tm_min).zfill(2))  # 替换文件名中的%minute%为当前分钟
        input = input.replace("%second%", str(now.tm_sec).zfill(2))  # 替换文件名中的%second%为当前秒
        return input

    if "%" in filename_prefix:
        filename_prefix = compute_vars(filename_prefix, image_width, image_height)  # 替换文件名中的%width%、%height%、%year%、%month%、%day%、%hour%、%minute%、%second%

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))  # 获取文件名前缀的子文件夹
    filename = os.path.basename(os.path.normpath(filename_prefix))  # 获取文件名前缀的文件名

    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:  # 如果文件名前缀的子文件夹与输出目录的公共路径不等于输出目录，则抛出异常
        err = "**** ERROR: Saving image outside the output folder is not allowed." + \
              "\n full_output_folder: " + os.path.abspath(full_output_folder) + \
              "\n         output_dir: " + output_dir + \
              "\n         commonpath: " + os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
        logging.error(err)
        raise Exception(err)

    try:
        counter = max(filter(lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename) and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix

def get_input_subfolders() -> list[str]:
    """Returns a list of all subfolder paths in the input directory, recursively.

    Returns:
        List of folder paths relative to the input directory, excluding the root directory
    """
    input_dir = get_input_directory()
    folders = []

    try:
        if not os.path.exists(input_dir):
            return []

        for root, dirs, _ in os.walk(input_dir):  # 遍历输入目录下的所有子目录
            rel_path = os.path.relpath(root, input_dir)  # 将root转换为相对于input_dir的相对路径
            if rel_path != ".":  # Only include non-root directories
                # Normalize path separators to forward slashes
                folders.append(rel_path.replace(os.sep, '/'))  # 将路径中的分隔符替换为斜杠

        return sorted(folders)
    except FileNotFoundError:
        return []
