# Prevent custom nodes from hooking anything important
import comfy.model_management

HOOK_BREAK = [(comfy.model_management, "cast_to")]


SAVED_FUNCTIONS = []


def save_functions():
    for f in HOOK_BREAK:
        SAVED_FUNCTIONS.append((f[0], f[1], getattr(f[0], f[1])))  # 保存(对象，对象的属性名称，对应对应的具体对应)


def restore_functions():
    for f in SAVED_FUNCTIONS:
        setattr(f[0], f[1], f[2])  # 将对象f[0]的属性f[1]设置为f[2]
