from collections import deque
from datetime import datetime
import io
import logging
import sys
import threading

logs = None
stdout_interceptor = None
stderr_interceptor = None


class LogInterceptor(io.TextIOWrapper):  #  自定义的日志拦截器，可拦截和处理标准输出和标准错误的日志流
    def __init__(self, stream,  *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(buffer, *args, **kwargs, encoding=encoding, line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}  # 为每条日志添加时间戳
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        super().write(data)

    def flush(self):  # 刷新日志
        super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):  # 注册刷新回调函数
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None:
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None:
        stderr_interceptor.on_flush(callback)

def setup_logger(log_level: str = 'INFO', capacity: int = 300, use_stdout: bool = False):
    global logs
    if logs:
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)  # 创建一个固定大小的队列，用于存储日志

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)  # 将sys.stdout替换为自定义的LogInterceptor，用于拦截和处理标准输出
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)  # 将sys.stderr替换为自定义的LogInterceptor，用于拦截和处理标准错误输出

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()  # 创建一个StreamHandler，用于将日志发送到控制台
    stream_handler.setFormatter(logging.Formatter("%(message)s"))  # 设置日志格式

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)  # 添加过滤器，错误及以上级别的日志发送到stderr

        # Lesser to stdout，错误以下级别的日志发送到stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)
