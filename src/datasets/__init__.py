"""数据层公共 API：schema 校验 / episode 读取 / 窗口切片 / 变换 / 过滤 / Dataset。

约定：本包不 import src/models、src/train、src/infer；不做隐式副作用（不读文件、不初始化 CUDA）。
"""

from .schema import EpisodeMeta, ValidateReport, validate_episode, validate_sample
from .transforms import apply_norm, build_transform, inverse_action, load_stats
from .window_builder import build_chunk, build_windows

__all__ = [
    "EpisodeMeta",
    "ValidateReport",
    "validate_episode",
    "validate_sample",
    "build_transform",
    "apply_norm",
    "inverse_action",
    "load_stats",
    "build_windows",
    "build_chunk",
]
