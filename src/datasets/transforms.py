"""图像/状态/动作的预处理与反预处理（训练、推理、评测三方共用）。

硬约束：
    * 推理侧禁止凭经验重写 resize / 归一化 / 通道顺序 / 角度单位，
      统一调用 build_transform(cfg) 与 inverse_action(...)；
    * 所有函数为纯函数，便于单元测试；
    * 归一化统计只来自 data/stats/normalization.json。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..utils.config import get, project_root
from ..utils.io_utils import read_json, sha256_file


class TransformError(ValueError):
    """预处理契约被破坏（shape/通道/统计不一致）。"""


@dataclass
class ImageTransform:
    """把 uint8 图像转成归一化 float32（与训练完全同一套参数）。"""

    height: int
    width: int
    channels: int
    color_space: str
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        squeeze = False
        if arr.ndim == 3:
            arr = arr[None, ...]
            squeeze = True
        if arr.ndim != 4:
            raise TransformError(f"图像维度应为 3 或 4，实际 {arr.shape}")
        if arr.shape[1] != self.channels:
            raise TransformError(f"通道数 {arr.shape[1]} 与配置 {self.channels} 不一致")
        if tuple(arr.shape[2:]) != (self.height, self.width):
            raise TransformError(f"图像尺寸 {arr.shape[2:]} 与配置 {(self.height, self.width)} 不一致")
        out = arr.astype(np.float32) / 255.0
        out = (out - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)
        return out[0] if squeeze else out


def build_transform(data_cfg: Mapping[str, Any]) -> ImageTransform:
    mean = np.asarray(get(data_cfg, "image.normalize.mean"), dtype=np.float32)
    std = np.asarray(get(data_cfg, "image.normalize.std"), dtype=np.float32)
    if mean.shape != std.shape:
        raise TransformError("mean/std 形状不一致")
    return ImageTransform(
        height=int(get(data_cfg, "image.height")),
        width=int(get(data_cfg, "image.width")),
        channels=int(get(data_cfg, "image.channels")),
        color_space=str(get(data_cfg, "image.color_space")),
        mean=mean,
        std=std,
    )


def load_stats(path: str | Path | None = None, data_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """加载归一化统计（训练/推理唯一来源）。"""
    if path is None:
        if data_cfg is not None and get(data_cfg, "stats_path"):
            path = get(data_cfg, "stats_path")
        else:
            path = project_root() / "data" / "stats" / "normalization.json"
    p = Path(path)
    if not p.exists():
        raise TransformError(f"缺少归一化统计文件: {p}（先运行 scripts/11_build_dataset.py）")
    stats = read_json(p)
    if "state_blocks" not in stats or "action" not in stats:
        raise TransformError(f"统计文件缺字段: {p}")
    return stats


def stats_fingerprint(stats_path: str | Path) -> str:
    """统计文件哈希：写入 checkpoint，推理加载时强校验。"""
    return sha256_file(stats_path)


def stats_path_for(paths: Mapping[str, Any], run_name: str = "base") -> Path:
    """归一化统计文件路径的唯一解析入口。

    为什么必须集中：消融运行（如 no_chunk / no_slam_input）的技能统计与 base **不同**
    （H 变了、状态维度变了 → 动作/状态分布都变），
    如果训练侧与评测侧各自解析路径，就会出现"训练用 base 统计、评测用消融统计"的错配。
    本工程曾因此被 checkpoint 的统计哈希校验拦下（见 .agent/failures.md），
    所以这里把路径规则收敛为一处，训练与评测都调用它。
    """
    base = Path(paths["stats_dir"])
    if run_name == "base":
        return base / "normalization.json"
    return base / "ablation" / f"{run_name}_normalization.json"


def apply_norm(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - mean) / std


def inverse_norm(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * std + mean


def norm_state(state: np.ndarray, stats: Mapping[str, Any], blocks: Mapping[str, slice],
               normalize_blocks: list[str] | None = None) -> np.ndarray:
    """按块归一化状态向量；passthrough 块（valid 标志、时间特征）保持原值。"""
    out = np.array(state, dtype=np.float32, copy=True)
    names = normalize_blocks if normalize_blocks is not None else list(stats.get("state_blocks", {}).keys())
    for name in names:
        if name not in blocks or name not in stats.get("state_blocks", {}):
            continue
        sl = blocks[name]
        mean = np.asarray(stats["state_blocks"][name]["mean"], dtype=np.float32)
        std = np.asarray(stats["state_blocks"][name]["std"], dtype=np.float32)
        if out.ndim == 1:
            out[sl] = apply_norm(out[sl].reshape(-1), mean, std)
        else:
            out[..., sl] = apply_norm(out[..., sl], mean, std)
    return out


def denorm_state(state: np.ndarray, stats: Mapping[str, Any], blocks: Mapping[str, slice],
                 normalize_blocks: list[str] | None = None) -> np.ndarray:
    out = np.array(state, dtype=np.float32, copy=True)
    names = normalize_blocks if normalize_blocks is not None else list(stats.get("state_blocks", {}).keys())
    for name in names:
        if name not in blocks or name not in stats.get("state_blocks", {}):
            continue
        sl = blocks[name]
        mean = np.asarray(stats["state_blocks"][name]["mean"], dtype=np.float32)
        std = np.asarray(stats["state_blocks"][name]["std"], dtype=np.float32)
        if out.ndim == 1:
            out[sl] = inverse_norm(out[sl].reshape(-1), mean, std)
        else:
            out[..., sl] = inverse_norm(out[..., sl], mean, std)
    return out


def norm_action(action: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    return apply_norm(action, np.asarray(stats["action"]["mean"], dtype=np.float32),
                      np.asarray(stats["action"]["std"], dtype=np.float32))


def inverse_action(norm_action_array: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """反归一化动作（物理量）。所有落盘/可视化/限幅前必须调用。"""
    return inverse_norm(norm_action_array, np.asarray(stats["action"]["mean"], dtype=np.float32),
                        np.asarray(stats["action"]["std"], dtype=np.float32))


def compute_stats(images: np.ndarray, state: np.ndarray, action: np.ndarray,
                  blocks: Mapping[str, slice], normalize_blocks: list[str],
                  clip_sigma: float = 4.0, min_std: float = 1e-6) -> dict[str, Any]:
    """由训练集统计归一化参数（只在 scripts/11_build_dataset.py 调用一次）。"""
    img = images.astype(np.float32) / 255.0
    flat_img = img.reshape(-1, img.shape[-3])
    state_blocks: dict[str, Any] = {}
    for name in normalize_blocks:
        if name not in blocks:
            continue
        sl = blocks[name]
        vals = np.asarray(state)[..., sl]
        vals = vals.reshape(-1, vals.shape[-1]) if vals.ndim > 1 else vals.reshape(-1, 1)
        lo = np.percentile(vals, 0.1, axis=0)
        hi = np.percentile(vals, 99.9, axis=0)
        clipped = np.clip(vals, lo, hi)
        state_blocks[name] = {
            "mean": clipped.mean(axis=0).tolist(),
            "std": np.maximum(clipped.std(axis=0), min_std).tolist(),
            "min": vals.min(axis=0).tolist(),
            "max": vals.max(axis=0).tolist(),
            "p001": lo.tolist(),
            "p999": hi.tolist(),
        }
    return {
        "image": {
            "mean": flat_img.mean(axis=0).tolist(),
            "std": np.maximum(flat_img.std(axis=0), min_std).tolist(),
            "clip_sigma": clip_sigma,
            "space": "uint8/255 后按通道标准化",
        },
        "state_blocks": state_blocks,
        "action": {
            "mean": action.mean(axis=0).tolist(),
            "std": np.maximum(action.std(axis=0), min_std).tolist(),
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
        },
        "note": "所有通道统计仅由 train split 计算；推理侧必须复用同一文件（哈希写入 checkpoint）。",
    }
