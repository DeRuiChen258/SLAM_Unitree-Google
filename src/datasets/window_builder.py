"""滑动窗口与动作块的唯一构造入口（训练 / 推理 / offline_eval 共用）。

样本定义（与 configs/data.yaml、dataset_schema.json 一致）：
    obs_index = [t-(K-1)*frame_stride, ..., t]
    state      = state[t]
    chunk      = action[t : t+H]
    mask       = 1 表示该步有真实动作，0 表示 episode 末尾 padding

硬约束：跨 episode 边界禁止拼接；末尾不足 H 步按配置 padding(mask) 或裁剪(crop)。
切片语义的唯一实现是 `build_chunk`，src/models/action_chunker.py 必须调用它，
禁止两处各写一份（tests/test_window_builder.py 会做一致性交叉验证）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..utils.chunking import build_chunk  # noqa: F401  (对外复用同一实现)
from ..utils.config import get


def obs_indices(t: int, num_frames: int, frame_stride: int, length: int) -> np.ndarray | None:
    """历史帧下标；起点不足时返回 None（该窗口被丢弃，禁止补帧）。"""
    start = t - (num_frames - 1) * frame_stride
    if start < 0:
        return None
    idx = np.arange(start, t + 1, frame_stride, dtype=np.int64)
    if idx.shape[0] != num_frames or idx[-1] > length - 1:
        return None
    return idx


def build_windows(length: int, num_frames: int, frame_stride: int, horizon: int,
                  stride: int, pad_mode: str = "mask") -> list[int]:
    """返回该 episode 所有合法窗口的时刻下标列表。"""
    first = (num_frames - 1) * frame_stride
    if length <= first:
        return []
    starts: list[int] = []
    last_full = length - horizon
    if last_full >= first:
        starts += list(range(first, last_full + 1, stride))
    if pad_mode == "mask":
        tail_start = max(first, (last_full + 1) if last_full >= first else first)
        starts += list(range(tail_start, length, stride))
    elif pad_mode != "crop":
        raise ValueError(f"未知 pad_mode={pad_mode!r}（可选 mask/crop）")
    return sorted({s for s in starts if first <= s < length})


def make_sample(episode: Mapping[str, np.ndarray], t: int, t_index: np.ndarray,
                data_cfg: Mapping[str, Any], state_selector: np.ndarray | None = None) -> dict[str, Any]:
    """由 episode 数组与时刻 t 组装单个样本（不含归一化，统计在 Dataset 层固化）。"""
    state = np.asarray(episode["state"])[t]
    if state_selector is not None:
        state = state[state_selector]
    chunk, mask = build_chunk(np.asarray(episode["action"]), t,
                              int(get(data_cfg, "chunk.H", 8)), int(get(data_cfg, "action.dim", 7)))
    return {
        "images": np.asarray(episode["images"])[t_index],
        "state": state,
        "action_chunk": chunk,
        "mask": mask,
        "slam_valid": np.asarray(episode["slam_valid"])[t_index],
        "meta": {
            "episode_id": str(episode.get("episode_id", "")),
            "t0": int(t),
            "timestamp": float(np.asarray(episode["timestamp"])[t]),
            "source": str(episode.get("source", "MOCK")),
        },
    }


def windows_per_episode(length: int, data_cfg: Mapping[str, Any]) -> int:
    """由配置推算窗口数（scripts 与测试用于核对切片是否漏/重）。"""
    return len(
        build_windows(
            length=length,
            num_frames=int(get(data_cfg, "observation.num_frames", 1)),
            frame_stride=int(get(data_cfg, "observation.frame_stride", 1)),
            horizon=int(get(data_cfg, "chunk.H", 8)),
            stride=int(get(data_cfg, "chunk.stride", 1)),
            pad_mode=str(get(data_cfg, "chunk.pad_mode", "mask")),
        )
    )
