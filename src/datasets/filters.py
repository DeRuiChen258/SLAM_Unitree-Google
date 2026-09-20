"""异常样本过滤：NaN/Inf、时间戳跳变、静止段、动作越界、SLAM 位姿丢失、图像全黑/过曝。

输出过滤统计（各类被过滤窗口数与占比）写入 logs/12_data_check.json。
过滤比例超过 `filters.max_reject_ratio` 时抛出 FilterError —— 宁可失败也不把数据洗没。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.chunking import build_chunk
from ..utils.config import get


class FilterError(RuntimeError):
    """过滤比例异常，禁止继续训练（数据或时钟有系统性问题）。"""


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[int]] = field(default_factory=dict)

    def reject(self, reason: str, index: int) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        bucket = self.examples.setdefault(reason, [])
        if len(bucket) < 5:
            bucket.append(int(index))

    @property
    def reject_ratio(self) -> float:
        return 1.0 - (self.kept / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "rejected": self.total - self.kept,
            "reject_ratio": round(self.reject_ratio, 6),
            "reasons": self.reasons,
            "example_indices": self.examples,
        }


def window_reject_reason(episode: Mapping[str, np.ndarray], t: int, obs_index: np.ndarray,
                         data_cfg: Mapping[str, Any], state_selector: np.ndarray | None = None) -> str | None:
    """返回拒绝原因；None 表示该窗口合法。原因名与 FilterStats.reasons 一一对应。"""
    state = np.asarray(episode["state"])[t]
    if state_selector is not None:
        state = state[state_selector]
    images = np.asarray(episode["images"])[obs_index]
    if not np.isfinite(state).all() or not np.isfinite(images.astype(np.float32)).all():
        return "nan_inf"

    # 时间戳跳变（观测窗口内任意相邻帧）
    ts = np.asarray(episode["timestamp"])[obs_index]
    max_gap = float(get(data_cfg, "filters.max_gap_s", 0.35))
    if ts.size > 1 and float(np.max(np.diff(ts))) > max_gap:
        return "time_gap"

    # 图像全黑 / 过曝（像素标准差过低，且该判定只针对有效观测帧）
    black_thr = float(get(data_cfg, "filters.black_frame_std", 0.01))
    if float(np.std(images.astype(np.float32) / 255.0)) < black_thr:
        return "blank_image"

    # 动作块统计（只在有真实动作的步上统计）
    horizon = int(get(data_cfg, "chunk.H", 8))
    dim = int(get(data_cfg, "action.dim", 7))
    chunk, mask = build_chunk(np.asarray(episode["action"]), t, horizon, dim)
    valid = mask > 0
    if valid.any():
        trans = np.abs(chunk[valid][:, :3]).max()
        rot = np.abs(chunk[valid][:, 3:6]).max()
        lim_t = float(get(data_cfg, "action.delta_translation_limit", 0.06)) * float(
            get(data_cfg, "filters.action_bound_sigma", 3.0)
        )
        lim_r = float(get(data_cfg, "action.delta_rotation_limit", 0.12)) * float(
            get(data_cfg, "filters.action_bound_sigma", 3.0)
        )
        if trans > lim_t or rot > lim_r:
            return "action_out_of_bound"
        if float(np.linalg.norm(chunk[valid], axis=1).mean()) < float(get(data_cfg, "filters.min_motion", 1e-4)):
            return "static_robot"

    # SLAM 位姿缺失
    slam_valid = np.asarray(episode["slam_valid"])[obs_index]
    max_slam_gap = float(get(data_cfg, "filters.max_slam_gap_s", 0.35))
    if max_slam_gap > 0 and ts.size > 1:
        invalid_idx = np.where(slam_valid == 0)[0]
        if invalid_idx.size and invalid_idx.size == slam_valid.size:
            return "slam_unavailable"
        for j in invalid_idx:
            span = ts[min(j + 1, ts.size - 1)] - ts[max(j - 1, 0)]
            if abs(span) > max_slam_gap:
                return "slam_gap_exceeded"
    return None


def filter_windows(episode: Mapping[str, np.ndarray], windows: Sequence[int], obs_matrix: np.ndarray,
                   data_cfg: Mapping[str, Any], stats: FilterStats | None = None,
                   state_selector: np.ndarray | None = None) -> list[int]:
    """按 `filters` 配置过滤窗口；`obs_matrix[i]` 为第 i 个窗口的历史帧下标。"""
    stats = stats or FilterStats()
    reject_invalid_slam = bool(get(data_cfg, "filters.reject_invalid_slam", False))
    kept: list[int] = []
    for i, t in enumerate(windows):
        stats.total += 1
        reason = window_reject_reason(episode, int(t), obs_matrix[i], data_cfg, state_selector)
        if reason is None and reject_invalid_slam:
            if not np.asarray(episode["slam_valid"])[obs_matrix[i]].all():
                reason = "slam_invalid_strict"
        if reason is None:
            kept.append(int(t))
            stats.kept += 1
        else:
            stats.reject(reason, int(t))
    max_ratio = float(get(data_cfg, "filters.max_reject_ratio", 0.35))
    if stats.total and stats.reject_ratio > max_ratio:
        raise FilterError(
            f"过滤比例 {stats.reject_ratio:.1%} 超过阈值 {max_ratio:.1%}，"
            f"原因分布 {stats.reasons}。先修数据/时钟，不要直接训练。"
        )
    return kept
