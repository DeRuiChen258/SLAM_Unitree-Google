"""单条 episode 的读取：图像序列 / 状态 / 动作 / SLAM 位姿 / 成功标记 / 元数据。

支持懒加载（逐帧读图，避免一次性占满内存）与 memmap；解析时间戳与频率；
检测丢帧、乱序、重复时间戳并产生告警统计（不修改原始文件，raw/ 只读）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class EpisodeAlerts:
    """读取过程中发现的异常，必须随 manifest 一起落盘。"""

    duplicate_timestamps: int = 0
    non_monotonic_pairs: int = 0
    gaps: int = 0
    dropped_frames_est: int = 0
    max_gap_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate_timestamps": self.duplicate_timestamps,
            "non_monotonic_pairs": self.non_monotonic_pairs,
            "gaps": self.gaps,
            "dropped_frames_est": self.dropped_frames_est,
            "max_gap_s": self.max_gap_s,
            "notes": self.notes,
        }


class EpisodeReader:
    """读取单个 episode npz；`lazy=True` 时图像使用 mmap 懒加载。"""

    REQUIRED = ("images", "state", "action", "timestamp", "base_pose", "slam_pose", "slam_valid")

    def __init__(self, path: str | Path, lazy: bool = False, expected_dt: float | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"episode 不存在: {self.path}")
        self.lazy = lazy
        self.expected_dt = expected_dt
        mmap = "r" if lazy else None
        self._handle = np.load(self.path, allow_pickle=False, mmap_mode=mmap)
        missing = [name for name in self.REQUIRED if name not in self._handle.files]
        if missing:
            raise KeyError(f"{self.path.name} 缺少字段: {missing}")
        self.alerts = EpisodeAlerts()
        self._check_timestamps()

    # -- 属性 -------------------------------------------------------------------
    @property
    def length(self) -> int:
        return int(self._handle["images"].shape[0])

    @property
    def episode_id(self) -> str:
        for key in ("episode_id", "_episode_id"):
            if key in self._handle.files:
                return str(self._handle[key])
        return self.path.stem

    @property
    def source(self) -> str:
        for key in ("source", "_source"):
            if key in self._handle.files:
                return str(self._handle[key])
        return "UNKNOWN"

    @property
    def success(self) -> int:
        return int(self._handle["success"]) if "success" in self._handle.files else -1

    @property
    def fps(self) -> float:
        ts = np.asarray(self._handle["timestamp"], dtype=np.float64)
        if ts.size < 2:
            return 0.0
        dt = float(np.median(np.diff(ts)))
        return 1.0 / dt if dt > 0 else 0.0

    # -- 数据访问 ---------------------------------------------------------------
    def timestamps(self) -> np.ndarray:
        return np.asarray(self._handle["timestamp"], dtype=np.float64)

    def state_matrix(self) -> np.ndarray:
        return np.asarray(self._handle["state"], dtype=np.float32)

    def actions(self) -> np.ndarray:
        return np.asarray(self._handle["action"], dtype=np.float32)

    def slam_poses(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self._handle["slam_pose"], dtype=np.float32),
                np.asarray(self._handle["slam_valid"], dtype=np.uint8))

    def base_poses(self) -> np.ndarray:
        return np.asarray(self._handle["base_pose"], dtype=np.float32)

    def image_at(self, index: int) -> np.ndarray:
        """单帧图像（懒加载时按需读取，避免整段载入内存）。"""
        return np.asarray(self._handle["images"][index])

    def images(self) -> np.ndarray:
        return np.asarray(self._handle["images"])

    def iter_frames(self, batch: int = 16) -> Iterator[dict[str, Any]]:
        """按批流式迭代，用于大 episode 的预处理与回放。"""
        for start in range(0, self.length, batch):
            end = min(start + batch, self.length)
            yield {
                "index": np.arange(start, end),
                "images": np.asarray(self._handle["images"][start:end], dtype=np.uint8),
                "state": np.asarray(self._handle["state"][start:end], dtype=np.float32),
                "action": np.asarray(self._handle["action"][start:end], dtype=np.float32),
                "timestamp": np.asarray(self._handle["timestamp"][start:end], dtype=np.float64),
                "slam_pose": np.asarray(self._handle["slam_pose"][start:end], dtype=np.float32),
                "slam_valid": np.asarray(self._handle["slam_valid"][start:end], dtype=np.uint8),
            }

    # -- 内部 -------------------------------------------------------------------
    def _check_timestamps(self) -> None:
        ts = self.timestamps()
        if ts.size < 2:
            return
        diff = np.diff(ts)
        self.alerts.duplicate_timestamps = int(np.sum(diff == 0))
        self.alerts.non_monotonic_pairs = int(np.sum(diff < 0))
        self.alerts.max_gap_s = float(diff.max())
        if self.expected_dt:
            gaps = diff > 2.0 * self.expected_dt
            self.alerts.gaps = int(gaps.sum())
            self.alerts.dropped_frames_est = int(np.sum(np.maximum(0, np.round(diff / self.expected_dt) - 1)))
            if self.alerts.non_monotonic_pairs:
                self.alerts.notes.append("存在时间戳回退：请检查采集时钟是否与单调时钟混用")
            if self.alerts.duplicate_timestamps:
                self.alerts.notes.append("存在重复时间戳：可能是同一帧被写入两次")

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> "EpisodeReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def load_episode(path: str | Path, lazy: bool = False, expected_dt: float | None = None) -> dict[str, Any]:
    """便捷函数：返回可被 window_builder 直接消费的字典。"""
    with EpisodeReader(path, lazy=lazy, expected_dt=expected_dt) as reader:
        return {
            "images": reader.images(),
            "state": reader.state_matrix(),
            "action": reader.actions(),
            "timestamp": reader.timestamps(),
            "base_pose": reader.base_poses(),
            "slam_pose": reader.slam_poses()[0],
            "slam_valid": reader.slam_poses()[1],
            "episode_id": reader.episode_id,
            "source": reader.source,
            "success": reader.success,
            "alerts": reader.alerts.to_dict(),
        }
