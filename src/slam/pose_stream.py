"""位姿流（跨解释器边界）：ROS2 侧写入，训练/推理侧读取。

为什么需要它：ROS2/Cartographer 侧必须用带 rclpy 的解释器，训练/推理侧用 conda `unitree_rt`，
两者禁止互相 pip 安装对方的包（见提示词【二】3）。因此位姿以 JSONL（或 UDP）流传递。

硬约束：取不到位姿时返回 None + 有效性标志，由调用方决定丢弃样本或标记无效；
**禁止默认返回零位姿**（`never_publish_zero_pose`）。
"""

from __future__ import annotations

import json
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..utils.io_utils import append_jsonl, read_jsonl
from ..utils.time_sync import AlignReport, align_nearest


@dataclass
class PoseSample:
    """统一位姿样本（两个时间戳缺一不可：采集时钟 + 单调时钟）。"""

    t_capture: float          # 采集时钟（数据集时间轴，s）
    t_mono: float             # 单调时钟（s），用于在线 watchdog 与对齐
    x: float
    y: float
    yaw: float
    z: float = 0.0
    source: str = "cartographer"
    valid: int = 1
    frame_id: str = "map"
    child_frame_id: str = "base_link"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def pose3(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> "PoseSample":
        known = {f: record[f] for f in cls.__dataclass_fields__ if f in record and f != "extra"}
        extra = {k: v for k, v in record.items() if k not in cls.__dataclass_fields__}
        return cls(**known, extra=extra)

    @classmethod
    def unavailable(cls, t_capture: float, t_mono: float, marker: str = "unavailable",
                    frame_id: str = "map") -> "PoseSample":
        """显式不可用样本：valid=0 且 source 标记降级原因，绝不伪装成有效零位姿。"""
        return cls(t_capture=t_capture, t_mono=t_mono, x=float("nan"), y=float("nan"),
                   yaw=float("nan"), source=marker, valid=0, frame_id=frame_id)


class PoseStreamWriter:
    """ROS2 侧写入器：JSONL 追加（带 flush）或 UDP 发送。"""

    def __init__(self, transport: str = "jsonl", jsonl_path: str | Path | None = None,
                 udp_host: str = "127.0.0.1", udp_port: int = 45001, flush_every: int = 1) -> None:
        if transport not in ("jsonl", "udp"):
            raise ValueError(f"未知位姿流传输方式 {transport!r}")
        self.transport = transport
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[dict[str, Any]] = []
        self._sock: socket.socket | None = None
        self.n_written = 0
        if transport == "udp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = (udp_host, int(udp_port))

    def write(self, sample: PoseSample) -> None:
        record = sample.to_json()
        if self.transport == "jsonl":
            self._buffer.append(record)
            if len(self._buffer) >= self.flush_every:
                append_jsonl(self.jsonl_path, self._buffer)  # type: ignore[arg-type]
                self._buffer.clear()
        else:
            assert self._sock is not None
            self._sock.sendto(json.dumps(record).encode("utf-8"), self._addr)
        self.n_written += 1

    def close(self) -> None:
        if self.transport == "jsonl" and self._buffer and self.jsonl_path:
            append_jsonl(self.jsonl_path, self._buffer)
            self._buffer.clear()
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class PoseStreamClient:
    """训练/推理侧读取器：维护滑动缓冲，支持按时间查询与统计。"""

    def __init__(self, transport: str = "jsonl", jsonl_path: str | Path | None = None,
                 udp_host: str = "127.0.0.1", udp_port: int = 45001, buffer_size: int = 512) -> None:
        self.transport = transport
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.buffer_size = int(buffer_size)
        self._samples: list[PoseSample] = []
        self._times: np.ndarray = np.zeros(0, dtype=np.float64)
        self.n_out_of_order = 0
        self.n_duplicate = 0
        self.n_invalid = 0
        self._last_t: float | None = None
        if transport == "udp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((udp_host, int(udp_port)))
            self._sock.settimeout(0.0)
        else:
            self._sock = None

    # -- 装载 ------------------------------------------------------------------
    def load_file(self, path: str | Path | None = None) -> int:
        """一次性读取 JSONL（离线评测/复现用）。返回读入的样本数。

        离线装载**不做滑窗截断**：`buffer_size` 只用于在线模式（限制内存）。
        早期版本对离线数据也套用滑窗，导致回放前段位姿被丢弃、闭环查询全部超容差
        （本工程实测踩过：1037 个样本只剩最后 512 个，t=0~60s 全部查不到）。
        """
        p = Path(path) if path else self.jsonl_path
        if p is None or not p.exists():
            return 0
        records = read_jsonl(p)
        for rec in records:
            self._append(PoseSample.from_json(rec), truncate=False)
        return len(records)

    def poll(self, max_messages: int = 64) -> int:
        """UDP 模式下把内核缓冲里的消息取空（在线模式调用）。"""
        if self.transport != "udp" or self._sock is None:
            return 0
        count = 0
        while count < max_messages:
            try:
                payload, _ = self._sock.recvfrom(65536)
            except (BlockingIOError, socket.timeout):
                break
            self._append(PoseSample.from_json(json.loads(payload.decode("utf-8"))))
            count += 1
        return count

    def _append(self, sample: PoseSample, truncate: bool = True) -> None:
        if self._last_t is not None and sample.t_mono < self._last_t:
            self.n_out_of_order += 1
        if self._last_t is not None and abs(sample.t_mono - self._last_t) < 1e-9:
            self.n_duplicate += 1
        if not sample.valid:
            self.n_invalid += 1
        self._last_t = sample.t_mono
        self._samples.append(sample)
        if truncate and len(self._samples) > self.buffer_size:
            self._samples = self._samples[-self.buffer_size :]
        self._times = np.asarray([s.t_mono for s in self._samples], dtype=np.float64)

    # -- 查询 ------------------------------------------------------------------
    def get_pose(self, t: float, tol: float | None = None, clock: str = "mono") -> PoseSample | None:
        """按时间取最近位姿；超容差或全部无效时返回 None（禁止返回零位姿）。"""
        if not self._samples:
            return None
        times = self._times if clock == "mono" else np.asarray([s.t_capture for s in self._samples])
        if times.size == 0:
            return None
        idx = int(np.argmin(np.abs(times - t)))
        sample = self._samples[idx]
        if tol is not None and abs(times[idx] - t) > tol:
            return None
        if not sample.valid or not np.isfinite(sample.x) or not np.isfinite(sample.y):
            return None
        return sample

    def latest(self) -> PoseSample | None:
        for sample in reversed(self._samples):
            if sample.valid and np.isfinite(sample.x):
                return sample
        return None

    def poses(self) -> np.ndarray:
        return np.asarray([s.pose3 for s in self._samples], dtype=np.float64)

    def capture_times(self) -> np.ndarray:
        return np.asarray([s.t_capture for s in self._samples], dtype=np.float64)

    def mono_times(self) -> np.ndarray:
        return self._times.copy()

    def alignment_report(self, query_times: Iterable[float], tol: float) -> dict[str, Any]:
        """把位姿流对齐到策略时钟，输出偏移分布与超容差比例（写入 logs/slam_align.json）。"""
        report = AlignReport()
        align_nearest(self.mono_times(), np.asarray(list(query_times), dtype=np.float64), tol, report)
        summary = report.summary()
        summary.update(
            {
                "n_pose_samples": len(self._samples),
                "n_out_of_order": self.n_out_of_order,
                "n_duplicate": self.n_duplicate,
                "n_invalid": self.n_invalid,
                "tolerance_s": tol,
            }
        )
        return summary

    def stats(self) -> dict[str, Any]:
        valid = np.asarray([s.valid for s in self._samples], dtype=np.float64) if self._samples else np.zeros(1)
        sources = sorted({s.source for s in self._samples})
        rate = 0.0
        if self._times.size > 1:
            span = float(self._times[-1] - self._times[0])
            rate = (self._times.size - 1) / span if span > 0 else 0.0
        return {
            "n_samples": len(self._samples),
            "valid_ratio": float(valid.mean()),
            "sources": sources,
            "rate_hz": rate,
            "n_out_of_order": self.n_out_of_order,
            "n_duplicate": self.n_duplicate,
            "n_invalid": self.n_invalid,
            "first_t": float(self._times[0]) if self._times.size else None,
            "last_t": float(self._times[-1]) if self._times.size else None,
        }

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def monotonic_now() -> float:
    return time.monotonic()
