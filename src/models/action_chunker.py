"""动作块的语义与调度（不含神经网络）。

职责：
    * `ActionSpec`：动作维度、单位、坐标系、是否增量（与 configs/data.yaml 同源）；
    * `build_chunk`：切片语义（复用 src/utils/chunking.py 的唯一实现）；
    * `ChunkScheduler(H, n_exec)`：决定何时重新预测、执行多少步、何时丢弃尾部；
    * `split_chunk_to_steps`：把动作块拆成逐步执行序列。

H 的影响（详细推导见 docs/algorithm_notes.md）：
    H 越大 → 单次推理摊销到更多步，推理频率与延迟降低，但动作对观测变化的响应变慢（滞后↑）；
    H 越小 → 响应快但推理频繁，且逐步噪声更容易体现在执行轨迹上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..utils.chunking import build_chunk as _build_chunk
from ..utils.config import get


@dataclass(frozen=True)
class ActionSpec:
    """动作语义（单位与坐标系必须写清楚，禁止下游各自假设）。"""

    dim: int
    layout: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    is_delta: bool

    @classmethod
    def from_config(cls, data_cfg: Mapping[str, Any]) -> "ActionSpec":
        layout = tuple(get(data_cfg, "action.layout", []) or [])
        units = tuple("rad" if name.startswith("r") else ("m" if name.startswith("d") else "1") for name in layout)
        return cls(
            dim=int(get(data_cfg, "action.dim", 7)),
            layout=layout,
            units=units,
            frame=str(get(data_cfg, "action.frame", "base")),
            is_delta=bool(get(data_cfg, "action.is_delta", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "layout": list(self.layout),
            "units": list(self.units),
            "frame": self.frame,
            "is_delta": self.is_delta,
        }


def build_chunk(actions: np.ndarray, t: int, horizon: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """与 datasets.window_builder 完全同一实现（同一函数、同一语义）。"""
    return _build_chunk(actions, t, horizon, dim)


class ChunkScheduler:
    """动作块调度：`n_exec ≤ H` 步执行一次预测，剩余尾部按配置丢弃或保留。"""

    def __init__(self, horizon: int, n_exec: int, drop_remainder: bool = True) -> None:
        if not 1 <= n_exec <= horizon:
            raise ValueError(f"要求 1 ≤ n_exec ≤ H，实际 n_exec={n_exec}, H={horizon}")
        self.horizon = horizon
        self.n_exec = n_exec
        self.drop_remainder = drop_remainder

    @classmethod
    def from_config(cls, data_cfg: Mapping[str, Any], infer_cfg: Mapping[str, Any]) -> "ChunkScheduler":
        return cls(
            horizon=int(get(data_cfg, "chunk.H", 8)),
            n_exec=int(get(infer_cfg, "chunk_exec.n_exec", 1)),
        )

    def should_replan(self, step: int) -> bool:
        """每 n_exec 步重新预测一次（step 从 0 开始）。"""
        return step % self.n_exec == 0

    def execute(self, chunk: np.ndarray, available_steps: int | None = None) -> np.ndarray:
        """截取本次要执行的 n_exec 步（并按 available_steps 截断 episode 尾部）。"""
        steps = np.asarray(chunk, dtype=np.float32)[: self.n_exec]
        if available_steps is not None:
            steps = steps[: max(0, available_steps)]
        return steps

    def effective_horizon(self) -> int:
        """实际被执行的步数（等价于重预测周期）。"""
        return self.n_exec

    def tail(self, chunk: np.ndarray) -> np.ndarray:
        """未被执行的尾部；drop_remainder=True 时返回空数组（显式丢弃而不是偷偷使用）。"""
        if self.drop_remainder:
            return np.zeros((0, np.asarray(chunk).shape[-1]), dtype=np.float32)
        return np.asarray(chunk, dtype=np.float32)[self.n_exec :]

    def describe(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "n_exec": self.n_exec,
            "replan_every_steps": self.n_exec,
            "theoretical_replan_rate_hz": None,
        }


def split_chunk_to_steps(chunk: np.ndarray) -> list[np.ndarray]:
    """把 [H, d_a] 拆成 H 个 [d_a] 的动作（保持顺序，不做任何平滑）。"""
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"chunk 应为 [H,d_a]，实际 {arr.shape}")
    return [arr[i] for i in range(arr.shape[0])]


class ActionChunker:
    """动作块的对外门面：语义（ActionSpec）+ 调度（ChunkScheduler）+ 逐步拆分。

    这是 src/models 对外的稳定接口（推理、评测、可视化都通过它，避免各处各自解释 H/n_exec）。
    """

    def __init__(self, spec: ActionSpec, scheduler: ChunkScheduler) -> None:
        self.spec = spec
        self.scheduler = scheduler

    @classmethod
    def from_config(cls, data_cfg: Mapping[str, Any], infer_cfg: Mapping[str, Any]) -> "ActionChunker":
        return cls(ActionSpec.from_config(data_cfg), ChunkScheduler.from_config(data_cfg, infer_cfg))

    @property
    def horizon(self) -> int:
        return self.scheduler.horizon

    @property
    def n_exec(self) -> int:
        return self.scheduler.n_exec

    def should_replan(self, step: int) -> bool:
        return self.scheduler.should_replan(step)

    def steps(self, chunk: np.ndarray) -> list[np.ndarray]:
        """本次实际执行的逐步动作（n_exec 步，已按 episode 尾部截断）。"""
        return split_chunk_to_steps(self.scheduler.execute(chunk))

    def describe(self) -> dict[str, Any]:
        return {"action": self.spec.to_dict(), **self.scheduler.describe()}
