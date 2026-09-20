"""时间集成融合器（核心文件）：把最近若干次预测在时间轴上重叠的动作按权重融合。

支持三种权重模式（提示词【六】6.4）：
    uniform      : w = 1                       （同一 chunk 内每步等权）
    exponential  : w = decay ** age            （age = t - t0_pred，按步计）
    inverse_age  : w = 1 / (1 + age)
融合公式：action(t) = Σ_j w_j · a_j(t) / Σ_j w_j

硬约束：
    * 权重必须归一化；冷启动（仅 1 条预测）时退化为该预测本身（不产生除零）；
    * 过期预测（t0_pred + H <= t）必须被丢弃；缺步与低权预测显式裁剪，**禁止补零**
      （补零会系统性把动作拉向原点，是这类实现最常见的隐蔽 bug）；
    * 融合只作用于「未来将被执行」的步，禁止回改已执行动作（由调用方的 t 单调递增保证）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class EnsembleRecord:
    """一次预测的完整记录（用于融合与复盘：动作来源必须可追溯）。"""

    chunk: np.ndarray                 # [H, d_a]
    t0: int                           # 该 chunk 首步对应的策略时钟（步）
    n_exec: int                       # 本次承诺执行的步数
    meta: dict[str, Any] = field(default_factory=dict)
    mask: np.ndarray | None = None    # [H]，0 表示该步无效（episode 末尾 padding）

    @property
    def horizon(self) -> int:
        return int(self.chunk.shape[0])

    def covers(self, t: int) -> bool:
        return self.t0 <= t < self.t0 + self.horizon

    def step_index(self, t: int) -> int:
        return t - self.t0


class TemporalEnsembler:
    """在线时间集成器。"""

    MODES = ("uniform", "exponential", "inverse_age")

    def __init__(self, mode: str = "exponential", decay: float = 0.5, window: int = 8,
                 min_weight: float = 1e-3, normalize: bool = True) -> None:
        if mode not in self.MODES:
            raise ValueError(f"未知时间集成模式 {mode!r}（可选 {self.MODES}）")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay 必须在 (0,1] 内，实际 {decay}")
        self.mode = mode
        self.decay = float(decay)
        self.window = int(window)
        self.min_weight = float(min_weight)
        self.normalize = bool(normalize)
        self._records: deque[EnsembleRecord] = deque(maxlen=self.window)
        self._n_updates = 0
        self._n_expired = 0
        self._n_no_source = 0
        self._weight_history: list[dict[str, Any]] = []

    # ----------------------------------------------------------------------------------
    # 在线更新
    # ----------------------------------------------------------------------------------
    def update(self, chunk: np.ndarray, t0: int, n_exec: int | None = None,
               meta: Mapping[str, Any] | None = None, mask: np.ndarray | None = None) -> None:
        """写入一次新预测；同时丢弃已过期（完全落在当前时刻之前）的预测。"""
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"chunk 应为 [H,d_a]，实际 {arr.shape}")
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 0:
                raise ValueError("mask 必须是逐步数组 [H]；标量掩码请在调用方广播，避免静默语义错误")
            if mask.shape[0] != arr.shape[0]:
                raise ValueError(f"mask 长度 {mask.shape[0]} 与 chunk 步数 {arr.shape[0]} 不一致")
        # 过期裁剪：不再覆盖任何未来步的记录必须离开
        while self._records and (self._records[0].t0 + self._records[0].horizon) <= t0:
            self._records.popleft()
            self._n_expired += 1
        self._records.append(
            EnsembleRecord(chunk=arr, t0=int(t0), n_exec=int(n_exec or arr.shape[0]),
                           meta=dict(meta or {}), mask=None if mask is None else np.asarray(mask, dtype=np.float32))
        )
        self._n_updates += 1

    def reset(self) -> None:
        self._records.clear()
        self._n_updates = 0
        self._n_expired = 0
        self._n_no_source = 0
        self._weight_history.clear()

    # ----------------------------------------------------------------------------------
    # 权重与查询
    # ----------------------------------------------------------------------------------
    def _weight(self, age: int) -> float:
        if self.mode == "uniform":
            return 1.0
        if self.mode == "exponential":
            return float(self.decay ** max(0, age))
        return 1.0 / (1.0 + max(0, age))

    def weights_at(self, t: int, with_sources: bool = False) -> Any:
        """返回覆盖 t 的 (记录下标, 权重) 列表；权重已剔除低于 min_weight 的项。"""
        pairs: list[tuple[int, float]] = []
        for i, rec in enumerate(self._records):
            if not rec.covers(t):
                continue
            idx = rec.step_index(t)
            if rec.mask is not None and rec.mask[idx] <= 0:
                continue
            w = self._weight(age=t - rec.t0)
            if w < self.min_weight:
                continue
            pairs.append((i, w))
        if with_sources:
            return [(i, w, dict(self._records[i].meta)) for i, w in pairs]
        return pairs

    def action_at(self, t: int, with_sources: bool = False) -> np.ndarray | None:
        """返回时刻 t 的融合动作；没有任何贡献者时返回 None（禁止返回 0 动作）。"""
        pairs = self.weights_at(t)
        if not pairs:
            self._n_no_source += 1
            return None
        dim = self._records[0].chunk.shape[1]
        acc = np.zeros(dim, dtype=np.float64)
        wsum = 0.0
        contributors: list[dict[str, Any]] = []
        for i, w in pairs:
            rec = self._records[i]
            acc += w * rec.chunk[rec.step_index(t)].astype(np.float64)
            wsum += w
            if with_sources:
                contributors.append({"record": i, "t0": rec.t0, "age": t - rec.t0, "weight": w,
                                     "meta": dict(rec.meta)})
        if wsum <= 0:
            self._n_no_source += 1
            return None
        action = acc / wsum if self.normalize else acc
        self._weight_history.append(
            {"t": int(t), "n_contributors": len(pairs), "weight_sum": float(wsum),
             "weights": [float(w) for _, w in pairs],
             "sources": contributors if with_sources else None}
        )
        return action.astype(np.float32)

    # ----------------------------------------------------------------------------------
    # 统计与查询
    # ----------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        per_step = [h["n_contributors"] for h in self._weight_history]
        return {
            "mode": self.mode,
            "decay": self.decay,
            "window": self.window,
            "n_updates": self._n_updates,
            "n_expired": self._n_expired,
            "n_no_source": self._n_no_source,
            "n_live_records": len(self._records),
            "contributors_mean": float(np.mean(per_step)) if per_step else 0.0,
            "contributors_max": int(max(per_step)) if per_step else 0,
        }

    def weight_matrix(self, t_start: int, t_end: int, rows: int | None = None) -> np.ndarray:
        """构造 [n_steps, n_predictions] 的权重热图数据（供可视化）。"""
        n_rows = rows or max(self.window, self._n_updates)
        mat = np.zeros((max(0, t_end - t_start), max(1, n_rows)), dtype=np.float32)
        for i, t in enumerate(range(t_start, t_end)):
            for rec_idx, w in self.weights_at(t):
                if rec_idx < mat.shape[1]:
                    mat[i, rec_idx] = w
        return mat

    def history(self) -> list[dict[str, Any]]:
        return list(self._weight_history)

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_config(cls, infer_cfg: Mapping[str, Any]) -> "TemporalEnsembler":
        from ..utils.config import get

        return cls(
            mode=str(get(infer_cfg, "temporal_ensemble.mode", "exponential")),
            decay=float(get(infer_cfg, "temporal_ensemble.decay", 0.5)),
            window=int(get(infer_cfg, "temporal_ensemble.window", 8)),
            min_weight=float(get(infer_cfg, "temporal_ensemble.min_weight", 1e-3)),
            normalize=bool(get(infer_cfg, "temporal_ensemble.normalize_weights", True)),
        )


def uniform_baseline(chunks: list[np.ndarray]) -> np.ndarray:
    """无时间集成基线：直接使用最新一次预测（A3 对照用）。"""
    if not chunks:
        raise ValueError("chunks 不能为空")
    return np.asarray(chunks[-1], dtype=np.float32)
