"""KL 项权重 β 调度器：linear / cosine / cyclical + warmup steps。

β(t) 从 start 升到 end（warmup 内完成），之后再按 scheme 周期或保持；
训练初期 β 小可以让重建先学好，避免 posterior collapse。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..utils.config import get


class BetaAnnealer:
    """β 调度；`value(step)` 在任意 step 上都是纯函数（可复现、可画曲线）。"""

    def __init__(self, scheme: str = "linear", warmup_steps: int = 0, start: float = 0.0,
                 end: float = 1.0, cycle_steps: int | None = None) -> None:
        if scheme not in ("linear", "cosine", "cyclical", "none"):
            raise ValueError(f"未知退火方案 {scheme!r}")
        self.scheme = scheme
        self.warmup_steps = max(0, int(warmup_steps))
        self.start = float(start)
        self.end = float(end)
        self.cycle_steps = int(cycle_steps or max(1, self.warmup_steps))

    @classmethod
    def from_config(cls, train_cfg: Mapping[str, Any]) -> "BetaAnnealer":
        kl = get(train_cfg, "loss.kl", {}) or {}
        if not kl.get("enabled", True):
            # A1 消融：KL 关闭时 β 恒为 0（而不是偷偷保留权重）
            return cls(scheme="none", warmup_steps=0, start=0.0, end=0.0)
        anneal = kl.get("anneal", {}) or {}
        return cls(
            scheme=str(anneal.get("scheme", "linear")),
            warmup_steps=int(anneal.get("warmup_steps", 0)),
            start=float(anneal.get("start", 0.0)),
            end=float(anneal.get("end", kl.get("weight", 1.0))),
        )

    def value(self, step: int) -> float:
        if self.scheme == "none" or self.end == self.start:
            return self.end
        if step < 0:
            step = 0
        if self.warmup_steps <= 0:
            return self.end
        ratio = min(1.0, step / float(self.warmup_steps))
        if self.scheme == "linear":
            frac = ratio
        elif self.scheme == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * ratio))
        else:  # cyclical：按 cycle_steps 循环上升
            cycle_pos = (step % max(1, self.cycle_steps)) / float(max(1, self.cycle_steps))
            frac = cycle_pos if step < self.warmup_steps else 1.0
        return self.start + (self.end - self.start) * frac

    def curve(self, steps: int) -> list[float]:
        return [self.value(s) for s in range(steps)]

    def describe(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "warmup_steps": self.warmup_steps,
            "start": self.start,
            "end": self.end,
            "cycle_steps": self.cycle_steps,
        }
