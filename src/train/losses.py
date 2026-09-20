"""总损失装配：把配置里的权重/退火翻译成模型可用的损失（公式在 src/models/losses.py）。

总损失：L = w_recon · L_recon + β(t) · L_kl + λ_smooth · L_smooth + λ_tc · L_tc
每一项都能被 configs/ablation/*.yaml 独立关闭（消融 A1/A3 等需要）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..models.losses import masked_recon_loss, smoothness_loss, temporal_consistency_loss
from ..utils.config import get


@dataclass
class LossWeights:
    """由 train.yaml 派生的损失权重容器（无隐藏默认值，全部显式可见）。"""

    recon_type: str = "huber"
    huber_delta: float = 1.0
    recon_weight: float = 1.0
    step_weights: torch.Tensor | None = None
    kl_enabled: bool = True
    free_bits: float = 0.0
    smooth_enabled: bool = True
    smooth_weight: float = 0.05
    smooth_order: int = 1
    tc_enabled: bool = False
    tc_weight: float = 0.0
    tc_shift: int = 1
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recon_type": self.recon_type,
            "huber_delta": self.huber_delta,
            "recon_weight": self.recon_weight,
            "kl_enabled": self.kl_enabled,
            "free_bits": self.free_bits,
            "smooth_enabled": self.smooth_enabled,
            "smooth_weight": self.smooth_weight,
            "smooth_order": self.smooth_order,
            "tc_enabled": self.tc_enabled,
            "tc_weight": self.tc_weight,
            "tc_shift": self.tc_shift,
            "step_weights": self.step_weights,
        }


def build_loss_weights(train_cfg: Mapping[str, Any], horizon: int) -> LossWeights:
    """把 train.yaml 的 loss 段翻译成 LossWeights（逐步权重也在这里构造）。"""
    mode = str(get(train_cfg, "loss.recon.step_weight_mode", "uniform"))
    decay = float(get(train_cfg, "loss.recon.step_weight_decay", 0.95))
    if mode == "decay":
        step_weights = torch.tensor([decay**i for i in range(horizon)], dtype=torch.float32)
        step_weights = step_weights / step_weights.mean()
    else:
        step_weights = torch.ones(horizon, dtype=torch.float32)
    return LossWeights(
        recon_type=str(get(train_cfg, "loss.recon.type", "huber")),
        huber_delta=float(get(train_cfg, "loss.recon.huber_delta", 1.0)),
        recon_weight=float(get(train_cfg, "loss.recon.weight", 1.0)),
        step_weights=step_weights,
        kl_enabled=bool(get(train_cfg, "loss.kl.enabled", True)),
        free_bits=float(get(train_cfg, "loss.kl.free_bits", 0.0)),
        smooth_enabled=bool(get(train_cfg, "loss.smooth.enabled", True)),
        smooth_weight=float(get(train_cfg, "loss.smooth.weight", 0.0)),
        smooth_order=int(get(train_cfg, "loss.smooth.order", 1)),
        tc_enabled=bool(get(train_cfg, "loss.temporal_consistency.enabled", False)),
        tc_weight=float(get(train_cfg, "loss.temporal_consistency.weight", 0.0)),
        tc_shift=int(get(train_cfg, "loss.temporal_consistency.shift", 1)),
    )


def compute_total_loss(policy, batch: Mapping[str, Any], beta: float, weights: LossWeights) -> dict[str, Any]:
    """调用策略的 compute_loss，保持损失公式只实现一份（models/losses.py）。"""
    return policy.compute_loss(batch, beta=beta, weights=weights.as_dict(), need_smooth=weights.smooth_enabled)


def standalone_regularizers(pred_chunk: torch.Tensor, mask: torch.Tensor, gt_chunk: torch.Tensor,
                            weights: LossWeights) -> dict[str, torch.Tensor]:
    """不含 KL 的独立正则项，供评测脚本在无模型时单独计算（例如对缓存预测打分）。"""
    return {
        "recon": masked_recon_loss(pred_chunk, gt_chunk, mask, weights.recon_type, weights.huber_delta,
                                   weights.step_weights)[0],
        "smooth": smoothness_loss(pred_chunk, mask, weights.smooth_order),
    }


__all__ = [
    "LossWeights",
    "build_loss_weights",
    "compute_total_loss",
    "standalone_regularizers",
    "temporal_consistency_loss",
    "smoothness_loss",
    "masked_recon_loss",
]
