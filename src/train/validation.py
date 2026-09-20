"""验证循环：重建误差 / KL / 平滑度 / 采样一致性 / 采样多样性。

同时报告 "确定性解码 vs 随机采样" 的差异：CVAE 若真的学到多模态，
多次采样的分散度应显著大于 0；若接近 0，说明潜变量塌缩（需检查 β 与 free-bits）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ..models.losses import masked_recon_loss, smoothness_loss
from ..utils.metrics import jitter
from .losses import LossWeights, compute_total_loss


@torch.no_grad()
def run_validation(model, loader, beta: float, weights: LossWeights, device: torch.device,
                   batch_limit: int | None = None, n_samples: int = 4) -> dict[str, Any]:
    """返回验证指标字典（全部来自真实前向，不做事后修饰）。"""
    was_training = model.training
    model.eval()
    totals: dict[str, list[float]] = {"total": [], "recon": [], "kl": [], "smooth": []}
    diversity: list[float] = []
    consistency: list[float] = []
    chunk_jitter: list[float] = []

    for i, batch in enumerate(loader):
        if batch_limit is not None and i >= batch_limit:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        losses = compute_total_loss(model, batch, beta, weights)
        for key in totals:
            totals[key].append(float(losses[key].detach().cpu()))

        pred_mean = model.predict_chunk(batch["images"], batch["state"], mode="mean")
        samples = model.predict_chunk(batch["images"], batch["state"], mode="sample", n_samples=n_samples)
        if samples.dim() == 4:  # [N, B, H, d_a]
            # 只在**同一观测**的多次采样之间统计分散度：
            # 跨观测两两相减会把"不同样本本身差异大"混进多模态指标（曾经的度量错误）。
            per_sample = samples.permute(1, 0, 2, 3)                     # [B, N, H, d_a]
            pairwise = (per_sample[:, :, None] - per_sample[:, None, :]).abs().mean(dim=(2, 3, 4))
            diversity.append(float(pairwise.mean().cpu()))
            consistency.append(float((samples.mean(dim=0) - pred_mean).abs().mean().cpu()))
        chunk_jitter.append(jitter(pred_mean[0].cpu().numpy(), order=2))

    model.train(was_training)
    out = {f"val/{k}": (float(np.mean(v)) if v else float("nan")) for k, v in totals.items()}
    out["val/sample_diversity"] = float(np.mean(diversity)) if diversity else 0.0
    out["val/mean_vs_sample_gap"] = float(np.mean(consistency)) if consistency else 0.0
    out["val/pred_chunk_jitter"] = float(np.mean(chunk_jitter)) if chunk_jitter else 0.0
    out["val/beta"] = float(beta)
    return out


def per_step_recon(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """逐步重建误差 [H]（用于画“误差随预测步数增长”的曲线）。"""
    return masked_recon_loss(pred, target, mask, "l1")[1]


def smoothness_of(chunk: torch.Tensor, mask: torch.Tensor, order: int = 2) -> float:
    return float(smoothness_loss(chunk, mask, order).detach().cpu())
