"""模型侧损失原语（张量级，无配置解析，无 I/O）。

分工说明（DRY）：
    * 本文件：重建 / KL / 平滑 / 时序一致性的**公式实现**；
    * src/train/losses.py：权重装配、β 退火与总损失组合（调用本文件的公式）。
这样既满足「模型不依赖 train」的依赖方向，又不重复实现同一公式。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      kind: str = "huber", huber_delta: float = 1.0,
                      step_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """掩码加权的重建损失。

    返回 (总损失, 逐步损失 [H])；mask=0 的 padding 步被完全排除，
    不会把「0 动作」当作监督信号（提示词【十三】列出的隐蔽 bug 之一）。
    """
    if pred.shape != target.shape:
        raise ValueError(f"预测与标签 shape 不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if kind == "l1":
        per_elem = (pred - target).abs()
    elif kind == "mse":
        per_elem = (pred - target) ** 2
    elif kind == "huber":
        per_elem = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"未知重建损失类型 {kind!r}（l1/mse/huber）")

    per_step = per_elem.mean(dim=-1)                      # [B, H]
    weight = mask.float()
    if step_weights is not None:
        weight = weight * step_weights.reshape(1, -1).to(weight.device)
    denom = weight.sum().clamp_min(1e-8)
    total = (per_step * weight).sum() / denom
    step_mean = (per_step * weight).sum(dim=0) / weight.sum(dim=0).clamp_min(1e-8)
    return total, step_mean.detach()


def smoothness_loss(chunk: torch.Tensor, mask: torch.Tensor, order: int = 1) -> torch.Tensor:
    """动作平滑损失：掩码加权的一阶/二阶差分（jerk）惩罚。

    chunk [B,H,d_a]，mask [B,H]；只在两端都有效的相邻步上计算差分。
    """
    if order not in (1, 2):
        raise ValueError(f"order 只能为 1 或 2，实际 {order}")
    diff = chunk
    weight = mask.float()
    for _ in range(order):
        diff = diff[:, 1:] - diff[:, :-1]
        weight = weight[:, 1:] * weight[:, :-1]
    if diff.shape[1] == 0:
        return chunk.new_zeros(())
    per_step = diff.abs().mean(dim=-1)
    denom = weight.sum().clamp_min(1e-8)
    return (per_step * weight).sum() / denom


def temporal_consistency_loss(chunk_current: torch.Tensor, chunk_shifted: torch.Tensor,
                              mask_current: torch.Tensor, mask_shifted: torch.Tensor,
                              shift: int = 1) -> torch.Tensor:
    """时序一致性：相邻窗口重叠部分（错开 shift 步）的预测应当一致。"""
    if shift < 1:
        raise ValueError("shift 必须 ≥ 1")
    if chunk_current.shape[1] <= shift:
        return chunk_current.new_zeros(())
    a = chunk_current[:, :-shift]
    b = chunk_shifted[:, shift:]
    n = min(a.shape[1], b.shape[1])
    if n <= 0:
        return chunk_current.new_zeros(())
    a, b = a[:, :n], b[:, :n]
    w = (mask_current[:, :-shift][:, :n] * mask_shifted[:, shift:][:, :n]).float()
    per_step = (a - b).abs().mean(dim=-1)
    denom = w.sum().clamp_min(1e-8)
    return (per_step * w).sum() / denom


def kl_from_params(mu_q: torch.Tensor, log_var_q: torch.Tensor,
                   mu_p: torch.Tensor, log_var_p: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL(q||p) 的解析式（对 batch 求均值）。"""
    from .layers import kl_diag_gaussian

    return kl_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p, free_bits=free_bits).mean()
