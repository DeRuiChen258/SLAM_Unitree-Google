"""共享基础块：MLP 构造器、激活、权重初始化、重参数化工具。

避免在多处重复实现同一结构（DRY）；不做任何 I/O。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def activation(name: str) -> nn.Module:
    table = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
    if name not in table:
        raise ValueError(f"未知激活函数 {name!r}（可选 {sorted(table)}）")
    return table[name]()


def norm_layer(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"未知归一化类型 {kind!r}")


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation_name: str = "relu",
        dropout: float = 0.0, final_activation: bool = False) -> nn.Sequential:
    """构造 MLP：Linear → Activation → Dropout 重复，最后一层为 Linear(out_dim)。"""
    layers: list[nn.Module] = []
    dims: Iterable[int] = [in_dim, *hidden]
    prev = in_dim
    for width in list(dims)[1:]:
        layers += [nn.Linear(prev, int(width)), activation(activation_name)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = int(width)
    layers.append(nn.Linear(prev, out_dim))
    if final_activation:
        layers.append(activation(activation_name))
    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """正交初始化 Linear，常数初始化 Norm；卷积未显式处理（由各自模块决定）。"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """重参数化采样：z = mu + exp(0.5 * log_var) * eps，eps ~ N(0, I)。"""
    std = torch.exp(0.5 * log_var)
    if generator is None:
        eps = torch.randn_like(std)
    else:
        eps = torch.randn(std.shape, generator=generator, dtype=std.dtype, device=std.device)
    return mu + std * eps


def kl_diag_gaussian(mu_q: torch.Tensor, log_var_q: torch.Tensor,
                     mu_p: torch.Tensor, log_var_p: torch.Tensor,
                     free_bits: float = 0.0) -> torch.Tensor:
    """KL(q || p) 的解析式（对角高斯），返回逐样本 KL 之和 [B]。

    KL = 0.5 * Σ ( var_q/var_p + (mu_p-mu_q)^2/var_p - 1 + log_var_p - log_var_q )
    free_bits > 0 时对每一维做 max(kl_d, free_bits) 下限（避免潜变量塌缩）。
    """
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    kl_dims = 0.5 * (var_q / var_p + (mu_p - mu_q) ** 2 / var_p - 1.0 + log_var_p - log_var_q)
    if free_bits > 0:
        kl_dims = torch.clamp(kl_dims, min=free_bits)
    return kl_dims.sum(dim=-1)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
