"""CVAE 后验编码器 q(z | o, s, a_chunk) 与先验网络 p(z | o, s)。

⚠️ teacher forcing 边界：后验编码器**只在训练阶段被调用**，
    推理路径（predict_chunk）绝不访问 action_chunk，也不调用本类的 PosteriorEncoder。
    该边界由 tests/test_cvae_shapes.py 的“后验不可达”检查与 src/infer 的代码审查共同拦截。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ..utils.config import get
from .layers import mlp


class PosteriorEncoder(nn.Module):
    """q(z | o, s, a_chunk)：输入视觉特征、状态编码与真实未来动作块，输出对角高斯参数。"""

    def __init__(self, cond_dim: int, chunk_dim: int, latent_dim: int, hidden: Sequence[int],
                 dropout: float = 0.1, logvar_clamp: tuple[float, float] = (-6.0, 4.0)) -> None:
        super().__init__()
        self.chunk_dim = chunk_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp
        self.net = mlp(cond_dim + chunk_dim, hidden, 2 * latent_dim, dropout=dropout)
        self._zero_init_head()

    def _zero_init_head(self) -> None:
        """把输出层零初始化：初始时 mu=0, log_var=0 → q=p=N(0,I)，KL≈0。

        这是 VAE 训练的稳定化技巧（不用 LayerNorm，避免强制 mu/log_var 具有单位方差，
        那会让初始 KL 高达几十并压制重建项）。
        """
        head = self.net[-1]
        if isinstance(head, nn.Linear):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, cond: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action_chunk.dim() != 3:
            raise ValueError(f"action_chunk 应为 [B,H,d_a]，实际 {tuple(action_chunk.shape)}")
        flat = action_chunk.reshape(action_chunk.shape[0], -1)
        out = self.net(torch.cat([cond, flat], dim=-1))
        mu, log_var = out.chunk(2, dim=-1)
        # log_var 必须 clamp（防数值爆炸，见提示词 6.2 数值条款）
        log_var = torch.clamp(log_var, self.logvar_clamp[0], self.logvar_clamp[1])
        return mu, log_var


class PriorNetwork(nn.Module):
    """p(z | o, s)：只依赖当前观测与状态，训练与推理都使用。"""

    def __init__(self, cond_dim: int, latent_dim: int, hidden: Sequence[int],
                 dropout: float = 0.1, logvar_clamp: tuple[float, float] = (-6.0, 4.0)) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp
        self.net = mlp(cond_dim, hidden, 2 * latent_dim, dropout=dropout)
        head = self.net[-1]
        if isinstance(head, nn.Linear):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(cond)
        mu, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, self.logvar_clamp[0], self.logvar_clamp[1])
        return mu, log_var


def build_prior(model_cfg: Mapping[str, Any], cond_dim: int):
    latent_dim = int(get(model_cfg, "latent.dim", 16))
    clamp = tuple(get(model_cfg, "latent.logvar_clamp", [-6.0, 4.0]))
    return PriorNetwork(
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden=list(get(model_cfg, "latent.prior_hidden", [128, 128])),
        dropout=float(get(model_cfg, "decoder.dropout", 0.1)),
        logvar_clamp=(float(clamp[0]), float(clamp[1])),
    )


def build_posterior(model_cfg: Mapping[str, Any], cond_dim: int, chunk_dim: int):
    latent_dim = int(get(model_cfg, "latent.dim", 16))
    clamp = tuple(get(model_cfg, "latent.logvar_clamp", [-6.0, 4.0]))
    return PosteriorEncoder(
        cond_dim=cond_dim,
        chunk_dim=chunk_dim,
        latent_dim=latent_dim,
        hidden=list(get(model_cfg, "latent.posterior_hidden", [256, 256])),
        dropout=float(get(model_cfg, "decoder.dropout", 0.1)),
        logvar_clamp=(float(clamp[0]), float(clamp[1])),
    )
