"""CVAE 策略主体（核心文件）：先验 / 后验 / 重参数化 / 动作块解码 / 损失装配。

显式实现的要素（逐条对应提示词【六】6.2）：
  (1) 先验网络 p(z | o, s)          —— 训练与推理都使用；
  (2) 后验网络 q(z | o, s, a_chunk) —— **仅训练**使用（teacher forcing 边界）；
  (3) 重参数化 z = mu + exp(0.5*log_var) * eps；
  (4) 解码器/策略头 pi(a_chunk | o, s, z) —— 一次输出未来 H 步动作块；
  (5) 损失装配：重建 + KL（权重与退火由 train.yaml 提供，可逐项关闭）；
  (6) 推理纯度：`predict_chunk` 只走先验，绝不接触真实未来动作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from ..utils.config import get, state_block_layout
from .encoder import build_posterior, build_prior
from .layers import count_parameters, mlp, reparameterize
from .losses import kl_from_params, masked_recon_loss, smoothness_loss, temporal_consistency_loss
from .vision_backbone import MultiFrameVisionEncoder


@dataclass
class CVAEOutput:
    """前向输出容器（便于测试与日志逐项读取）。"""

    pred_chunk: torch.Tensor                  # [B, H, d_a]
    z: torch.Tensor                           # [B, latent_dim]
    mu_prior: torch.Tensor
    log_var_prior: torch.Tensor
    mu_post: torch.Tensor | None = None
    log_var_post: torch.Tensor | None = None
    aux: dict[str, Any] | None = None


class ActionDecoder(nn.Module):
    """解码器 / 策略头：输入 (视觉+状态+latent)，输出未来 H 步动作块。"""

    def __init__(self, cond_dim: int, latent_dim: int, horizon: int, action_dim: int,
                 hidden: Sequence[int], dropout: float, activation_name: str) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = mlp(cond_dim + latent_dim, hidden, horizon * action_dim, activation_name, dropout)

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([cond, z], dim=-1))
        return out.reshape(-1, self.horizon, self.action_dim)


class CVAEPolicy(nn.Module):
    """条件 VAE 策略：观测 → 动作块。"""

    def __init__(self, data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.data_cfg = dict(data_cfg)
        self.horizon = int(get(data_cfg, "chunk.H", 8))
        self.action_dim = int(get(data_cfg, "action.dim", 7))
        self.state_dim = int(state_block_layout(data_cfg)[-1][2])
        self.latent_enabled = bool(get(model_cfg, "latent.enabled", True))
        self.latent_dim = int(get(model_cfg, "latent.dim", 16)) if self.latent_enabled else 1
        self.logvar_clamp = tuple(get(model_cfg, "latent.logvar_clamp", [-6.0, 4.0]))
        self.free_bits = float(get(model_cfg, "latent.free_bits", 0.0)) if get(model_cfg, "latent.free_bits") else 0.0

        self.vision = MultiFrameVisionEncoder(data_cfg, model_cfg)
        st_hidden = list(get(model_cfg, "state_encoder.hidden", [128, 128]))
        self.state_encoder = mlp(self.state_dim, st_hidden, st_hidden[-1],
                                 str(get(model_cfg, "state_encoder.activation", "relu")),
                                 float(get(model_cfg, "state_encoder.dropout", 0.0)))
        self.cond_dim = self.vision.feature_dim + st_hidden[-1]

        self.prior = build_prior(model_cfg, self.cond_dim)
        self.posterior = build_posterior(model_cfg, self.cond_dim, self.horizon * self.action_dim) \
            if self.latent_enabled else None
        self.decoder = ActionDecoder(
            cond_dim=self.cond_dim,
            latent_dim=self.latent_dim,
            horizon=self.horizon,
            action_dim=self.action_dim,
            hidden=list(get(model_cfg, "decoder.hidden", [256, 256])),
            dropout=float(get(model_cfg, "decoder.dropout", 0.1)),
            activation_name=str(get(model_cfg, "decoder.activation", "relu")),
        )

    # ----------------------------------------------------------------------------------
    # 前向
    # ----------------------------------------------------------------------------------
    def encode(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """观测编码：视觉特征 + 状态编码 → 条件向量 cond [B, d_c]。"""
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state 应为 [B,{self.state_dim}]，实际 {tuple(state.shape)}")
        visual = self.vision(images)
        return torch.cat([visual, self.state_encoder(state)], dim=-1)

    def forward(self, images: torch.Tensor, state: torch.Tensor,
                action_chunk: torch.Tensor | None = None, mode: str = "train",
                sample_z: bool = True) -> CVAEOutput:
        """前向。

        mode="train"：允许（且应当）传入真实动作块以计算后验（teacher forcing）；
        mode="eval"/"infer"：禁止传后验，z 只能来自先验。
        """
        cond = self.encode(images, state)
        mu_prior, log_var_prior = self.prior(cond)
        mu_post = log_var_post = None

        if self.latent_enabled:
            if mode == "train":
                if action_chunk is None:
                    raise ValueError("训练模式的 forward 必须提供 action_chunk（teacher forcing 边界）")
                assert self.posterior is not None
                mu_post, log_var_post = self.posterior(cond, action_chunk)
                z = reparameterize(mu_post, log_var_post) if sample_z else mu_post
            else:
                if action_chunk is not None:
                    raise ValueError("推理/验证模式禁止传入真实动作块（后验只能用于训练）")
                z = reparameterize(mu_prior, log_var_prior) if sample_z else mu_prior
        else:
            # A1 消融：关闭 latent，退化为确定性动作块回归（无 KL）
            z = cond.new_zeros((cond.shape[0], self.latent_dim))

        pred = self.decoder(cond, z)
        return CVAEOutput(pred_chunk=pred, z=z, mu_prior=mu_prior, log_var_prior=log_var_prior,
                          mu_post=mu_post, log_var_post=log_var_post)

    # ----------------------------------------------------------------------------------
    # 推理接口（不得访问后验）
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def predict_chunk(self, images: torch.Tensor, state: torch.Tensor,
                      mode: str = "mean", n_samples: int = 1) -> torch.Tensor:
        """从先验解码动作块。

        mode="mean"：z 取先验均值（推荐用于闭环控制，减小抖动）；
        mode="sample"：从先验采样；n_samples>1 时返回 [n_samples, B, H, d_a] 展示多模态。
        """
        if mode not in ("mean", "sample"):
            raise ValueError(f"mode 只能为 mean/sample，实际 {mode!r}")
        was_training = self.training
        self.eval()
        try:
            if n_samples == 1:
                return self.forward(images, state, None, mode="infer", sample_z=(mode == "sample")).pred_chunk
            samples = [
                self.forward(images, state, None, mode="infer", sample_z=True).pred_chunk
                for _ in range(n_samples)
            ]
            return torch.stack(samples, dim=0)
        finally:
            self.train(was_training)

    # ----------------------------------------------------------------------------------
    # 损失
    # ----------------------------------------------------------------------------------
    def compute_loss(self, batch: Mapping[str, torch.Tensor], beta: float = 1.0,
                     weights: Mapping[str, float] | None = None,
                     need_smooth: bool = True) -> dict[str, torch.Tensor]:
        """返回 {recon, kl, smooth, temporal, reg, total, beta}，每一项都独立可关。"""
        weights = dict(weights or {})
        images, state = batch["images"], batch["state"]
        target = batch["action_chunk"]
        mask = batch["mask"]

        out = self.forward(images, state, target, mode="train", sample_z=True)
        recon, step_losses = masked_recon_loss(
            out.pred_chunk, target, mask,
            kind=str(weights.get("recon_type", "huber")),
            huber_delta=float(weights.get("huber_delta", 1.0)),
            step_weights=weights.get("step_weights"),
        )
        loss = float(weights.get("recon_weight", 1.0)) * recon
        loss_dict: dict[str, torch.Tensor] = {
            "recon": recon,
            "step_losses": step_losses,
            "beta": torch.as_tensor(beta, device=recon.device),
        }

        if self.latent_enabled and bool(weights.get("kl_enabled", True)):
            kl = kl_from_params(out.mu_post, out.log_var_post, out.mu_prior, out.log_var_prior,
                                free_bits=float(weights.get("free_bits", 0.0)))
            loss = loss + beta * kl
            loss_dict["kl"] = kl
        else:
            loss_dict["kl"] = torch.zeros((), device=recon.device)

        reg = torch.zeros((), device=recon.device)
        if need_smooth and bool(weights.get("smooth_enabled", True)) and float(weights.get("smooth_weight", 0.0)) > 0:
            smooth = smoothness_loss(out.pred_chunk, mask, order=int(weights.get("smooth_order", 1)))
            loss = loss + float(weights["smooth_weight"]) * smooth
            reg = reg + float(weights["smooth_weight"]) * smooth
            loss_dict["smooth"] = smooth
        else:
            loss_dict["smooth"] = torch.zeros((), device=recon.device)

        if bool(weights.get("tc_enabled", False)) and float(weights.get("tc_weight", 0.0)) > 0 and "action_chunk_next" in batch:
            tc = temporal_consistency_loss(out.pred_chunk, batch["action_chunk_next"], mask,
                                           batch["mask_next"], shift=int(weights.get("tc_shift", 1)))
            loss = loss + float(weights["tc_weight"]) * tc
            reg = reg + float(weights["tc_weight"]) * tc
            loss_dict["temporal"] = tc
        else:
            loss_dict["temporal"] = torch.zeros((), device=recon.device)

        loss_dict["reg"] = reg
        loss_dict["total"] = loss
        return loss_dict

    # ----------------------------------------------------------------------------------
    # 元信息
    # ----------------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "params_total": count_parameters(self, trainable_only=False),
            "params_trainable": count_parameters(self, trainable_only=True),
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "latent_enabled": self.latent_enabled,
            "latent_dim": self.latent_dim,
            "vision_feature_dim": self.vision.feature_dim,
            "cond_dim": self.cond_dim,
            "aggregation": get(self.data_cfg, "observation.num_frames"),
        }
