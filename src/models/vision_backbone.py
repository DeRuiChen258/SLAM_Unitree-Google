"""多帧视觉编码：共享权重 CNN 主干 + 多帧聚合 + 特征投影。

输入（由 configs/data.yaml 决定）：
    单相机   images [B, K, C, H, W]
    多相机   images [B, N_cam, K, C, H, W]（按 `observation.multi_camera_mode` 拼接或堆叠）
输出：
    features [B, d_v]

硬约束：不得在本模块内做 resize/归一化（预处理在 datasets 层完成）；
        前向必须断言输入 shape 与数值健康。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..utils.config import get
from .layers import activation, norm_layer


class SmallCNN(nn.Module):
    """轻量 CNN 主干（无外部权重依赖，离线可复现）。"""

    def __init__(self, in_channels: int, feature_dim: int, norm: str = "group") -> None:
        super().__init__()
        widths = (32, 64, 128)
        layers: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(prev, width, kernel_size=3, stride=2, padding=1),
                norm_layer(norm, width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                norm_layer(norm, width),
                nn.ReLU(inplace=True),
            ]
            prev = width
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(widths[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"SmallCNN 期望 [N,C,H,W]，实际 {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("视觉输入含 NaN/Inf，请检查预处理是否与训练同源")
        feat = self.body(x)
        return self.proj(self.pool(feat).flatten(1))


class MultiFrameVisionEncoder(nn.Module):
    """多帧 / 多相机视觉编码器。

    **CUDA 为主的输入契约**：模型直接接收 **uint8** 图像（`[B,K,C,H,W]`，0–255），
    在**设备侧**完成 /255 与标准化；因此：
        * 主机→显存的数据量比 float32 方案少 4 倍（uint8 vs float32）；
        * CPU 不再做逐样本浮点转换与归一化，把 CPU 从数据路径上摘出来；
        * 训练与推理共用同一份归一化实现（就在本模块内），不存在"两边各写一份"的风险。
    float 输入被视为**已归一化**（供单元测试与外部预处理路径使用），不做二次归一化。
    """

    def __init__(self, data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.num_frames = int(get(data_cfg, "observation.num_frames", 1))
        self.num_cameras = int(get(data_cfg, "observation.num_cameras", 1))
        self.channels = int(get(data_cfg, "image.channels", 3))
        self.height = int(get(data_cfg, "image.height", 64))
        self.width = int(get(data_cfg, "image.width", 64))
        self.feature_dim = int(get(model_cfg, "vision.feature_dim", 128))
        self.aggregation = str(get(model_cfg, "aggregation.mode", "last"))
        self.freeze = bool(get(model_cfg, "vision.freeze", False))
        mean = torch.tensor(get(data_cfg, "image.normalize.mean"), dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(get(data_cfg, "image.normalize.std"), dtype=torch.float32).view(1, -1, 1, 1)
        # persistent=False：这两个常量由配置决定，不必写进 checkpoint（避免与 config hash 重复校验）
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)
        encoder_dim = self.feature_dim
        self.encoder = SmallCNN(
            in_channels=self.channels,
            feature_dim=encoder_dim,
            norm=str(get(model_cfg, "vision.norm", "group")),
        )
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if self.aggregation == "gru":
            self.gru = nn.GRU(encoder_dim, int(get(model_cfg, "aggregation.hidden", encoder_dim)), batch_first=True)
            out_dim = int(get(model_cfg, "aggregation.hidden", encoder_dim))
        else:
            self.gru = None
            out_dim = encoder_dim
        self.out_dim = out_dim * self.num_cameras
        self.post = nn.Sequential(nn.Linear(self.out_dim, self.feature_dim), activation("relu"))

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        """把 [B,(N_cam),K,C,H,W] 统一成 [B*N_cam*K, C, H, W] 并记录维度信息。"""
        if images.dim() == 5:
            b, k, c, h, w = images.shape
            cams = 1
        elif images.dim() == 6:
            b, cams, k, c, h, w = images.shape
        else:
            raise ValueError(f"images 维度应为 5 或 6，实际 {tuple(images.shape)}")
        if (k, c, h, w) != (self.num_frames, self.channels, self.height, self.width):
            raise ValueError(
                f"观测 shape {(k, c, h, w)} 与配置 {(self.num_frames, self.channels, self.height, self.width)} 不一致"
            )
        flat = self.normalize_images(images.reshape(b * cams * k, c, h, w))
        return flat, b, cams, k

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """设备侧归一化：uint8 → float → 标准化；float 输入直接透传（视为已归一化）。"""
        if images.dtype == torch.uint8:
            x = images.to(torch.float32).div_(255.0)
            mean = self.img_mean.to(x.dtype)
            std = self.img_std.to(x.dtype)
            return (x - mean) / std
        if images.dtype != torch.float32:
            return images.to(torch.float32)
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat, b, cams, k = self._prepare(images)
        emb = self.encoder(flat)                      # [B*N*K, d_e]
        emb = emb.reshape(b * cams, k, -1)
        if self.aggregation == "gru":
            _, hidden = self.gru(emb)
            agg = hidden[-1]
        elif self.aggregation == "mean":
            agg = emb.mean(dim=1)
        elif self.aggregation == "last":
            agg = emb[:, -1]
        else:
            raise ValueError(f"未知多帧聚合方式 {self.aggregation!r}（last/mean/gru）")
        features = agg.reshape(b, cams * agg.shape[-1])
        return self.post(features)
