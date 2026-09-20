"""前向 shape 与数值健康：输出 [B,H,d_a]、无 NaN、log_var 被 clamp、推理不碰后验。"""

from __future__ import annotations

import pytest
import torch

from src.models.cvae_policy import CVAEPolicy


@pytest.fixture(scope="module")
def policy(data_cfg, model_cfg):
    torch.manual_seed(0)
    return CVAEPolicy(data_cfg, model_cfg).eval()


def test_forward_shapes(policy, tiny_batch):
    out = policy(tiny_batch["images"], tiny_batch["state"], tiny_batch["action_chunk"], mode="train")
    h = policy.horizon
    d_a = policy.action_dim
    latent = policy.latent_dim
    assert out.pred_chunk.shape == (tiny_batch["state"].shape[0], h, d_a)
    assert out.mu_prior.shape == (tiny_batch["state"].shape[0], latent)
    assert out.log_var_prior.shape == (tiny_batch["state"].shape[0], latent)
    assert out.mu_post.shape == out.mu_prior.shape
    assert torch.isfinite(out.pred_chunk).all()


def test_predict_chunk_from_prior(policy, tiny_batch):
    chunk_mean = policy.predict_chunk(tiny_batch["images"], tiny_batch["state"], mode="mean")
    assert chunk_mean.shape == (tiny_batch["state"].shape[0], policy.horizon, policy.action_dim)
    multi = policy.predict_chunk(tiny_batch["images"], tiny_batch["state"], mode="sample", n_samples=3)
    assert multi.shape == (3, tiny_batch["state"].shape[0], policy.horizon, policy.action_dim)
    assert torch.isfinite(multi).all()
    # 先验采样必须真的是随机的（否则多模态展示无意义）
    assert not torch.allclose(multi[0], multi[1])


def test_inference_rejects_action_chunk(policy, tiny_batch):
    """推理路径禁止访问后验：传入 action_chunk 必须报错，而不是被静默忽略。"""
    with pytest.raises(ValueError):
        policy(tiny_batch["images"], tiny_batch["state"], tiny_batch["action_chunk"], mode="infer")


def test_train_mode_requires_action_chunk(policy, tiny_batch):
    with pytest.raises(ValueError):
        policy(tiny_batch["images"], tiny_batch["state"], None, mode="train")


def test_logvar_is_clamped(data_cfg, model_cfg):
    """构造极端输入，log_var 必须被 clamp 到配置区间内。"""
    torch.manual_seed(0)
    net = CVAEPolicy(data_cfg, model_cfg)
    # 破坏输出层：把最后一层权重放大，迫使 log_var 打满上下界
    for name in ("prior", "posterior"):
        head = getattr(net, name).net[-1]
        with torch.no_grad():
            head.weight.mul_(50.0)
    images = torch.randn(3, net.vision.num_frames, 3, 64, 64) * 3
    state = torch.randn(3, net.state_dim) * 5
    chunk = torch.randn(3, net.horizon, net.action_dim) * 5
    out = net(images, state, chunk, mode="train")
    lo, hi = net.logvar_clamp
    assert out.log_var_prior.max().item() <= hi + 1e-6
    assert out.log_var_prior.min().item() >= lo - 1e-6
    assert out.log_var_post.max().item() <= hi + 1e-6
    assert out.log_var_post.min().item() >= lo - 1e-6


def test_state_dim_validation(policy, tiny_batch):
    with pytest.raises(ValueError):
        policy.encode(tiny_batch["images"], torch.zeros(2, 3))


def test_no_cvae_variant_is_deterministic(data_cfg, model_cfg):
    """A1 消融：latent 关闭时同一观测必须给出完全相同的动作块。"""
    cfg = {**model_cfg, "latent": {**model_cfg["latent"], "enabled": False}}
    torch.manual_seed(0)
    net = CVAEPolicy(data_cfg, cfg).eval()
    images = torch.randn(2, int(data_cfg["observation"]["num_frames"]), 3, 64, 64)
    state = torch.randn(2, net.state_dim)
    a = net.predict_chunk(images, state, mode="sample", n_samples=2)
    assert torch.allclose(a[0], a[1])
    assert net.posterior is None
