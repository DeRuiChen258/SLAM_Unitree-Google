"""损失正确性：先验=后验时 KL≈0、完美预测重建≈0、β 缩放生效、掩码排除 padding。"""

from __future__ import annotations

import pytest
import torch

from src.models.layers import kl_diag_gaussian
from src.models.losses import masked_recon_loss, smoothness_loss


def test_kl_zero_when_identical():
    mu = torch.randn(5, 4)
    log_var = torch.randn(5, 4) * 0.5
    kl = kl_diag_gaussian(mu, log_var, mu, log_var)
    assert torch.allclose(kl, torch.zeros(5), atol=1e-6)


def test_kl_positive_when_shifted():
    mu_q = torch.zeros(3, 4)
    mu_p = torch.ones(3, 4)
    log_var = torch.zeros(3, 4)
    kl = kl_diag_gaussian(mu_q, log_var, mu_p, log_var)
    # 每维贡献 0.5*Δμ² = 0.5，4 维共 2.0
    assert torch.allclose(kl, torch.full((3,), 2.0), atol=1e-6)


def test_free_bits_floor():
    mu_q = torch.zeros(1, 4)
    mu_p = torch.full((1, 4), 1e-3)     # 极小差异 → 每维 KL 远小于 free_bits
    log_var = torch.zeros(1, 4)
    kl_free = kl_diag_gaussian(mu_q, log_var, mu_p, log_var, free_bits=0.5)
    assert kl_free.item() == pytest.approx(4 * 0.5, abs=1e-6)


def test_recon_zero_for_perfect_prediction():
    target = torch.randn(4, 8, 7)
    mask = torch.ones(4, 8)
    loss, per_step = masked_recon_loss(target.clone(), target, mask, "huber")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(per_step, torch.zeros(8), atol=1e-6)


def test_mask_excludes_padding():
    target = torch.ones(2, 4, 3)
    pred = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    loss, per_step = masked_recon_loss(pred, target, mask, "l1")
    # 只有 3 个有效步，每步误差 1.0 → 总损失 1.0，且后两步的“逐步损失”为 0（无有效样本）
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
    assert per_step[0].item() == pytest.approx(1.0, abs=1e-6)
    assert per_step[-1].item() == pytest.approx(0.0, abs=1e-6)


def test_smoothness_zero_for_constant_chunk():
    chunk = torch.ones(2, 6, 7) * 0.3
    mask = torch.ones(2, 6)
    assert smoothness_loss(chunk, mask, order=1).item() == pytest.approx(0.0, abs=1e-6)
    assert smoothness_loss(chunk, mask, order=2).item() == pytest.approx(0.0, abs=1e-6)


def test_smoothness_penalises_jitter():
    flat = torch.zeros(1, 6, 2)
    jittery = flat.clone()
    jittery[0, ::2] = 1.0
    mask = torch.ones(1, 6)
    assert smoothness_loss(jittery, mask, order=1) > smoothness_loss(flat, mask, order=1)


def test_beta_scales_kl(data_cfg, model_cfg, tiny_batch):
    from src.models.cvae_policy import CVAEPolicy

    torch.manual_seed(0)
    policy = CVAEPolicy(data_cfg, model_cfg)
    # 初始时先验/后验输出层都被零初始化（KL 恒为 0，这是有意的稳定化设计），
    # 因此需要先把后验输出层扰动成非零，才能验证 beta 的缩放关系。
    with torch.no_grad():
        policy.posterior.net[-1].weight.normal_(0.0, 0.5)
    weights = {"kl_enabled": True, "smooth_weight": 0.0, "smooth_enabled": False, "recon_weight": 1.0}
    # 注意：两次独立前向会各自重参数化采样，不能跨前向比较 total（差异会被采样噪声掩盖）。
    # 正确做法是在同一次前向里核对装配恒等式：total = recon + beta*kl + reg。
    beta = 0.5
    out = policy.compute_loss(tiny_batch, beta=beta, weights=weights)
    assert torch.allclose(out["total"], out["recon"] + beta * out["kl"] + out["reg"], atol=1e-6)
    assert out["kl"].item() > 0.0


def test_kl_is_zero_at_initialisation(data_cfg, model_cfg, tiny_batch):
    """零初始化先验/后验输出层 → 初始 KL 必须为 0（防止初始 KL 压制重建项）。"""
    from src.models.cvae_policy import CVAEPolicy

    torch.manual_seed(0)
    policy = CVAEPolicy(data_cfg, model_cfg)
    out = policy.compute_loss(tiny_batch, beta=1.0, weights={"kl_enabled": True, "smooth_enabled": False})
    assert out["kl"].item() == pytest.approx(0.0, abs=1e-9)


def test_kl_can_be_switched_off(data_cfg, model_cfg, tiny_batch):
    from src.models.cvae_policy import CVAEPolicy

    torch.manual_seed(0)
    policy = CVAEPolicy(data_cfg, model_cfg)
    out = policy.compute_loss(tiny_batch, beta=1.0, weights={"kl_enabled": False, "smooth_enabled": False})
    assert out["kl"].item() == 0.0
    assert out["total"].item() == pytest.approx(out["recon"].item(), rel=1e-6)
