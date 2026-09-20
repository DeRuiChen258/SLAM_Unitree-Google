"""时间集成正确性：权重归一化、冷启动退化、过期预测被丢弃、不补零。"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.temporal_ensemble import TemporalEnsembler


def test_cold_start_returns_single_prediction():
    ens = TemporalEnsembler(mode="exponential", decay=0.5)
    chunk = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    ens.update(chunk, t0=0, n_exec=4)
    for t in range(8):
        assert np.allclose(ens.action_at(t), chunk[t])   # 只有 1 条预测时退化为原始值


def test_uniform_fusion_equals_mean():
    ens = TemporalEnsembler(mode="uniform")
    c1 = np.zeros((8, 2), dtype=np.float32)
    c2 = np.full((8, 2), 1.0, dtype=np.float32)
    ens.update(c1, t0=0, n_exec=8)
    ens.update(c2, t0=2, n_exec=8)
    # t=2 同时被两条覆盖（权重 1:1）→ 均值 0.5
    assert np.allclose(ens.action_at(2), np.full(2, 0.5), atol=1e-6)
    # t=1 只被第一条覆盖 → 0
    assert np.allclose(ens.action_at(1), np.zeros(2), atol=1e-6)


def test_exponential_decay_weights_newer_more():
    ens = TemporalEnsembler(mode="exponential", decay=0.5, min_weight=0.0)
    old = np.zeros((8, 2), dtype=np.float32)
    new = np.ones((8, 2), dtype=np.float32)
    ens.update(old, t0=0, n_exec=8)
    ens.update(new, t0=4, n_exec=8)
    # t=4：old 的 age=4 → w=0.0625，new 的 age=0 → w=1；结果应远大于 0.5
    value = ens.action_at(4)
    assert 0.85 < value[0] < 1.0
    expected = (0.5**4 * 0.0 + 1.0 * 1.0) / (0.5**4 + 1.0)
    assert value[0] == pytest.approx(expected, abs=1e-6)


def test_expired_predictions_dropped():
    ens = TemporalEnsembler(mode="uniform")
    ens.update(np.ones((8, 2), dtype=np.float32), t0=0, n_exec=8)
    ens.update(np.zeros((8, 2), dtype=np.float32), t0=100, n_exec=8)   # 会把第一条挤掉
    assert len(ens) == 1
    assert ens.stats()["n_expired"] == 1
    assert ens.action_at(0) is None              # 已执行的过去步不再有贡献者
    assert np.allclose(ens.action_at(100), np.zeros(2))   # 新 chunk 的值


def test_no_zero_fill_when_no_source():
    ens = TemporalEnsembler(mode="exponential", decay=0.5)
    assert ens.action_at(0) is None              # 没有任何预测时必须返回 None，而不是 0 动作
    assert ens.stats()["n_no_source"] == 1


def test_mask_removes_invalid_steps():
    ens = TemporalEnsembler(mode="uniform")
    chunk = np.ones((4, 2), dtype=np.float32)
    ens.update(chunk, t0=0, n_exec=4, mask=np.array([1, 1, 0, 0], dtype=np.float32))
    assert ens.action_at(2) is None
    assert np.allclose(ens.action_at(1), np.ones(2))


def test_min_weight_filters_old_predictions():
    ens = TemporalEnsembler(mode="exponential", decay=0.1, min_weight=0.05)
    ens.update(np.zeros((16, 2), dtype=np.float32), t0=0, n_exec=16)
    ens.update(np.ones((16, 2), dtype=np.float32), t0=10, n_exec=16)
    # t=10 处 old 的权重 0.1^10 = 1e-10 < min_weight → 被裁剪，只剩新预测
    assert np.allclose(ens.action_at(10), np.ones(2))
    assert len(ens.weights_at(10)) == 1


def test_scalar_mask_is_rejected():
    ens = TemporalEnsembler(mode="uniform")
    with pytest.raises(ValueError):
        ens.update(np.ones((4, 2), dtype=np.float32), t0=0, n_exec=4, mask=np.float32(1.0))


def test_reset_clears_state():
    ens = TemporalEnsembler(mode="uniform")
    ens.update(np.ones((4, 2), dtype=np.float32), t0=0, n_exec=4)
    ens.reset()
    assert len(ens) == 0 and ens.stats()["n_updates"] == 0


def test_weight_matrix_shape():
    ens = TemporalEnsembler(mode="exponential", decay=0.5, window=4)
    for i in range(3):
        ens.update(np.full((4, 2), float(i), dtype=np.float32), t0=i, n_exec=4)
    mat = ens.weight_matrix(0, 6)
    assert mat.shape[0] == 6 and mat.shape[1] >= 1
