"""数据契约校验：合法数据通过；缺失字段/形状错误/NaN 必须被拒绝。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.schema import (
    SchemaError,
    observation_spec,
    raw_block_layout,
    state_column_selector,
    validate_episode,
    validate_sample,
)

MOCK = Path("data/mock/mock_0000.npz")


def _copy_episode(tmp_path: Path, **mutations) -> Path:
    with np.load(MOCK, allow_pickle=False) as src:
        arrays = {k: np.asarray(src[k]) for k in src.files}
    arrays.update(mutations)
    out = tmp_path / "episode.npz"
    np.savez_compressed(out, **arrays)
    return out


@pytest.mark.skipif(not MOCK.exists(), reason="需要先运行 scripts/10_gen_mock_data.py")
def test_valid_episode_passes(data_cfg):
    report = validate_episode(MOCK, data_cfg)
    assert report.ok, report.errors
    assert report.stats["state_dim"] == 29
    assert 0.0 <= report.stats["slam_valid_ratio"] <= 1.0


@pytest.mark.skipif(not MOCK.exists(), reason="需要先运行 scripts/10_gen_mock_data.py")
def test_missing_field_rejected(tmp_path, data_cfg):
    with np.load(MOCK, allow_pickle=False) as src:
        arrays = {k: np.asarray(src[k]) for k in src.files if k != "action"}
    path = tmp_path / "missing.npz"
    np.savez_compressed(path, **arrays)
    report = validate_episode(path, data_cfg)
    assert not report.ok
    assert any("action" in e for e in report.errors)


@pytest.mark.skipif(not MOCK.exists(), reason="需要先运行 scripts/10_gen_mock_data.py")
def test_wrong_shape_rejected(tmp_path, data_cfg):
    path = _copy_episode(tmp_path, images=np.zeros((5, 3, 32, 32), dtype=np.uint8))
    report = validate_episode(path, data_cfg)
    assert not report.ok
    assert any("images shape" in e for e in report.errors)


@pytest.mark.skipif(not MOCK.exists(), reason="需要先运行 scripts/10_gen_mock_data.py")
def test_nan_rejected(tmp_path, data_cfg):
    with np.load(MOCK, allow_pickle=False) as src:
        state = np.asarray(src["state"]).copy()
    state[3, 1] = np.nan
    report = validate_episode(_copy_episode(tmp_path, state=state), data_cfg)
    assert not report.ok
    assert any("NaN" in e for e in report.errors)


@pytest.mark.skipif(not MOCK.exists(), reason="需要先运行 scripts/10_gen_mock_data.py")
def test_non_monotonic_timestamp_rejected(tmp_path, data_cfg):
    with np.load(MOCK, allow_pickle=False) as src:
        ts = np.asarray(src["timestamp"]).copy()
    ts[10] = ts[9]
    report = validate_episode(_copy_episode(tmp_path, timestamp=ts), data_cfg)
    assert not report.ok
    assert any("单调" in e for e in report.errors)


def test_validate_sample_rejects_bad_shapes(data_cfg):
    obs = observation_spec(data_cfg)
    with pytest.raises(SchemaError):
        validate_sample(
            {
                "images": np.zeros((1, 3, 64, 64), dtype=np.float32),   # K 应为 2
                "state": np.zeros(29, dtype=np.float32),
                "action_chunk": np.zeros((8, 7), dtype=np.float32),
                "mask": np.ones(8, dtype=np.float32),
            },
            data_cfg,
        )
    assert obs.image_shape == (2, 3, 64, 64)


def test_mask_must_be_binary(data_cfg):
    with pytest.raises(SchemaError):
        validate_sample(
            {
                "images": np.zeros((2, 3, 64, 64), dtype=np.float32),
                "state": np.zeros(29, dtype=np.float32),
                "action_chunk": np.zeros((8, 7), dtype=np.float32),
                "mask": np.full(8, 0.5, dtype=np.float32),
            },
            data_cfg,
        )


def test_state_column_selector_drops_slam_blocks(data_cfg):
    """A4 消融：关掉 slam_input 后，状态列选择器必须精确去掉 slam_pose/slam_valid 两列组。"""
    layout = raw_block_layout(
        ["joint_pos", "joint_vel", "gripper", "ee_pose", "slam_pose", "slam_valid", "time_feat"],
        [7, 7, 1, 7, 4, 1, 2],
    )
    full = state_column_selector(layout, data_cfg)
    assert full.size == 29
    cfg_no_slam = dict(data_cfg)
    cfg_no_slam["slam_input"] = {**data_cfg["slam_input"], "enabled": False}
    reduced = state_column_selector(layout, cfg_no_slam)
    assert reduced.size == 24
    assert set(range(22, 27)).isdisjoint(set(reduced.tolist()))   # slam_pose(22:26)+slam_valid(26)
