"""窗口切片正确性：起止边界、stride、跨 episode 不拼接、掩码语义。"""

from __future__ import annotations

import numpy as np

from src.datasets.window_builder import build_windows, obs_indices, windows_per_episode
from src.utils.chunking import build_chunk


def test_windows_start_after_history():
    """K=2 时首个窗口必须落在 t=1（不允许用补帧凑历史）。"""
    windows = build_windows(length=20, num_frames=2, frame_stride=1, horizon=8, stride=2, pad_mode="mask")
    assert windows[0] == 1
    assert all(w >= 1 for w in windows)


def test_obs_indices_respects_history():
    assert obs_indices(0, 2, 1, 10) is None          # 起点不足 → 丢弃，而不是补帧
    idx = obs_indices(3, 2, 1, 10)
    assert idx.tolist() == [2, 3]
    idx_strided = obs_indices(4, 2, 2, 10)
    assert idx_strided.tolist() == [2, 4]


def test_stride_is_honoured():
    windows = build_windows(length=40, num_frames=1, frame_stride=1, horizon=8, stride=4, pad_mode="crop")
    assert windows == list(range(0, 33, 4))


def test_pad_mode_crop_excludes_tail():
    windows = build_windows(length=12, num_frames=1, frame_stride=1, horizon=8, stride=1, pad_mode="crop")
    assert windows == [0, 1, 2, 3, 4]
    windows_mask = build_windows(length=12, num_frames=1, frame_stride=1, horizon=8, stride=1, pad_mode="mask")
    assert windows_mask[-1] == 11     # mask 模式保留尾部窗口


def test_no_cross_episode_concatenation():
    """窗口的每个历史帧下标都必须落在 [0, length-1] 内（禁止跨 episode 拼接）。"""
    length = 15
    for t in build_windows(length, 2, 1, 8, 2, "mask"):
        idx = obs_indices(t, 2, 1, length)
        assert idx is not None
        assert idx.min() >= 0 and idx.max() <= length - 1


def test_build_chunk_mask_marks_padding():
    actions = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    chunk, mask = build_chunk(actions, t=17, horizon=8, dim=3)
    assert chunk.shape == (8, 3)
    assert mask[:3].tolist() == [1.0, 1.0, 1.0]
    assert mask[3:].tolist() == [0.0] * 5
    # padding 步必须是 0 而不是上一帧的重复值（防止静默污染监督信号）
    assert np.allclose(chunk[3:], 0.0)


def test_windows_per_episode_matches_build_windows(data_cfg):
    length = 120
    assert windows_per_episode(length, data_cfg) == len(
        build_windows(length, int(data_cfg["observation"]["num_frames"]),
                      int(data_cfg["observation"]["frame_stride"]),
                      int(data_cfg["chunk"]["H"]), int(data_cfg["chunk"]["stride"]),
                      str(data_cfg["chunk"]["pad_mode"]))
    )
