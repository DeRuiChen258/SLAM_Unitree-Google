"""动作块构造与重预测调度：H/n_exec 语义、尾部裁剪、与 window_builder 语义一致。"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.window_builder import build_chunk as wb_build_chunk
from src.models.action_chunker import ActionChunker, ActionSpec, ChunkScheduler, build_chunk, split_chunk_to_steps


def test_action_spec_from_config(data_cfg):
    spec = ActionSpec.from_config(data_cfg)
    assert spec.dim == 7
    assert spec.layout[0] == "dx" and spec.layout[-1] == "gripper"
    assert spec.is_delta is True
    assert spec.units[:3] == ("m", "m", "m")
    assert spec.units[3] == "rad"


def test_scheduler_replan_cadence():
    scheduler = ChunkScheduler(horizon=8, n_exec=4)
    assert [scheduler.should_replan(t) for t in range(8)] == [True, False, False, False, True, False, False, False]
    assert scheduler.effective_horizon() == 4


def test_scheduler_rejects_invalid_n_exec():
    with pytest.raises(ValueError):
        ChunkScheduler(horizon=4, n_exec=5)
    with pytest.raises(ValueError):
        ChunkScheduler(horizon=4, n_exec=0)


def test_execute_and_tail():
    scheduler = ChunkScheduler(horizon=8, n_exec=3)
    chunk = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    steps = scheduler.execute(chunk)
    assert steps.shape == (3, 2)
    assert np.allclose(steps, chunk[:3])
    assert scheduler.tail(chunk).shape == (0, 2)          # drop_remainder=True 时显式丢弃尾部
    keep = ChunkScheduler(horizon=8, n_exec=3, drop_remainder=False)
    assert keep.tail(chunk).shape == (5, 2)


def test_execute_respects_episode_end():
    scheduler = ChunkScheduler(horizon=8, n_exec=4)
    chunk = np.zeros((8, 7), dtype=np.float32)
    assert scheduler.execute(chunk, available_steps=2).shape == (2, 7)
    assert scheduler.execute(chunk, available_steps=0).shape == (0, 7)


def test_split_chunk_to_steps():
    chunk = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    steps = split_chunk_to_steps(chunk)
    assert len(steps) == 4
    assert steps[2].tolist() == [6.0, 7.0, 8.0]


def test_build_chunk_matches_window_builder():
    """切片语义唯一实现：两处入口必须给出完全一致的 chunk 与 mask。"""
    actions = np.random.default_rng(0).normal(size=(30, 7)).astype(np.float32)
    for t in (0, 5, 23, 29):
        a1, m1 = build_chunk(actions, t, 8, 7)
        a2, m2 = wb_build_chunk(actions, t, 8, 7)
        assert np.array_equal(a1, a2) and np.array_equal(m1, m2)


def test_action_chunker_facade(data_cfg, infer_cfg):
    chunker = ActionChunker.from_config(data_cfg, infer_cfg)
    assert chunker.horizon == int(data_cfg["chunk"]["H"])
    assert chunker.n_exec == int(infer_cfg["chunk_exec"]["n_exec"])
    steps = chunker.steps(np.ones((chunker.horizon, 7), dtype=np.float32))
    assert len(steps) == chunker.n_exec
