"""pytest 公共配置：把工程根目录加入 sys.path，并集中放置跨测试的 fixture。

测试约束（提示词【五】5.11）：不依赖 GPU（除显式标记 cuda 的用例）、不依赖网络、不依赖真机。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def data_cfg():
    from src.utils.config import load_config

    return load_config("data")


@pytest.fixture(scope="session")
def model_cfg():
    from src.utils.config import load_config

    return load_config("model")


@pytest.fixture(scope="session")
def train_cfg():
    from src.utils.config import load_config

    return load_config("train")


@pytest.fixture(scope="session")
def infer_cfg():
    from src.utils.config import load_config

    return load_config("infer")


@pytest.fixture(scope="session")
def paths_cfg():
    from src.utils.config import load_paths

    return load_paths()


@pytest.fixture(scope="session")
def tiny_batch(data_cfg):
    """构造一个不依赖磁盘数据的小 batch（用于形状与损失测试）。"""
    import torch

    from src.utils.config import state_block_layout

    generator = torch.Generator().manual_seed(0)
    b, k = 4, int(data_cfg["observation"]["num_frames"])
    c = int(data_cfg["image"]["channels"])
    h, w = int(data_cfg["image"]["height"]), int(data_cfg["image"]["width"])
    horizon = int(data_cfg["chunk"]["H"])
    a_dim = int(data_cfg["action"]["dim"])
    state_dim = state_block_layout(data_cfg)[-1][2]
    mask = torch.ones(b, horizon)
    mask[:, -1] = 0.0  # 构造末尾 padding，验证掩码路径
    return {
        "images": torch.randn(b, k, c, h, w, generator=generator),
        "state": torch.randn(b, state_dim, generator=generator),
        "action_chunk": torch.randn(b, horizon, a_dim, generator=generator) * 0.1,
        "mask": mask,
        "meta": [{"episode_id": "unit", "t0": i, "timestamp": float(i), "source": "UNIT_TEST"} for i in range(b)],
    }
