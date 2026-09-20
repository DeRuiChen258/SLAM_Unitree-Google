"""保存→加载一致性：权重、config hash、统计哈希一致；不一致时必须报错。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.cvae_policy import CVAEPolicy
from src.train.checkpoint import CheckpointError, load_checkpoint, load_for_inference, save_checkpoint
from src.utils.config import config_hash

STATS = Path("data/stats/normalization.json")

pytestmark = pytest.mark.skipif(not STATS.exists(), reason="需要先运行 scripts/11_build_dataset.py")


def _tiny_model(data_cfg, model_cfg):
    torch.manual_seed(0)
    return CVAEPolicy(data_cfg, model_cfg)


def test_save_load_roundtrip(tmp_path, data_cfg, model_cfg):
    hashes = {"data+model": config_hash(data_cfg, model_cfg)}
    model = _tiny_model(data_cfg, model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    meta = save_checkpoint(tmp_path / "last.pt", model, optimizer, step=12, epoch=3,
                           config_hashes=hashes, stats_path=STATS, seed=7,
                           metrics={"val/total": 0.5}, data_version="unit-test")
    assert meta["stats_sha256"]
    assert (tmp_path / "meta.json").exists()

    payload = load_checkpoint(tmp_path / "last.pt")
    assert payload["meta"]["step"] == 12
    assert payload["meta"]["seed"] == 7
    assert "optimizer_state" in payload

    fresh = _tiny_model(data_cfg, model_cfg)
    loaded_meta = load_for_inference(tmp_path / "last.pt", fresh, data_cfg=data_cfg,
                                     model_cfg=model_cfg, stats_path=STATS)
    assert loaded_meta["step"] == 12
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)


def test_config_hash_mismatch_is_rejected(tmp_path, data_cfg, model_cfg):
    model = _tiny_model(data_cfg, model_cfg)
    save_checkpoint(tmp_path / "best.pt", model, None, step=1, epoch=1,
                    config_hashes={"data+model": "deadbeefcafe"}, stats_path=STATS, seed=0)
    fresh = _tiny_model(data_cfg, model_cfg)
    with pytest.raises(CheckpointError):
        load_for_inference(tmp_path / "best.pt", fresh, data_cfg=data_cfg, model_cfg=model_cfg,
                           stats_path=STATS, strict=True)


def test_stats_hash_mismatch_is_rejected(tmp_path, data_cfg, model_cfg):
    model = _tiny_model(data_cfg, model_cfg)
    save_checkpoint(tmp_path / "best.pt", model, None, step=1, epoch=1,
                    config_hashes={"data+model": config_hash(data_cfg, model_cfg)},
                    stats_path=STATS, seed=0)
    fake_stats = tmp_path / "other_stats.json"
    fake_stats.write_text('{"state_blocks": {}, "action": {"mean": [0], "std": [1]}}', encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_for_inference(tmp_path / "best.pt", model, data_cfg=data_cfg, model_cfg=model_cfg,
                           stats_path=fake_stats, strict=True)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.pt")


def test_model_config_change_detected(tmp_path, data_cfg, model_cfg):
    """消融导致结构变化（latent 关闭）后，base checkpoint 必须拒绝加载。"""
    model = _tiny_model(data_cfg, model_cfg)
    save_checkpoint(tmp_path / "best.pt", model, None, step=1, epoch=1,
                    config_hashes={"data+model": config_hash(data_cfg, model_cfg)},
                    stats_path=STATS, seed=0)
    other = {**model_cfg, "latent": {**model_cfg["latent"], "enabled": False}}
    fresh = CVAEPolicy(data_cfg, other)
    with pytest.raises(CheckpointError):
        load_for_inference(tmp_path / "best.pt", fresh, data_cfg=data_cfg, model_cfg=other,
                           stats_path=STATS, strict=True)
