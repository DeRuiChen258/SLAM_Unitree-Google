"""checkpoint 保存与加载：权重 / 优化器 / 进度 / config hash / 数据版本 / 统计哈希 / git commit。

`load_for_inference` 严格校验 config hash 与归一化统计哈希，不一致直接报错
（禁止“强行加载”导致推理侧用错归一化统计——那会让指标虚高且难以察觉）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from ..datasets.transforms import stats_fingerprint
from ..utils.config import config_hash, project_root
from ..utils.io_utils import atomic_write_json, ensure_dir, sha256_file


class CheckpointError(RuntimeError):
    """checkpoint 与当前配置/统计不匹配。"""


def git_commit(path: Path | None = None) -> str:
    """当前 git commit（不可用时返回 'unknown'，不抛异常）。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(path or project_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, *, step: int, epoch: int,
                    config_hashes: Mapping[str, str], stats_path: str | Path, seed: int,
                    metrics: Mapping[str, Any] | None = None, data_version: str | None = None,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """保存权重与同目录 meta.json（权重只允许存在一份，禁止复制到多处）。"""
    p = Path(path)
    ensure_dir(p.parent)
    meta = {
        "step": int(step),
        "epoch": int(epoch),
        "config_hashes": dict(config_hashes),
        "seed": int(seed),
        "stats_path": str(stats_path),
        "stats_sha256": stats_fingerprint(stats_path),
        "data_version": data_version or "unknown",
        "git_commit": git_commit(),
        "metrics": dict(metrics or {}),
    }
    payload: dict[str, Any] = {"model_state": model.state_dict(), "meta": meta}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload.update(dict(extra))
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)  # 原子替换：中断不会留下半个权重
    atomic_write_json(p.parent / "meta.json",
                      {**meta, "weights": p.name, "weights_sha256": sha256_file(p)})
    return meta


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"checkpoint 不存在: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    if "model_state" not in payload or "meta" not in payload:
        raise CheckpointError(f"checkpoint 结构异常（缺少 model_state/meta）: {p}")
    return payload


def load_for_inference(path: str | Path, model, *, data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any],
                       stats_path: str | Path, strict: bool = True) -> dict[str, Any]:
    """加载权重并强校验 config hash / 统计哈希；不一致即抛 CheckpointError。"""
    payload = load_checkpoint(path)
    meta = payload["meta"]
    expected = config_hash(data_cfg, model_cfg)
    recorded = meta.get("config_hashes", {}).get("data+model")
    if strict and recorded and recorded != expected:
        raise CheckpointError(
            f"config hash 不一致：checkpoint={recorded} 当前={expected}。"
            "推理必须使用与训练完全相同的 data/model 配置（含消融开关）。"
        )
    current_stats = stats_fingerprint(stats_path)
    if strict and meta.get("stats_sha256") and meta["stats_sha256"] != current_stats:
        raise CheckpointError(
            f"归一化统计哈希不一致：checkpoint={str(meta['stats_sha256'])[:12]} 当前={current_stats[:12]}。"
            "禁止用未对齐的统计做推理。"
        )
    model.load_state_dict(payload["model_state"])
    return meta
