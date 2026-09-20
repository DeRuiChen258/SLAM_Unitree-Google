#!/usr/bin/env python
"""scripts/11_build_dataset.py —— 预处理与窗口化：raw/mock → processed/splits/stats/manifest。

流程（每一步都可复现）：
    1. 枚举 episode（data/raw 优先，否则 data/mock）
    2. schema 校验（错误即中止）
    3. 固定 seed 划分 train/val/test
    4. 逐 episode 切窗 + 过滤（src/datasets/filters.py）
    5. 用 **train split** 统计归一化参数 → data/stats/normalization.json
    6. 固化归一化后写 processed/{train,val,test}.npz
    7. 写 splits/*.json、data/manifest.json

用法：
    python scripts/11_build_dataset.py                       # 默认配置
    python scripts/11_build_dataset.py --ablation no_slam_input
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.episode_reader import EpisodeReader  # noqa: E402
from src.datasets.filters import FilterStats, filter_windows  # noqa: E402
from src.datasets.schema import (  # noqa: E402
    raw_block_layout,
    state_column_selector,
    state_spec,
    validate_episode,
)
from src.datasets.transforms import compute_stats, norm_action, norm_state  # noqa: E402
from src.datasets.window_builder import build_windows, make_sample, obs_indices  # noqa: E402
from src.utils.config import config_hash, get, load_config, load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_json, ensure_dir, save_npy, sha256_file, write_jsonl  # noqa: E402
from src.utils.logging_utils import StageLogger, banner  # noqa: E402


def discover_episodes(paths: dict[str, Any]) -> tuple[list[Path], str]:
    """data/raw 优先（真实数据），为空则回退 MOCK；来源必须显式记录。"""
    raw = sorted(Path(paths["raw_dir"]).glob("*.npz"))
    if raw:
        return raw, "REAL"
    mock = sorted(Path(paths["mock_dir"]).glob("*.npz"))
    if mock:
        return mock, "MOCK"
    raise FileNotFoundError(
        f"未找到任何 episode：{paths['raw_dir']} 与 {paths['mock_dir']} 均为空。"
        "先运行 python scripts/10_gen_mock_data.py"
    )


def split_episodes(paths: list[Path], cfg: dict[str, Any]) -> dict[str, list[int]]:
    """按配置比例与 seed 划分 episode；同一 seed 必须得到同一划分（可重生成验证）。"""
    seed = int(get(cfg, "splits.seed", 0))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n = len(paths)
    n_train = int(round(n * float(get(cfg, "splits.train", 0.7))))
    n_val = int(round(n * float(get(cfg, "splits.val", 0.15))))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": sorted(order[:n_train].tolist()),
        "val": sorted(order[n_train : n_train + n_val].tolist()),
        "test": sorted(order[n_train + n_val :].tolist()),
    }


def episode_frames(path: Path) -> int:
    """只读元信息取帧数（mmap，不把图像载入内存）。"""
    with np.load(path, allow_pickle=False, mmap_mode="r") as handle:
        return int(handle["images"].shape[0])


def collect_split(episode_paths: list[Path], indices: list[int], cfg: dict[str, Any], selector: np.ndarray,
                  filter_stats: FilterStats, lazy: bool = False) -> dict[str, Any]:
    """把某个 split 的所有窗口读进内存（窗口数量已由过滤与切片决定）。"""
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ep_index: list[int] = []
    t0s: list[int] = []
    times: list[float] = []
    per_episode_counts: dict[str, int] = {}
    sub_index: dict[int, int] = {ep: i for i, ep in enumerate(indices)}

    num_frames = int(get(cfg, "observation.num_frames", 1))
    frame_stride = int(get(cfg, "observation.frame_stride", 1))
    horizon = int(get(cfg, "chunk.H", 8))
    stride = int(get(cfg, "chunk.stride", 1))
    pad_mode = str(get(cfg, "chunk.pad_mode", "mask"))

    for ep in indices:
        path = episode_paths[ep]
        with EpisodeReader(path, lazy=lazy, expected_dt=1.0 / float(get(cfg, "control.hz", 10.0))) as reader:
            episode = {
                "images": reader.images(),
                "state": reader.state_matrix(),
                "action": reader.actions(),
                "timestamp": reader.timestamps(),
                "slam_valid": reader.slam_poses()[1],
                "episode_id": reader.episode_id,
                "source": reader.source,
            }
            windows = build_windows(reader.length, num_frames, frame_stride, horizon, stride, pad_mode)
            obs_matrix = np.stack([obs_indices(t, num_frames, frame_stride, reader.length) for t in windows])
            kept = filter_windows(episode, windows, obs_matrix, cfg, filter_stats, selector)
            keep_map = {t: i for i, t in enumerate(windows)}
            for t in kept:
                sample = make_sample(episode, t, obs_matrix[keep_map[t]], cfg, selector)
                images.append(sample["images"].astype(np.uint8, copy=False))
                states.append(sample["state"].astype(np.float32, copy=False))
                actions.append(sample["action_chunk"])
                masks.append(sample["mask"])
                ep_index.append(sub_index[ep])
                t0s.append(t)
                times.append(sample["meta"]["timestamp"])
        per_episode_counts[str(path.name)] = len(kept)

    if not images:
        raise RuntimeError("该 split 没有任何合法窗口，请检查过滤阈值与数据长度")
    return {
        "images": np.stack(images, axis=0),
        "state": np.stack(states, axis=0),
        "action_chunk": np.stack(actions, axis=0),
        "mask": np.stack(masks, axis=0),
        "episode_index": np.asarray(ep_index, dtype=np.int32),
        "t0": np.asarray(t0s, dtype=np.int32),
        "timestamp": np.asarray(times, dtype=np.float64),
        "per_episode_counts": per_episode_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预处理与窗口化，生成可训练数据")
    parser.add_argument("--ablation", default=None, help="消融名（影响状态块选择与产物路径）")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-name", default=None, help="产物子目录名（默认 base 或消融名）")
    args = parser.parse_args(argv)

    paths = load_paths()
    cfg = load_config("data", ablation=args.ablation, overrides=args.override)
    run_name = args.run_name or (args.ablation or "base")
    log = StageLogger("build_dataset", paths["logs_dir"], stage="11")
    print(banner(f"S3 数据管线：run={run_name} ablation={args.ablation or 'base'}"))

    episodes, source = discover_episodes(paths)
    log.info(f"发现 {len(episodes)} 条 episode，来源={source}", episodes=[p.name for p in episodes[:5]])

    # --- schema 校验（错误直接中止）---
    reports = [validate_episode(p, cfg) for p in episodes]
    errors = [(r.path, r.errors) for r in reports if not r.ok]
    if errors:
        for path, errs in errors[:5]:
            log.error(f"schema 校验失败: {path}: {errs}")
        raise RuntimeError(f"{len(errors)} 条 episode 未通过 schema 校验，先修数据")
    warnings = sum(len(r.warnings) for r in reports)
    log.info(f"schema 校验通过（{len(reports)} 条，warning {warnings} 条）")

    # --- 划分 ---
    splits = split_episodes(episodes, cfg)
    log.info(f"划分：train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # --- 原始块布局 → 当前配置所需列 ---
    with np.load(episodes[0], allow_pickle=False) as handle:
        raw_layout = raw_block_layout(handle["state_block_names"], handle["state_block_dims"])
    selector = state_column_selector(raw_layout, cfg)
    target_dim = int(state_spec(cfg).dim)
    log.info(f"状态块选择：raw_dim={sum(raw_layout[b].stop - raw_layout[b].start for b in raw_layout)} "
             f"→ cfg_dim={target_dim}（slam_input={get(cfg, 'slam_input.enabled')}）")

    # --- 逐 split 切窗与过滤 ---
    filter_stats = FilterStats()
    collected = {
        split: collect_split(episodes, idx, cfg, selector, filter_stats)
        for split, idx in splits.items()
    }
    log.info(f"过滤统计：{filter_stats.to_dict()}")

    # --- 归一化统计（只用 train）---
    train = collected["train"]
    block_map = {name: slice(start, end) for name, start, end in state_spec(cfg).blocks}
    normalize_blocks = list(get(cfg, "state.normalize_blocks", []))
    stats = compute_stats(
        images=train["images"].reshape(-1, *train["images"].shape[2:]),
        state=train["state"],
        action=train["action_chunk"][train["mask"] > 0],
        blocks=block_map,
        normalize_blocks=normalize_blocks,
        clip_sigma=float(get(cfg, "stats.clip_sigma", 4.0)),
        min_std=float(get(cfg, "stats.min_std", 1e-6)),
    )
    stats["schema_version"] = str(get(cfg, "schema_version", "unknown"))
    stats["config_hash"] = config_hash(cfg)
    stats["state_blocks_order"] = [name for name, _, _ in state_spec(cfg).blocks]
    stats["data_version"] = f"{source.lower()}-v1-{len(episodes)}ep-seed{int(get(cfg, 'splits.seed', 0))}"
    stats_path = Path(paths["stats_dir"]) / "normalization.json"
    if run_name != "base":
        stats_path = ensure_dir(Path(paths["stats_dir"]) / "ablation") / f"{run_name}_normalization.json"
    atomic_write_json(stats_path, stats)
    log.info(f"归一化统计 → {stats_path}（仅用 train split 计算）")

    # --- 归一化 + 落盘 ---
    processed_dir = ensure_dir(Path(paths["processed_dir"]))
    splits_dir = ensure_dir(Path(paths["splits_dir"]))
    suffix = "" if run_name == "base" else f".{run_name}"
    file_hashes: dict[str, str] = {}
    split_records: dict[str, Any] = {}
    for split, data in collected.items():
        state_n = norm_state(data["state"], stats, block_map, normalize_blocks)
        action_n = norm_action(data["action_chunk"], stats)
        # 每字段一个 .npy：只有 .npy 支持真正的 mmap 懒加载（npz 每次访问成员都会整体解压）
        stem = processed_dir / f"{split}{suffix}"
        arrays = {
            "images": data["images"],
            "state": state_n.astype(np.float32),
            "action_chunk": action_n.astype(np.float32),
            "mask": data["mask"].astype(np.float32),
            "episode_index": data["episode_index"],
            "t0": data["t0"],
            "timestamp": data["timestamp"],
        }
        for field, array in arrays.items():
            field_path = save_npy(stem.with_name(f"{stem.name}.{field}.npy"), array)
            file_hashes[field_path.name] = sha256_file(field_path)
        episode_ids = [episodes[ep].stem for ep in splits[split]]
        atomic_write_json(stem.with_suffix(".meta.json"),
                          {"episode_ids": episode_ids, "split": split, "source": source,
                           "schema_version": str(get(cfg, "schema_version", "unknown"))})
        split_records[split] = {
            "processed_stem": str(stem),
            "windows": int(data["state"].shape[0]),
            "episode_ids": episode_ids,
            "per_episode_windows": data["per_episode_counts"],
            "state_dim": int(data["state"].shape[1]),
            "action_dim": int(data["action_chunk"].shape[-1]),
            "horizon": int(data["action_chunk"].shape[1]),
            "mask_ratio": float(data["mask"].mean()),
        }
        json_path = splits_dir / f"{split}{suffix}.json"
        atomic_write_json(
            json_path,
            {
                "split": split,
                "run_name": run_name,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": int(get(cfg, "splits.seed", 0)),
                "schema_version": str(get(cfg, "schema_version", "unknown")),
                "source": source,
                "num_episodes": len(episode_ids),
                "num_windows": split_records[split]["windows"],
                "episode_ids": episode_ids,
                "episode_indices": [int(i) for i in splits[split]],
                "config_hash": config_hash(cfg),
            },
        )
        log.info(f"{split}: {split_records[split]['windows']} 个窗口 → {stem.name}.*.npy")

    manifest = {
        "version": stats["data_version"],
        "source": source,
        "schema_version": str(get(cfg, "schema_version", "unknown")),
        "generated_command": "python scripts/11_build_dataset.py" + (f" --ablation {args.ablation}" if args.ablation else ""),
        "config_hash": config_hash(cfg),
        "slam_input_enabled": bool(get(cfg, "slam_input.enabled", True)),
        "num_episodes": len(episodes),
        "num_frames": int(sum(episode_frames(p) for p in episodes)),
        "fps": float(get(cfg, "control.hz", 10.0)),
        "state_dim": target_dim,
        "action_dim": int(get(cfg, "action.dim", 7)),
        "horizon": int(get(cfg, "chunk.H", 8)),
        "num_windows": {k: v["windows"] for k, v in split_records.items()},
        "filter_stats": filter_stats.to_dict(),
        "splits": split_records,
        "stats_path": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "file_hashes": file_hashes,
        "episode_files": [str(p) for p in episodes],
    }
    manifest_path = Path(paths["data_dir"]) / ("manifest.json" if run_name == "base" else f"manifest.{run_name}.json")
    atomic_write_json(manifest_path, manifest)
    write_jsonl(Path(paths["logs_dir"]) / "11_build_dataset.jsonl",
                [{"source": source, "episodes": len(episodes), **split_records[s]} for s in split_records])
    log.info(f"manifest → {manifest_path}")
    log.flush()
    print(json.dumps({"status": "OK", "source": source, "windows": {k: v["windows"] for k, v in split_records.items()},
                      "state_dim": target_dim, "stats": str(stats_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
