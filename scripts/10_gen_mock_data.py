#!/usr/bin/env python
"""scripts/10_gen_mock_data.py —— 生成合成演示数据（MOCK）与激光扫描流。

产物：
    data/mock/mock_XXXX.npz       每条 episode 的观测/状态/动作/位姿
    data/slam/scans.npz           与真值位姿同步的激光扫描（供真正的 Cartographer 回放）
    data/slam/world.json          2D 世界几何（可视化与复现）
    data/mock_manifest.json       生成参数、统计口径、逐 episode 哈希

用法：python scripts/10_gen_mock_data.py [--episodes N] [--length L] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.mock_generator import generate_episodes  # noqa: E402
from src.utils.config import config_hash, get, load_config, load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_json, sha256_file  # noqa: E402
from src.utils.logging_utils import StageLogger, banner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 MOCK 数据")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    paths = load_paths()
    overrides = []
    if args.episodes:
        overrides.append(f"mock.num_episodes={args.episodes}")
    if args.length:
        overrides.append(f"mock.episode_len={args.length}")
    if args.seed is not None:
        overrides.append(f"mock.seed={args.seed}")
    cfg = load_config("data", overrides=overrides)
    log = StageLogger("gen_mock_data", paths["logs_dir"], stage="10")
    print(banner("S2 生成 MOCK 数据（含激光扫描流）"))

    manifest = generate_episodes(cfg)
    manifest.update(
        {
            "generated_at_unix": __import__("time").time(),
            "config_hash": config_hash(cfg),
            "episodes_per_sec": None,
            "data_version": f"mock-v1-{manifest['num_episodes']}x{manifest['episode_len']}-seed{manifest['seed']}",
            "files": {
                "episodes": [ep["path"] for ep in manifest["episodes"]],
                "scans": manifest["scan_file"],
            },
            "hashes": {
                "scans_sha256": sha256_file(manifest["scan_file"]),
            },
        }
    )
    out = Path(paths["data_dir"]) / "mock_manifest.json"
    atomic_write_json(out, manifest)
    log.info(
        f"MOCK 数据生成完成：{manifest['num_episodes']} episodes × {manifest['episode_len']} 帧，"
        f"多模态 episode {manifest['multimodal_episodes']} 条",
        manifest=str(out), scan_file=manifest["scan_file"],
    )
    log.flush()
    print(f"manifest → {out}")
    print(f"source=MOCK（所有下游结论必须标注 MOCK，禁止当作真实数据结论）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
