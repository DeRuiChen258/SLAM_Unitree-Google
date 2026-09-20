#!/usr/bin/env python
"""scripts/43_gen_cartographer_config.py —— 生成自包含的 Cartographer Lua 配置。

背景（本机实测，证据在 logs/40_slam_bringup.log）：
    RoboStack `ros-jazzy-cartographer-ros 2.0.9003` 的 Lua `include` 解析有缺陷，
    连官方自带示例配置都会抛 `basic_filebuf::underflow ... Is a directory`。
    因此把官方默认值**内联**，再叠加本实验的 override，生成一个不依赖 include 的配置文件。

做法（单一事实源）：
    官方默认（只读）           configs/cartographer/g1_2d.overrides.lua（人工编辑的唯一覆盖层）
            └──────────────┬──────────────┘
                  生成的 g1_2d.lua（自包含，禁止手改）

产物：configs/cartographer/g1_2d.lua（含来源路径 + 各文件 sha256 + 生成命令）
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_text  # noqa: E402

# 内联顺序即依赖顺序：先默认值，最后是 override 与 return options
CORE_FILES = [
    "pose_graph.lua",
    "trajectory_builder_2d.lua",
    "trajectory_builder_3d.lua",
    "trajectory_builder.lua",
    "map_builder.lua",
]


def strip_includes(text: str) -> str:
    """删掉 include 行（本机 include 解析不可用），其余原样保留。"""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("include "))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    paths = load_paths()
    carto_share = Path(paths["cartographer_prefix"]) / "share" / "cartographer" / "configuration_files"
    if not carto_share.exists():
        print(f"BLOCKED: 未找到官方默认配置目录 {carto_share}（先 conda env create -f environment_cartographer.yaml）")
        return 3

    out_path = Path(paths["configs_dir"]) / "cartographer" / "g1_2d.lua"
    overrides_path = Path(paths["configs_dir"]) / "cartographer" / "g1_2d.overrides.lua"
    overrides = overrides_path.read_text(encoding="utf-8")

    header = [
        "-- ============================================================================",
        "-- configs/cartographer/g1_2d.lua",
        "-- 【自动生成，禁止手改】编辑 configs/cartographer/g1_2d.overrides.lua 后重新运行：",
        "--     python scripts/43_gen_cartographer_config.py",
        "--",
        "-- 为什么自包含：本机 cartographer_ros 2.0.9003 的 Lua include 解析有缺陷",
        "-- （官方 backpack_2d.lua 也会失败：basic_filebuf::underflow, Is a directory），",
        "-- 因此把官方默认值内联，仅覆盖本实验需要的参数。",
        "--",
        "-- 内联来源（官方 Cartographer 默认配置，只读）：",
    ]
    body: list[str] = []
    for name in CORE_FILES:
        src = carto_share / name
        if not src.exists():
            print(f"BLOCKED: 缺少官方默认配置 {src}")
            return 3
        text = src.read_text(encoding="utf-8")
        # 注释里不写本机绝对路径：记录「环境内相对位置 + 内容哈希」同样可追溯，且不泄露个人路径
        try:
            shown = src.relative_to(Path(paths["cartographer_prefix"]))
        except ValueError:
            shown = Path(src.name)
        header.append(f"--     {shown}  (sha256:{sha256_text(text)})")
        body.append(f"-- >>>>>>>>>> BEGIN {name} （官方原文，已移除 include 行） >>>>>>>>>>")
        body.append(strip_includes(text))
        body.append(f"-- <<<<<<<<<< END {name} <<<<<<<<<<\n")

    content = "\n".join(header) + "\n" + "\n".join(body) + "\n" + overrides
    atomic_write_text(out_path, content)
    print(f"OK 生成 {out_path}（{len(content.splitlines())} 行）")
    print(f"   覆盖层: {overrides_path}  sha256:{sha256_text(overrides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
