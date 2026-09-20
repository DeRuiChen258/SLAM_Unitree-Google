#!/usr/bin/env python
"""scripts/92_build_viz_page.py —— 生成可直接在浏览器打开的本地可视化页面。

产物：outputs/figures/index.html
     = 内联结果看板（outputs/figures/results-dashboard.html 片段）
     + 全部静态图（outputs/figures/*.png、outputs/slam/*.png）的图集，
       每张图标注来源文件与生成脚本，便于逐一核对。

为什么单独做一个页面：内联看板适合在对话里快速看结论，
而"启动可视化"需要在浏览器里同时浏览曲线/热图/地图等原始大图——
两者数据同源（都由 scripts/91_export_viz_data.py 与各阶段脚本产出），不做任何手工填数。

用法：python scripts/92_build_viz_page.py [--port 8788]（--port 只用于打印访问地址）
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_text  # noqa: E402

# 图的来源说明（作者维护：每张图由哪个阶段命令产生），未登记的图会标注"未登记"
FIGURE_SOURCE = {
    "loss_curve.png": "bash scripts/20_train.sh（logs/train_metrics.jsonl）",
    "device_split.png": "bash scripts/20_train.sh（logs/20_device_usage.json）",
    "12_data_samples.png": "python scripts/12_data_check.py（抽样核查数据是否像样）",
    "30_action_compare_base.png": "bash scripts/30_infer_offline.sh（GT vs 无集成 vs 时间集成）",
    "30_chunk_weights_base.png": "bash scripts/30_infer_offline.sh（每次预测在每步的融合权重）",
    "30_per_step_error_base.png": "bash scripts/30_infer_offline.sh（误差随 chunk 内步数增长）",
    "cartographer_map_with_trajectory.png": "bash scripts/42_slam_replay_eval.sh（真实 Cartographer 地图 + 轨迹）",
    "trajectory_overlay.png": "bash scripts/42_slam_replay_eval.sh（刚体对齐后的轨迹对比）",
    "ate_over_time.png": "bash scripts/42_slam_replay_eval.sh（误差随回放时间）",
}


def figure_cards(figures: list[tuple[Path, str]]) -> str:
    cards: list[str] = []
    for path, rel in figures:
        src = FIGURE_SOURCE.get(path.name, "未登记（见 logs/ 对应阶段日志）")
        cards.append(
            '<figure class="fig">'
            f'<a href="{html.escape(rel)}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(path.stem)}"></a>'
            f'<figcaption><b>{html.escape(path.name)}</b><span>{html.escape(src)}</span></figcaption>'
            "</figure>"
        )
    return "\n".join(cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成本地可视化页面")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)

    paths = load_paths()
    fig_dir = Path(paths["figures_dir"])
    slam_dir = Path(paths["slam_out_dir"])
    fragment_path = fig_dir / "results-dashboard.html"
    if not fragment_path.exists():
        print(f"BLOCKED: 缺少看板片段 {fragment_path}（先运行 scripts/91_export_viz_data.py 并生成看板）")
        return 3
    fragment = fragment_path.read_text(encoding="utf-8")

    # 图集：目录内的 PNG（含 slam 子目录里的图，用相对路径引用以适配 http.server 根目录）
    figures: list[tuple[Path, str]] = []
    for png in sorted(fig_dir.glob("*.png")):
        if png.name.startswith(("results-", "index")):
            continue
        figures.append((png, png.name))
    for png in sorted(slam_dir.glob("*.png")):
        rel = f"../slam/{png.name}"
        figures.append((png, rel))

    page = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CVAE × Cartographer SLAM · 可视化</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0b0d12;color:#e8eaf0;font-family:ui-sans-serif,system-ui,"Segoe UI","Noto Sans CJK SC",sans-serif}}
header{{padding:18px 22px 6px}}
h1{{margin:0 0 6px;font-size:18px}}
.meta{{color:#98a0b3;font-size:12px;line-height:1.7}}
.meta code{{color:#c9d4e6;background:#161a22;border:1px solid #232936;border-radius:5px;padding:1px 5px}}
section{{padding:10px 22px 26px}}
h2{{font-size:14px;margin:18px 0 10px;color:#c9d4e6;font-weight:600}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.fig{{margin:0;background:#12151b;border:1px solid #232936;border-radius:10px;overflow:hidden}}
.fig img{{display:block;width:100%;height:auto;background:#0f1115}}
.fig figcaption{{padding:8px 10px;font-size:11.5px;display:flex;flex-direction:column;gap:3px}}
.fig figcaption span{{color:#98a0b3}}
a{{color:#4ea1ff}}
</style>
</head>
<body>
<header>
  <h1>CVAE 策略 × Google Cartographer SLAM · 可视化总览</h1>
  <div class="meta">
    路径：全部相对项目根（<code>configs/paths.yaml</code>，其中 <code>project_root</code> 由代码位置自动推导）；
    外部环境根用 <code>CVSLAM_ENV_ROOT</code> 覆盖<br>
    数据来源：<b>MOCK</b>（合成世界 / 合成观测 / 合成激光）· 无真机 · 动作下发默认关闭 ·
    图与看板均由脚本从真实产物生成（<code>logs/</code>、<code>outputs/</code>），无手工填数
  </div>
</header>
<section>
  {fragment}
  <h2>静态图集（{len(figures)} 张，点击可看原图）</h2>
  <div class="gallery">
  {figure_cards(figures)}
  </div>
</section>
</body>
</html>
"""
    out = fig_dir / "index.html"
    atomic_write_text(out, page)
    # 注意：页面用相对路径引用 ../slam/*.png，因此 HTTP 服务必须挂在 outputs/ 这一级，
    # 挂在 figures/ 会让 ../slam 触发目录穿越保护（404，本工程实测踩过）。
    out_dir = Path(paths["outputs_dir"])
    print(f"OK 可视化页面 → {out}")
    print(f"   图集 {len(figures)} 张；启动服务：python3 -m http.server {args.port} "
          f"--bind 127.0.0.1 --directory {out_dir}")
    print(f"   浏览器地址：http://127.0.0.1:{args.port}/figures/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
