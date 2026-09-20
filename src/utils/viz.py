"""可视化：loss/beta 曲线、动作对比、chunk 权重热图、时间集成轨迹、SLAM 地图与轨迹、对齐分布。

约束（提示词【五】5.9）：
    * 每张图必须带轴标签 + 单位 + 图例 + 数据来源（config hash / 日志文件名 / source 标记）；
    * 图必须可由 logs/ 与 outputs/ 中的真实产物重建，禁止手工补图；
    * MOCK 数据的图必须显式标注 MOCK。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io_utils import ensure_dir

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def setup_matplotlib():
    """统一 matplotlib 后端与中文字体；所有绘图入口都必须先调用它。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 中文字体：本机存在 Noto Sans CJK；缺失时退回默认字体并只影响标注文字，不影响数据。
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "AR PL UMing CN"):
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def _mpl():
    return setup_matplotlib()


def _footer(fig, source: str, extra: str = "") -> None:
    """统一在图底部标注数据来源（便于报告溯源）。"""
    text = f"source: {source}"
    if extra:
        text += f" | {extra}"
    fig.text(0.01, 0.005, text, fontsize=7, color="#555555")


def plot_loss_curves(metrics: Sequence[Mapping[str, Any]], out_path: str | Path,
                     title: str = "训练曲线", source: str = "", smooth: int = 9) -> Path:
    """total / recon / kl / beta 四联图（数据来自 logs/*_metrics.jsonl）。"""
    plt = _mpl()
    steps = np.asarray([m.get("step", i) for i, m in enumerate(metrics)], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    series = {
        "loss/total": ("total loss", axes[0, 0]),
        "loss/recon": ("recon loss (huber)", axes[0, 1]),
        "loss/kl": ("KL (nats)", axes[1, 0]),
        "beta": ("beta (KL weight)", axes[1, 1]),
    }
    for key, (label, ax) in series.items():
        values = np.asarray([float(m.get(key, np.nan)) for m in metrics], dtype=np.float64)
        ax.plot(steps, values, alpha=0.35, color=_COLORS[0], label="raw")
        if smooth > 1 and values.size > smooth:
            kernel = np.ones(smooth) / smooth
            ax.plot(steps[smooth - 1 :], np.convolve(values, kernel, mode="valid"),
                    color=_COLORS[1], label=f"moving avg ({smooth})")
        ax.set_xlabel("training step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_action_comparison(gt: np.ndarray, predictions: Mapping[str, np.ndarray], out_path: str | Path,
                           title: str = "动作对比", source: str = "", channels: Sequence[int] | None = None,
                           channel_names: Sequence[str] | None = None) -> Path:
    """GT vs 多路预测的逐步动作曲线（用于判断抖动与滞后）。"""
    plt = _mpl()
    names = channel_names or ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
    chans = list(channels or range(min(4, gt.shape[1])))
    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 2.1 * len(chans)), sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(gt.shape[0])
    for ax, ch in zip(axes, chans):
        ax.plot(t, gt[:, ch], color="black", lw=2.0, label="ground truth")
        for i, (label, values) in enumerate(predictions.items()):
            vals = np.asarray(values)
            if vals.ndim == 2 and vals.shape[1] > ch:
                ax.plot(t[: vals.shape[0]], vals[: gt.shape[0], ch], color=_COLORS[i % len(_COLORS)],
                        lw=1.2, alpha=0.9, label=label)
        ax.set_ylabel(f"{names[ch] if ch < len(names) else f'a{ch}'} (m or rad per step)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("execution step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_chunk_weight_heatmap(matrix: np.ndarray, out_path: str | Path, title: str = "时间集成权重热图",
                              source: str = "") -> Path:
    """[n_steps, n_predictions] 权重热图：直观看到"每个时刻由哪几次预测融合"。"""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("execution step t")
    ax.set_ylabel("prediction index (older → newer, left → right)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("normalized weight")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_trajectories(trajectories: Mapping[str, np.ndarray], out_path: str | Path,
                      title: str = "轨迹对比", source: str = "", reference: str | None = None) -> Path:
    """多路 2D 轨迹叠加（GT / Cartographer / 里程计降级…）。"""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for i, (label, traj) in enumerate(trajectories.items()):
        arr = np.asarray(traj)
        style = "--" if reference and label == reference else "-"
        ax.plot(arr[:, 0], arr[:, 1], style, color="black" if reference and label == reference else _COLORS[i % len(_COLORS)],
                lw=2.0 if reference and label == reference else 1.4, label=label)
        ax.scatter(arr[0, 0], arr[0, 1], marker="o", s=30, color=_COLORS[i % len(_COLORS)])
        ax.scatter(arr[-1, 0], arr[-1, 1], marker="X", s=40, color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_slam_map(occupancy: np.ndarray, origin: tuple[float, float], resolution: float,
                  trajectories: Mapping[str, np.ndarray], out_path: str | Path,
                  title: str = "Cartographer 建图与轨迹", source: str = "") -> Path:
    """占用栅格地图 + 轨迹叠加（occupancy: [H,W]，0=未知 0.5=空闲 1=占用）。"""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 8))
    height, width = occupancy.shape
    extent = [
        origin[0],
        origin[0] + width * resolution,
        origin[1],
        origin[1] + height * resolution,
    ]
    ax.imshow(occupancy, cmap="gray_r", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    for i, (label, traj) in enumerate(trajectories.items()):
        arr = np.asarray(traj)
        ax.plot(arr[:, 0], arr[:, 1], color=_COLORS[i % len(_COLORS)], lw=1.6, label=label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_alignment(offsets_s: Sequence[float], out_path: str | Path, tolerance: float,
                   title: str = "位姿流与策略时钟对齐偏移", source: str = "") -> Path:
    """对齐偏移直方图 + 容差线（用于判断时间同步是否可接受）。"""
    plt = _mpl()
    arr = np.asarray(offsets_s, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(arr, bins=40, color=_COLORS[0], alpha=0.85)
    axes[0].axvline(tolerance, color=_COLORS[1], ls="--", label=f"+tol {tolerance:.3f}s")
    axes[0].axvline(-tolerance, color=_COLORS[1], ls="--", label=f"-tol {-tolerance:.3f}s")
    axes[0].set_xlabel("alignment offset (s)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.arange(arr.size), np.abs(arr), color=_COLORS[2], lw=1.0)
    axes[1].axhline(tolerance, color=_COLORS[1], ls="--")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("|offset| (s)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_latency_and_per_step(latency_ms: Sequence[float], per_step_error: Mapping[str, np.ndarray],
                              out_path: str | Path, source: str = "") -> Path:
    """推理延迟分布 + 误差随预测步数增长（chunk 长度取舍的核心证据）。"""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lat = np.asarray(latency_ms, dtype=np.float64)
    axes[0].hist(lat, bins=30, color=_COLORS[3], alpha=0.85)
    axes[0].axvline(np.percentile(lat, 50), color="black", ls="--", label=f"p50={np.percentile(lat, 50):.2f} ms")
    axes[0].axvline(np.percentile(lat, 95), color=_COLORS[1], ls=":", label=f"p95={np.percentile(lat, 95):.2f} ms")
    axes[0].set_xlabel("per-prediction latency (ms)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    for i, (label, values) in enumerate(per_step_error.items()):
        axes[1].plot(np.arange(len(values)), np.asarray(values), marker="o", ms=3,
                     color=_COLORS[i % len(_COLORS)], label=label)
    axes[1].set_xlabel("steps into the predicted chunk (H index)")
    axes[1].set_ylabel("masked L1 action error (normalized units)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.suptitle("推理延迟与逐步误差")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_sample_grid(images: np.ndarray, out_path: str | Path, title: str = "观测示例",
                     source: str = "", n_cols: int = 8, denorm: bool = False,
                     stats: Mapping[str, Any] | None = None) -> Path:
    """观测图像网格（核查预处理是否与训练同源）。"""
    plt = _mpl()
    arr = np.asarray(images)
    if arr.ndim == 5:
        arr = arr[:, 0]
    n = min(arr.shape[0], n_cols * 2)
    cols = min(n_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        img = arr[i]
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if denorm and stats is not None:
            mean = np.asarray(stats["image"]["mean"], dtype=np.float32)
            std = np.asarray(stats["image"]["std"], dtype=np.float32)
            img = np.clip(img * std + mean, 0.0, 1.0)
        ax.imshow(np.squeeze(img), cmap=None if img.ndim == 3 else "gray")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_multimodal_samples(samples: np.ndarray, gt: np.ndarray, out_path: str | Path,
                            source: str = "", channel_names: Sequence[str] | None = None) -> Path:
    """同一观测下多次先验采样的动作块（验证 CVAE 是否学到多模态）。"""
    plt = _mpl()
    names = channel_names or ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for ax, ch in zip(axes, range(min(4, samples.shape[-1]))):
        for i in range(samples.shape[0]):
            ax.plot(samples[i, :, ch], lw=1.0, alpha=0.8, color=_COLORS[i % len(_COLORS)],
                    label=f"prior sample {i}")
        ax.plot(gt[:, ch], "k--", lw=2.0, label="ground truth")
        ax.set_title(f"{names[ch] if ch < len(names) else ch}")
        ax.set_xlabel("step into chunk")
        ax.set_ylabel("normalized action")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("同一观测的多模态动作块（先验采样 vs GT）")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _footer(fig, source)
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_device_split(series: Mapping[str, Sequence[float]], summary: Mapping[str, Any],
                      out_path: str | Path, source: str = "") -> Path:
    """CPU/GPU 分工实测图：每步耗时堆叠 + 时间占比饼图 + 判定结论。

    左图回答"GPU 在等 CPU 还是在算"，右图回答"整体比例是否偏科"。
    """
    plt = _mpl()
    data = np.asarray(series.get("data_ms", []), dtype=np.float64)
    h2d = np.asarray(series.get("h2d_ms", []), dtype=np.float64)
    comp = np.asarray(series.get("compute_ms", []), dtype=np.float64)
    n = min(len(data), len(h2d), len(comp))
    data, h2d, comp = data[:n], h2d[:n], comp[:n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(n)
    axes[0].fill_between(x, 0, data, color="#d62728", alpha=0.85, label="data wait (CPU workers)")
    axes[0].fill_between(x, data, data + h2d, color="#ff7f0e", alpha=0.85, label="host→device copy")
    axes[0].fill_between(x, data + h2d, data + h2d + comp, color="#1f77b4", alpha=0.85,
                         label="compute (GPU forward/backward)")
    axes[0].set_xlabel("measured step index")
    axes[0].set_ylabel("time per step (ms)")
    axes[0].set_title("每步耗时拆分")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.25)
    ratios = [summary.get("data_ratio", 0.0), summary.get("h2d_ratio", 0.0), summary.get("compute_ratio", 0.0)]
    axes[1].pie(
        ratios,
        labels=[f"data wait {ratios[0]:.1%}", f"H2D {ratios[1]:.1%}", f"compute {ratios[2]:.1%}"],
        colors=["#d62728", "#ff7f0e", "#1f77b4"],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
    )
    axes[1].set_title(f"时间占比 · 判定：{summary.get('verdict', 'UNKNOWN')}")
    fig.suptitle("CPU/GPU 分工实测（不偏科：GPU 计算 vs CPU 供给）")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _footer(fig, source, extra=str(summary.get("advice", ""))[:120])
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p
