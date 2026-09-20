#!/usr/bin/env python
"""scripts/12_data_check.py —— 数据体检：schema / 形状 / 时间戳 / 取值域 / 对齐 / 抽样可视化。

输出：
    logs/12_data_check.json     结构化体检报告（含被过滤样本明细与原因分布）
    outputs/figures/12_data_samples.png  抽样图像-状态-动作对（人工核查）
    logs/slam_align.json        SLAM 位姿与策略时钟的对齐报告（存在位姿流时）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.episode_reader import EpisodeReader  # noqa: E402
from src.datasets.schema import observation_spec, state_spec, validate_episode, validate_sample  # noqa: E402
from src.datasets.transforms import build_transform  # noqa: E402
from src.datasets.window_builder import build_windows, make_sample, obs_indices, windows_per_episode  # noqa: E402
from src.slam.pose_utils import angle_wrap  # noqa: E402
from src.utils.config import get, load_config, load_paths, state_block_layout  # noqa: E402
from src.utils.io_utils import atomic_write_json, append_jsonl, ensure_dir, read_jsonl  # noqa: E402
from src.utils.logging_utils import StageLogger, banner  # noqa: E402
from src.utils.time_sync import AlignReport, align_nearest  # noqa: E402


def check_episode(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """单条 episode 的形状/时间戳/取值域检查（不抛异常，全部进报告）。"""
    report = validate_episode(path, cfg)
    out = {"path": str(path), "ok": report.ok, "errors": report.errors, "warnings": report.warnings,
           "stats": report.stats}
    if not report.ok:
        return out
    with EpisodeReader(path, lazy=True, expected_dt=1.0 / float(get(cfg, "control.hz", 10.0))) as reader:
        ts = reader.timestamps()
        out["timestamps"] = {
            "n": int(ts.size),
            "dt_median_s": float(np.median(np.diff(ts))) if ts.size > 1 else None,
            "dt_max_s": float(np.max(np.diff(ts))) if ts.size > 1 else None,
            "monotonic": bool(np.all(np.diff(ts) > 0)) if ts.size > 1 else True,
        }
        out["alerts"] = reader.alerts.to_dict()
        pose, valid = reader.slam_poses()
        out["slam"] = {
            "valid_ratio": float(valid.mean()),
            "max_xy_step": float(np.abs(np.diff(pose[:, :2], axis=0)).max()) if pose.shape[0] > 1 else 0.0,
            "max_yaw_step_rad": float(np.abs(np.diff(angle_wrap(pose[:, 2]))).max()) if pose.shape[0] > 1 else 0.0,
        }
        images = reader.images()
        out["images"] = {
            "shape": list(images.shape),
            "mean": float(images.mean()),
            "std": float(images.std()),
            "all_black_frames": int(np.sum(images.reshape(images.shape[0], -1).std(axis=1) < 1.0)),
        }
    return out


def window_consistency(episode_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """窗口数是否与「episode 长度 + stride + H」推算一致，并做一次样本级校验。"""
    with EpisodeReader(episode_path, lazy=True) as reader:
        length = reader.length
        expected = windows_per_episode(length, cfg)
        windows = build_windows(
            length,
            int(get(cfg, "observation.num_frames", 1)),
            int(get(cfg, "observation.frame_stride", 1)),
            int(get(cfg, "chunk.H", 8)),
            int(get(cfg, "chunk.stride", 1)),
            str(get(cfg, "chunk.pad_mode", "mask")),
        )
        idx = [obs_indices(t, int(get(cfg, "observation.num_frames", 1)),
                           int(get(cfg, "observation.frame_stride", 1)), length) for t in windows]
        assert all(i is not None for i in idx), "窗口历史帧下标越界（禁止补帧）"
        sample = make_sample(
            {
                "images": reader.images(),
                "state": reader.state_matrix(),
                "action": reader.actions(),
                "timestamp": reader.timestamps(),
                "slam_valid": reader.slam_poses()[1],
                "episode_id": reader.episode_id,
            },
            windows[len(windows) // 2], idx[len(windows) // 2], cfg,
        )
        validate_sample(sample, cfg)
    return {
        "episode": episode_path.name,
        "length": length,
        "expected_windows": expected,
        "actual_windows": len(windows),
        "consistent": expected == len(windows),
        "sample_shapes": {k: list(np.asarray(v).shape) for k, v in sample.items() if k != "meta"},
        "mask_sum_last": float(np.asarray(sample["mask"]).sum()),
    }


def alignment_report(cfg: dict[str, Any], paths: dict[str, Any], slam_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """SLAM 位姿与 episode 时间戳的对齐报告（超容差比例 > 阈值即必须先修时钟）。"""
    pose_stream = Path(paths["slam_data_dir"]) / "pose_stream.jsonl"
    if not pose_stream.exists():
        return {"status": "NOT_MEASURED", "reason": f"位姿流不存在: {pose_stream}",
                "hint": "先运行 scripts/40_slam_bringup.sh（或 mock 位姿流）"}
    records = read_jsonl(pose_stream)
    if not records:
        return {"status": "NOT_MEASURED", "reason": "位姿流为空"}
    times = np.asarray([float(r.get("t_mono", r.get("t_capture", 0.0))) for r in records])
    tol = float(get(slam_cfg or {}, "sync.max_align_tolerance_s", 0.1))
    report = AlignReport()
    # 用数据集的 timestamp 作为查询时钟（与训练对齐口径一致）
    episodes = sorted(Path(paths["mock_dir"]).glob("*.npz")) or sorted(Path(paths["raw_dir"]).glob("*.npz"))
    if not episodes:
        return {"status": "NOT_MEASURED", "reason": "没有 episode 可比对"}
    with EpisodeReader(episodes[0], lazy=True) as reader:
        query = reader.timestamps()
    _, valid = align_nearest(times, query, tol, report)
    summary = report.summary()
    summary["status"] = "OK"
    summary["n_pose_samples"] = len(records)
    summary["tolerance_s"] = tol
    summary["compared_episode"] = episodes[0].name
    summary["pass"] = summary["out_of_tolerance_ratio"] < 0.05
    return summary


def sample_figure(paths: dict[str, Any], cfg: dict[str, Any], out_path: Path) -> bool:
    """抽样可视化：图像序列 + 状态块 + 动作块（人工核查"数据是否像样"）。"""
    try:
        from src.utils.viz import setup_matplotlib

        plt = setup_matplotlib()
    except Exception:
        return False
    episodes = sorted(Path(paths["mock_dir"]).glob("*.npz")) or sorted(Path(paths["raw_dir"]).glob("*.npz"))
    if not episodes:
        return False
    transform = build_transform(cfg)
    layout = state_block_layout(cfg)
    with EpisodeReader(episodes[0], lazy=True) as reader:
        t = min(reader.length - 1, int(reader.length * 0.6))
        num_frames = int(get(cfg, "observation.num_frames", 1))
        stride = int(get(cfg, "observation.frame_stride", 1))
        idx = obs_indices(t, num_frames, stride, reader.length)
        images = reader.images()[idx]
        state_raw = reader.state_matrix()[t]
        action = reader.actions()
        chunk = action[t : t + int(get(cfg, "chunk.H", 8))]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i in range(min(num_frames, 3)):
        ax = axes[0, i]
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.set_title(f"obs frame {i} (t={t - (num_frames - 1 - i) * stride})")
        ax.axis("off")
    axes[0, 2].plot(state_raw, marker="o", ms=3)
    axes[0, 2].set_title(f"state @ t={t} (dim={state_raw.size})")
    for name, start, end in layout:
        axes[0, 2].axvspan(start - 0.5, end - 0.5, alpha=0.08)
        axes[0, 2].text((start + end) / 2 - 0.5, float(np.max(state_raw)), name, fontsize=6, rotation=90,
                        ha="center", va="bottom")
    for d in range(min(chunk.shape[1], 3)):
        axes[1, 0].plot(chunk[:, d], marker="o", ms=3, label=f"a[{d}]")
    axes[1, 0].set_title("action chunk: translation deltas")
    axes[1, 0].legend(fontsize=7)
    for d in range(3, min(chunk.shape[1], 6)):
        axes[1, 1].plot(chunk[:, d], marker="o", ms=3, label=f"a[{d}]")
    axes[1, 1].set_title("action chunk: rotation deltas")
    axes[1, 1].legend(fontsize=7)
    if chunk.shape[1] >= 7:
        axes[1, 2].plot(action[:, 6], label="all steps")
        axes[1, 2].plot(np.arange(t, t + chunk.shape[0]), chunk[:, 6], label="chunk window")
        axes[1, 2].set_title("gripper")
        axes[1, 2].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle(f"12_data_check · {episodes[0].name} · source=MOCK（合成数据，非真实采集）")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据体检")
    parser.add_argument("--config", default="data")
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    paths = load_paths()
    cfg = load_config("data", ablation=args.ablation, overrides=args.override)
    slam_cfg = load_config("slam", overrides=args.override)
    log = StageLogger("data_check", paths["logs_dir"], stage="12")
    print(banner("S3 数据体检：schema / 形状 / 时间戳 / 对齐 / 抽样"))

    episodes = sorted(Path(paths["mock_dir"]).glob("*.npz")) or sorted(Path(paths["raw_dir"]).glob("*.npz"))
    if not episodes:
        raise FileNotFoundError("没有 episode 可检查，先运行 scripts/10_gen_mock_data.py")

    checks = [check_episode(p, cfg) for p in episodes]
    consistency = [window_consistency(p, cfg) for p in episodes[:3]]
    obs = observation_spec(cfg)
    st = state_spec(cfg)
    alignment = alignment_report(cfg, paths, slam_cfg)

    n_errors = sum(len(c["errors"]) for c in checks)
    n_warnings = sum(len(c["warnings"]) for c in checks)
    report = {
        "stage": "12",
        "source": "MOCK" if "mock" in str(episodes[0]) else "REAL",
        "config_hash": get(cfg, "schema_version"),
        "contract": {
            "images": list(obs.image_shape),
            "state_dim": st.dim,
            "action_dim": int(get(cfg, "action.dim", 7)),
            "horizon": int(get(cfg, "chunk.H", 8)),
            "slam_input": bool(get(cfg, "slam_input.enabled", True)),
        },
        "num_episodes": len(episodes),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "episodes": checks,
        "window_consistency": consistency,
        "alignment": alignment,
        "status": "OK" if n_errors == 0 else "ERROR",
    }
    out = Path(paths["logs_dir"]) / "12_data_check.json"
    atomic_write_json(out, report)
    align_out = Path(paths["logs_dir"]) / "slam_align.json"
    atomic_write_json(align_out, alignment)

    fig_ok = sample_figure(paths, cfg, Path(paths["figures_dir"]) / "12_data_samples.png")
    append_jsonl(Path(paths["logs_dir"]) / "12_data_check.jsonl",
                 [{"episode": c["path"], "ok": c["ok"], **(c.get("stats") or {})} for c in checks])

    log.info(f"体检完成：episodes={len(episodes)} errors={n_errors} warnings={n_warnings} "
             f"figure={'OK' if fig_ok else 'UNAVAILABLE'} alignment={alignment.get('status')}")
    log.flush()
    print(json.dumps({"status": report["status"], "n_errors": n_errors, "n_warnings": n_warnings,
                      "contract": report["contract"],
                      "window_consistency": [c["consistent"] for c in consistency],
                      "alignment": alignment.get("status"),
                      "report": str(out)}, ensure_ascii=False, indent=2))
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
