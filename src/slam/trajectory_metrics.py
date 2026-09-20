"""SLAM 轨迹质量评估：ATE / RPE / 累计漂移 / 与参考轨迹的偏差统计。

硬规则（提示词【五】5.6）：
    * 参考轨迹来源必须显式声明（真值 / 里程计 / mock），无参考时**禁止给出 ATE**；
    * 结果落到 outputs/slam/（CSV + 图 + JSON），数字不得手工编辑。

参考轨迹来源（本实验）：`data/slam/ground_truth.json` 中的 `MOCK` 世界真值位姿
（由 src/datasets/mock_generator.py 生成，与驱动 Cartographer 的激光同源），
因此 ATE 的"真值"是仿真真值，必须在报告中标注 MOCK，不得当作真实场地精度。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..utils.config import get, load_config, load_paths
from ..utils.io_utils import atomic_write_json, ensure_dir, read_jsonl
from ..utils.logging_utils import StageLogger, banner
from ..utils.metrics import write_csv
from .pose_utils import angle_wrap, apply_transform2d, umeyama_alignment


def _portable(path_like: Any, paths: Mapping[str, Any]) -> str:
    """把本机绝对路径转成相对项目根的路径（无法相对化时只保留文件名）。

    JSON 指标与图页脚会随仓库公开，因此禁止写入本机绝对路径。
    """
    p = Path(path_like)
    try:
        return str(p.resolve().relative_to(Path(paths["project_root"]).resolve()))
    except (ValueError, OSError):
        return p.name


class MissingReferenceError(RuntimeError):
    """没有参考轨迹却要求 ATE —— 必须报错而不是编数字。"""


def load_reference(paths: Mapping[str, Any], source: str = "ground_truth") -> dict[str, Any]:
    """加载参考轨迹（真值/里程计/mock），并显式返回来源标签。"""
    if source != "ground_truth":
        raise MissingReferenceError(
            f"当前工程只有 MOCK 世界真值可作为参考（请求的 reference={source}）。"
            "无参考轨迹时禁止给出 ATE。"
        )
    path = Path(paths["slam_data_dir"]) / "ground_truth.json"
    if not path.exists():
        raise MissingReferenceError(f"缺少参考轨迹: {path}（先运行 scripts/10_gen_mock_data.py）")
    data = json.loads(path.read_text(encoding="utf-8"))
    traj = np.asarray(data["trajectory"], dtype=np.float64)
    hz = float(data.get("fps", 10.0))
    return {
        "path": traj[:, :2],
        "yaw": traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
        "t": np.arange(traj.shape[0], dtype=np.float64) / hz,
        "source": str(data.get("source", "MOCK")),
        "file": str(path),
    }


def load_estimate(pose_stream: str | Path) -> dict[str, Any]:
    """从位姿流 JSONL 读估计轨迹（只取 valid=1 的样本）。"""
    records = read_jsonl(pose_stream)
    if not records:
        raise FileNotFoundError(f"位姿流为空或不存在: {pose_stream}")
    valid = [r for r in records if int(r.get("valid", 1)) == 1 and np.isfinite(r.get("x", np.nan))]
    if not valid:
        raise RuntimeError(f"位姿流中没有有效样本（全部 invalid）: {pose_stream}")
    return {
        "t": np.asarray([float(r.get("t_capture", 0.0)) for r in valid], dtype=np.float64),
        "xy": np.asarray([[float(r["x"]), float(r["y"])] for r in valid], dtype=np.float64),
        "yaw": np.asarray([float(r["yaw"]) for r in valid], dtype=np.float64),
        "source": sorted({str(r.get("source", "unknown")) for r in valid}),
        "n_records": len(records),
        "n_invalid": len(records) - len(valid),
        "file": str(pose_stream),
    }


def nearest_reference(ref_t: np.ndarray, est_t: np.ndarray, tol: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """按时间最近邻把估计样本对齐到参考轨迹，返回 (参考下标, 有效掩码)。"""
    idx = np.zeros(est_t.shape[0], dtype=np.int64)
    ok = np.zeros(est_t.shape[0], dtype=bool)
    order = np.argsort(ref_t)
    ref_sorted = ref_t[order]
    for i, t in enumerate(est_t):
        pos = int(np.searchsorted(ref_sorted, t))
        cand = [c for c in (pos - 1, pos) if 0 <= c < ref_sorted.size]
        if not cand:
            continue
        best = min(cand, key=lambda c: abs(ref_sorted[c] - t))
        idx[i] = order[best]
        ok[i] = abs(ref_sorted[best] - t) <= tol
    return idx, ok


def compute_metrics(est_xy: np.ndarray, ref_xy: np.ndarray, est_yaw: np.ndarray,
                    ref_yaw: np.ndarray, rpe_delta_s: float = 1.0, sample_dt: float = 0.1,
                    yaw_offset_rad: float = 0.0,
                    est_xy_unaligned: np.ndarray | None = None) -> dict[str, Any]:
    """计算 ATE（对齐/未对齐）与 RPE 及漂移统计。

    `est_xy` 必须是**已对齐到参考系**的估计轨迹（方向：est → ref），
    `est_xy_unaligned` 为原始（未对齐）估计轨迹，用于报告"坐标系差异"这一独立事实。
    `yaw_offset_rad` 是把估计轨迹从 map 系旋转到参考系所用的角度（est→ref）。
    因此航向误差按 |(est_yaw + yaw_offset) - ref_yaw| 计算——
    否则会把"map 系与参考系之间的固定旋转"错误地算成航向误差（本工程踩过这个坑）。
    """
    n = min(est_xy.shape[0], ref_xy.shape[0])
    est, ref = est_xy[:n], ref_xy[:n]
    err_aligned = np.linalg.norm(est - ref, axis=1)
    if est_xy_unaligned is not None and est_xy_unaligned.shape[0] >= n:
        err_raw = np.linalg.norm(est_xy_unaligned[:n] - ref, axis=1)
    else:
        err_raw = err_aligned
    transform_meta = {"mode": "umeyama_rigid_aligned_by_caller"}

    step = max(1, int(round(rpe_delta_s / sample_dt)))
    if n > step:
        d_est = est[step:] - est[:-step]
        d_ref = ref[step:] - ref[:-step]
        rpe = np.linalg.norm(d_est - d_ref, axis=1)
    else:
        rpe = np.array([np.nan])
    path_len = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1))) if n > 1 else 0.0
    yaw_err = np.abs(angle_wrap(est_yaw[:n] + yaw_offset_rad - ref_yaw[:n]))
    return {
        "n_samples": int(n),
        "ate_rmse_m": float(np.sqrt(np.mean(err_aligned**2))),
        "ate_mean_m": float(np.mean(err_aligned)),
        "ate_max_m": float(np.max(err_aligned)),
        "ate_rmse_unaligned_m": float(np.sqrt(np.mean(err_raw**2))),
        "ate_mean_unaligned_m": float(np.mean(err_raw)),
        "rpe_rmse_m": float(np.sqrt(np.nanmean(rpe**2))),
        "rpe_mean_m": float(np.nanmean(rpe)),
        "rpe_delta_s": float(rpe_delta_s),
        "yaw_err_mean_rad": float(np.mean(yaw_err)),
        "yaw_err_max_rad": float(np.max(yaw_err)),
        "reference_path_length_m": path_len,
        "final_drift_m": float(err_aligned[-1]) if n else float("nan"),
        "alignment": transform_meta,
        "yaw_offset_rad": float(yaw_offset_rad),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cartographer 轨迹质量评估（ATE/RPE/漂移）")
    parser.add_argument("--pose-stream", default=None, help="位姿流 JSONL（默认 configs/slam.yaml 的路径）")
    parser.add_argument("--reference", default=None, help="参考来源，默认取 slam.yaml:metrics.reference")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tol", type=float, default=0.2, help="时间对齐容差(s)")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    paths = load_paths()
    slam_cfg = load_config("slam", overrides=args.override)
    out_dir = ensure_dir(args.out_dir or get(slam_cfg, "metrics.output_dir", paths["slam_out_dir"]))
    log = StageLogger("trajectory_metrics", paths["logs_dir"], stage="42", jsonl_name="42_trajectory_metrics.jsonl")
    print(banner("S8 SLAM 轨迹评估：ATE / RPE / 漂移"))

    pose_stream = args.pose_stream or str(get(slam_cfg, "pose_stream.jsonl_path"))
    reference = args.reference or str(get(slam_cfg, "metrics.reference", "ground_truth"))
    ref = load_reference(paths, reference)
    est = load_estimate(pose_stream)
    idx, ok = nearest_reference(ref["t"], est["t"], tol=float(args.tol))
    n_ok = int(ok.sum())
    if n_ok < 5:
        log.error(f"时间对齐后有效样本仅 {n_ok} 个：位姿流与参考时间轴不匹配，禁止给出 ATE")
        atomic_write_json(out_dir / "trajectory_metrics.json",
                          {"status": "BLOCKED", "reason": "insufficient_time_aligned_samples",
                           "n_aligned": n_ok, "n_estimate": len(est["t"]), "tol_s": float(args.tol)})
        return 4

    est_xy = est["xy"][ok]
    ref_xy = ref["path"][idx[ok]]
    est_yaw = est["yaw"][ok]
    ref_yaw = ref["yaw"][idx[ok]]
    # 显式做一次 Umeyama 刚体对齐（方向：**估计 → 参考**），结果同时用于指标、航向偏移与可视化
    ref_centered = ref_xy - ref_xy.mean(axis=0)
    if est_xy.shape[0] >= 3:
        align_matrix = umeyama_alignment(est_xy - est_xy.mean(axis=0), ref_centered, with_scale=False)
        est_aligned = apply_transform2d(est_xy - est_xy.mean(axis=0), align_matrix)
        yaw_offset = float(np.arctan2(align_matrix[1, 0], align_matrix[0, 0]))
    else:
        est_aligned, yaw_offset = est_xy, 0.0
    metrics = compute_metrics(est_aligned, ref_centered, est_yaw, ref_yaw,
                              rpe_delta_s=1.0, sample_dt=0.1, yaw_offset_rad=yaw_offset,
                              est_xy_unaligned=est_xy)
    metrics["alignment"] = {"mode": "umeyama_rigid", "matrix": align_matrix.tolist()
                            if est_xy.shape[0] >= 3 else None,
                            "direction": "estimate→reference", "with_scale": False}
    metrics.update({
        "status": "OK",
        "reference_source": ref["source"],
        "reference_file": _portable(ref["file"], paths),
        "estimate_file": _portable(est["file"], paths),
        "estimate_sources": est["source"],
        "time_alignment": {
            "method": "nearest_neighbor",
            "tolerance_s": float(args.tol),
            "n_estimate_samples": int(len(est["t"])),
            "n_invalid_samples": int(est["n_invalid"]),
            "n_aligned_used": n_ok,
            "time_span_s": float(est["t"][-1] - est["t"][0]) if len(est["t"]) > 1 else 0.0,
        },
        "evidence_level": "MOCK 真值 + 真实 Cartographer 输出（精度数字仅代表仿真世界，不代表真实场地）",
    })
    atomic_write_json(out_dir / "trajectory_metrics.json", metrics)
    write_csv(out_dir / "trajectory_metrics.csv", [metrics])

    # 图：地图 + 轨迹叠加；ATE 随时间
    try:
        from ..utils.viz import plot_trajectories

        # 图上必须画**对齐后**的轨迹，否则会让人误以为两条轨迹在空间上差了几米
        est_aligned_traj = est_aligned
        ref_traj = ref_centered
        plot_trajectories(
            {"reference (MOCK ground truth)": ref_traj, "Cartographer estimate": est_aligned_traj},
            out_dir / "trajectory_overlay.png",
            title=f"轨迹对比（已做刚体对齐）· ATE(RMSE)={metrics['ate_rmse_m']:.4f} m · "
                  f"RPE={metrics['rpe_rmse_m']:.4f} m",
            source=f"{_portable(est['file'], paths)} vs {_portable(ref['file'], paths)} · MOCK 仿真世界",
            reference="reference (MOCK ground truth)",
        )
        import matplotlib

        from ..utils.viz import setup_matplotlib

        plt = setup_matplotlib()
        err = np.linalg.norm(est_aligned_traj - ref_traj, axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(est["t"][ok], err, color="#1f77b4", lw=1.2)
        ax.axhline(metrics["ate_rmse_m"], color="#d62728", ls="--",
                   label=f"ATE RMSE = {metrics['ate_rmse_m']:.4f} m")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position error (m)")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_title("Cartographer 轨迹误差随时间（对齐后）")
        fig.tight_layout()
        fig.savefig(out_dir / "ate_over_time.png", dpi=130)
        plt.close(fig)
    except Exception as exc:
        log.warning(f"轨迹图绘制失败：{exc}")

    # 地图叠加（若已导出 pgm/yaml）
    map_yaml = out_dir / "cartographer_map.yaml"
    if map_yaml.exists():
        try:
            import matplotlib

            from ..utils.viz import setup_matplotlib

            plt = setup_matplotlib()
            meta = _read_ros_map_yaml(map_yaml)
            pgm = out_dir / f"{meta['image']}"
            img = plt.imread(str(pgm))
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.imshow(img, cmap="gray", origin="upper",
                      extent=[meta["origin"][0], meta["origin"][0] + img.shape[1] * meta["resolution"],
                              meta["origin"][1], meta["origin"][1] + img.shape[0] * meta["resolution"]])
            # 地图在 Cartographer 的 map 系里；要把参考真值叠上去必须用**完整刚体变换**
            # （只做平移会漏掉 map 系相对世界系的固定旋转，图会看起来"两条轨迹不重合"）。
            # 变换方向：ref(世界系) → est(map 系)，即对齐矩阵的逆。
            if est_xy.shape[0] >= 3:
                inv_align = np.linalg.inv(align_matrix)
                ref_in_map = apply_transform2d(ref_xy - ref_xy.mean(axis=0), inv_align) + est_xy.mean(axis=0)
            else:
                ref_in_map = ref_xy + (est_xy[0] - ref_xy[0])
            # 参考真值用更粗的半透明虚线打底：两条轨迹几乎重合时（ATE≈2.7cm）才看得出"确实重叠"，
            # 否则细虚线会被实线盖住，图上看不出它们一致（本工程踩过这个可视化坑）。
            ax.plot(ref_in_map[:, 0], ref_in_map[:, 1], "--", color="#444444", lw=3.0, alpha=0.55,
                    label="reference (MOCK truth, 刚体对齐到 map 系)")
            ax.plot(est_xy[:, 0], est_xy[:, 1], color="#d62728", lw=1.3, label="Cartographer (map 系)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Cartographer 占用栅格地图 + 轨迹")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "cartographer_map_with_trajectory.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            log.warning(f"地图叠加图绘制失败：{exc}")

    log.info(f"轨迹评估完成：ATE(RMSE)={metrics['ate_rmse_m']:.4f} m，"
             f"RPE={metrics['rpe_rmse_m']:.4f} m，对齐样本 {n_ok}/{len(est['t'])}，来源={ref['source']}")
    log.metric_now(message="trajectory metrics", **{k: v for k, v in metrics.items()
                                                   if isinstance(v, (int, float))})
    log.flush()
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str)[:1600])
    return 0


def _read_ros_map_yaml(path: Path) -> dict[str, Any]:
    """极简解析 ROS map yaml（避免为一个字段引入额外依赖）。"""
    import re

    text = path.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    m = re.search(r"image:\s*(\S+)", text)
    out["image"] = m.group(1) if m else "cartographer_map.pgm"
    m = re.search(r"resolution:\s*([0-9.eE+-]+)", text)
    out["resolution"] = float(m.group(1)) if m else 0.05
    m = re.search(r"origin:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)", text)
    out["origin"] = [float(m.group(1)), float(m.group(2))] if m else [0.0, 0.0]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
