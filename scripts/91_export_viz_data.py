#!/usr/bin/env python
"""scripts/91_export_viz_data.py —— 导出「窗口内可视化」所需的紧凑数据。

用途：把 logs/ 与 outputs/ 里的真实产物压缩成一个小 JSON（默认每序列 ≤120 点、保留 4 位有效数字），
供内联可视化（HTML 片段）直接嵌入，避免把整份日志塞进对话。**不做任何补数**：
缺失的序列直接不出现，并在 missing 字段里登记。

产物：outputs/figures/viz_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.slam.pose_utils import apply_transform2d, umeyama_alignment  # noqa: E402
from src.slam.trajectory_metrics import load_estimate, load_reference, nearest_reference  # noqa: E402
from src.utils.config import load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_json, read_json, read_jsonl  # noqa: E402


def _round(values, nd: int = 4) -> list[float]:
    return [float(f"{float(v):.{nd}g}") for v in values]


def _downsample(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, target).astype(int))


def train_curve(logs: Path, target: int = 100) -> dict | None:
    records = read_jsonl(logs / "train_metrics.jsonl") if (logs / "train_metrics.jsonl").exists() else []
    # 同一个 JSONL 里也会落验证记录（key 为 val/*），这里只保留训练步记录，避免出现空值
    records = [r for r in records if r.get("loss/total") is not None]
    if not records:
        return None
    idx = _downsample(len(records), target)
    steps = [int(records[i].get("step", i)) for i in idx]
    out = {"step": steps}
    for key, name in (("loss/total", "total"), ("loss/recon", "recon"),
                      ("loss/kl", "kl"), ("beta", "beta"), ("grad_norm", "grad_norm")):
        vals = [records[i].get(key) for i in idx]
        if all(v is not None for v in vals):
            out[name] = _round(vals)
    return out


def val_curve(logs: Path) -> dict | None:
    records = read_jsonl(logs / "val_metrics.jsonl") if (logs / "val_metrics.jsonl").exists() else []
    if not records:
        return None
    return {
        "epoch": [int(r.get("epoch", i)) for i, r in enumerate(records)],
        "total": _round([r.get("val/total", float("nan")) for r in records]),
        "recon": _round([r.get("val/recon", float("nan")) for r in records]),
        "kl": _round([r.get("val/kl", float("nan")) for r in records]),
    }


def device_usage(logs: Path) -> dict | None:
    d = read_json(logs / "20_device_usage.json") if (logs / "20_device_usage.json").exists() else None
    if not d:
        return None
    policy = d.get("policy") or {}
    return {
        "data_ratio": round(float(d.get("data_ratio", 0.0)), 4),
        "h2d_ratio": round(float(d.get("h2d_ratio", 0.0)), 4),
        "compute_ratio": round(float(d.get("compute_ratio", 0.0)), 4),
        "data_ms": round(float(d.get("data_ms_mean", 0.0)), 3),
        "h2d_ms": round(float(d.get("h2d_ms_mean", 0.0)), 3),
        "compute_ms": round(float(d.get("compute_ms_mean", 0.0)), 3),
        "verdict": d.get("verdict"),
        "n_steps": int(d.get("n_steps", 0)),
        "batch_size": policy.get("batch_size"),
        "workers": policy.get("num_workers"),
        "torch_threads": policy.get("torch_threads"),
        "device": d.get("device"),
    }


def ablation_summary(out_dir: Path) -> list[dict]:
    path = out_dir / "ablation" / "summary.csv"
    if not path.exists():
        return []
    import csv

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            def num(key):
                v = raw.get(key)
                try:
                    return float(v) if v not in (None, "", "—") else None
                except ValueError:
                    return None

            rows.append({
                "run": raw.get("run"),
                "status": raw.get("status"),
                "recon": num("eval_recon_l1"),
                "latency_per_step_ms": num("latency_per_step_ms"),
                "jitter_no_ensemble": num("eval_jitter_no_ensemble"),
                "jitter_ensemble": num("eval_jitter_ensemble"),
                "best_val": num("best_val_total"),
                "n_exec": num("n_exec"),
            })
    return rows


def slam_panel(paths: dict, target: int = 220) -> dict | None:
    metrics_path = Path(paths["slam_out_dir"]) / "trajectory_metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    if not metrics or metrics.get("status") != "OK":
        return None
    # metrics.reference_source 记录的是参考轨迹的**数据来源**（MOCK/REAL），
    # 而 load_reference 需要的是"参考类型"（ground_truth），两者语义不同，不要混用。
    ref = load_reference(paths, "ground_truth")
    est = load_estimate(str(paths["slam_data_dir"]) + "/pose_stream.jsonl")
    idx, ok = nearest_reference(ref["t"], est["t"], tol=0.2)
    est_xy = est["xy"][ok]
    ref_xy = ref["path"][idx[ok]]
    if est_xy.shape[0] < 3:
        return None
    ref_centered = ref_xy - ref_xy.mean(axis=0)
    align = umeyama_alignment(est_xy - est_xy.mean(axis=0), ref_centered, with_scale=False)
    est_aligned = apply_transform2d(est_xy - est_xy.mean(axis=0), align)
    take = _downsample(est_aligned.shape[0], target)
    error = np.linalg.norm(est_aligned - ref_centered, axis=1)
    return {
        "est": [[round(float(est_aligned[i, 0]), 3), round(float(est_aligned[i, 1]), 3)] for i in take],
        "ref": [[round(float(ref_centered[i, 0]), 3), round(float(ref_centered[i, 1]), 3)] for i in take],
        "error": _round([error[i] for i in take], 3),
        "t": _round([float(est["t"][ok][i]) for i in take], 4),
        "metrics": {
            "ate_rmse_m": metrics.get("ate_rmse_m"),
            "ate_mean_m": metrics.get("ate_mean_m"),
            "ate_max_m": metrics.get("ate_max_m"),
            "rpe_rmse_m": metrics.get("rpe_rmse_m"),
            "yaw_err_mean_rad": metrics.get("yaw_err_mean_rad"),
            "final_drift_m": metrics.get("final_drift_m"),
            "path_length_m": metrics.get("reference_path_length_m"),
            "n_samples": metrics.get("n_samples"),
            "reference_source": metrics.get("reference_source"),
        },
    }


def main() -> int:
    paths = load_paths()
    logs = Path(paths["logs_dir"])
    out_dir = Path(paths["outputs_dir"])
    payload: dict = {
        # 公开产物不写本机绝对路径：项目根恒为仓库根（"."），外部根只用占位符
        "project_root": ".",
        "env_root": "<external env root>",
        "source": "MOCK",
        "generated_from": {
            "train": "logs/train_metrics.jsonl",
            "val": "logs/val_metrics.jsonl",
            "device": "logs/20_device_usage.json",
            "ablation": "outputs/ablation/summary.csv",
            "slam": "outputs/slam/trajectory_metrics.json + data/slam/pose_stream.jsonl",
        },
        "missing": [],
    }
    for key, fn in (("train_curve", train_curve), ("val_curve", val_curve)):
        value = fn(logs)
        if value is None:
            payload["missing"].append(key)
        else:
            payload[key] = value
    value = device_usage(logs)
    if value is None:
        payload["missing"].append("device")
    else:
        payload["device"] = value
    payload["ablation"] = ablation_summary(out_dir)
    value = slam_panel(paths)
    if value is None:
        payload["missing"].append("slam")
    else:
        payload["slam"] = value
    summary = read_json(logs / "train_summary.json") if (logs / "train_summary.json").exists() else None
    if summary:
        payload["train_summary"] = {
            "status": summary.get("status"),
            "steps": summary.get("steps"),
            "duration_s": summary.get("duration_s"),
            "best_val": summary.get("best_val"),
            "monitor": summary.get("monitor"),
            "params": (summary.get("model") or {}).get("params_trainable"),
            "config_hash": (summary.get("config_hashes") or {}).get("data+model"),
            "seed": summary.get("seed"),
            "git_commit": summary.get("git_commit"),
        }
    gate = read_json(logs / "21_overfit_gate.json") if (logs / "21_overfit_gate.json").exists() else None
    if gate:
        payload["gate"] = {k: gate.get(k) for k in
                           ("status", "loss_first_window", "loss_last_window", "relative_drop",
                            "threshold", "batch_size", "steps")}
    evalj = read_json(logs / "infer_eval.json") if (logs / "infer_eval.json").exists() else None
    if evalj:
        wm = evalj.get("window_metrics") or {}
        roll = evalj.get("rollouts") or {}
        payload["eval"] = {
            "recon_l1": wm.get("recon_l1_mean"),
            "latency": wm.get("latency"),
            "diversity": wm.get("sample_diversity"),
            "hit_rate": wm.get("multimodal_hit_rate"),
            "jitter_no_ensemble": (roll.get("no_ensemble") or {}).get("jitter_order2_mean"),
            "jitter_ensemble": (roll.get("ensemble") or {}).get("jitter_order2_mean"),
        }
    close = read_json(logs / "31_closed_loop_summary.json") if (logs / "31_closed_loop_summary.json").exists() else None
    if close:
        payload["closed_loop"] = {
            "executed_steps": close.get("executed_steps"),
            "publish_allowed": (close.get("safety") or {}).get("publish_allowed"),
            "clip_events": (close.get("limiter") or {}).get("n_clip_events"),
            "nan_events": (close.get("limiter") or {}).get("n_nan_events"),
            "events": len(close.get("events") or []),
            "pose_source": close.get("pose_source"),
        }
    out = Path(paths["figures_dir"]) / "viz_data.json"
    atomic_write_json(out, payload)
    print(json.dumps({"status": "OK", "out": str(out), "missing": payload["missing"],
                      "keys": sorted(payload.keys())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
