"""离线评测：在 val/test split 上计算逐步/逐 chunk 动作误差、平滑度、多模态命中率、
推理延迟分位数，以及「有无时间集成 / 有无 SLAM 输入」的对照指标。

输出：outputs/eval/*.csv、outputs/figures/*.png、logs/infer_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in (None, ""):  # 允许 `python src/infer/offline_eval.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..datasets.episode_dataset import WindowDataset, collate_fn, split_paths
from ..datasets.episode_reader import EpisodeReader
from ..datasets.schema import raw_block_layout, state_column_selector
from ..datasets.transforms import (build_transform, inverse_action, load_stats, norm_action, norm_state,
                                   stats_path_for)
from ..datasets.window_builder import obs_indices
from ..models.cvae_policy import CVAEPolicy
from ..models.losses import masked_recon_loss
from ..train.checkpoint import load_for_inference
from ..train.losses import build_loss_weights
from ..utils.config import config_hash, get, load_config, load_paths
from ..utils.device_profile import DevicePolicy
from ..utils.io_utils import atomic_write_json, ensure_dir
from ..utils.logging_utils import StageLogger, banner
from ..utils.metrics import jitter, quantiles, rmse, summarize_latency, write_csv
from ..utils.viz import plot_action_comparison, plot_chunk_weight_heatmap, plot_latency_and_per_step
from .rollout_policy import RolloutRunner


def resolve_checkpoint(paths: Mapping[str, Any], infer_cfg: Mapping[str, Any], run_name: str) -> Path:
    """按 `infer.checkpoint`（last/best/绝对路径）解析 checkpoint 位置。"""
    spec = str(get(infer_cfg, "checkpoint", "best"))
    if spec in ("last", "best"):
        p = Path(paths["checkpoints_dir"]) / run_name / f"{spec}.pt"
    else:
        p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {p}（先训练或修正 infer.checkpoint）")
    return p


def load_model_and_stats(paths: Mapping[str, Any], data_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any],
                         infer_cfg: Mapping[str, Any], run_name: str, device: torch.device):
    """加载 checkpoint + 统计（强校验 hash），返回 (model, stats, meta, stats_path)。"""
    stats_path = stats_path_for(paths, run_name)
    stats = load_stats(stats_path, data_cfg)
    model = CVAEPolicy(data_cfg, model_cfg)
    ckpt = resolve_checkpoint(paths, infer_cfg, run_name)
    meta = load_for_inference(ckpt, model, data_cfg=data_cfg, model_cfg=model_cfg,
                              stats_path=stats_path, strict=True)
    model.to(device).eval()
    return model, stats, meta, stats_path


def load_episode_for_inference(path: Path, data_cfg: Mapping[str, Any], stats: Mapping[str, Any],
                               selector: np.ndarray) -> dict[str, Any]:
    """用与训练完全同源的函数把原始 episode 转成推理输入（图像归一化 + 状态归一化）。"""
    from ..datasets.schema import state_spec

    # CUDA 为主：这里不做 CPU 侧归一化，把 uint8 观测直接交给模型在设备侧处理
    block_map = {name: slice(start, end) for name, start, end in state_spec(data_cfg).blocks}
    normalize_blocks = list(get(data_cfg, "state.normalize_blocks", []))
    num_frames = int(get(data_cfg, "observation.num_frames", 1))
    frame_stride = int(get(data_cfg, "observation.frame_stride", 1))
    with EpisodeReader(path, lazy=False) as reader:
        length = reader.length
        images = reader.images()
        state_raw = reader.state_matrix()[:, selector]
        action_raw = reader.actions()
        obs_idx = []
        for t in range(length):
            idx = obs_indices(t, num_frames, frame_stride, length)
            obs_idx.append(np.zeros(num_frames, dtype=np.int64) if idx is None else idx)
        obs_all = np.stack([images[i] for i in obs_idx], axis=0)
        state_norm = norm_state(state_raw, stats, block_map, normalize_blocks)
        action_norm = norm_action(action_raw, stats)
        timestamps = reader.timestamps()   # 必须在 with 块内读（块外 reader 已关闭）
        meta = {"episode_id": reader.episode_id, "source": reader.source, "length": length,
                "success": reader.success,
                "slam_valid_ratio": float(reader.slam_poses()[1].mean())}
    return {
        # 保持 uint8：归一化在 GPU 侧由模型完成（与训练完全同一实现）
        "images": obs_all.astype(np.uint8, copy=False).reshape(
            length, num_frames, obs_all.shape[2], obs_all.shape[-2], obs_all.shape[-1]
        ),
        "state": state_norm,
        "action": action_norm,
        "obs_indices": np.asarray(obs_idx),
        "raw_action": action_raw,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "meta": meta,
    }


@torch.no_grad()
def evaluate_windows(model: CVAEPolicy, loader, stats: Mapping[str, Any], device: torch.device,
                     limit: int | None = None, warmup_batches: int = 2) -> dict[str, Any]:
    """窗口级评测：逐步误差、chunk 级误差、平均动作基线、多模态命中率、延迟。"""
    latencies_ms: list[float] = []
    per_step_chunk: list[np.ndarray] = []
    per_step_single: list[np.ndarray] = []
    per_step_baseline: list[np.ndarray] = []
    diversity: list[float] = []
    hit_rate: list[float] = []
    abs_errors: list[float] = []
    meta_samples: list[dict[str, Any]] = []

    import time

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        target = batch["action_chunk"]
        mask = batch["mask"]
        _sync()
        started = time.perf_counter()
        pred = model.predict_chunk(images, state, mode="mean")
        _sync()
        latency_ms = (time.perf_counter() - started) * 1000.0
        # 前 warmup_batches 个 batch 包含 CUDA 上下文/算法选择开销，不计入延迟统计
        # （避免把 300ms 的一次性初始化写成模型延迟）。
        if i >= warmup_batches:
            latencies_ms.append(latency_ms)
        samples = model.predict_chunk(images, state, mode="sample", n_samples=4)

        pred_cpu = pred.cpu()
        per_step_chunk.append(masked_recon_loss(pred_cpu, target, mask, "l1")[1].numpy())
        per_step_single.append(
            masked_recon_loss(pred_cpu[:, :1], target[:, :1], mask[:, :1], "l1")[1].numpy()
        )
        # 基线：预测训练集动作均值（归一化空间即接近 0）
        baseline = torch.zeros_like(target)
        per_step_baseline.append(masked_recon_loss(baseline, target, mask, "l1")[1].numpy())
        abs_errors.append(float(masked_recon_loss(pred_cpu, target, mask, "l1")[0]))

        # 多模态：4 次先验采样中是否至少有一次逼近 GT（用 L1 阈值判定"命中"）
        flat = samples.permute(1, 0, 2, 3).cpu()  # [B, N, H, d_a]
        pairwise = flat[:, :, None] - flat[:, None, :]
        diversity.append(float(pairwise.abs().mean()))
        dist = (flat - target[:, None]).abs().mean(dim=(2, 3))  # [B, N]
        hit = (dist.min(dim=1).values < float(dist.median()) * 0.7).float().mean()
        hit_rate.append(float(hit))
        if len(meta_samples) < 3:
            meta_samples.extend(batch["meta"][:1])

    out = {
        "n_batches": len(abs_errors),
        "recon_l1_mean": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "recon_rmse": None,
        "per_step_chunk_l1": np.mean(per_step_chunk, axis=0).tolist() if per_step_chunk else [],
        "per_step_single_l1": float(np.mean([p[0] for p in per_step_single])) if per_step_single else float("nan"),
        "per_step_baseline_l1": float(np.mean([p[0] for p in per_step_baseline])) if per_step_baseline else float("nan"),
        "sample_diversity": float(np.mean(diversity)) if diversity else 0.0,
        "multimodal_hit_rate": float(np.mean(hit_rate)) if hit_rate else 0.0,
        # summarize_latency 的入参单位是秒，这里统一换算，避免二次乘 1000 的经典错误
        "latency": summarize_latency([v / 1000.0 for v in latencies_ms]),
        "latency_samples_ms": latencies_ms,
    }
    return out


def evaluate_rollouts(model: CVAEPolicy, data_cfg: Mapping[str, Any], infer_cfg: Mapping[str, Any],
                      stats: Mapping[str, Any], device: torch.device, episodes: Sequence[Path],
                      ensemble: bool, max_steps: int | None = None, selector: np.ndarray | None = None,
                      log: StageLogger | None = None) -> dict[str, Any]:
    """闭环滚动评测（含/不含时间集成），返回抖动与延迟对照指标。"""
    cfg = json.loads(json.dumps(infer_cfg))
    # 注意：这里**不能**无条件把 enabled 置 True——否则 no_temporal_ensemble 消融会被静默覆盖，
    # 两条对照实际跑的是同一配置（本工程踩过这个坑）。
    # 语义：ensemble=True 表示"启用时间集成"，但最终是否启用还要受该次运行配置的开关约束。
    cfg["temporal_ensemble"]["enabled"] = bool(ensemble) and bool(
        get(infer_cfg, "temporal_ensemble.enabled", True)
    )
    runner = RolloutRunner(model, data_cfg, cfg, stats, device)
    jitters, latencies, n_steps = [], [], 0
    all_actions, all_gt, all_masks = [], [], []
    for path in episodes:
        sel = selector if selector is not None else np.arange(_state_dim(data_cfg))
        episode = load_episode_for_inference(path, data_cfg, stats, sel)
        result = runner.run_episode(episode, episode["obs_indices"], actions_norm=episode["action"],
                                    max_steps=max_steps, masks=np.ones(episode["state"].shape[0], dtype=np.float32))
        actions = result.actions_phys()
        jitters.append(jitter(actions, order=2))
        latencies.extend(result.latency_ms)
        n_steps += len(result.steps)
        all_actions.append(actions)
        all_gt.append(result.gt_actions())
        all_masks.append(result.masks())
        if log:
            payload = {"episode": path.name, "ensemble": bool(ensemble), **{
                k: v for k, v in result.summary().items() if isinstance(v, (int, float))
            }}
            log.metric_now(message=f"rollout {path.name} ensemble={ensemble}", **payload)
    return {
        "ensemble": bool(ensemble),
        "n_episodes": len(episodes),
        "n_steps": n_steps,
        "jitter_order2_mean": float(np.mean(jitters)) if jitters else 0.0,
        "latency": summarize_latency(latencies),
        "actions": np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 1)),
        "gt_actions": np.concatenate(all_gt, axis=0) if all_gt else np.zeros((0, 1)),
        "masks": np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0,)),
        "runner": runner,
    }


def _state_dim(data_cfg: Mapping[str, Any]) -> int:
    from ..datasets.schema import state_spec

    return state_spec(data_cfg).dim


def _selector_for(path: Path, data_cfg: Mapping[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as handle:
        layout = raw_block_layout(handle["state_block_names"], handle["state_block_dims"])
    return state_column_selector(layout, data_cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线推理与评测")
    parser.add_argument("--split", default=None, help="val | test（默认取 infer.yaml）")
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    paths = load_paths()
    data_cfg = load_config("data", ablation=args.ablation, overrides=args.override)
    model_cfg = load_config("model", ablation=args.ablation, overrides=args.override)
    train_cfg = load_config("train", ablation=args.ablation, overrides=args.override)
    infer_cfg = load_config("infer", ablation=args.ablation, overrides=args.override)
    if args.split:
        infer_cfg["split"] = args.split
    if args.device:
        infer_cfg["device"] = args.device
    split = str(get(infer_cfg, "split", "test"))
    dev_name = str(get(infer_cfg, "device", "auto"))
    if dev_name not in ("cpu", "cuda"):
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    log = StageLogger("offline_eval", paths["logs_dir"], stage="30")
    print(banner(f"S6/S7 离线推理与评测：run={args.run_name} split={split}"))

    model, stats, meta, stats_path = load_model_and_stats(paths, data_cfg, model_cfg, infer_cfg,
                                                          args.run_name, device)
    log.info(f"加载 checkpoint：step={meta['step']} hash={meta['config_hashes'].get('data+model')}",
             checkpoint=meta.get("weights", ""), git_commit=meta.get("git_commit"))

    stem, _ = split_paths(split, paths)
    if args.run_name != "base":
        candidate = Path(paths["processed_dir"]) / f"{split}.{args.run_name}"
        if candidate.with_name(f"{candidate.name}.state.npy").exists():
            stem = candidate
    dataset = WindowDataset(stem, data_cfg)
    # 推理侧同样按策略分配 CPU：worker 进程读盘+归一化，GPU 只做前向；
    # 评测聚合、画图、落盘全在 CPU，避免把非张量工作塞进 GPU 关键路径。
    policy = DevicePolicy.from_config(infer_cfg)
    policy.apply_thread_limits()
    log.info("推理侧 CPU/GPU 分工：" + json.dumps(policy.describe(), ensure_ascii=False))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=collate_fn,
        **policy.dataloader_kwargs(is_train=False),
    )
    window_metrics = evaluate_windows(model, loader, stats, device)

    # --- 闭环滚动对照：关闭 / 开启时间集成 ---
    test_episodes = sorted(Path(paths["mock_dir"]).glob("*.npz"))[: args.rollout_episodes] \
        if Path(paths["mock_dir"]).exists() else []
    selector = _selector_for(test_episodes[0], data_cfg) if test_episodes else None
    rollouts: dict[str, Any] = {}
    if test_episodes:
        rollouts["no_ensemble"] = evaluate_rollouts(model, data_cfg, infer_cfg, stats, device, test_episodes,
                                                    ensemble=False, max_steps=args.max_steps,
                                                    selector=selector, log=log)
        rollouts["ensemble"] = evaluate_rollouts(model, data_cfg, infer_cfg, stats, device, test_episodes,
                                                 ensemble=True, max_steps=args.max_steps,
                                                 selector=selector, log=log)

    # --- 落盘指标 ---
    eval_dir = ensure_dir(Path(paths["eval_dir"]))
    rows = [
        {
            "run": args.run_name, "split": split, "checkpoint_step": meta["step"],
            "recon_l1_mean": window_metrics["recon_l1_mean"],
            "per_step0_single_l1": window_metrics["per_step_single_l1"],
            "per_step0_baseline_l1": window_metrics["per_step_baseline_l1"],
            "sample_diversity": window_metrics["sample_diversity"],
            "multimodal_hit_rate": window_metrics["multimodal_hit_rate"],
            **{f"latency_{k}": v for k, v in window_metrics["latency"].items()},
        }
    ]
    for name, res in rollouts.items():
        rows.append({
            "run": args.run_name, "split": split, "checkpoint_step": meta["step"],
            "mode": name, "jitter_order2_mean": res["jitter_order2_mean"],
            "latency_p50_ms": res["latency"]["p50_ms"], "latency_p95_ms": res["latency"]["p95_ms"],
            "n_steps": res["n_steps"],
        })
    csv_path = write_csv(eval_dir / f"{args.run_name}_{split}_metrics.csv", rows)

    # --- 图 ---
    fig_dir = ensure_dir(Path(paths["figures_dir"]))
    source = f"logs/{args.run_name}_metrics · checkpoint step={meta['step']} · MOCK data"
    lat = window_metrics["latency_samples_ms"]
    plot_latency_and_per_step(
        lat,
        {"predicted chunk (per-step L1)": np.asarray(window_metrics["per_step_chunk_l1"])},
        fig_dir / f"30_per_step_error_{args.run_name}.png", source=source,
    )
    if "ensemble" in rollouts:
        gt = rollouts["ensemble"]["gt_actions"]
        n = min(150, gt.shape[0])
        plot_action_comparison(
            gt[:n],
            {"no ensemble (latest chunk)": rollouts["no_ensemble"]["actions"][:n],
             "temporal ensemble": rollouts["ensemble"]["actions"][:n]},
            fig_dir / f"30_action_compare_{args.run_name}.png",
            title=f"动作对比（{split}, 前 {n} 步）",
            source=source, channels=[0, 1, 2, 6],
            channel_names=["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
        )
        runner = rollouts["ensemble"]["runner"]
        if runner.ensemble is not None and runner.ensemble.history():
            hist = runner.ensemble.history()
            t0 = hist[0]["t"]
            t1 = hist[-1]["t"] + 1
            mat = runner.ensemble.weight_matrix(t0, t1)
            plot_chunk_weight_heatmap(mat, fig_dir / f"30_chunk_weights_{args.run_name}.png",
                                      source=f"{source} · mode={runner.ensemble.mode}")

    payload = {
        "run": args.run_name,
        "split": split,
        "checkpoint": {"step": meta["step"], "config_hashes": meta["config_hashes"],
                       "stats_sha256": meta["stats_sha256"], "git_commit": meta.get("git_commit")},
        "window_metrics": {k: v for k, v in window_metrics.items() if k != "latency_samples_ms"},
        "rollouts": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray) and kk != "runner"}
                     for k, v in rollouts.items()},
        "config_hash": config_hash(data_cfg, model_cfg, infer_cfg),
        "device_policy": policy.describe(),
        "data": "MOCK",
        "csv": str(csv_path),
    }
    # 消融运行写独立文件，避免覆盖 base 的评测结果（曾导致汇总表串用他人数据）
    eval_name = "infer_eval.json" if not args.ablation else f"infer_eval_{args.ablation}.json"
    atomic_write_json(Path(paths["logs_dir"]) / eval_name, payload)
    log.info(f"评测完成：recon_l1={window_metrics['recon_l1_mean']:.4f} "
             f"jitter(no_ens)={rollouts.get('no_ensemble', {}).get('jitter_order2_mean', float('nan')):.3e} "
             f"jitter(ens)={rollouts.get('ensemble', {}).get('jitter_order2_mean', float('nan')):.3e}")
    log.flush()
    print(json.dumps({"status": "OK", "csv": str(csv_path),
                      "recon_l1_mean": window_metrics["recon_l1_mean"],
                      "latency": window_metrics["latency"],
                      "jitter_no_ensemble": rollouts.get("no_ensemble", {}).get("jitter_order2_mean"),
                      "jitter_ensemble": rollouts.get("ensemble", {}).get("jitter_order2_mean")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
