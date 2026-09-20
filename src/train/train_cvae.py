"""训练入口：配置解析 → 数据/模型/损失/优化器 → 训练 → 周期验证 → checkpoint → 指标落盘。

退出码语义（脚本 wrapper 依赖它做门禁）：
    0  成功
    2  配置错误（ConfigError）
    3  数据错误（缺文件 / schema / 过滤比例超限）
    4  OOM（显存不足）
    5  训练发散（NaN/Inf）
    6  过拟合门禁未通过
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..datasets.episode_dataset import WindowDataset, collate_fn, split_paths
from ..datasets.transforms import load_stats, stats_path_for
from ..models.cvae_policy import CVAEPolicy
from ..utils.config import ConfigError, config_hash, get, load_config, load_paths
from ..utils.cuda_check import assert_cuda_available, cuda_report
from ..utils.device_profile import DevicePolicy, StepTiming
from ..utils.io_utils import atomic_write_json, ensure_dir, read_json, write_jsonl
from ..utils.logging_utils import StageLogger, banner
from ..utils.seed import make_worker_init, set_seed
from .checkpoint import git_commit, save_checkpoint
from .kl_anneal import BetaAnnealer
from .losses import build_loss_weights
from .validation import run_validation

EXIT_OK, EXIT_CONFIG, EXIT_DATA, EXIT_OOM, EXIT_DIVERGED, EXIT_GATE = 0, 2, 3, 4, 5, 6


def build_optimizer(model, train_cfg: Mapping[str, Any]):
    name = str(get(train_cfg, "optimizer.name", "adamw"))
    lr = float(get(train_cfg, "optimizer.lr", 3e-4))
    wd = float(get(train_cfg, "optimizer.weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=float(get(train_cfg, "optimizer.momentum", 0.9)))
    raise ConfigError(f"未知优化器 {name!r}")


def build_scheduler(optimizer, train_cfg: Mapping[str, Any], total_steps: int):
    name = str(get(train_cfg, "scheduler.name", "none"))
    if name == "none":
        return None
    warmup = int(get(train_cfg, "scheduler.warmup_steps", 0))
    min_ratio = float(get(train_cfg, "scheduler.min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-3, (step + 1) / warmup)
        if name == "cosine":
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "step":
            return 0.5 ** ((step - warmup) // max(1, total_steps // 4))
        raise ConfigError(f"未知调度器 {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        # 显式启用 CUDA：不可用即报错（禁止静默降级到 CPU 后仍按 GPU 口径汇报）
        report = assert_cuda_available("cuda")
        if report.get("errors"):
            raise ConfigError(f"CUDA 自检未通过: {report['errors']}")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(data_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], paths: Mapping[str, Any],
                  splits: tuple[str, ...] = ("train", "val"), seed: int = 0,
                  policy: DevicePolicy | None = None, run_name: str = "base") -> dict[str, Any]:
    """构造 DataLoader；划分只读 data/splits（禁止训练时重新随机划分）。"""
    policy = policy or DevicePolicy.from_config(train_cfg)
    loaders: dict[str, Any] = {}
    batch_size = int(get(train_cfg, "batch_size", 64))
    for split in splits:
        stem, json_path = split_paths(split, paths)
        # 消融运行使用消融专属的 processed 分片（状态维度/H 可能不同），禁止误读 base 数据
        if run_name != "base":
            candidate = Path(paths["processed_dir"]) / f"{split}.{run_name}"
            if candidate.with_name(f"{candidate.name}.state.npy").exists():
                stem = candidate
        if not stem.with_name(f"{stem.name}.state.npy").exists() or not json_path.exists():
            raise FileNotFoundError(
                f"缺少 {split} split（{stem.name}.*.npy / {json_path.name}），先运行 scripts/11_build_dataset.py"
            )
        dataset = WindowDataset(stem, data_cfg)
        generator = torch.Generator()
        generator.manual_seed(seed + (0 if split == "train" else 1000))
        kwargs = policy.dataloader_kwargs(is_train=(split == "train"))
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else min(batch_size, 64),
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            drop_last=bool(get(train_cfg, "dataloader.drop_last", True)) and split == "train",
            worker_init_fn=make_worker_init(seed) if kwargs["num_workers"] > 0 else None,
            generator=generator,
            **kwargs,
        )
    return loaders


def metrics_paths(paths: Mapping[str, Any], run_name: str) -> dict[str, Path]:
    logs = Path(paths["logs_dir"])
    if run_name == "base":
        return {
            "train": logs / "train_metrics.jsonl",
            "val": logs / "val_metrics.jsonl",
            "summary": logs / "train_summary.json",
        }
    out = ensure_dir(logs / "ablation")
    return {
        "train": out / f"{run_name}_train_metrics.jsonl",
        "val": out / f"{run_name}_val_metrics.jsonl",
        "summary": out / f"{run_name}_summary.json",
    }


def train(args: argparse.Namespace) -> int:
    paths = load_paths()
    overrides = args.override or []
    data_cfg = load_config("data", ablation=args.ablation, overrides=overrides)
    model_cfg = load_config("model", ablation=args.ablation, overrides=overrides)
    train_cfg = load_config("train", ablation=args.ablation, overrides=overrides)
    if args.max_steps:
        train_cfg["max_steps"] = int(args.max_steps)
    if args.batch_size:
        train_cfg["batch_size"] = int(args.batch_size)

    run_name = args.run_name or (args.ablation or "base")
    log = StageLogger("train_cvae", paths["logs_dir"], stage="20", jsonl_name=metrics_paths(paths, run_name)["train"].name)
    print(banner(f"S5 训练：run={run_name} ablation={args.ablation or 'base'}"))

    hashes = {
        "data": config_hash(data_cfg),
        "model": config_hash(model_cfg),
        "train": config_hash(train_cfg),
        "data+model": config_hash(data_cfg, model_cfg),
    }
    seed = int(get(train_cfg, "seed", 0))
    seed_state = set_seed(seed, deterministic=bool(get(train_cfg, "deterministic", False)))
    device = resolve_device(str(get(train_cfg, "device", "auto")))
    cuda_info = cuda_report() if device.type == "cuda" else {"requested": False, "device": "cpu"}
    log.info(f"device={device} seed={seed} config_hashes={hashes}",
             stage="20", run=run_name, hashes=hashes, seed_state={k: str(v) for k, v in seed_state.items()})
    log.info("CUDA 自检：" + json.dumps(
        {k: v for k, v in cuda_info.items() if k != "arch_list"}, ensure_ascii=False))

    # 统计路径必须与数据构建一致（消融运行使用消融专属统计），否则训练/评测口径会错配
    stats_path = stats_path_for(paths, run_name)
    stats = load_stats(stats_path, data_cfg)
    policy = DevicePolicy.from_config(train_cfg)
    policy.apply_thread_limits()
    timing = StepTiming()
    loaders = build_loaders(data_cfg, train_cfg, paths, ("train", "val"), seed, policy, run_name)
    model = CVAEPolicy(data_cfg, model_cfg).to(device)
    log.info(f"模型参数量 {model.describe()['params_trainable']}，状态维度 {model.state_dim}，H={model.horizon}",
             **model.describe())
    optimizer = build_optimizer(model, train_cfg)
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * int(get(train_cfg, "epochs", 1))
    if train_cfg.get("max_steps"):
        total_steps = min(total_steps, int(train_cfg["max_steps"]))
    scheduler = build_scheduler(optimizer, train_cfg, total_steps)
    annealer = BetaAnnealer.from_config(train_cfg)
    weights = build_loss_weights(train_cfg, model.horizon)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(get(train_cfg, "amp", False)) and device.type == "cuda")
    log.info("CPU/GPU 分工策略：" + json.dumps(policy.describe(), ensure_ascii=False))

    # ---- 过拟合门禁模式 --------------------------------------------------------------
    if args.overfit:
        return run_overfit_gate(args, model, optimizer, loaders["train"], train_cfg, weights, annealer,
                                device, paths, log, run_name)

    mpaths = metrics_paths(paths, run_name)
    ckpt_dir = ensure_dir(Path(paths["checkpoints_dir"]) / run_name)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    step = 0
    best_value = float("inf") if str(get(train_cfg, "checkpoint.mode", "min")) == "min" else -float("inf")
    epochs_without_improve = 0
    monitor = str(get(train_cfg, "checkpoint.monitor", "val/total"))
    started = time.time()
    status = "OK"

    for epoch in range(1, int(get(train_cfg, "epochs", 1)) + 1):
        model.train()
        data_iter = iter(loaders["train"])
        while True:
            # 数据等待时间：反映 CPU 侧（worker 读取 + 归一化 + collate）能否喂饱 GPU
            t_data = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            data_ms = (time.perf_counter() - t_data) * 1000.0
            step += 1
            measure = (step % policy.timing_every == 0)
            t_phase = time.perf_counter()
            batch = {k: policy.to_device(v, device) for k, v in batch.items()}
            h2d_ms = (time.perf_counter() - t_phase) * 1000.0
            t_phase = time.perf_counter()
            beta = annealer.value(step)
            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    losses = model.compute_loss(batch, beta=beta, weights=weights.as_dict(),
                                                need_smooth=weights.smooth_enabled)
                total = losses["total"]
                if not torch.isfinite(total):
                    log.error(f"step {step} 出现 NaN/Inf，立即中止（不得用 0 掩盖）", step=step,
                              recon=float(losses["recon"]), kl=float(losses["kl"]))
                    status = "DIVERGED"
                    break
                if scaler.is_enabled():
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                     float(get(train_cfg, "grad_clip", 1.0))))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                     float(get(train_cfg, "grad_clip", 1.0))))
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if measure:
                    if device.type == "cuda":
                        torch.cuda.synchronize()   # 精确计时点：确保计算真的完成再读秒
                    timing.add(data_ms=data_ms, h2d_ms=h2d_ms,
                               compute_ms=(time.perf_counter() - t_phase) * 1000.0)
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - 依赖硬件
                log.error(f"CUDA OOM: {exc}。按【十】5 阶梯降低 batch/分辨率后重试", step=step)
                torch.cuda.empty_cache()
                return EXIT_OOM

            rec = {
                "stage": "20", "run": run_name, "epoch": epoch, "step": step,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "beta": beta,
                "loss/total": float(total.detach()),
                "loss/recon": float(losses["recon"].detach()),
                "loss/kl": float(losses["kl"].detach()),
                "loss/smooth": float(losses["smooth"].detach()),
                "grad_norm": grad_norm,
                "config_hash": hashes["train"],
            }
            train_records.append(rec)
            if step % 10 == 0 or step == 1:
                log.metric_now(message=f"epoch {epoch} step {step}", **rec)
            if train_cfg.get("max_steps") and step >= int(train_cfg["max_steps"]):
                break
        if status == "DIVERGED" or (train_cfg.get("max_steps") and step >= int(train_cfg["max_steps"])):
            break

        every_val = int(get(train_cfg, "validation.every_n_epochs", 1))
        if epoch % every_val == 0:
            val = run_validation(model, loaders["val"], annealer.value(step), weights, device,
                                 batch_limit=int(get(train_cfg, "validation.batch_limit", 20)))
            val.update({"epoch": epoch, "step": step, "run": run_name, "stage": "20"})
            val_records.append(val)
            log.metric_now(message=f"epoch {epoch} 验证", **val)
            current = float(val.get(monitor, float("nan")))
            mode = str(get(train_cfg, "checkpoint.mode", "min"))
            improved = current < best_value if mode == "min" else current > best_value
            if improved and np.isfinite(current):
                best_value = current
                epochs_without_improve = 0
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, step=step, epoch=epoch,
                                config_hashes=hashes, stats_path=stats_path, seed=seed,
                                metrics={monitor: best_value}, data_version=stats.get("data_version", "unknown"))
            else:
                epochs_without_improve += 1

        every_ckpt = int(get(train_cfg, "checkpoint.every_n_epochs", 5))
        if epoch % every_ckpt == 0:
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step=step, epoch=epoch,
                            config_hashes=hashes, stats_path=stats_path, seed=seed,
                            metrics={"train/total": float(train_records[-1]["loss/total"])},
                            data_version=stats.get("data_version", "unknown"))

        patience = int(get(train_cfg, "checkpoint.early_stop_patience", 1000))
        if epochs_without_improve >= patience:
            log.warning(f"早停：连续 {epochs_without_improve} 次验证无提升（monitor={monitor}）")
            break

    # 收尾：至少产生 last 与 best
    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, step=step,
                    epoch=int(get(train_cfg, "epochs", 1)), config_hashes=hashes, stats_path=stats_path,
                    seed=seed, metrics={"train/total": float(train_records[-1]["loss/total"]) if train_records else None},
                    data_version=stats.get("data_version", "unknown"))
    if not (ckpt_dir / "best.pt").exists():
        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, step=step, epoch=1,
                        config_hashes=hashes, stats_path=stats_path, seed=seed, metrics={},
                        data_version=stats.get("data_version", "unknown"))

    write_jsonl(mpaths["train"], train_records)
    write_jsonl(mpaths["val"], val_records)
    summary = {
        "status": status,
        "run": run_name,
        "ablation": args.ablation,
        "config_hashes": hashes,
        "seed": seed,
        "device": str(device),
        "cuda": cuda_info,
        "git_commit": git_commit(),
        "steps": step,
        "epochs": int(get(train_cfg, "epochs", 1)),
        "duration_s": time.time() - started,
        "final_train_loss": train_records[-1]["loss/total"] if train_records else None,
        "best_val": best_value if np.isfinite(best_value) else None,
        "monitor": monitor,
        "checkpoint_dir": str(ckpt_dir),
        "metrics": {"train": str(mpaths["train"]), "val": str(mpaths["val"])},
        "model": model.describe(),
        "beta_final": annealer.value(step),
        "params": train_cfg,
        "data": {"state_dim": model.state_dim, "horizon": model.horizon,
                 "slam_input": bool(get(data_cfg, "slam_input.enabled", True))},
    }
    atomic_write_json(mpaths["summary"], summary)
    # CPU/GPU 分工实测：把"谁在等谁"写成可核查的数字，并给出瓶颈判定与调参建议
    usage = timing.summary(warn_ratio=float(get(train_cfg, "device_policy.gpu_idle_warn_ratio", 0.30)))
    usage.update({"policy": policy.describe(), "device": str(device), "cuda": cuda_info,
                  "steps_total": step, "steps_measured": timing.n,
                  "config_hash": hashes["train"], "run": run_name, "seed": seed})
    atomic_write_json(Path(paths["logs_dir"]) / "20_device_usage.json", usage)
    if usage.get("n_steps"):
        log.info(f"CPU/GPU 比例：data={usage['data_ratio']:.1%} h2d={usage['h2d_ratio']:.1%} "
                 f"compute={usage['compute_ratio']:.1%} → {usage['verdict']}（{usage['advice']}）")
        summary["device_usage"] = {k: v for k, v in usage.items() if k != "policy"}
    # 训练曲线（提示词 S5 门禁要求 outputs/figures/loss_curve.png 由训练阶段直接产出）
    try:
        from ..utils.viz import plot_device_split, plot_loss_curves

        fig = plot_loss_curves(
            train_records,
            Path(paths["figures_dir"]) / ("loss_curve.png" if run_name == "base" else f"loss_curve_{run_name}.png"),
            title=f"训练曲线 · run={run_name} · ablation={args.ablation or 'base'}",
            source=f"logs/{mpaths['train'].name} · seed={seed} · hash={hashes['train']} · data=MOCK",
        )
        summary["loss_curve"] = str(fig)
        if usage.get("n_steps"):
            fig2 = plot_device_split(
                timing.series(), usage,
                Path(paths["figures_dir"]) / ("device_split.png" if run_name == "base"
                                              else f"device_split_{run_name}.png"),
                source=f"logs/20_device_usage.json · run={run_name} · workers={policy.num_workers}",
            )
            summary["device_split_figure"] = str(fig2)
        atomic_write_json(mpaths["summary"], summary)
    except Exception as exc:  # 图失败不能掩盖训练结果，但必须显式记录
        log.warning(f"训练曲线绘制失败（{exc}），请用 scripts/90_report.py 重新生成")
    log.flush()
    log.info(f"训练完成 status={status} steps={step} best_{monitor}={best_value}")
    return EXIT_OK if status == "OK" else EXIT_DIVERGED


def run_overfit_gate(args: argparse.Namespace, model, optimizer, train_loader, train_cfg, weights,
                     annealer, device, paths, log: StageLogger, run_name: str) -> int:
    """单 batch 过拟合门禁：固定一个 batch，要求 loss 显著下降，否则禁止进入正式训练。"""
    gate = dict(get(train_cfg, "overfit_gate", {}) or {})
    steps = int(gate.get("steps", 200))
    gate_batch = int(gate.get("batch_size", 16))
    threshold = float(gate.get("loss_drop_threshold", 0.5))
    log_every = int(gate.get("log_every", 50))
    batch = next(iter(train_loader))
    # 只取 overfit_gate.batch_size 条样本：门禁考察的是"能否记住一个小批量"，
    # 直接用整个 train batch（128 条）在 300 步内不可能拟合，会让门禁失去意义。
    batch = {k: (v[:gate_batch] if isinstance(v, (torch.Tensor, list)) else v) for k, v in batch.items()}
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    # 注意：LambdaLR 在构造时就会调用一次 lr_lambda(0)，把 LR 变成 base_lr * warmup 比例
    # （本项目 warmup_steps=50 → 1/50）。单 batch 门禁不做 warmup，必须显式恢复基准学习率，
    # 否则门禁会在 2e-5 的学习率下"假失败"。
    base_lr = float(get(train_cfg, "optimizer.lr", 1e-3))
    for group in optimizer.param_groups:
        group["lr"] = base_lr
    records: list[dict[str, Any]] = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = model.compute_loss(batch, beta=annealer.value(step), weights=weights.as_dict())
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(get(train_cfg, "grad_clip", 1.0)))
        optimizer.step()
        records.append({
            "step": step, "loss/total": float(losses["total"].detach()),
            "loss/recon": float(losses["recon"].detach()), "loss/kl": float(losses["kl"].detach()),
            "beta": annealer.value(step),
        })
        if step % log_every == 0 or step == 1:
            log.metric_now(message=f"overfit step {step}", run=run_name, **records[-1])
    first = float(np.mean([r["loss/total"] for r in records[: max(1, steps // 10)]]))
    last = float(np.mean([r["loss/total"] for r in records[-max(1, steps // 10):]]))
    passed = last <= first * (1.0 - threshold)
    out = {
        "gate": "single_batch_overfit",
        "run": run_name,
        "steps": steps,
        "batch_size": int(batch["images"].shape[0]),
        "loss_first_window": first,
        "loss_last_window": last,
        "relative_drop": (first - last) / first if first else 0.0,
        "threshold": threshold,
        "status": "PASS" if passed else "FAIL",
        "records": records,
    }
    atomic_write_json(Path(paths["logs_dir"]) / "21_overfit_gate.json", out)
    log.flush()
    log.info(f"单 batch 过拟合门禁：{out['status']}（{first:.6g} → {last:.6g}，阈值下降 {threshold:.0%}）")
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, ensure_ascii=False, indent=2))
    return EXIT_OK if passed else EXIT_GATE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVAE + 动作分块策略训练")
    parser.add_argument("--config", default="train", help="主配置名（默认 train）")
    parser.add_argument("--ablation", default=None, help="configs/ablation/<name>.yaml 的消融名")
    parser.add_argument("--override", action="append", default=[], help="key=value 覆盖，可重复")
    parser.add_argument("--run-name", default=None, help="运行名（决定 checkpoint 与日志路径）")
    parser.add_argument("--max-steps", type=int, default=0, help="限制总步数（smoke / 复现实验用）")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--overfit", action="store_true", help="只跑单 batch 过拟合门禁")
    parser.add_argument("--resume", default=None, help="从 checkpoint 继续（仅恢复权重/优化器）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return train(args)
    except ConfigError as exc:
        print(json.dumps({"status": "CONFIG_ERROR", "message": str(exc)}, ensure_ascii=False))
        return EXIT_CONFIG
    except (FileNotFoundError, KeyError) as exc:
        print(json.dumps({"status": "DATA_ERROR", "message": str(exc)}, ensure_ascii=False))
        return EXIT_DATA


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
