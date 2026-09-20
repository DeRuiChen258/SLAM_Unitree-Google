#!/usr/bin/env python
"""scripts/52_slam_input_usage_probe.py —— 诊断"模型是否真的在使用 SLAM 位姿输入"。

为什么需要：A4 消融（no_slam_input）的指标与 base 接近，可能有两种完全不同的原因：
    (a) 位姿输入对当前 MOCK 任务确实不重要；
    (b) 模型把 slam_pose 块学成了"死输入"（权重不敏感），此时 A4 的对照没有信息量。
两者对报告的含义完全不同，必须用**受控扰动实验**区分：
    对同一个已训练模型，在推理时把 slam_pose 块替换为 ①零值 ②batch 内乱序值 ③加噪，
    观察验证集重建误差是否显著变化：
        * 误差显著变大 → 模型依赖该输入（原因 a 成立，A4 的"无差异"是真结论）；
        * 误差几乎不变 → 该输入实际上是死输入（原因 b，必须在报告中说明）。

产物：logs/52_slam_input_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.episode_dataset import WindowDataset, collate_fn, split_paths  # noqa: E402
from src.datasets.schema import state_spec  # noqa: E402
from src.infer.offline_eval import load_model_and_stats  # noqa: E402
from src.models.losses import masked_recon_loss  # noqa: E402
from src.utils.config import load_config, load_paths  # noqa: E402
from src.utils.io_utils import atomic_write_json  # noqa: E402
from src.utils.logging_utils import StageLogger  # noqa: E402


@torch.no_grad()
def recon_with_perturbation(model, loader, device, blocks: dict[str, slice], mode: str,
                            n_batches: int = 8, seed: int = 0) -> float:
    """在指定扰动下计算验证集重建 L1（与 30_infer_offline 同口径）。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errors: list[float] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        images = batch["images"].to(device)
        state = batch["state"].clone()
        sl = blocks["slam_pose"]
        if mode == "zero":
            state[:, sl] = 0.0
        elif mode == "shuffle":
            perm = torch.randperm(state.shape[0], generator=gen)
            state[:, sl] = state[perm][:, sl]
        elif mode == "noise":
            state[:, sl] = state[:, sl] + torch.randn(state[:, sl].shape, generator=gen) * 1.0
        state = state.to(device)
        pred = model.predict_chunk(images, state, mode="mean").cpu()
        errors.append(float(masked_recon_loss(pred, batch["action_chunk"], batch["mask"], "l1")[0]))
    return float(np.mean(errors)) if errors else float("nan")


def main() -> int:
    paths = load_paths()
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    infer_cfg = load_config("infer")
    log = StageLogger("slam_input_probe", paths["logs_dir"], stage="52")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, stats, meta, _ = load_model_and_stats(paths, data_cfg, model_cfg, infer_cfg, "base", device)
    model.eval()
    stem, _ = split_paths("val", paths)
    dataset = WindowDataset(stem, data_cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    blocks = {name: slice(start, end) for name, start, end in state_spec(data_cfg).blocks}
    if "slam_pose" not in blocks:
        log.error("当前配置没有 slam_pose 块（是不是用了 no_slam_input 消融？本探针只对 base 有意义）")
        return 3

    baseline = recon_with_perturbation(model, loader, device, blocks, "none")
    results = {"baseline": baseline}
    for mode in ("zero", "shuffle", "noise"):
        results[mode] = recon_with_perturbation(model, loader, device, blocks, mode)
    deltas = {k: (v - baseline) / baseline for k, v in results.items() if k != "baseline"}
    verdict = "USED" if max(deltas.values()) > 0.05 else "DEAD_INPUT"
    payload = {
        "checkpoint_step": meta["step"],
        "state_dim": int(state_spec(data_cfg).dim),
        "slam_pose_slice": [blocks["slam_pose"].start, blocks["slam_pose"].stop],
        "recon_l1": results,
        "relative_change": deltas,
        "verdict": verdict,
        "interpretation": (
            "扰动 slam_pose 后误差显著上升 → 模型依赖该输入，A4 的『无差异』可解释为任务本身对该输入不敏感"
            if verdict == "USED" else
            "扰动 slam_pose 后误差几乎不变 → 该输入实际是死输入（权重不敏感）。"
            "此时 A4 的对照没有信息量，必须在报告中说明，而不能宣称『SLAM 输入无收益』"
        ),
        "note": "本探针只做受控扰动诊断，不改变模型权重；误差统计口径与 30_infer_offline 一致（masked L1）",
    }
    atomic_write_json(Path(paths["logs_dir"]) / "52_slam_input_probe.json", payload)
    log.info(f"SLAM 输入使用诊断：{verdict}；baseline={baseline:.4f}，"
             f"相对变化={ {k: round(v, 4) for k, v in deltas.items()} }")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
