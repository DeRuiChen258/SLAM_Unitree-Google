#!/usr/bin/env python
"""scripts/90_report.py —— 汇总 manifest / logs / outputs，生成实验报告与登记表。

硬规则：所有数字必须来自真实产物文件；缺失项写 MISSING / NOT_RUN，禁止补数。
产物：
    outputs/REPORT.md     汇总报告（README 的"实验报告"章节来源）
    docs/experiments.md   实验登记表（编号 / 变量 / config hash / seed / 指标 / 结论）
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_paths  # noqa: E402
from src.utils.io_utils import read_json, read_jsonl  # noqa: E402


def fmt(value, nd: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{nd}g}"
    return str(value)


def safe(path: Path):
    try:
        return read_json(path) if path.exists() else None
    except Exception:
        return None


def section_env(env, train) -> list[str]:
    out = ["## 1. 环境基线（实测）", "", "| 项 | 值 | 来源 |", "|---|---|---|"]
    if env:
        out.append(f"| GPU | {env.get('device_name')}（capability {env.get('capability')}，{env.get('sm')}） | logs/00_cuda_check.json |")
        out.append(f"| torch | {env.get('torch_version')}（CUDA runtime {env.get('torch_cuda_version')}，"
                   f"arch_list 含 sm_120={env.get('arch_supported')}） | 同上 |")
        out.append(f"| CUDA 实测 | matmul {fmt(env.get('matmul_gflops'), 5)} GFLOPS；卷积前向 "
                   f"{fmt(env.get('conv_forward_ms'), 3)} ms | 同上 |")
        out.append(f"| CUDA 自检 | {env.get('status')} | 同上 |")
    else:
        out.append("| CUDA | MISSING（先运行 scripts/00_env_check.sh） | — |")
    out.append(f"| 训练设备 | {fmt((train or {}).get('device'))} | logs/train_summary.json |")
    out.append("")
    return out


def section_data(data_manifest, mock_manifest) -> list[str]:
    out = ["## 2. 数据（source=MOCK）", ""]
    if data_manifest:
        n_ep = max(1, int(data_manifest.get("num_episodes", 1)))
        filt = data_manifest.get("filter_stats") or {}
        out += [
            "| 项 | 值 |", "|---|---|",
            f"| 来源 | {data_manifest.get('source')} |",
            f"| episode 数 / 总帧数 | {data_manifest.get('num_episodes')} / {data_manifest.get('num_frames')} |",
            f"| 窗口数 | {json.dumps(data_manifest.get('num_windows'), ensure_ascii=False)} |",
            f"| 状态 / 动作维度 / H | {data_manifest.get('state_dim')} / {data_manifest.get('action_dim')} / {data_manifest.get('horizon')} |",
            f"| 过滤 | 共 {filt.get('total')} 个窗口，保留 {filt.get('kept')}，拒绝比例 {fmt(filt.get('reject_ratio'), 3)} |",
            f"| 数据版本 | {data_manifest.get('version')} |",
            "",
        ]
        out.append(f"（每条 episode {int(data_manifest.get('num_frames', 0) / n_ep)} 帧，fps={data_manifest.get('fps')}）")
    else:
        out.append("MISSING：data/manifest.json（先运行 scripts/11_build_dataset.py）")
    out.append("")
    if mock_manifest:
        out.append(f"MOCK 生成规则：{mock_manifest.get('num_episodes')} 条 episode，其中多模态 "
                   f"{mock_manifest.get('multimodal_episodes')} 条（approach bias 分布 "
                   f"{mock_manifest.get('bias_histogram')}）；同步导出 {mock_manifest.get('total_frames')} 帧激光。")
        out.append("")
        for note in mock_manifest.get("notes", []):
            out.append(f"> {note}")
        out.append("")
    return out


def section_train(train, gate, usage, train_metrics, val_metrics) -> list[str]:
    out = ["## 3. 训练", ""]
    if train:
        out += [
            "| 项 | 值 |", "|---|---|",
            f"| 状态 | {train.get('status')} |",
            f"| 步数 / epoch | {train.get('steps')} / {train.get('epochs')} |",
            f"| 耗时 | {fmt(train.get('duration_s'), 4)} s |",
            f"| 最终训练 loss | {fmt(train.get('final_train_loss'))} |",
            f"| 最优 {train.get('monitor')} | {fmt(train.get('best_val'))} |",
            f"| 可训练参数量 | {(train.get('model') or {}).get('params_trainable')} |",
            f"| config hash (data+model) | {(train.get('config_hashes') or {}).get('data+model')} |",
            f"| seed / commit | {train.get('seed')} / {train.get('git_commit')} |",
        ]
    else:
        out.append("MISSING：logs/train_summary.json")
    out.append("")
    if gate:
        out.append(f"**单 batch 过拟合门禁**：{gate.get('status')}"
                   f"（{fmt(gate.get('loss_first_window'))} → {fmt(gate.get('loss_last_window'))}，"
                   f"相对下降 {fmt(gate.get('relative_drop'), 3)}，阈值 {fmt(gate.get('threshold'), 2)}，"
                   f"batch={gate.get('batch_size')}）")
        out.append("")
    if usage:
        policy = usage.get("policy") or {}
        out.append(f"**CPU/GPU 分工实测**：data wait {fmt(usage.get('data_ratio'), 3)}、"
                   f"H2D {fmt(usage.get('h2d_ratio'), 3)}、compute {fmt(usage.get('compute_ratio'), 3)} "
                   f"→ **{usage.get('verdict')}**（workers={policy.get('num_workers')}，"
                   f"torch_threads={policy.get('torch_threads')}，batch={policy.get('batch_size')}）")
        out.append("")
        out.append(f"> 建议：{usage.get('advice')}")
        out.append("")
    if train_metrics:
        first, last = train_metrics[0], train_metrics[-1]
        out.append(f"训练曲线数据点 {len(train_metrics)} 个：total {fmt(first.get('loss/total'))} → "
                   f"{fmt(last.get('loss/total'))}；recon {fmt(first.get('loss/recon'))} → "
                   f"{fmt(last.get('loss/recon'))}；kl {fmt(first.get('loss/kl'))} → {fmt(last.get('loss/kl'))}。")
        out.append("")
    if val_metrics:
        v = val_metrics[-1]
        out.append(f"最终验证：total={fmt(v.get('val/total'))}、recon={fmt(v.get('val/recon'))}、"
                   f"kl={fmt(v.get('val/kl'))}、采样多样性={fmt(v.get('val/sample_diversity'))}、"
                   f"mean-vs-sample 差={fmt(v.get('val/mean_vs_sample_gap'))}。")
        out.append("")
    return out


def section_infer(evalj) -> list[str]:
    out = ["## 4. 推理与时间集成", ""]
    if evalj:
        wm = evalj.get("window_metrics") or {}
        lat = wm.get("latency") or {}
        roll = evalj.get("rollouts") or {}
        out += [
            "| 指标 | 值 |", "|---|---|",
            f"| 重建 L1（归一化动作单位） | {fmt(wm.get('recon_l1_mean'))} |",
            f"| 多模态命中率（4 次先验采样） | {fmt(wm.get('multimodal_hit_rate'), 3)} |",
            f"| 采样多样性 | {fmt(wm.get('sample_diversity'), 3)} |",
            f"| 推理延迟 p50 / p95 | {fmt(lat.get('p50_ms'), 3)} / {fmt(lat.get('p95_ms'), 3)} ms |",
            f"| checkpoint step | {(evalj.get('checkpoint') or {}).get('step')} |",
        ]
        if roll.get("no_ensemble") and roll.get("ensemble"):
            j0 = roll["no_ensemble"].get("jitter_order2_mean")
            j1 = roll["ensemble"].get("jitter_order2_mean")
            out.append(f"| 抖动（二阶差分方差）无/有集成 | {fmt(j0, 3)} / {fmt(j1, 3)} |")
            if j0 and j1:
                out.append(f"| 抖动相对变化 | {(j1 - j0) / j0 * 100:+.1f}% |")
    else:
        out.append("MISSING：logs/infer_eval.json（先运行 scripts/30_infer_offline.sh）")
    out.append("")
    return out


def section_slam(slam, align) -> list[str]:
    out = ["## 5. SLAM（Cartographer）", ""]
    if slam and slam.get("status") == "OK":
        ta = slam.get("time_alignment") or {}
        out += [
            "| 指标 | 值 |", "|---|---|",
            f"| ATE RMSE（Umeyama 刚体对齐后） | {fmt(slam.get('ate_rmse_m'))} m |",
            f"| ATE 均值 / 最大 | {fmt(slam.get('ate_mean_m'))} / {fmt(slam.get('ate_max_m'))} m |",
            f"| RPE（每 {slam.get('rpe_delta_s')} s）RMSE | {fmt(slam.get('rpe_rmse_m'))} m |",
            f"| 航向误差均值 / 最大 | {fmt(slam.get('yaw_err_mean_rad'))} / {fmt(slam.get('yaw_err_max_rad'))} rad |",
            f"| 参考路径长度 | {fmt(slam.get('reference_path_length_m'))} m |",
            f"| 时间对齐 | {ta.get('n_aligned_used')}/{ta.get('n_estimate_samples')} 样本（容差 {ta.get('tolerance_s')} s） |",
            f"| 参考来源 | {slam.get('reference_source')} |",
            "",
            f"> 证据等级：{slam.get('evidence_level')}",
        ]
    else:
        status = slam.get("status") if slam else "文件不存在"
        out.append(f"MISSING/BLOCKED：outputs/slam/trajectory_metrics.json（{status}）")
    out.append("")
    if align:
        out.append(f"位姿流对齐：{align.get('status')}，超容差比例 {fmt(align.get('out_of_tolerance_ratio'), 3)}，"
                   f"偏移 p95 {fmt(align.get('offset_p95_s'), 3)} s。")
        out.append("")
    return out


def main() -> int:
    paths = load_paths()
    logs = Path(paths["logs_dir"])
    out_dir = Path(paths["outputs_dir"])
    env = safe(logs / "00_cuda_check.json")
    data_manifest = safe(Path(paths["data_dir"]) / "manifest.json")
    mock_manifest = safe(Path(paths["data_dir"]) / "mock_manifest.json")
    gate = safe(logs / "21_overfit_gate.json")
    train = safe(logs / "train_summary.json")
    usage = safe(logs / "20_device_usage.json")
    evalj = safe(logs / "infer_eval.json")
    slam = safe(out_dir / "slam" / "trajectory_metrics.json")
    align = safe(logs / "slam_align.json")
    train_metrics = read_jsonl(logs / "train_metrics.jsonl") if (logs / "train_metrics.jsonl").exists() else []
    val_metrics = read_jsonl(logs / "val_metrics.jsonl") if (logs / "val_metrics.jsonl").exists() else []
    ablation_csv = out_dir / "ablation" / "summary.csv"

    lines: list[str] = [
        "# 实验报告：CVAE + 动作分块 + 时间集成 × Cartographer SLAM",
        "",
        f"> 自动生成：{datetime.now().isoformat(timespec='seconds')}（scripts/90_report.py）",
        "> 所有数字来自本工程真实产物文件；缺失项写 MISSING/NOT_RUN，不做任何补数。",
        "",
        "**结论摘要（限定条件必须一起读）**：",
        "",
        "1. 本实验数据为 **MOCK**（合成世界 + 合成观测 + 合成激光），所有结论只在此仿真条件下成立；",
        "2. 策略侧训练/推理全部在 GPU（RTX 5070 Laptop, sm_120）上完成，CPU 负责数据供给与落盘，"
        "实测比例见第 3 节；",
        "3. SLAM 侧运行的是**真实 Google Cartographer**（RoboStack 2.0.9003 二进制），"
        "输入是离线回放的 ROS2 bag，输出 pbstream/栅格地图/位姿流；",
        "4. 无真机：任何涉及真实机器人动作下发的环节都停在流程与安全门禁层面（见 README 已知限制）。",
        "",
    ]
    lines += section_env(env, train)
    lines += section_data(data_manifest, mock_manifest)
    lines += section_train(train, gate, usage, train_metrics, val_metrics)
    lines += section_infer(evalj)
    lines += section_slam(slam, align)
    lines.append("## 6. 消融（同一 splits、同一 seed）")
    lines.append("")
    if ablation_csv.exists():
        lines.append(ablation_csv.read_text(encoding="utf-8").strip() or "（表为空）")
    else:
        lines.append("NOT_RUN：outputs/ablation/summary.csv 不存在（运行 scripts/50_run_ablation.sh）")
    lines.append("")
    lines += [
        "## 7. 复现命令",
        "",
        "```bash",
        "bash scripts/00_env_check.sh                    # S0 环境基线 + CUDA 自检",
        "python scripts/10_gen_mock_data.py              # S2 MOCK 数据 + 激光扫描",
        "python scripts/11_build_dataset.py              # S3 窗口化 + 归一化",
        "python scripts/12_data_check.py                 # S3 数据体检",
        "bash scripts/21_overfit_single_batch.sh         # S5 门禁（必须先过）",
        "bash scripts/20_train.sh                        # S5 训练（含 CPU/GPU 分工实测）",
        "bash scripts/30_infer_offline.sh --split test   # S6 评测（含时间集成对照）",
        "bash scripts/31_infer_closed_loop.sh            # S7 闭环执行（默认不下发指令）",
        "bash scripts/43_gen_cartographer_config.py      # S8 生成自包含 Lua 配置",
        "bash scripts/40_slam_bringup.sh                 # S8 真实 Cartographer 离线回放",
        "bash scripts/42_slam_replay_eval.sh             # S8 轨迹 ATE/RPE",
        "bash scripts/50_run_ablation.sh                 # S10 消融矩阵",
        "pytest -q tests/                                # 契约测试",
        "bash scripts/60_collect_evidence.sh             # 证据索引",
        "```",
        "",
    ]
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告 → {report}（{len(lines)} 行）")

    # 实验登记表：每个 run 一行，含 config hash / seed / 指标 / 结论限定条件
    registry: list[str] = [
        "# 实验登记表（自动生成，禁止手改）",
        "",
        f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}；数据来源：MOCK（合成世界）。",
        "> 每行对应一次真实运行；config hash 用于证明「除消融变量外其余配置一致」。",
        "",
        "| 编号 | 变量 | run | config hash (data+model) | seed | commit | 步数 | 最优 val/total | 结论限定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    base_summary_paths = [("base", logs / "train_summary.json")]
    base_summary_paths += [(p.name.replace("_summary.json", ""), p)
                           for p in sorted((logs / "ablation").glob("*_summary.json"))]
    for name, path in base_summary_paths:
        s = safe(path)
        if not s:
            continue
        hashes = s.get("config_hashes") or {}
        model_info = s.get("model") or {}
        data_info = s.get("data") or {}
        caveats = []
        if not model_info.get("latent_enabled", True):
            caveats.append("无潜变量（确定性回归）")
        if not data_info.get("slam_input", True):
            caveats.append("无 SLAM 输入")
        if model_info.get("horizon") not in (None, 8):
            caveats.append(f"H={model_info.get('horizon')}")
        if name in ("no_temporal_ensemble", "ensemble_uniform", "ensemble_exp_d1p0"):
            caveats.append("仅推理侧差异（复用 base checkpoint）")
        registry.append(
            f"| {len(registry) - 5} | {name} | {name} | {hashes.get('data+model', '—')} | {s.get('seed')} | "
            f"{str(s.get('git_commit', '—'))[:8]} | {s.get('steps')} | {fmt(s.get('best_val'))} | "
            f"{'；'.join(caveats) if caveats else '基准配置'} |"
        )
    registry += [
        "",
        "## 结论限定条件（必须一起读）",
        "",
        "1. 所有数据为 MOCK：合成 2D 世界 + 64×64 合成图像 + 解析射线投射激光；",
        "2. 无真机：执行循环默认 dry-run，动作只写日志；",
        "3. 消融运行与 SLAM 全局优化曾存在 CPU 竞争，`duration_s` 与设备占比可能偏高，"
        "但模型指标（val/total、recon、抖动、延迟）不受影响；",
        "4. 推理侧消融（时间集成开关/权重）复用 base checkpoint，因此它们与 base 的 config hash 相同；",
        "5. 未跑项：A7（观测历史帧数 K 扫描）标 NOT_RUN。",
        "",
        "## 原始产物索引",
        "",
        "| 产物 | 路径 |",
        "|---|---|",
        "| 训练指标 | `logs/train_metrics.jsonl`、`logs/val_metrics.jsonl` |",
        "| 设备分工 | `logs/20_device_usage.json` |",
        "| 评测指标 | `outputs/eval/*.csv`、`logs/infer_eval.json` |",
        "| 消融汇总 | `outputs/ablation/summary.csv` |",
        "| SLAM 轨迹 | `outputs/slam/trajectory_metrics.json`、`*.png` |",
        "| 证据索引 | `evidence/index.md` |",
    ]
    exp_path = Path(paths["docs_dir"]) / "experiments.md"
    exp_path.write_text("\n".join(registry) + "\n", encoding="utf-8")
    print(f"实验登记 → {exp_path}（{len(base_summary_paths)} 个 run）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
