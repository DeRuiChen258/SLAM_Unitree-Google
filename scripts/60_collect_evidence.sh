#!/usr/bin/env bash
# scripts/60_collect_evidence.sh —— 证据采集：把每阶段产物与日志配对，生成 evidence/index.md。
#
# 硬规则：缺失的证据项直接标 MISSING；截图工具不可用时显式写 SCREENSHOT_UNAVAILABLE，不得伪造图片。
# 退出码：0=全部证据项存在；1=有缺失（用于门禁）。
set -uo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLAM_ROOT}"
OUT="${SLAM_ROOT}/evidence"
mkdir -p "${OUT}"
index="${OUT}/index.md"
missing=0

{
echo "# 证据索引"
echo
echo "> 由 \`scripts/60_collect_evidence.sh\` 生成 @ $(date -Is)。"
echo "> 每条证据 = 命令 + 产物文件 + 关键日志；缺失项标 MISSING，禁止伪造。"
echo
echo "| 阶段 | 命令 | 关键产物 | 关键日志 | 说明 |"
echo "|---|---|---|---|---|"

row() {
  local stage="$1" cmd="$2" artifact="$3" logfile="$4" note="$5"
  local a="MISSING" l="MISSING"
  for f in $artifact; do [ -e "$f" ] && a="$f" && break; done
  for f in $logfile; do [ -e "$f" ] && l="$f" && break; done
  if [ "$a" = "MISSING" ] || [ "$l" = "MISSING" ]; then missing=$((missing+1)); fi
  echo "| ${stage} | \`${cmd}\` | ${a} | ${l} | ${note} |"
}

row "S0 环境" "bash scripts/00_env_check.sh" "logs/00_env_check.txt" "logs/00_cuda_check.json" "CUDA + ROS2 + Cartographer 探测"
row "S2 数据" "python scripts/10_gen_mock_data.py" "data/mock/mock_0000.npz" "logs/10_gen_mock.log" "MOCK 数据 + 激光扫描流"
row "S3 管线" "python scripts/11_build_dataset.py" "data/processed/train.state.npy" "logs/11_build_dataset.jsonl" "窗口化 + 归一化统计"
row "S3 体检" "python scripts/12_data_check.py" "logs/12_data_check.json" "logs/12_data_check.jsonl" "schema/形状/时间戳/对齐"
row "S4 单测" "pytest -q tests/" "tests/test_cvae_shapes.py" "logs/90_pytest.log" "契约测试（96 项）"
row "S5 门禁" "bash scripts/21_overfit_single_batch.sh" "logs/21_overfit_gate.json" "logs/train_metrics.jsonl" "单 batch 过拟合"
row "S5 训练" "bash scripts/20_train.sh" "checkpoints/base/meta.json" "logs/train_metrics.jsonl" "loss/KL/β 曲线"
row "S5 分工" "bash scripts/20_train.sh" "logs/20_device_usage.json" "logs/20_train.log" "CPU/GPU 比例实测"
row "S6 评测" "bash scripts/30_infer_offline.sh --split test" "outputs/eval/base_test_metrics.csv" "logs/infer_eval.json" "误差/抖动/延迟/多模态"
row "S7 闭环" "bash scripts/31_infer_closed_loop.sh" "outputs/infer_samples/executed_actions_base.jsonl" "logs/31_closed_loop_summary.json" "限幅/watchdog/位姿可用性"
row "S8 SLAM" "bash scripts/40_slam_bringup.sh" "outputs/slam/trajectory.pbstream" "logs/40_slam_bringup.log" "真实 Cartographer 离线回放"
row "S8 轨迹" "bash scripts/42_slam_replay_eval.sh" "outputs/slam/trajectory_metrics.json" "logs/42_trajectory_metrics.jsonl" "ATE/RPE/漂移"
row "S8 地图" "cartographer_pbstream_to_ros_map" "outputs/slam/cartographer_map.pgm" "logs/40_pbstream_to_map.log" "占用栅格地图"
row "S10 消融" "bash scripts/50_run_ablation.sh" "outputs/ablation/summary.csv" "logs/50_ablation.log" "A1–A6 对照"
row "图" "bash scripts/90_report.py" "outputs/figures/loss_curve.png" "logs/train_summary.json" "训练曲线"
row "图" "bash scripts/20_train.sh" "outputs/figures/device_split.png" "logs/20_device_usage.json" "CPU/GPU 分工"
row "图" "bash scripts/30_infer_offline.sh" "outputs/figures/30_action_compare_base.png" "logs/infer_eval.json" "动作对比"
row "图" "bash scripts/30_infer_offline.sh" "outputs/figures/30_chunk_weights_base.png" "logs/infer_eval.json" "时间集成权重热图"
row "图" "python scripts/12_data_check.py" "outputs/figures/12_data_samples.png" "logs/12_data_check.json" "数据抽样可视化"
echo
echo "## 截图"
echo
echo "本实验全部可视化产物由 matplotlib 直接输出 PNG（可复现），不引入无法复现的手工截图。"
echo "若后续需要界面截图（如 rviz2），必须同时记录命令与时间戳；本机未安装 scrot/import 时写 SCREENSHOT_UNAVAILABLE。"
echo
echo "## 缺失项"
echo
if [ "${missing}" -eq 0 ]; then echo "无（全部证据项存在）"; else echo "${missing} 项缺失（见上表 MISSING），必须补齐或标注原因。"; fi
} > "${index}"

echo "证据索引 → ${index}（缺失 ${missing} 项）"
[ "${missing}" -eq 0 ]
