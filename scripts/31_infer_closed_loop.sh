#!/usr/bin/env bash
# scripts/31_infer_closed_loop.sh —— 闭环推理（mock/回放位姿流）：滚动预测 + 时间集成 + 限幅 + watchdog。
# 默认 dry-run：只写 logs/ 与 outputs/infer_samples/，不发布任何控制指令。
# 用法：bash scripts/31_infer_closed_loop.sh [--run-name base] [--episode 0] [--steps 120] [--pose-stream FILE]
set -euo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"

exec "${TRAIN_PY}" -m src.infer.execution_loop "$@"
