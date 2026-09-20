#!/usr/bin/env bash
# scripts/30_infer_offline.sh —— 离线推理与评测（val/test），产出指标表与曲线。
# 用法：bash scripts/30_infer_offline.sh [--split test] [--run-name base] [--ablation NAME]
set -euo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"

exec "${TRAIN_PY}" -m src.infer.offline_eval "$@"
