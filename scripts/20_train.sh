#!/usr/bin/env bash
# scripts/20_train.sh —— 训练入口（参数全部来自 configs/，可叠加消融覆盖）。
# 用法：bash scripts/20_train.sh [--ablation NAME] [--override k=v] [--max-steps N] [--run-name NAME]
set -euo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"

exec "${TRAIN_PY}" -m src.train.train_cvae "$@"
