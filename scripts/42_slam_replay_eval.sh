#!/usr/bin/env bash
# scripts/42_slam_replay_eval.sh —— 离线评估 Cartographer 轨迹质量（ATE/RPE/漂移）+ 出图。
# 说明：位姿流是在线（前端 + 局部优化）位姿，正是策略实际可用的位姿；
#       outputs/slam/*.pbstream 与栅格地图是含全局优化的最终结果，两者分别标注、不可混用。
set -euo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"

exec "${TRAIN_PY}" -m src.slam.trajectory_metrics "$@"
