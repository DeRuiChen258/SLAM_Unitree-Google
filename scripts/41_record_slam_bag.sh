#!/usr/bin/env bash
# scripts/41_record_slam_bag.sh —— 单独录制扫描数据包（网络/传感器链路验证用）。
# 说明：本实验没有真实激光雷达，因此数据源是 src/slam/ros2_scan_player.py 发布的合成扫描流
#       （与 data/slam/scans.npz 同源，由 scripts/10_gen_mock_data.py 生成）。
#       真实传感器接入时把 --source 换成真实雷达驱动即可，录制与后续流程完全一致。
set -euo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-${HOME}/Workspace/miniconda}"
PREFIX="${CONDA_BASE}/envs/${CARTOGRAPHER_ENV:-cartographer_ros}"
export PATH="${PREFIX}/bin:${PATH}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CONDA_PREFIX="${PREFIX}"
export AMENT_PREFIX_PATH="${PREFIX}"
export CMAKE_PREFIX_PATH="${PREFIX}"
export COLCON_PREFIX_PATH="${PREFIX}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_VERSION=2 ROS_DISTRO=jazzy
OUT="${1:-${SLAM_ROOT}/data/bags/manual_record}"
MAX_FRAMES="${MAX_FRAMES:-0}"

rm -rf "${OUT}"
"${PREFIX}/bin/python" -m src.slam.ros2_scan_player --max-frames "${MAX_FRAMES}" --speed 8 --start-delay 1 &
PLAYER=$!
sleep 2
timeout 120 ros2 bag record -s sqlite3 -o "${OUT}" /scan /tf_static &
REC=$!
wait "${PLAYER}"
sleep 1
kill -INT "${REC}" 2>/dev/null || true
wait "${REC}" 2>/dev/null || true
echo "bag → ${OUT}"
ros2 bag info "${OUT}" 2>&1 | head -20
