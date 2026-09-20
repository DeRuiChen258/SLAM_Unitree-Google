#!/usr/bin/env bash
# scripts/40_slam_bringup.sh —— 启动 SLAM 数据链并跑通真实 Google Cartographer（降级链逐级尝试）。
#
# 本机实测事实（2026-09-20，写入 logs/40_slam_bringup.log）：
#   * 本机 ROS2 Lyrical rootless 安装包内 **没有** cartographer 包 → a/b 级不可行；
#   * 官方 cartographer_ros 仓库只有 master / release-1.0 / ros2-dashing（2019）→ 无 Lyrical 分支；
#   * RoboStack 提供 ros-jazzy-cartographer-ros（Cartographer 2.0.9003 构建产物）→ 采用；
#   * 该产物中 `cartographer_node` 与 `cartographer_assets_writer` 因 gflags 重复定义无法启动（打包缺陷），
#     但 `cartographer_offline_node` 可用 → 走 **离线回放** 路径（提示词【十】c 级）。
#
# 流程：录制 ROS2 bag（/scan + /tf_static）→ 启动位姿桥接 → cartographer_offline_node 建图并写 pbstream
#       → 导出栅格地图（pgm/yaml）→ 位姿流落盘供训练/推理侧读取。
#
# 用法：bash scripts/40_slam_bringup.sh [--max-frames N] [--speed X]
set -uo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-${HOME}/Workspace/miniconda}"
CARTO_ENV="${CARTOGRAPHER_ENV:-cartographer_ros}"
PREFIX="${CONDA_BASE}/envs/${CARTO_ENV}"
export PATH="${PREFIX}/bin:${PATH}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"          # 仿真/离线回放 domain，与实机隔离
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
# RoboStack 环境需要通过 activate.d 脚本导出 AMENT_PREFIX_PATH / LD_LIBRARY_PATH 等，
# 否则 `ros2 bag` 等 CLI 会直接报 "AMENT_PREFIX_PATH is not set"。
# 显式设定 ROS2 运行环境（不依赖 conda activate 的 activate.d：其中的 AMENT_PREFIX_PATH
# 会指向调用方的 CONDA_PREFIX，导致 rosbag2 插件加载失败——本机实测踩过这个坑）。
export CONDA_PREFIX="${PREFIX}"
export AMENT_PREFIX_PATH="${PREFIX}"
export CMAKE_PREFIX_PATH="${PREFIX}"
export COLCON_PREFIX_PATH="${PREFIX}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_VERSION=2
export ROS_DISTRO=jazzy

MAX_FRAMES="${MAX_FRAMES:-0}"
SPEED="${SPEED:-8.0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --speed) SPEED="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

LOG="${SLAM_ROOT}/logs/40_slam_bringup.log"
BAG_DIR="${SLAM_ROOT}/data/bags/cvae_slam_replay"
PBS="${SLAM_ROOT}/outputs/slam/trajectory.pbstream"
MAP_PREFIX="${SLAM_ROOT}/outputs/slam/cartographer_map"
mkdir -p "${SLAM_ROOT}/logs" "${SLAM_ROOT}/outputs/slam" "$(dirname "${BAG_DIR}")"

{
echo "=== SLAM bringup @ $(date -Is) ==="
echo "PREFIX=${PREFIX}"

echo "--- 0. 环境与二进制可用性实测 ---"
for b in cartographer_node cartographer_offline_node cartographer_pbstream_to_ros_map cartographer_occupancy_grid_node; do
  if [ -x "${PREFIX}/lib/cartographer_ros/${b}" ]; then echo "  present: ${b}"; else echo "  MISSING: ${b}"; fi
done

echo "--- 1. 回放扫描流并写成 ROS2 bag（/scan + /tf_static + /clock）---"
rm -rf "${BAG_DIR}"
"${PREFIX}/bin/python" -m src.slam.ros2_scan_player --max-frames "${MAX_FRAMES}" --speed "${SPEED}" \
  --start-delay 1.0 --record-bag "${BAG_DIR}" \
  > "${SLAM_ROOT}/logs/41_scan_player.log" 2>&1 &
wait $!
echo "scan player 退出码=$?"
ls -la "${BAG_DIR}" | head -5
echo "bag 大小：$(du -sh "${BAG_DIR}" 2>/dev/null | cut -f1)"

echo "--- 2. 启动位姿桥接（消息驱动，不丢帧）---"
"${PREFIX}/bin/python" -m src.slam.cartographer_bridge --mode replay --trigger tf --min-period 0.05 \
  --reset-stream --duration 1800 > "${SLAM_ROOT}/logs/40_bridge.log" 2>&1 &
BRIDGE_PID=$!
sleep 3
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "桥接进程启动失败："; tail -20 "${SLAM_ROOT}/logs/40_bridge.log"; exit 4
fi

echo "--- 3. cartographer_offline_node 建图（真实 Cartographer 2D SLAM）---"
rm -f "${PBS}"
"${PREFIX}/lib/cartographer_ros/cartographer_offline_node" \
  -configuration_directory="${SLAM_ROOT}/configs/cartographer" \
  -configuration_basenames=g1_2d.lua \
  -bag_filenames="${BAG_DIR}/$(ls "${BAG_DIR}" | grep -E '\.db3$|\.mcap$' | head -1)" \
  -save_state_filename="${PBS}" \
  > "${SLAM_ROOT}/logs/40_cartographer_offline.log" 2>&1
echo "offline_node 退出码=$?"
tail -25 "${SLAM_ROOT}/logs/40_cartographer_offline.log"

sleep 2
kill -INT "${BRIDGE_PID}" 2>/dev/null
wait "${BRIDGE_PID}" 2>/dev/null
echo "--- 4. 位姿流统计 ---"
echo "pose stream: $(wc -l < "${SLAM_ROOT}/data/slam/pose_stream.jsonl" 2>/dev/null || echo 0) 行"
tail -3 "${SLAM_ROOT}/logs/40_bridge.log"

echo "--- 5. 导出栅格地图（pgm + yaml）---"
if [ -f "${PBS}" ]; then
  "${PREFIX}/lib/cartographer_ros/cartographer_pbstream_to_ros_map" \
    -pbstream_filename="${PBS}" -map_filestem="${MAP_PREFIX}" \
    > "${SLAM_ROOT}/logs/40_pbstream_to_map.log" 2>&1
  echo "导出退出码=$?"
  ls -la "${SLAM_ROOT}/outputs/slam/" | head -10
else
  echo "BLOCKED: 未生成 pbstream，无法导出地图"
fi
echo "=== done @ $(date -Is) ==="
} 2>&1 | tee "${LOG}"
