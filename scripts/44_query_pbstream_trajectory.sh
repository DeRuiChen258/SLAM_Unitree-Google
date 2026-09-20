#!/usr/bin/env bash
# scripts/44_query_pbstream_trajectory.sh —— 从建好的 pbstream 里导出**优化后**的完整轨迹到位姿流。
#
# 原理：以 `-load_state_filename=<pbstream> -keep_running=true` 启动 cartographer_offline_node
#       作为服务端，再用 src/slam/cartographer_bridge.py --mode pbstream_query 调 trajectory_query 服务。
# 为什么不用订阅 TF：高倍速离线回放时 TF 高频发布，订阅侧丢包严重（实测 1200 帧只收到 34 个），
#       对"必须完整采样"的轨迹评估是不可接受的。
#
# 用法：bash scripts/44_query_pbstream_trajectory.sh [--pbstream FILE] [--trajectory-id N]
set -uo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-${HOME}/Workspace/miniconda}"
PREFIX="${CONDA_BASE}/envs/${CARTOGRAPHER_ENV:-cartographer_ros}"
export PATH="${PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${PREFIX}"
export AMENT_PREFIX_PATH="${PREFIX}"
export CMAKE_PREFIX_PATH="${PREFIX}"
export COLCON_PREFIX_PATH="${PREFIX}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_VERSION=2
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

cd "${SLAM_ROOT}"
PBS="${SLAM_ROOT}/outputs/slam/trajectory.pbstream"
TRAJ_ID=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pbstream) PBS="$2"; shift 2 ;;
    --trajectory-id) TRAJ_ID="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

echo "=== 查询 pbstream 轨迹：${PBS}（trajectory_id=${TRAJ_ID}）==="
"${PREFIX}/lib/cartographer_ros/cartographer_offline_node" \
  -configuration_directory="${SLAM_ROOT}/configs/cartographer" \
  -configuration_basenames=g1_2d.lua \
  -load_state_filename="${PBS}" \
  -keep_running=true \
  > "${SLAM_ROOT}/logs/44_pbstream_server.log" 2>&1 &
SERVER_PID=$!
sleep 6
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "服务端启动失败："; tail -20 "${SLAM_ROOT}/logs/44_pbstream_server.log"; exit 4
fi

"${PREFIX}/bin/python" -m src.slam.cartographer_bridge --mode pbstream_query \
  --trajectory-id "${TRAJ_ID}" --reset-stream --service-timeout 60 \
  > "${SLAM_ROOT}/logs/44_query.log" 2>&1
rc=$?
echo "查询退出码=${rc}"; tail -5 "${SLAM_ROOT}/logs/44_query.log"

kill -INT "${SERVER_PID}" 2>/dev/null
sleep 2
kill -9 "${SERVER_PID}" 2>/dev/null
echo "位姿流：$(wc -l < "${SLAM_ROOT}/data/slam/pose_stream.jsonl" 2>/dev/null || echo 0) 行"
exit "${rc}"
