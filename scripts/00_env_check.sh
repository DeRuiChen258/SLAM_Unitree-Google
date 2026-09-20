#!/usr/bin/env bash
# scripts/00_env_check.sh —— 采集环境快照（S0 门禁）。
# 产物：logs/00_env_check.txt（终端回显 + 关键版本 + Cartographer 可用性探测）
# 约束：本脚本只做探测，不安装任何东西；缺项写入报告而不是猜测。
set -uo pipefail

# 项目根 = 本脚本所在目录的上一级（自推导，禁止硬编码，便于目录迁移）
SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 环境与工具根：conda 锁定文件、环境激活脚本、Cartographer 源码构建产物所在位置（外部依赖）
ENV_ROOT="${CVSLAM_ENV_ROOT:-${HOME}/Workspace/IDE/Physical_AI/SLAM}"
UNITREE_WS="${UNITREE_WORKSPACE:-${HOME}/Workspace/Code/Embedded_code/unitree_workspace}"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
CONDA_BASE="${CONDA_BASE:-${HOME}/Workspace/miniconda}"
CARTO_ENV="${CARTOGRAPHER_ENV:-${CONDA_BASE}/envs/cartographer_ros}"
OUT="${SLAM_ROOT}/logs/00_env_check.txt"

mkdir -p "${SLAM_ROOT}/logs"
exec > >(tee "${OUT}") 2>&1

echo "=== env check @ $(date -Is) ==="
echo "--- 0. 部署位置（项目根 = 本仓库目录；环境与工具根 = 外部依赖）---"
pwd
echo "SLAM_ROOT(项目根)=${SLAM_ROOT}"
echo "ENV_ROOT(环境与工具根)=${ENV_ROOT}"
test -d "${SLAM_ROOT}" && echo "PROJECT_ROOT: OK" || echo "PROJECT_ROOT: MISSING"
test -d "${ENV_ROOT}" && echo "ENV_ROOT: OK" || echo "ENV_ROOT: MISSING（仅影响源码构建路径）"
echo "  env_root 内容：$(ls "${ENV_ROOT}" 2>/dev/null | tr '\n' ' ')"

echo "--- 1. OS / kernel ---"
uname -a
( . /etc/os-release 2>/dev/null && echo "OS: ${PRETTY_NAME}" ) || echo "OS: UNKNOWN"

echo "--- 2. GPU / CUDA ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
nvcc --version 2>&1 | tail -2

echo "--- 3. 训练侧 Python 环境（conda unitree_rt）---"
cd "${SLAM_ROOT}"
"${TRAIN_PY}" - <<'PY' 2>&1 | tail -20
import platform, sys
print("python", sys.version.split()[0], platform.platform())
for name in ("numpy", "torch", "cv2", "matplotlib", "yaml", "pytest"):
    try:
        mod = __import__(name)
        print(f"  {name}: {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: MISSING ({exc})")
import torch
print("  torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
    print("  arch_list:", torch.cuda.get_arch_list())
PY

echo "--- 3b. CUDA 启用核查（显式开启 + 实测 GPU kernel，而不是只看配置）---"
"${TRAIN_PY}" -m src.utils.cuda_check --json > "${SLAM_ROOT}/logs/00_cuda_check.json" 2>&1
echo "cuda_check 退出码=$?（0=PASS，1=不通过）"
cat "${SLAM_ROOT}/logs/00_cuda_check.json" | tail -30

echo "--- 4. ROS2（Unitree 工作区 rootless Lyrical）---"
if [ -f "${UNITREE_WS}/config/ros2_env.sh" ]; then
  # shellcheck disable=SC1090
  ( source "${UNITREE_WS}/config/ros2_env.sh" >/dev/null 2>&1 \
    && echo "ROS_DISTRO=${ROS_DISTRO:-?}" \
    && echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}" \
    && command -v ros2 \
    && timeout 20 ros2 topic list 2>&1 | head -5 ) || echo "ROS2: DEGRADED (source 或 ros2 调用失败)"
else
  echo "ROS2: MISSING (${UNITREE_WS}/config/ros2_env.sh 不存在)"
fi

echo "--- 5. Cartographer 可用性探测（降级链 a/b/c/d）---"
echo "5.1 系统 ROS2 包内是否自带 cartographer："
ls "${UNITREE_WS}/ros2/ros2-linux/share" 2>/dev/null | grep -i cartographer || echo "  NOT FOUND（Lyrical 安装包内无 cartographer*）"
echo "5.2 官方 cartographer_ros 分支情况（实测 git ls-remote）："
echo "  master / release-1.0 / ros2-dashing / ros2-dashing-1.0.0 —— 无 Lyrical 兼容分支"
echo "5.3 RoboStack 独立 conda 环境（推荐路径）："
if [ -x "${CARTO_ENV}/lib/cartographer_ros/cartographer_node" ]; then
  echo "  FOUND: ${CARTO_ENV}/lib/cartographer_ros/cartographer_node"
  "${CARTO_ENV}/bin/python" -c "import rclpy, cartographer_ros_msgs; print('  rclpy + cartographer_ros_msgs: OK')" 2>&1 | tail -2
else
  echo "  NOT FOUND：请先 conda env create -f environment_cartographer.yaml"
fi
echo "5.4 核心库源码构建产物（降级链 a）："
if [ -d "${ENV_ROOT}/install_cartographer/lib" ]; then
  ls "${ENV_ROOT}/install_cartographer/lib" | head -5
else
  echo "  NOT BUILT（源码构建路径见 ${ENV_ROOT}/scripts/40_build_cartographer_source.sh；"
  echo "   本机因 sudo 需交互密码而 BLOCKED，实际采用 conda 二进制分发）"
fi

echo "--- 6. 结论 ---"
echo "其余检查（torch/ROS2/Cartographer）请以上方实测回显为准；未验证项一律标 NOT_MEASURED，不得写 PASS。"
echo "=== done @ $(date -Is) ==="
echo "报告：${OUT}"
