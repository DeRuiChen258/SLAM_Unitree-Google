#!/usr/bin/env bash
# scripts/21_overfit_single_batch.sh —— 单 batch 过拟合门禁（S5 前置条件）。
# 阈值在 configs/train.yaml:overfit_gate.loss_drop_threshold；未通过则返回码 6，禁止进入正式训练。
set -uo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"

set +e
"${TRAIN_PY}" -m src.train.train_cvae --overfit --run-name overfit_gate "$@"
rc=$?
set -e
if [ "${rc}" -eq 0 ]; then
  echo "[21_overfit_gate] PASS"
else
  echo "[21_overfit_gate] FAIL（rc=${rc}）：先查数据/损失/学习率，禁止继续正式训练" >&2
fi
exit "${rc}"
