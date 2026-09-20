#!/usr/bin/env bash
# scripts/50_run_ablation.sh —— 按【九】消融矩阵批量运行（同一份 splits、同一组 seed）。
#
# 每个消融项：构建数据（如需要）→ 单 batch 门禁通过后的正式训练 → 离线评测 → 汇总表。
# 结果写入 outputs/ablation/summary.csv 与 docs/experiments.md 登记表。
#
# 用法：
#   bash scripts/50_run_ablation.sh                 # 跑全部已定义消融
#   bash scripts/50_run_ablation.sh no_cvae no_chunk # 只跑指定项
set -uo pipefail

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PY="${CVSLAM_TRAIN_PYTHON:-${HOME}/Workspace/miniconda/envs/unitree_rt/bin/python}"
cd "${SLAM_ROOT}"
export PYTHONPATH="${SLAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# 消融矩阵（配置覆盖文件见 configs/ablation/）
ALL_ABLATIONS=(no_cvae no_chunk no_temporal_ensemble no_slam_input chunk_len_H4 chunk_len_H16 chunk_len_H32 ensemble_uniform ensemble_exp_d1p0)
if [ $# -gt 0 ]; then
  ABLATIONS=("$@")
else
  ABLATIONS=("${ALL_ABLATIONS[@]}")
fi
mkdir -p outputs/ablation logs/ablation

for name in "${ABLATIONS[@]}"; do
  echo "==================== 消融 ${name} ===================="
  # 纯推理侧消融：只改 infer.*，data/model/train 配置与 base 完全一致，
  # 因此复用 base 的 checkpoint 与 processed 数据，不重复训练（省时且保证变量唯一）。
  case "${name}" in
    no_temporal_ensemble|ensemble_uniform|ensemble_exp_d0p5|ensemble_exp_d1p0)
      "${TRAIN_PY}" -m src.infer.offline_eval --ablation "${name}" --run-name base --split test \
        > "logs/ablation/${name}_eval.log" 2>&1
      echo "  推理侧消融完成（复用 base checkpoint）：$(tail -3 logs/ablation/${name}_eval.log | head -1)"
      continue
      ;;
  esac
  # 1) 数据侧：H 与状态维度变化需要重建 processed（同 seed、同 splits 规则）
  if [ -f "configs/ablation/${name}.yaml" ]; then
    DATA_FLAG="--ablation ${name} --run-name ${name}"
    "${TRAIN_PY}" scripts/11_build_dataset.py ${DATA_FLAG} > "logs/ablation/${name}_build.log" 2>&1 || {
      echo "  数据构建失败，见 logs/ablation/${name}_build.log"; continue; }
  else
    echo "  缺少 configs/ablation/${name}.yaml，跳过"; continue
  fi
  # 2) 训练（与 base 完全相同的超参，只差消融项；config hash 会不同）
  "${TRAIN_PY}" -m src.train.train_cvae --ablation "${name}" --run-name "${name}" \
    > "logs/ablation/${name}_train.log" 2>&1
  rc=$?
  if [ "${rc}" -ne 0 ]; then echo "  训练失败 rc=${rc}，见 logs/ablation/${name}_train.log"; continue; fi
  # 3) 评测
  "${TRAIN_PY}" -m src.infer.offline_eval --ablation "${name}" --run-name "${name}" --split test \
    > "logs/ablation/${name}_eval.log" 2>&1 || { echo "  评测失败"; continue; }
  echo "  完成：$(tail -3 logs/ablation/${name}_eval.log | head -1)"
done

# 4) 汇总（只汇总真实存在的产物，缺失项在表里显式留空而不是补数）
"${TRAIN_PY}" scripts/51_summarize_ablation.py
