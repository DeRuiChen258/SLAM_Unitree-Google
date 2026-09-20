# Makefile —— 统一命令入口（只转发到 scripts/，不内联任何业务逻辑）。
SHELL := /bin/bash
# 训练侧解释器：默认取环境变量 CVSLAM_TRAIN_PYTHON，其次用 PATH 上的 python
TRAIN_PY ?= $(or $(CVSLAM_TRAIN_PYTHON),python3)
CARTO_ENV ?= cartographer_ros

.PHONY: help env data dataset check gate train eval closed-loop slam slam-eval ablation test report evidence viz

help:
	@echo "make env         # S0 环境基线 + CUDA 自检"
	@echo "make data        # S2 生成 MOCK 数据 + 激光扫描"
	@echo "make dataset     # S3 窗口化 / 归一化 / 划分"
	@echo "make check       # S3 数据体检"
	@echo "make gate        # S5 单 batch 过拟合门禁"
	@echo "make train       # S5 训练"
	@echo "make eval        # S6 离线评测（含时间集成对照）"
	@echo "make closed-loop # S7 闭环执行（默认不下发指令）"
	@echo "make slam        # S8 生成配置 + 离线回放建图"
	@echo "make slam-eval   # S8 轨迹 ATE/RPE"
	@echo "make ablation    # S10 消融矩阵"
	@echo "make test        # 契约测试"
	@echo "make report      # 汇总报告"
	@echo "make evidence    # 证据索引"
	@echo "make viz         # 生成可视化数据 + 页面（浏览器打开 http://127.0.0.1:8788/figures/index.html）"

env:
	bash scripts/00_env_check.sh

data:
	$(TRAIN_PY) scripts/10_gen_mock_data.py

dataset:
	$(TRAIN_PY) scripts/11_build_dataset.py

check:
	$(TRAIN_PY) scripts/12_data_check.py

gate:
	bash scripts/21_overfit_single_batch.sh

train:
	bash scripts/20_train.sh

eval:
	bash scripts/30_infer_offline.sh --split test

closed-loop:
	bash scripts/31_infer_closed_loop.sh

slam:
	$(TRAIN_PY) scripts/43_gen_cartographer_config.py
	bash scripts/40_slam_bringup.sh

slam-eval:
	bash scripts/44_query_pbstream_trajectory.sh
	bash scripts/42_slam_replay_eval.sh

ablation:
	bash scripts/50_run_ablation.sh

test:
	$(TRAIN_PY) -m pytest -q tests/ | tee logs/90_pytest.log

report:
	$(TRAIN_PY) scripts/90_report.py

evidence:
	bash scripts/60_collect_evidence.sh

# 可视化：导出看板数据 → 生成 index.html → 提示本地服务命令（服务需常驻，故不在此后台启动）
viz:
	$(TRAIN_PY) scripts/91_export_viz_data.py
	$(TRAIN_PY) scripts/92_build_viz_page.py
	@echo "启动服务：python3 -m http.server 8788 --bind 127.0.0.1 --directory outputs"
	@echo "浏览器：http://127.0.0.1:8788/figures/index.html"
