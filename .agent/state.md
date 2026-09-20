# state

- 2026-09-20：任务摄入；读取 workflow README / unitree workflow / contract / registry。
- 2026-09-20：S0 环境勘察（RTX 5070 sm_120 / CUDA 13.2 / torch 2.14.0+cu130 / ROS2 Lyrical rootless / 无 cartographer 包）。
- 2026-09-20：S1–S3 工程骨架、配置、MOCK 数据（60×120 帧）与数据管线；schema 体检 0 错误。
- 2026-09-20：S4 模型与单测（96 项契约测试）。
- 2026-09-20：S5 训练（含单 batch 门禁）。
- 2026-09-20：S6/S7 离线评测与闭环执行。
- 2026-09-20：S8 Cartographer 接入（降级链 a→BLOCKED、c→PASS），离线回放建图 + 轨迹评估。
- 2026-09-20：S10 消融矩阵；S11 复现与证据；S12 报告素材。
- 用户中途指令：**启用 CUDA 版本** → 显式 `device: cuda` + CUDA 自检（arch/kernel 实测）。
- 用户中途指令：**协调 CPU/GPU 比例** → 新增 `device_policy` 与每步耗时拆分 + 瓶颈判定。
- 用户中途指令：**主要使用 CUDA** → 图像归一化从 CPU 迁到设备侧（uint8 上显存）、启用 AMP、batch 192；
  实测 data wait 29.2% → 0.3%、compute 70.6% → 99.6%（GPU_BOUND）。
- 2026-09-20（后续指令）：**目录职责重构** —— 环境与工具（conda 锁定、环境激活脚本、Cartographer
  源码构建脚本与产物）留在 `<环境与工具根>`；
  项目算法代码/配置/数据/产物/文档迁到
  `<项目根>`（git 历史随项目迁移）。
  配置侧新增 `paths.yaml:env_root`（外部依赖，可用 `CVSLAM_ENV_ROOT` 覆盖）；
  `scripts/00_env_check.sh` 同时校验两个根；流水线从新位置整体重跑并复现指标。
- 2026-09-20：新增 `scripts/91_export_viz_data.py`（导出窗口内可视化所需的紧凑数据）
  与 `outputs/figures/results-dashboard.html`（内联结果看板）。
- 2026-09-20（用户指令：启动可视化）：新增 `scripts/92_build_viz_page.py`，
  生成 `outputs/figures/index.html`（内联看板 + 39 张静态图图集）；
  本地 HTTP 服务 `python3 -m http.server 8788 --directory outputs`，
  访问 `http://127.0.0.1:8788/figures/index.html`（服务必须挂在 outputs/ 一级，
  因为页面引用 `../slam/*.png`；挂在 figures/ 会触发目录穿越保护返回 404）。
  验证：服务日志 86 次 GET 全部 200，浏览器实际加载页面与全部图。

## 当前检查点

全部阶段完成，产物落盘，测试全绿（96 项），证据索引 0 缺失。
目录重构后从新位置整体重跑：数据→训练→评测→闭环→Cartographer→消融，指标可复现
（ATE 0.0275 m、best val/total 0.0694、compute 占比 99.3%）。
恢复点：如需重跑，按 README §5.4 顺序执行（先读本文件与 TASK.md）。

- 2026-09-21（用户指令：详细 README + 去个人信息 + 发布到 GitHub）：
  重写 `README.md`（含全部 39 张产物图与逐图说明、快速开始、证据链、限制清单）；
  去个人信息化：`configs/paths.yaml` 改 `project_root: auto`（代码位置推导）+ `${HOME}`/环境变量覆盖，
  `scripts/*.sh`、`Makefile`、文档与产物页脚不再写本机绝对路径，
  `scripts/43` 的配置注释改用环境内相对路径 + sha256，`scripts/91/92` 不再输出本机路径；
  重新生成 `configs/cartographer/g1_2d.lua`、`outputs/slam/*`（指标数值不变）、`outputs/figures/{viz_data.json,index.html}`；
  验证：`pytest -q tests/` 96 passed，配置解析出的路径与本机实际路径一致（行为未变）。
  发布：公开快照提交 `86e578b`（父提交为远程 `Initial commit`，fast-forward，无强推），
  本地 `main` 已对齐远程；原始本地历史保留在 `backup/local-history-pre-publish`（49365f5）。
