# TASK：CVAE + 动作分块 + 时间集成 × Cartographer SLAM 端到端实验

> 中断恢复入口：先读本文件与 `.agent/state.md`，从"当前检查点"继续，禁止从头重跑。

## 0. 任务定义

| 项 | 内容 |
|---|---|
| 目标 | 在可执行工程中完成「视觉观测 → CVAE 策略 → 动作分块 → 时间集成 → 动作输出 → Cartographer SLAM 反馈」的端到端实验 |
| 项目根 | `<项目根>`（唯一，禁止第二份工程） |
| 环境与工具根 | `<环境与工具根>`（conda 锁定 / 环境激活脚本 / Cartographer 构建脚本与产物，作为外部依赖引用） |
| 输入资料 | `Prompt/提示词‘.txt`（大纲）、`Prompt/CVAE_动作分块_时间集成_Cartographer_SLAM_详细执行提示词.txt`（详细规格） |
| 交付物 | 代码 + 配置 + 数据管线 + 训练/推理/评测 + SLAM 集成 + 消融 + 文档 + 证据索引 |
| 约束 | 无真机；动作下发默认关闭；训练侧 conda 与 ROS2 侧 Python 严格隔离；MOCK 结论必须标注 MOCK |
| 验收标准 | 阶段门禁逐条通过；`pytest -q tests/` 全绿；产物与日志可复现；缺失项显式标注 |

## 1. 阶段清单与门禁

| 阶段 | 内容 | 门禁 | 状态 |
|---|---|---|---|
| S0 | 环境勘察与基线复测（含 CUDA 显式启用自检） | CPU 流水线可跑；GPU 项有实测证据 | ✅ 完成（`logs/00_env_check.txt`、`logs/00_cuda_check.json`） |
| S1 | 工程骨架与配置 | `project_root` 精确匹配；配置校验拒绝非法配置 | ✅ 完成（`tests/test_config_validation.py`） |
| S2 | 数据契约与 MOCK 数据 | schema 0 错误；多模态样本存在 | ✅ 完成（60 episodes，多模态标记见 `data/mock_manifest.json`） |
| S3 | 数据管线 | 窗口数可推算一致；同 seed 可重生成；超容差 < 5% | ✅ 完成（`logs/12_data_check.json`） |
| S4 | 模型实现 | shape / KL / 集成器单测通过 | ✅ 完成（`tests/test_cvae_*.py`、`test_temporal_ensemble.py`） |
| S5 | 训练闭环 | 单 batch 门禁通过；loss 下降；last/best 可加载 | ✅ 完成（门禁 PASS 83.9%；`checkpoints/base/`） |
| S6 | 分块与集成验证 | 开集成后抖动下降或有诚实解释 | ✅ 完成（−4.3%，限定条件见 `docs/algorithm_notes.md` §3.3） |
| S7 | 推理闭环 | 连续运行无中断；延迟分位记录；未访问后验 | ✅ 完成（`logs/31_closed_loop_summary.json`） |
| S8 | Cartographer 接入 | 位姿流可用或明确记录降级；时间同步量化 | ✅ 完成（离线回放 + pbstream 查询；在线节点 BLOCKED 已记录） |
| S9 | SLAM 位姿参与策略 | 除位姿输入外其余配置一致（可 diff） | ✅ 完成（A4 消融，状态 29→24 维） |
| S10 | 消融实验 | 每项有配置/日志/指标 | ✅ 完成（`outputs/ablation/summary.csv`） |
| S11 | 验证与复现 | 一键复现命令可跑；清单逐条有证据 | ✅ 完成（`evidence/index.md`、`logs/reproduce_*.log`） |
| S12 | 报告素材与收尾 | 图有源数据与生成命令 | ✅ 完成（`outputs/REPORT.md`、`outputs/figures/`） |

## 2. 检查点

- 当前检查点：**全部阶段完成，已提交 git（`066f32e`）**。
- 中断恢复方式：读本文件 → 读 `.agent/state.md` → 按第 1 节状态列继续未完成项。

## 3. Known Issues

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 1 | RoboStack `cartographer_node`/`assets_writer` 因 gflags 重复定义无法启动 | 无法走在线 SLAM | 已走离线回放 + `trajectory_query`；在线路径代码保留并标 `NOT_MEASURED` |
| 2 | 该构建 Lua `include` 解析异常 | 官方示例配置不可直接运行 | 生成自包含配置（含来源 sha256） |
| 3 | 官方源码构建 Cartographer 需 `sudo` 交互密码 | a 级降级链不可行 | 标 `BLOCKED`，改用 conda 二进制（c 级） |
| 4 | 高倍速回放时 TF 订阅丢包（1200 帧仅收到 34 个） | 轨迹评估样本不足 | 改用 `trajectory_query` 服务取 pbstream 优化后轨迹（1037 节点） |
| 5 | A7（观测历史帧数 K 扫描）未跑 | 多帧视觉的必要性未量化 | 标 `NOT_RUN`，README §8.4 声明 |
| 6 | 消融运行期间与 SLAM 全局优化存在 CPU 竞争 | 个别 run 的 `duration_s` 与设备占比受干扰 | `docs/experiments.md` 已注明；模型指标不受影响（仅耗时） |
| 7 | `g1_3d.lua`、rviz 配置未实测 | 3D 建图与人工核查缺失 | 标 `NOT_MEASURED` / `NOT_RUN` |
| 8 | 本地可视化服务需手动常驻 | 关闭终端后页面不可访问 | `make viz` 生成页面；服务用 `python3 -m http.server 8788 --directory outputs` 启动，停止用 `pkill -f "http.server 8788"` |

## 4. 安全门禁（涉及动作下发的检查点）

| 检查项 | 状态 | 说明 |
|---|---|---|
| 急停可达 | N/A（无真机） | 无真机，永不发布动作 |
| 场地隔离 | N/A | 同上 |
| 限速限幅生效 | ✅ 已实现并测试 | `src/infer/execution_loop.py:ActionLimiter`，超阈值即中止闭环 |
| watchdog 生效 | ✅ 已实现 | 观测/位姿超时即停止输出并记录事件 |
| 仿真场景先行 | ✅ | 全部实验在 MOCK/离线回放中完成 |
| 动作发布开关 | ✅ 默认 false | `infer.allow_command_publish=false`；打开时需同时满足 estop+watchdog 门禁（配置校验强制） |
