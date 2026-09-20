# CVAE + 动作分块 + 时间集成 × Google Cartographer 2D 激光 SLAM

> 端到端可运行工程：**视觉观测 → CVAE 策略 → 动作分块 → 时间集成 → 动作输出**，
> 并由**真实 Google Cartographer**（2D 激光 SLAM）提供位姿输入与轨迹精度评估。

**读任何数字之前，请先接受三条硬限定**：

1. **数据全部是 MOCK**：合成 2D 世界 + 64×64 合成观测 + 解析射线投射激光，结论只在仿真条件下成立；
2. **无真机**：执行循环默认 dry-run，动作只写日志，从不发布控制指令（`allow_command_publish=false`）；
3. **每个数字都有产物**：报告与图表由脚本从 `logs/`、`outputs/` 汇总生成，缺失项写 `MISSING / NOT_RUN`，不做手工填数。

<p align="center">
  <img src="outputs/slam/cartographer_map_with_trajectory.png" alt="Cartographer 占用栅格地图 + 轨迹" width="640">
</p>

<p align="center"><i>真实 Cartographer 输出的 MOCK 场地占用栅格地图与估计轨迹（红）；灰色虚线为 MOCK 世界真值，
刚体对齐到 map 系后两者几乎完全重合。</i></p>

---

## 目录

- [1. 这个工程在做什么](#1-这个工程在做什么)
- [2. 结果速览](#2-结果速览)
- [3. 系统架构与数据流](#3-系统架构与数据流)
- [4. 目录结构](#4-目录结构)
- [5. 快速开始](#5-快速开始)
- [6. 数据契约（MOCK）](#6-数据契约mock)
- [7. 方法与实现](#7-方法与实现)
- [8. 训练与 CPU/GPU 分工](#8-训练与-cpugpu-分工)
- [9. 推理与评测](#9-推理与评测)
- [10. Cartographer SLAM 集成](#10-cartographer-slam-集成)
- [11. 消融实验（A1–A6）](#11-消融实验a1a6)
- [12. 可复现性与证据链](#12-可复现性与证据链)
- [13. 已知限制与不适用场景](#13-已知限制与不适用场景)
- [14. 许可与致谢](#14-许可与致谢)

---

## 1. 这个工程在做什么

四条技术主线，每条都必须给出**可验证的产物**，而不是只写"已实现"：

| 主线 | 要回答的问题 | 实现 | 回答方式 |
|---|---|---|---|
| **CVAE 策略** | 同一观测对应多种合理动作时，策略能否表达多模态？ | 先验 `p(z\|o,s)` / 后验 `q(z\|o,s,a)` + 解码器一次输出整块动作 | A1 消融（关潜变量）+ 先验采样多样性/命中率指标 |
| **动作分块** | 一次预测 H 步，能省多少推理、会损失多少精度？ | `ChunkScheduler(H, n_exec)`、掩码末段填充 | A5 消融：H ∈ {1, 4, 8, 16, 32} 全量对照 |
| **时间集成** | 融合最近多次预测能否降低执行抖动？代价是什么？ | `uniform / exponential / inverse_age` 三种权重 + 过期丢弃、禁止补零 | A6 消融：开/关集成、三种权重、3 条 test episode 闭环 |
| **真实 Cartographer** | SLAM 位姿能否作为策略输入、精度如何？ | ROS2 桥接 + 离线回放建图 + `trajectory_query` 查轨迹 | A4 消融（29→24 维状态可 diff）+ ATE/RPE/漂移 |

工程侧的同等目标是**可复现、可审计**：配置哈希、全局 seed、结构化 JSONL 日志、数据 manifest、
阶段门禁（单 batch 过拟合门禁、schema 校验、配置校验）与证据索引（命令 ↔ 产物 ↔ 日志）。

---

## 2. 结果速览

| 问题 | 实测结论（限定条件见括号） | 证据文件 |
|---|---|---|
| CVAE 是否学到多模态？ | 4 次先验采样之间存在差异（`sample_diversity=0.0585`），但"至少一次采样接近真值"的命中率仅 `5.97%`；关闭潜变量（A1）后重构误差与基准相当（0.2305 vs 0.2351）——**多模态能力有限，未构成优势** | `logs/infer_eval.json`、`outputs/ablation/summary.csv` |
| 动作分块的 H 怎么选？ | 逐步误差随 H 单调变差：H=1 `0.163` < H=4 `0.190` < H=8 `0.235` < H=16 `0.313` < H=32 `0.373`；`n_exec=4` 下每步摊薄延迟约 `0.74 ms`（各 run 几乎相同） | `outputs/ablation/summary.csv` |
| 时间集成降低抖动了吗？ | 降低但幅度小：`−4.7%`（二阶差分方差 `9.14e-7 → 8.71e-7`）。原因是推理用先验均值（`latent.mode=mean`），预测本身已经确定；换成随机采样策略收益会变大 | `logs/infer_eval.json` |
| SLAM 位姿可用吗？ | 可用：`1037` 个位姿样本、有效位姿比例 `100%`、时间对齐失效 `0`；位姿进入状态向量（A4 消融使状态 29→24 维） | `outputs/slam/trajectory_metrics.json` |
| Cartographer 轨迹精度？ | `ATE(RMSE)=0.0275 m`、`RPE(RMSE)=0.0360 m`、末端漂移 `1.59 mm`、航向误差均值 `0.0024 rad`（MOCK 真值 + 真实 Cartographer，仅代表仿真世界） | `outputs/slam/trajectory_metrics.json` |
| 算力是否真的压在 GPU？ | `GPU_BOUND`：训练 compute 占比 `99.45%`，data wait `0.33%`、H2D `0.23%`；8 GB 显存内 batch=192 | `logs/20_device_usage.json` |

> 一句话总结：**"能被真实 SLAM 驱动的多模态动作策略"这条链路已经打通并全程留痕**；
> 但策略侧的收益（多模态、时间集成）在本 MOCK 数据与当前的执行配置下**幅度有限**，本仓库如实标注，不夸大。

---

## 3. 系统架构与数据流

```mermaid
graph TB
    subgraph SIM["MOCK 世界（source=MOCK）"]
        W["2D 世界模型<br/>场地 + 障碍物"]
        TR["基座轨迹"]
        IMG["64×64 俯视观测"]
        LZ["激光扫描流<br/>data/slam/scans.npz"]
    end

    subgraph ROS2["ROS2 侧（conda env cartographer_ros，无 torch）"]
        BAG["ROS2 bag<br/>/scan /tf_static /clock"]
        CART["cartographer_offline_node<br/>真实 Google Cartographer 2D"]
        BR["cartographer_bridge<br/>TF map→odom→base_link"]
        PS["pose_stream.jsonl<br/>双时间戳"]
        MP["占用栅格地图<br/>pbstream → pgm/yaml"]
    end

    subgraph TRAIN["训练/推理侧（conda env unitree_rt）"]
        EP["episode npz / processed npy"]
        DS["WindowDataset<br/>K 帧图像 + state + action_chunk + mask"]
        MODEL["CVAEPolicy<br/>vision → state → q(z)/p(z) → decoder"]
        CHUNK["ActionChunker<br/>H / n_exec 调度"]
        ENS["TemporalEnsembler<br/>uniform / exponential / inverse_age"]
        OUT["执行动作 → 限幅/watchdog → 日志"]
    end

    W --> IMG
    W --> LZ
    TR --> IMG
    TR --> LZ
    LZ --> BAG --> CART
    CART --> BR --> PS
    CART --> MP
    W --> EP
    TR --> EP
    IMG --> EP
    PS --> EP
    EP --> DS --> MODEL --> CHUNK --> ENS --> OUT
```

三条约束贯穿整个工程：

- **位姿回路是真实闭环**：策略输入里带 Cartographer 位姿（`slam_pose` 4 维 + `slam_valid` 1 维），
  而 Cartographer 的输入是同一 MOCK 世界的激光扫描——"策略用了 SLAM"这件事由 A4 消融证明（状态维度可 diff）。
- **两侧 Python 严格隔离**：训练侧 `unitree_rt`（torch/CUDA）与 ROS2 侧 `cartographer_ros`（rclpy，不装 torch）
  互不污染；桥接通过 JSONL/TF 与命令行完成，不做进程内 import。
- **外部依赖可替换**：外部环境根（conda 锁定、Cartographer 构建脚本与产物）通过 `configs/paths.yaml` 的
  `env_root` 引用，可用环境变量覆盖；仓库本身不含任何本机绝对路径。

---

## 4. 目录结构

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `configs/` | YAML/Lua 配置与消融覆盖（唯一超参事实源） | 代码逻辑、运行状态 |
| `src/datasets/` | episode 读取、窗口切片、过滤、归一化、mock 生成 | 训练循环 |
| `src/models/` | CVAE、视觉主干、动作分块、时间集成、损失 | 数据读写 |
| `src/train/` | 训练循环、checkpoint、KL 退火、验证 | 推理调度 |
| `src/infer/` | 离线评测、闭环执行、rollout | 训练更新 |
| `src/slam/` | Cartographer 桥接、位姿流、轨迹指标、扫描回放 | 策略逻辑 |
| `src/utils/` | 配置、日志、seed、指标、CUDA 自检、可视化 | 业务逻辑 |
| `scripts/` | 编号化命令入口（幂等，00–92） | 算法实现 |
| `tests/` | 契约测试（96 项，不依赖网络与真机） | 长训练用例 |
| `data/` | `schema/` 数据契约、`slam/` 真值与世界定义；大文件不入库 | 代码、权重 |
| `outputs/` | `figures/`、`eval/`、`ablation/`、`slam/`、报告 | 源数据 |
| `docs/` | 架构、算法笔记、数据格式、SLAM 集成、实验登记、排错 | 可执行代码 |
| `evidence/` | 证据索引（命令 ↔ 产物 ↔ 日志） | 手工截图 |
| `Prompt/`、`TASK.md`、`.agent/` | 任务规格与过程台账（阶段门禁、决策、失败记录） | 代码与产物 |

---

## 5. 快速开始

### 5.1 环境要求

| 项 | 本工程实测基线 | 备注 |
|---|---|---|
| OS / GPU | Ubuntu 26.04 LTS / NVIDIA RTX 5070 Laptop 8 GB（capability `(12,0)` = `sm_120`） | 训练侧 |
| CUDA | toolkit 13.2（nvcc）；`torch 2.14.0+cu130` | matmul 实测 8.8–14.7 TFLOPS（两次不同规模自检） |
| 训练侧 Python | conda env `unitree_rt`，Python 3.12.9 | 需要 numpy / torch / pyyaml / matplotlib |
| ROS2 侧 | RoboStack `ros-jazzy-cartographer-ros 2.0.9003`（conda env `cartographer_ros`，**不装 torch**） | 仅 S8 阶段需要 |
| RMW / domain | `rmw_cyclonedds_cpp` / `ROS_DOMAIN_ID=1` | 与实机隔离，避免误连 |

> 只有 S8（Cartographer 建图与轨迹评估）需要 ROS2 侧环境；S0–S7、S10 只需要训练侧环境。

### 5.2 配置

`configs/paths.yaml` 是唯一的路径事实源：

- `project_root: auto` —— 由 `src/utils/config.py` 的代码位置**自动推导**，克隆到任意目录都能跑，仓库里不写本机路径；
- 外部路径（`env_root`、`cartographer_prefix`、`external_bag_dir` 等）给的是带 `${HOME}` 的默认值，**换机器请用环境变量覆盖**。

| 环境变量 | 覆盖字段 | 用途 |
|---|---|---|
| `CVSLAM_ENV_ROOT` | `env_root` | 外部环境与工具根（conda 锁定、Cartographer 构建脚本与产物） |
| `CVSLAM_TRAIN_PYTHON` | 脚本中的 `TRAIN_PY` | 训练侧解释器路径 |
| `CVSLAM_CARTOGRAPHER_PREFIX` | `cartographer_prefix` | Cartographer 安装前缀 |
| `CVSLAM_BAG_DIR` | `external_bag_dir` | 外部 ROS2 bag 目录 |
| `UNITREE_WORKSPACE` | `unitree_workspace` | 上层工作区（ROS2 rootless 安装所在） |

### 5.3 一键命令

```bash
make help          # 列出全部入口

make env           # S0 环境基线 + CUDA 自检
make data          # S2 生成 MOCK 数据 + 激光扫描
make dataset       # S3 窗口化 / 归一化 / 划分
make check         # S3 数据体检（schema / 形状 / 时间戳 / 对齐）
make gate          # S5 单 batch 过拟合门禁
make train         # S5 训练
make eval          # S6 离线评测（含无集成/有集成对照）
make closed-loop   # S7 闭环执行（默认 dry-run，不下发指令）
make slam          # S8 生成自包含 Cartographer 配置 + 离线回放建图
make slam-eval     # S8 轨迹 ATE / RPE / 漂移 + 出图
make ablation      # S10 消融矩阵（A1–A6）
make test          # 契约测试（pytest）
make report        # 汇总报告（outputs/REPORT.md）
make evidence      # 证据索引（evidence/index.md）
make viz           # 导出看板数据 + 生成 outputs/figures/index.html
```

### 5.4 端到端复现顺序

```bash
# 0) 环境自检（两个根的存在性 + CUDA + ROS2 探测）
bash scripts/00_env_check.sh

# 1) 数据：MOCK 世界 → 观测/状态/动作块 + 激光扫描
python scripts/10_gen_mock_data.py
python scripts/11_build_dataset.py
python scripts/12_data_check.py          # 出图 outputs/figures/12_data_samples.png

# 2) 训练：先过单 batch 门禁，再正式训练
bash scripts/21_overfit_single_batch.sh  # 门禁：过拟合不上就不允许进入正式训练
bash scripts/20_train.sh                 # 出图 outputs/figures/loss_curve.png、device_split.png

# 3) 推理：离线评测 + 闭环执行（dry-run）
bash scripts/30_infer_offline.sh --split test
bash scripts/31_infer_closed_loop.sh

# 4) SLAM：离线回放建图 → 查询轨迹 → ATE/RPE
bash scripts/40_slam_bringup.sh
bash scripts/41_record_slam_bag.sh
bash scripts/44_query_pbstream_trajectory.sh
bash scripts/42_slam_replay_eval.sh      # 出图 outputs/slam/*.png

# 5) 消融与汇总
bash scripts/50_run_ablation.sh
python scripts/51_summarize_ablation.py
python scripts/90_report.py
bash scripts/60_collect_evidence.sh
```

### 5.5 测试

```bash
python -m pytest -q tests/     # 96 passed（本仓库实测）
```

契约测试覆盖：配置校验（未知字段/越界/互斥必须报错）、schema、窗口构建、动作分块一致性、
CVAE 形状与推理纯度、损失、时间集成硬约束、位姿工具、时间同步、checkpoint 往返、CUDA 可用性。

---

## 6. 数据契约（MOCK）

### 6.1 规格

| 项 | 值 | 来源 |
|---|---|---|
| 数据版本 | `mock-v1-60ep-seed0`（source=MOCK） | `data/manifest.json` |
| episode / 帧数 / 频率 | 60 / 7200 / 10 Hz | 同上 |
| 图像 | 64×64×3 RGB，均值 0.5 / 标准差 0.25 | `configs/data.yaml` |
| 观测历史 | K=2 帧，帧间隔 1 | 同上 |
| 状态维度 | **29**（含 SLAM 位姿 5 维） | 同上 |
| 动作维度 | **7** = `dx, dy, dz, rx, ry, rz, gripper`（增量，base 系） | 同上 |
| 动作块 | H=8，stride=2，末段 mask=0 | 同上 |
| 窗口数 | train 2478 / val 531 / test 531 | `data/manifest.json` |
| 过滤 | 3600 → 3540 窗口，拒绝率 1.67%（阈值 35%） | `logs/12_data_check.json` |
| 激光 | 180 束/帧，量程 0.15–8.0 m，σ=0.01 m | `configs/data.yaml` |
| 世界 | 6 m × 6 m，8 个障碍物，分辨率 0.05 m | 同上 |
| 多模态 | 60/60 episode 标注为多模态（approach bias 分布：−1 有 19 条、+1 有 41 条） | `data/manifest.json` |
| 噪声 | 动作噪声 3e-4 m/step、观测延迟 1 帧、SLAM 丢帧 2%、SLAM 漂移 σ=3e-3 | `configs/data.yaml` |

状态向量的**块顺序即张量顺序**（`data/schema/dataset_schema.json` 是机器可读契约）：

| # | 块 | 维度 | 归一化 | 说明 |
|---|---|---|---|---|
| 1 | `joint_pos` | 7 | ✓ | 关节位置 |
| 2 | `joint_vel` | 7 | ✓ | 关节速度 |
| 3 | `gripper` | 1 | ✓ | 夹爪开合 |
| 4 | `ee_pose` | 7 | ✓ | 末端位姿 |
| 5 | `slam_pose` | 4 | ✓ | `dx, dy, sinΔyaw, cosΔyaw`（**相对 episode 起点**；角度用 sin/cos 编码，禁止直接归一化弧度） |
| 6 | `slam_valid` | 1 | 直通 | 时间对齐是否有效（容差 0.10 s） |
| 7 | `time_feat` | 2 | 直通 | 时间特征 |

> A4 消融（`no_slam_input`）把第 5、6 块整体移除：状态维度 29 → 24，其余配置逐字段可 diff。

### 6.2 为什么用 MOCK

本工程的目标是**打通并审计"策略 + 真实 SLAM"的链路**，而不是刷真实机器人指标。数据由规则化隐式策略生成：

- 观测是 64×64 俯视示意图，只提供目标方位，**不提供尺度线索**；
- 激光由同一世界模型解析射线投射得到，可直接驱动真实 Cartographer，保证"真值—激光—参考轨迹"同源；
- MOCK 数据**不代表真实机器人动力学与真实相机成像**；
- 数据生成参数（含噪声依据）都写在 `configs/data.yaml`，失败记录保留在 `docs/experiments.md`：
  动作噪声最初取 0.004 时噪声标准差≈信号标准差的 94%，任务退化成"预测噪声"，
  `recon` 卡在 0.28 不下降；调到 3e-4 后噪声约为典型步长的 5%–15%。

### 6.3 数据抽样图

<p align="center">
  <img src="outputs/figures/12_data_samples.png" alt="数据抽样核查" width="900">
</p>

`outputs/figures/12_data_samples.png` · 生成命令：`python scripts/12_data_check.py` · 源日志：`logs/12_data_check.json`

这张图回答"数据到底长什么样"，六个子图逐条对应：

| 子图 | 内容 | 怎么读 |
|---|---|---|
| 左上 / 中上 `obs frame 0/1 (t=71/72)` | 同一时刻的两帧历史观测（俯视示意图） | 白/绿色方块是目标与末端，背景方块是场地纹理；两帧几乎重合，说明 K=2 的帧间隔为 1（10 Hz 下相差 0.1 s） |
| 右上 `state @ t=72 (dim=29)` | 该时刻 29 维状态向量 | 各块边界（`joint_pos … time_feat`）；`time_feat` 是 0/1 附近的时钟特征，`slam_valid=1` 表示该样本位姿有效 |
| 左下 `action chunk: translation deltas` | 动作块 `a[0..2]`（dx/dy/dz，归一化） | 8 个点就是 H=8 的整块预测目标；平移增量在 ±0.003 量级 |
| 中下 `action chunk: rotation deltas` | 动作块 `a[3..5]`（rx/ry/rz） | 旋转增量比平移大一个数量级（0.01 rad 级） |
| 右下 `gripper` | 整条 episode 的夹爪轨迹 + 当前 chunk 窗口 | 蓝线是全程曲线、橙线是"被本窗口覆盖的 8 步"，用于核对 chunk 切片是否对齐 |

> 结论：图像只给方位、状态块顺序与 schema 完全一致、动作块切片与 mask 语义正确。
> 这些只是"数据像样"的最低要求，**不能**用来支撑任何真实机器人结论。

---

## 7. 方法与实现

### 7.1 CVAE 策略

```
先验      p(z | o, s)     = N( μ_p(o,s), diag(σ_p²(o,s)) )      src/models/encoder.py:PriorNetwork
后验      q(z | o, s, a)  = N( μ_q(o,s,a), diag(σ_q²(o,s,a)) )   src/models/encoder.py:PosteriorEncoder
解码器    π(a | o, s, z)  = [B, H, 7]                            src/models/cvae_policy.py:ActionDecoder
重参数化  z = μ + σ ⊙ ε,  ε ~ N(0, I)                            src/models/layers.py:reparameterize
```

损失（公式：`src/models/losses.py`；权重装配：`src/train/losses.py`）：

```
L_recon  = Σ_h m_h · w_h · Huber(pred_h, gt_h) / Σ_h m_h w_h   （w_h = 0.95^h，越靠后的步权重越低）
L_kl     = 0.5 · Σ_d ( σ_q²/σ_p² + (μ_p−μ_q)²/σ_p² − 1 + log σ_p² − log σ_q² )，带 free-bits 下限
L_smooth = 掩码加权的一阶差分均值
L_total  = w_recon·L_recon + β(t)·L_kl + λ_smooth·L_smooth
```

**两个数值稳定化设计，都有实测依据**（详见 `docs/algorithm_notes.md`）：

1. **输出层零初始化**：早期版本在 `μ/log_var` 输出上加 LayerNorm，强制 `μ≈1`，初始 KL 高达 20–25 nats，
   直接压住重建项，单 batch 过拟合门禁 FAIL（recon 0.42 不下降）；改为零初始化后初始 KL=0，门禁 PASS。
2. **free-bits + 小 β**：`β` 终值 0.05、`free_bits=0.02 nats/维`、300 步线性退火。
   `β=1.0` 时后验直接塌缩（`val/kl ≡ 0`，退化为无条件 VAE）；改小 β 后 `val/kl ≈ 0.35`。

**推理纯度**：`predict_chunk(mode="mean"|"sample", n_samples=k)` 只走先验；`mode="infer"` 下传入真实动作块会
直接抛错（`tests/test_cvae_shapes.py` 双重拦截），保证"推理没有偷看后验"。

| 结构 | 取值 |
|---|---|
| 视觉主干 | `small_cnn`（自实现，`pretrained=false`，离线可复现），特征 128 维，GroupNorm |
| 多帧聚合 | GRU（hidden 128），K=2 帧共享 CNN 权重 |
| 状态编码 | 2×128，ReLU，dropout 0.1 |
| 潜变量 | dim 16；先验 2×128，后验 2×256，`logvar_clamp=[−6, 4]` |
| 解码器 | 2×256，`chunk_full`：一次输出 `[B, H, 7]` |
| 可训练参数 | 814,616 |

### 7.2 动作分块

- **切片语义唯一实现**：`src/utils/chunking.py:build_chunk()` 取 `action[t : t+H]`，越界步补零并置 `mask=0`；
  `WindowDataset` 与 `ActionChunker` 共用它，由 `tests/test_action_chunker.py` 交叉验证。
- **调度**：`ChunkScheduler(H, n_exec)` 每 `n_exec` 步重新预测，尾部（`H − n_exec`）显式丢弃，不做隐式复用。
- **H 的取舍**（A5 实测，误差为归一化空间 step-0 L1）：

| H | 1 | 4 | 8（基准） | 16 | 32 |
|---|---|---|---|---|---|
| 重构 L1 | **0.163** | 0.190 | 0.235 | 0.313 | 0.373 |
| 最优 val/total | **0.0327** | 0.0484 | 0.0694 | 0.108 | 0.144 |
| 含义 | 每步都要推理，精度最高、闭环开销最大 | 短块，响应快 | 折中 | 长块，误差明显上升 | 更长块，误差最大 |

> 关键限定：本轮所有 run 的 `n_exec=4`，因此"每步摊薄延迟"几乎相同（≈0.74 ms/步）；
> H 的差异主要体现在**误差与响应滞后**，而不是单步延迟。

### 7.3 时间集成

```
action(t) = Σ_j w_j · a_j(t) / Σ_j w_j     （只对 t0_j ≤ t < t0_j + H 且 mask>0 的预测求和）
uniform      w = 1
exponential  w = decay^(t − t0_j)
inverse_age  w = 1 / (1 + (t − t0_j))
```

实现：`src/models/temporal_ensemble.py`。硬约束（全部来自踩坑记录）：

- **冷启动**：只有一条预测时直接返回它，权重归一化不会除零；
- **过期丢弃**：`t0 + H ≤ t` 的记录在 `update` 时移除，并计入 `stats()['n_expired']`；
- **禁止补零**：没有贡献者时返回 `None`，绝不返回 0 动作（补零会把动作系统性拉向原点）；
- **逐步掩码**：`mask` 必须是 `[H]` 数组，标量掩码直接抛错；
- **不回改已执行动作**：融合只作用于调用方请求的时刻，调用方按时间单调推进。

代价是**用平滑换滞后**：`decay` 越小越信任最新预测（响应快、抖动大），`decay → 1` 等价于全体等权（最平滑、滞后最大）。

### 7.4 与 Cartographer 的耦合点

1. 位姿进入状态向量的形式是**相对 episode 起点的 `(dx, dy, sinΔyaw, cosΔyaw)` + `valid` 标志**；
2. 位姿既作为策略输入，也用于轨迹一致性评估（ATE/RPE）与 A4 消融对照；
3. 时间对齐容差：策略侧 0.10 s、评估侧 0.2 s，超容差样本 `slam_valid=0`；
4. 详细实现与实测见 `docs/cartographer_integration.md`。

---

## 8. 训练与 CPU/GPU 分工

### 8.1 训练摘要（run=base）

| 项 | 值 |
|---|---|
| 设备 | `cuda`（`device=cuda` 时若无 GPU 直接报错，禁止静默退回 CPU） |
| 步数 / epoch / 耗时 | 456 / 60 / 21.8 s |
| 最终 train loss / 最优 val total | 0.07161 / **0.06945** |
| 优化器 / 调度 | AdamW，lr 1e-3，weight decay 1e-4，cosine + 50 步 warmup |
| batch / AMP / 梯度裁剪 | 192 / 开启 / 1.0 |
| seed | 0 |
| config hash | data `01fd9245568d`、model `9ac9a2bbb306`、train `f25b306f3dbf`、data+model `6b5ee4759573` |
| 单 batch 过拟合门禁 | **PASS**：0.2779 → 0.04464，相对下降 83.9%（阈值 50%，batch=16） |

### 8.2 训练曲线

<p align="center">
  <img src="outputs/figures/loss_curve.png" alt="训练曲线" width="900">
</p>

`outputs/figures/loss_curve.png` · 生成命令：`python scripts/90_report.py` · 源数据：`logs/train_metrics.jsonl`、`logs/val_metrics.jsonl`

| 子图 | 横 / 纵轴 | 怎么读 | 支持的结论 |
|---|---|---|---|
| 左上 `total loss` | training step / total loss | 浅蓝为原始值，红线为 9 步滑动平均 | 0–100 步快速下降后进入平台期（0.07 量级），60 epoch 内无发散 |
| 右上 `recon loss (huber)` | training step / Huber | 与 total 形状一致 | 重建项主导总损失，KL 只占小权重（β=0.05） |
| 左下 `KL (nats)` | training step / KL | 起始 ≈0（零初始化的结果），40 步附近升至 ~0.95 后回落并稳定在 ≈0.35 | **后验没有塌缩**：潜变量确实被使用（塌缩会表现为 KL 恒为 0） |
| 右下 `beta (KL weight)` | training step / β | 0 → 0.05 线性退火，300 步到顶 | 退火策略与 `configs/train.yaml` 一致，前 300 步不惩罚 KL |

> 结论：训练在 MOCK 数据上稳定收敛、KL 未塌缩。但 `val/total` 是**归一化动作空间**的重建误差，
> 不能当作真实机器人精度，也不能与其它数据集的数值横向比较。

### 8.3 CPU/GPU 分工

<p align="center">
  <img src="outputs/figures/device_split.png" alt="CPU/GPU 分工实测" width="900">
</p>

`outputs/figures/device_split.png` · 生成命令：`bash scripts/20_train.sh` · 源数据：`logs/20_device_usage.json`

| 子图 | 内容 | 读数 |
|---|---|---|
| 左 `每步耗时拆分` | 每步耗时堆叠：data wait（CPU workers）+ host→device copy + compute（GPU forward/backward） | compute 柱几乎顶满，data wait 与 H2D 只有薄薄一层 |
| 右 `时间占比 · 判定` | 饼图给出三类时间占比与判定结论 | `compute 99.5%`、`data wait 0.3%`、`H2D 0.2%` → **GPU_BOUND** |

> 结论与建议（原文写在图页脚）：主算力已经在 CUDA 上且接近饱和（workers=8、torch_threads=8、batch=192）；
> 继续提速只能靠**减小模型/输入分辨率或增大 batch** 摊薄固定开销，再加 DataLoader worker 已无收益。

---

## 9. 推理与评测

### 9.1 指标（run=base，split=test，checkpoint step=168）

| 指标 | 值 | 说明 |
|---|---|---|
| 重构 L1（8 步掩码平均） | `0.2351` | 归一化动作空间，不是物理单位 |
| chunk 第 0 步 L1 | `0.2063` | 对比基线 `0.7157`（**零动作** ≈ 训练集动作均值），约改善 3.5× |
| 逐步误差（step 0→7） | 0.206 → 0.282 | 预测越靠后越难，是动作分块的固有代价 |
| 采样多样性（4 次先验采样） | `0.0585` | 越大说明多模态越明显 |
| 多模态命中率 | `0.0597` | "至少一次采样接近真值"的比例——**偏低，如实标注** |
| 单次预测延迟 | p50 `3.00 ms` / p95 `3.12 ms` | `n_exec=4` → 摊薄 `0.74 ms/步`（已排除 CUDA 预热批次） |
| 二阶差分方差（抖动） | `9.14e-7` → `8.71e-7` | 无集成 → 有集成（decay=0.5，3 条 test episode 闭环），`−4.7%` |

### 9.2 动作对比

<p align="center">
  <img src="outputs/figures/30_action_compare_base.png" alt="动作对比" width="900">
</p>

`outputs/figures/30_action_compare_base.png` · 生成命令：`bash scripts/30_infer_offline.sh --split test` · 源数据：`logs/infer_eval.json`

- 四条通道自上而下是 `dx / dy / dz / gripper`（归一化增量），横轴是执行步（test split 前 150 步）；
- **黑实线** = 真值动作，**蓝线** = 不用集成（每步取最新一次预测），**红线** = 时间集成（decay=0.5）；
- 怎么读：t≈120 处三条线同时出现尖峰（方向切换/抓取动作），说明集成没有把突变的真值平滑掉；
  平台段三线差异很小——这正是"集成收益有限"的直接体现；
- 限制：只有 3 条 test episode，个体差异大，不能据此给出统计显著性结论。

### 9.3 时间集成权重热图

<p align="center">
  <img src="outputs/figures/30_chunk_weights_base.png" alt="时间集成权重热图" width="760">
</p>

`outputs/figures/30_chunk_weights_base.png` · 生成命令：同上一节 · 数据来源：`TemporalEnsembler.weight_matrix()`

- 横轴 `execution step t`，纵轴 `prediction index`（越靠上越新），颜色是归一化融合权重；
- 当前记录窗口内只有末端若干步出现非零权重，说明融合只覆盖**最近 H 步内仍存活的预测**，
  过期预测按硬约束被丢弃（`n_expired` 计数）；
- 这张图的价值在**可核对性**：任何"集成到底融合了哪几次预测"的疑问都能落到矩阵上；
  同时它也暴露了本次可视化的一个局限——稀疏矩阵不利于观察权重衰减的形状。

### 9.4 推理延迟与逐步误差

<p align="center">
  <img src="outputs/figures/30_per_step_error_base.png" alt="推理延迟与逐步误差" width="900">
</p>

`outputs/figures/30_per_step_error_base.png` · 生成命令：同上一节 · 源数据：`logs/infer_eval.json`

| 子图 | 内容 | 怎么读 |
|---|---|---|
| 左 `per-prediction latency` | 每次预测的延迟直方图 | p50=3.00 ms、p95=3.12 ms 两条参考线；样本数很少（预热后的少数批次），只作量级参考 |
| 右 `per-step L1` | 块内第 0…7 步的掩码 L1 误差 | 单调上升 0.206 → 0.282：**分块预测的误差随预测跨度增长**，这是 H 不能无限放大的直接证据 |

### 9.5 闭环执行（dry-run）

`logs/31_closed_loop_summary.json`：执行 60 步、`publish_allowed=false`（只写日志）、
限幅事件 0 次、NaN 事件 0 次、位姿来源 `cartographer_pose_stream`（1037 样本、有效率 100%）、
位姿缺失事件 1 次（被 watchdog 记录，而不是静默忽略）。

---

## 10. Cartographer SLAM 集成

### 10.1 流程

```mermaid
graph LR
    A["MOCK 世界 + 激光<br/>data/slam/scans.npz"] --> B["scripts/41 录制 ROS2 bag<br/>/scan /tf_static /clock"]
    B --> C["scripts/40 离线回放<br/>cartographer_offline_node + g1_2d.lua"]
    C --> D["trajectory.pbstream<br/>+ 占用栅格地图"]
    C --> E["pose_stream.jsonl<br/>双时间戳位姿流"]
    D --> F["scripts/44 trajectory_query"]
    F --> G["scripts/42 ATE / RPE / 漂移<br/>+ 3 张图"]
```

要点：

1. **配置自包含**：本机 RoboStack 构建的 Lua `include` 解析有缺陷，因此 `scripts/43` 把官方默认配置
   内联进 `configs/cartographer/g1_2d.lua`（头部记录来源相对路径与 sha256），只允许改 `*.overrides.lua`；
2. **两类位姿分清楚**：`pose_stream.jsonl` 是**在线（前端 + 局部优化）位姿**——策略实际可用的那一种；
   `pbstream` 查询出的轨迹是**含全局优化的最终结果**。两者分别标注、不混用；
3. **对齐必须做刚体变换**：map 系与世界系之间存在固定旋转（本实验 `yaw_offset≈0.586 rad`），
   只做平移会让两条轨迹"看起来差几米"（未对齐 ATE 1.483 m → 对齐后 0.0275 m）。

### 10.2 轨迹精度指标

| 指标 | 值 | 说明 |
|---|---|---|
| 样本数 / 时长 | 1037 / 118.8 s | 时间对齐用最近邻，容差 0.2 s，失效 0 个 |
| 对齐方式 | `umeyama_rigid`（estimate → reference，无尺度） | 旋转 + 平移，禁止缩放 |
| ATE(RMSE) | **0.0275 m** | 对齐后的平均轨迹误差 |
| RPE(RMSE, Δt=1 s) | 0.0360 m | 相对位姿误差 |
| 末端漂移 | 0.00159 m | 终点偏差 |
| 航向误差（均值 / 最大） | 0.00236 / 0.155 rad | 角度用 sin/cos 编码，避免弧度线性归一化 |
| 证据等级 | MOCK 真值 + 真实 Cartographer 输出 | 精度数字只代表仿真世界，**不代表真实场地** |

### 10.3 图①：占用栅格地图 + 轨迹

<p align="center">
  <img src="outputs/slam/cartographer_map_with_trajectory.png" alt="Cartographer 地图与轨迹" width="620">
</p>

`outputs/slam/cartographer_map_with_trajectory.png` · 生成命令：`bash scripts/42_slam_replay_eval.sh` · 源数据：`outputs/slam/cartographer_map.yaml` + `data/slam/pose_stream.jsonl`

- 灰白底图是 Cartographer 用 180 束激光建立的占用栅格地图（黑=障碍，白=可通行）；
- 场地在 map 系里呈菱形，正是 map↔world 之间存在 0.586 rad 固定旋转的可视化结果；
- 红色实线是 Cartographer 估计轨迹，灰色粗虚线是 MOCK 世界真值（按对齐矩阵逆变换到 map 系）；
- 虚线用**更粗 + 半透明**是刻意为之：当 ATE≈2.7 cm 时两条线几乎重合，细虚线会被实线完全盖住，
  图上就"看不出它们一致"——这是本工程踩过的可视化坑；
- 局限：栅格地图与轨迹都来自 MOCK 激光，真实场地的建图质量不由此图背书。

### 10.4 图②：刚体对齐后的轨迹对比

<p align="center">
  <img src="outputs/slam/trajectory_overlay.png" alt="轨迹对比" width="760">
</p>

`outputs/slam/trajectory_overlay.png` · 生成命令：同上 · 源数据：`outputs/slam/trajectory_metrics.json`

- 横/纵轴是米（已对齐到参考系中心），标题即 ATE / RPE；
- 蓝点是起点、红叉是终点，两者几乎重合 → 末端漂移仅 1.6 mm；红线与黑虚线在整个 118.8 s 内贴合；
- **必须读作"对齐后"**：对齐是评估的必要步骤（消除 map 系定义的刚体自由度），不是"调参把误差调小"——
  完整对齐矩阵、方向、是否带尺度都写在 `trajectory_metrics.json` 的 `alignment` 字段里，可复核。

### 10.5 图③：误差随回放时间

<p align="center">
  <img src="outputs/slam/ate_over_time.png" alt="误差随时间" width="760">
</p>

`outputs/slam/ate_over_time.png` · 生成命令：同上 · 源数据：`outputs/slam/trajectory_metrics.json`

- 横轴回放时间（0–119 s），纵轴对齐后位置误差；红色虚线标 ATE(RMSE)=0.0275 m；
- 前 5 秒有两处尖峰（最大 0.44 m），随后收敛到 0 附近——对应 **Cartographer 建图初始化阶段**；
  这说明 ATE 均值很小但最大值不小，**只看 RMSE 会漏掉冷启动段**，因此本仓库同时保留 `ate_max_m` 与原始序列；
- 局限：只有一段离线回放、单一场景、单次运行，没有重复试验与方差，不能外推为"精度稳定性"。

---

## 11. 消融实验（A1–A6）

汇总表（`outputs/ablation/summary.csv`；每行一次真实运行，config hash 用于证明"除消融变量外其余配置一致"）：

| run | 变量 | H | 潜变量 | SLAM 输入 | 步数 | 最优 val/total | 最终 train loss | 重构 L1 | 延迟 p50 (ms) | 抖动（无→有集成） |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | 基准 | 8 | ✓ | ✓ | 456 | 0.0694 | 0.0716 | 0.2351 | 2.96 | 9.14e-7 → 8.71e-7 |
| `chunk_len_H4` | A5 | 4 | ✓ | ✓ | 432 | 0.0484 | 0.0573 | 0.1904 | 2.99 | 9.49e-7 → 9.49e-7 |
| `chunk_len_H16` | A5 | 16 | ✓ | ✓ | 480 | 0.1085 | 0.0977 | 0.3132 | 2.96 | 9.23e-7 → 8.67e-7 |
| `chunk_len_H32` | A5 | 32 | ✓ | ✓ | 528 | 0.1444 | 0.1203 | 0.3727 | 2.97 | 7.06e-7 → 6.58e-7 |
| `no_chunk` | A5 | 1 | ✓ | ✓ | 750 | 0.0327 | 0.0380 | 0.1628 | 3.06 | 2.57e-7 → 2.57e-7 |
| `no_cvae` | A1 | 8 | ✗ | ✓ | 576 | 0.0609 | 0.0608 | 0.2305 | 2.98 | 1.11e-6 → 1.06e-6 |
| `no_slam_input` | A4 | 8 | ✓ | ✗（29→24 维） | 432 | 0.0707 | 0.0801 | 0.2354 | 3.03 | 1.09e-6 → 1.05e-6 |
| `ensemble_uniform` | A6 | 8 | ✓ | ✓ | 仅推理 | — | — | — | — | 9.14e-7 → 7.93e-7 |
| `ensemble_exp_d1p0` | A6 | 8 | ✓ | ✓ | 仅推理 | — | — | — | — | 9.14e-7 → 7.93e-7 |
| `no_temporal_ensemble` | A6 | 8 | ✓ | ✓ | 仅推理 | — | — | — | — | 9.14e-7 → 9.14e-7（关集成，前后必须完全一致） |

### 结论（每条都带限定）

1. **潜变量（A1）没有带来重构优势**：`no_cvae` 的 0.2305 甚至略优于 `base` 的 0.2351。
   这与 CVAE 的设计目标一致——潜变量是**为多模态表达服务**（同一观测多种合理动作），不是为降低 L1；
   但本实验的多模态指标（命中率 5.97%）也偏低，因此**不能宣称"多模态能力已被验证"**。
2. **SLAM 输入（A4）的影响在噪声范围内**：`no_slam_input` 0.2354 vs `base` 0.2351。
   该消融的意义在于**证明位姿确实进入了策略输入**（状态 29→24 维、配置逐字段可 diff），而不是证明精度提升。
3. **H 的取舍（A5）清晰**：H 越小越容易拟合（0.163 @ H=1），H 越大逐步误差越大（0.373 @ H=32）；
   `n_exec=4` 固定时每步摊薄延迟几乎相同（≈0.74 ms/步），所以 H 应由**闭环响应要求**来选，而不是由延迟来选。
   说明：本轮 `n_exec` 未随 H 联动（都是 4），因此"长块能省多少推理"在本表里**没有被真正验证**，
   属于已记录的实验设计局限。
4. **时间集成（A6）稳健但增益小**：三种权重都把抖动降下来（uniform 与 decay=1.0 结果相同，
   符合"权重趋同"的直觉；关集成时前后完全一致，是自检项），相对降幅 `−4.7%`。
   原因是推理取先验均值（`latent.mode=mean`），预测本身已确定；若换成先验采样策略，集成收益应显著变大——
   这是"结论在什么条件下成立"的关键限定。
5. **未跑项**：A7（观测历史帧数 K 扫描）标 `NOT_RUN`，不写结论。

### 消融图集（每 run 5 张，点击展开）

<details>
<summary><b>chunk_len_H4</b>（H=4；重构 L1 0.1904，最优 val 0.0484）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_chunk_len_H4.png" width="380"> | 训练/验证曲线与 β 退火：与基准同形状，步数更少（432） |
| <img src="outputs/figures/device_split_chunk_len_H4.png" width="380"> | CPU/GPU 分工：仍为 `GPU_BOUND`（compute ≈99.5%） |
| <img src="outputs/figures/30_action_compare_chunk_len_H4.png" width="380"> | 动作对比：短块预测，尖峰处跟随更紧 |
| <img src="outputs/figures/30_chunk_weights_chunk_len_H4.png" width="380"> | 时间集成权重热图 |
| <img src="outputs/figures/30_per_step_error_chunk_len_H4.png" width="380"> | 延迟直方图 + 逐步误差（H=4 → 只有 4 个点） |

</details>

<details>
<summary><b>chunk_len_H16</b>（H=16；重构 L1 0.3132，最优 val 0.1085）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_chunk_len_H16.png" width="380"> | 训练/验证曲线：重建项更高，符合"预测跨度变长" |
| <img src="outputs/figures/device_split_chunk_len_H16.png" width="380"> | CPU/GPU 分工：GPU_BOUND |
| <img src="outputs/figures/30_action_compare_chunk_len_H16.png" width="380"> | 动作对比：块内更平滑，块边界处响应滞后 |
| <img src="outputs/figures/30_chunk_weights_chunk_len_H16.png" width="380"> | 时间集成权重热图 |
| <img src="outputs/figures/30_per_step_error_chunk_len_H16.png" width="380"> | 逐步误差：16 步内持续上升 |

</details>

<details>
<summary><b>chunk_len_H32</b>（H=32；重构 L1 0.3727，最优 val 0.1444）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_chunk_len_H32.png" width="380"> | 训练/验证曲线：长程预测更难，误差整体上移 |
| <img src="outputs/figures/device_split_chunk_len_H32.png" width="380"> | CPU/GPU 分工：GPU_BOUND |
| <img src="outputs/figures/30_action_compare_chunk_len_H32.png" width="380"> | 动作对比：长块更"平滑"，但滞后更明显 |
| <img src="outputs/figures/30_chunk_weights_chunk_len_H32.png" width="380"> | 时间集成权重热图 |
| <img src="outputs/figures/30_per_step_error_chunk_len_H32.png" width="380"> | 逐步误差：32 步内从 ~0.21 升到 ~0.42 |

</details>

<details>
<summary><b>no_chunk</b>（H=1；重构 L1 0.1628，最优 val 0.0327 —— 本轮最好）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_no_chunk.png" width="380"> | 训练曲线：步数最多（750），loss 最低 |
| <img src="outputs/figures/device_split_no_chunk.png" width="380"> | CPU/GPU 分工：GPU_BOUND |
| <img src="outputs/figures/30_action_compare_no_chunk.png" width="380"> | 动作对比：`n_exec=1`，每步重新预测，抖动最低（2.57e-7） |
| <img src="outputs/figures/30_chunk_weights_no_chunk.png" width="380"> | 权重热图：H=1 时集成几乎没有融合空间 |
| <img src="outputs/figures/30_per_step_error_no_chunk.png" width="380"> | 逐步误差只有一个点 |

</details>

<details>
<summary><b>no_cvae</b>（A1：关闭潜变量，退化为确定性回归；重构 L1 0.2305）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_no_cvae.png" width="380"> | 训练曲线：无 KL 项，只有重建 + 平滑 |
| <img src="outputs/figures/device_split_no_cvae.png" width="380"> | CPU/GPU 分工：GPU_BOUND |
| <img src="outputs/figures/30_action_compare_no_cvae.png" width="380"> | 动作对比：确定性预测更"敢跟"，但失去多模态 |
| <img src="outputs/figures/30_chunk_weights_no_cvae.png" width="380"> | 时间集成权重热图 |
| <img src="outputs/figures/30_per_step_error_no_cvae.png" width="380"> | 延迟与逐步误差 |

</details>

<details>
<summary><b>no_slam_input</b>（A4：状态 29 → 24 维，去掉 SLAM 位姿与有效位）</summary>

| 图 | 说明 |
|---|---|
| <img src="outputs/figures/loss_curve_no_slam_input.png" width="380"> | 训练曲线：与基准几乎重合 |
| <img src="outputs/figures/device_split_no_slam_input.png" width="380"> | CPU/GPU 分工：GPU_BOUND |
| <img src="outputs/figures/30_action_compare_no_slam_input.png" width="380"> | 动作对比：差异在噪声量级 |
| <img src="outputs/figures/30_chunk_weights_no_slam_input.png" width="380"> | 时间集成权重热图 |
| <img src="outputs/figures/30_per_step_error_no_slam_input.png" width="380"> | 延迟与逐步误差 |

</details>

> 39 张图全部位于 `outputs/figures/` 与 `outputs/slam/`；
> `outputs/figures/index.html` 是本地图集页面（`make viz` 生成，
> 再执行 `python -m http.server 8788 --directory outputs`，浏览器访问 `/figures/index.html`）。

---

## 12. 可复现性与证据链

### 12.1 证据索引（节选，完整表见 `evidence/index.md`）

| 阶段 | 命令 | 关键产物 | 关键日志 |
|---|---|---|---|
| S0 环境 | `bash scripts/00_env_check.sh` | `logs/00_env_check.txt` | `logs/00_cuda_check.json` |
| S2 数据 | `python scripts/10_gen_mock_data.py` | `data/mock/mock_0000.npz` | `logs/10_gen_mock.log` |
| S3 管线 | `python scripts/11_build_dataset.py` | `data/processed/train.state.npy` | `logs/11_build_dataset.jsonl` |
| S3 体检 | `python scripts/12_data_check.py` | `outputs/figures/12_data_samples.png` | `logs/12_data_check.json` |
| S4 单测 | `python -m pytest -q tests/` | `tests/`（96 项） | `logs/90_pytest.log` |
| S5 门禁 | `bash scripts/21_overfit_single_batch.sh` | `logs/21_overfit_gate.json` | `logs/train_metrics.jsonl` |
| S5 训练 | `bash scripts/20_train.sh` | `checkpoints/base/meta.json` | `logs/train_summary.json` |
| S6 评测 | `bash scripts/30_infer_offline.sh --split test` | `outputs/eval/base_test_metrics.csv` | `logs/infer_eval.json` |
| S7 闭环 | `bash scripts/31_infer_closed_loop.sh` | `outputs/infer_samples/executed_actions_base.jsonl` | `logs/31_closed_loop_summary.json` |
| S8 建图 | `bash scripts/40_slam_bringup.sh` | `outputs/slam/trajectory.pbstream` | `logs/40_slam_bringup.log` |
| S8 轨迹 | `bash scripts/42_slam_replay_eval.sh` | `outputs/slam/trajectory_metrics.json` | `logs/42_trajectory_metrics.jsonl` |
| S10 消融 | `bash scripts/50_run_ablation.sh` | `outputs/ablation/summary.csv` | `logs/50_ablation.log` |
| 可视化 | `make viz` | `outputs/figures/index.html` | `outputs/figures/viz_data.json` |

### 12.2 可复现性措施

- **配置哈希**：`data / model / train` 三份配置各自哈希 + `data+model` 组合哈希，写入 checkpoint 与实验登记表，
  用来证明"消融只改了目标变量"；
- **全局 seed**：数据生成（`mock.seed=1234`）、划分（`splits.seed=0`）、训练（`seed=0`）分别固定；
- **结构化日志**：每个阶段写 `logs/` 下的 JSONL/JSON（指标、设备占比、门禁结论、对齐统计）；
- **数据契约**：`data/schema/dataset_schema.json` 机器可读，`scripts/12_data_check.py` 校验 0 错误才允许继续；
- **门禁式流程**：schema 校验、配置校验（未知字段必须报错）、单 batch 过拟合门禁、推理纯度拦截
  （推理链不得访问后验）——任一不过就停止，而不是"跳过继续跑"；
- **产物可追溯**：每张图的页脚写源文件与 checkpoint 步数，每份报告末尾列出原始产物路径。

### 12.3 关于本仓库

- 本仓库是**公开快照**：`configs/paths.yaml` 用 `project_root: auto` 自动推导仓库根，
  其余机器相关路径用 `${HOME}` + 环境变量覆盖，仓库内不包含个人绝对路径；
- 实验登记表里的 `commit` 字段记录的是**当时本地开发历史**的短哈希，公开快照中不可解析，仅作登记；
- 大文件（`data/raw|mock|processed`、`data/bags`、`checkpoints/`、`logs/`、`*.pbstream`、`*.pgm`）
  按 `.gitignore` 排除，只保留**小体积、可复核**的图表、指标与配置。

---

## 13. 已知限制与不适用场景

| # | 限制 / 问题 | 影响 | 处理方式 |
|---|---|---|---|
| 1 | 数据为 MOCK（合成世界 + 合成观测 + 合成激光） | 所有精度/收益结论只在仿真条件下成立 | 全文与图表均标注 MOCK；不写真实场地结论 |
| 2 | 无真机、无真实相机与动力学 | 不能说明真机可用性 | 执行链默认 dry-run；`allow_command_publish=true` 必须同时启用急停与 watchdog 才允许加载配置 |
| 3 | 在线 `cartographer_node` / `assets_writer` 因 gflags 重复定义无法启动 | 在线建图路径未验证 | 走离线回放 + `trajectory_query`；在线路径代码保留并标 `NOT_MEASURED` |
| 4 | 官方源码构建 Cartographer 需要 `sudo` 交互密码 | a 级降级链不可行 | 标 `BLOCKED`，改用 conda 二进制（c 级），并在 `docs/troubleshooting.md` 写明 |
| 5 | 高倍速回放时 TF 订阅会丢包 | 直接订阅 TF 采样轨迹不可靠 | 改为 `trajectory_query` 查询轨迹 |
| 6 | 时间集成收益仅 `−4.7%` | 不能宣称"显著降低抖动" | 如实给出数值与原因（推理取先验均值）；改用随机策略后复测列为 TODO |
| 7 | 多模态指标偏低（命中率 5.97%） | CVAE 的多模态优势未被验证 | 结果速览与消融结论里显式写明"未构成优势" |
| 8 | A5 的 `n_exec` 未随 H 联动；A7（K 帧扫描）未跑 | 无法回答"长块到底省多少推理" | 在消融结论中标注实验设计局限；未跑项写 `NOT_RUN` |
| 9 | 消融运行与 SLAM 全局优化存在 CPU 竞争 | `duration_s` 与设备占比可能偏高 | 文档标注；模型指标（val/loss/重构/抖动/延迟）不受影响 |
| 10 | 单场景、单次运行、无重复试验 | 没有方差与显著性 | 不给统计结论；需要时按 `docs/experiments.md` 的登记方式补种子重复 |

### 不适用场景

- ❌ 不能用于论证真实机器人的抓取/操作成功率；
- ❌ 不能用于论证真实场地下的 SLAM 精度或建图质量；
- ❌ 不能用于论证"时间集成一定显著降低抖动"（本实验条件下仅 −4.7%）；
- ❌ 不能用于论证"CVAE 一定优于确定性回归"（本实验的 A1 未显示重构优势）。

---

## 14. 许可与致谢

- 本仓库代码以 **MIT License** 发布，见 [`LICENSE`](LICENSE)。
- 依赖与致谢：**Google Cartographer**（Apache-2.0）提供 2D 激光 SLAM；
  **ROS 2 / RoboStack** 提供 `cartographer_ros` 二进制环境；**PyTorch** 提供训练与推理框架；
  **Unitree 工作区**提供上层实验上下文（本工程自身不含真机控制逻辑）。
- 引用或复用本工程时，请连同限定条件一起引用：**数据为 MOCK、无真机、结论限于仿真**。
