# 排错手册

> 每条都来自本工程真实踩过的坑（含复现现象、定位命令、修复方式）。

## 1. 训练/推理

### 1.1 单 batch 过拟合门禁 FAIL
**现象**：`bash scripts/21_overfit_single_batch.sh` 返回码 6，`logs/21_overfit_gate.json` 里 `relative_drop` 接近 0。
**本工程真实原因（按概率排序）**：
1. **学习率被调度器"偷走"**：`LambdaLR` 构造时会立即调用一次 `lr_lambda(0)`，带 warmup 时 LR 变成
   `base_lr/warmup_steps`（本项目 1e-3/50 = 2e-5），门禁在极低 LR 下必然"假失败"。
   修复：`run_overfit_gate` 显式恢复基准 LR（`src/train/train_cvae.py`）。
2. **初始 KL 压制重建**：后验输出层若带 LayerNorm 或未零初始化，初始 KL 可达 20+ nats。
   修复：输出层零初始化 + `logvar_clamp`；证据 `logs/21_overfit_gate.json` 前后对比。
3. 数据本身不可学：看 `logs/12_data_check.json` 的动作/观测统计与噪声占比。

### 1.2 loss 不下降 / 卡在常数
**现象**：`logs/train_metrics.jsonl` 的 `loss/recon` 长期不动。
**排查顺序**：① `loss/kl` 是否远大于 recon（KL 权重过大 → 后验塌缩）；② 归一化是否与数据同源
（`data/stats/normalization.json` 的哈希与 checkpoint 是否一致）；③ 动作噪声是否淹没了信号
（用 `data/stats/normalization.json` 的 `action.std` 与 mock 噪声参数对比）。

### 1.3 `mat1 and mat2 shapes cannot be multiplied`
**现象**：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 128x128)`。
**本工程真实原因**：对**已经按时间对齐**的观测 `images[t]`（形状 `[K,C,H,W]`）又用历史帧下标二次索引，
得到 `[K,K,C,H,W]`，被视觉主干误判为多相机输入（`cams=K`）。
修复：`src/infer/rollout_policy.py:run_episode` 直接使用 `images[t]`，并把非 5 维输入判为错误。

### 1.4 checkpoint 加载报"统计哈希不一致"
**含义**：训练与推理用到了不同的 `normalization.json`（多半是消融运行）。
修复：统计路径统一走 `src/datasets/transforms.py:stats_path_for(paths, run_name)`；
**不要**为了跑通而删掉校验。

### 1.5 训练 CPU 100% / GPU 空转
**原因**：`data/processed` 用 npz 存储时 `mmap_mode` 无效，每次取样本都要解压整个数组。
修复：processed 改为每字段一个 `.npy`（真 mmap）；并把 worker 数按 CPU 核数自动推导。
度量：`logs/20_device_usage.json` 与 `outputs/figures/device_split.png`。

## 2. SLAM / ROS2

### 2.1 `ros2 topic list` 只有 /parameter_events 与 /rosout
按「domain → 网卡 → 消息类型」顺序排查：`echo $ROS_DOMAIN_ID`（本工程离线回放用 1，与实机隔离）、
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`ROS_LOCALHOST_ONLY=1` 时不会跨网卡发现。

### 2.2 `RuntimeError: Environment variable 'AMENT_PREFIX_PATH' is not set or empty`
**原因**：RoboStack 环境未激活时 `ros2` CLI 找不到插件。
修复：脚本内显式导出 `AMENT_PREFIX_PATH/CMAKE_PREFIX_PATH/COLCON_PREFIX_PATH/LD_LIBRARY_PATH/PYTHONPATH`
（见 `scripts/40_slam_bringup.sh`）。**不要**依赖调用方 shell 里残留的 `CONDA_PREFIX`
（实测会被解析到 `unitree_rt` 环境，导致 rosbag2 插件加载失败）。

### 2.3 `cartographer_node: ERROR: flag 'collect_metrics' was defined more than once`
**原因**：RoboStack 包 `ros-jazzy-cartographer-ros 2.0.9003` 的组合缺陷（`offline_node.cpp` 与
`node_main.cpp` 同时定义同名 gflags）。`cartographer_node` 与 `cartographer_assets_writer` 均无法启动。
**绕过**：使用 `cartographer_offline_node`（可用）走离线回放路径；
或改用上游源码/容器自行构建（见 `docs/cartographer_integration.md` 的降级链记录）。

### 2.4 `basic_filebuf::underflow error reading the file: Is a directory`
**原因**：该构建的 Lua `include` 解析异常，连官方 `backpack_2d.lua` 都会失败。
**绕过**：用 `scripts/43_gen_cartographer_config.py` 生成自包含（无 include）的 `g1_2d.lua`。

### 2.5 `No topics were listed in metadata` / `message_count: 0`
**原因**：用 `rosbag2_py` 写 bag 时 `TopicMetadata` 用了关键字传参 → 抛 TypeError → bag 里没有 topic。
修复：按位置传参 `TopicMetadata(id, name, type, serialization_format)`（见 `src/slam/ros2_scan_player.py`）。

### 2.6 TF 超时 / 位姿流为空
排查顺序：① `map → odom → base_link` 是否都被发布（`provide_odom_frame=true` 时 odom→base_link 由
Cartographer 提供）；② 传感器 frame 与 `tracking_frame` 之间是否有静态 TF；
③ 时间戳是否在同一时钟域（离线回放用消息自带的仿真时间戳）。

## 3. 环境与硬件

### 3.1 `torch.cuda.is_available() == False` 或 sm_120 kernel 缺失
`python -m src.utils.cuda_check --json` 会给出 `arch_list`、`capability`、`arch_supported`。
本项目要求 `sm_120 ∈ arch_list`；不满足时应显式降级并在报告中标注 `CPU_ONLY`，不得沿用 GPU 口径。

### 3.2 CUDA OOM
按阶梯降低 `batch_size` → 图像分辨率 → 多帧数 K；每次重试都要记录在
`logs/*.jsonl` 与 `TASK.md` 的 Known Issues 中。

### 3.3 `sudo: A terminal is required to authenticate`
本机 apt 安装需要交互式密码（无 NOPASSWD）。因此 Cartographer 的**源码构建**在本环境被标为
`BLOCKED`（见 `docs/cartographer_integration.md`），必须走 conda 二进制分发路线。
