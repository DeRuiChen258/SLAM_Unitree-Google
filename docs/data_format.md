# 数据格式与数据字典

> 机器可读契约优先：`data/schema/dataset_schema.json`。本文件是人类可读说明，两者冲突时以 schema 为准并修正本文件。

## 1. 目录布局

```
data/
├── raw/            # 真实采集 episode（只读；本实验为空，因为无真机）
├── mock/           # MOCK episode（mock_XXXX.npz）
├── processed/      # 窗口化 + 归一化后的训练数据（每字段一个 .npy）
├── splits/         # train/val/test 划分（episode 列表 + seed + config hash）
├── stats/          # 归一化统计（normalization.json 及其消融版本）
├── schema/         # dataset_schema.json（机器可读契约）
├── slam/           # scans.npz / world.json / ground_truth.json / pose_stream.jsonl
├── bags/           # ROS2 bag（Cartographer 离线回放输入）
├── manifest.json   # 数据版本事实源
└── mock_manifest.json
```

## 2. episode npz 字段（`data/mock/mock_XXXX.npz`）

| 字段 | shape | dtype | 单位/范围 | 说明 |
|---|---|---|---|---|
| `images` | `[T,3,64,64]` | uint8 | 0–255 | RGB 俯视观测（**合成图像**，非真实相机） |
| `state` | `[T,29]` | float32 | 混合 | 全量状态矩阵，块顺序见下 |
| `action` | `[T,7]` | float32 | m/rad | `[dx,dy,dz,rx,ry,rz,gripper]`，前 6 维为增量 |
| `timestamp` | `[T]` | float64 | s | 采集/仿真时钟，严格单调，10 Hz |
| `base_pose` | `[T,3]` | float32 | m,m,rad | 基座真值位姿（仅用于 SLAM 评估，**不进策略输入**） |
| `slam_pose` | `[T,3]` | float32 | m,m,rad | 相对 episode 起点的 SLAM 位姿 |
| `slam_valid` | `[T]` | uint8 | 0/1 | 位姿有效性；0 表示丢帧/超容差，**禁止当 0 位姿使用** |
| `episode_id` / `source` | 标量 | str | — | 数据来源，MOCK 数据结论必须标注 MOCK |
| `success` | 标量 | uint8 | 0/1 | 任务成功标记 |
| `state_block_names` / `state_block_dims` | `[7]` | str/int | — | 原始数据的自描述块布局（供消融选列） |

### 状态块顺序（`state` 的列语义）

| 块 | 维度 | 列区间 | 单位 | 是否归一化 | 来源 |
|---|---|---|---|---|---|
| `joint_pos` | 7 | 0–6 | rad | 是 | 由末端位置线性映射（**非真实 IK**） |
| `joint_vel` | 7 | 7–13 | rad/s | 是 | 差分 |
| `gripper` | 1 | 14 | 1 | 是 | 夹爪开合度 |
| `ee_pose` | 7 | 15–21 | m + 四元数 | 是 | `[x,y,z,qw,qx,qy,qz]` |
| `slam_pose` | 4 | 22–25 | m, m, 1, 1 | 是 | `[dx, dy, sin(yaw), cos(yaw)]`，相对 episode 起点 |
| `slam_valid` | 1 | 26 | 0/1 | 否（passthrough） | 位姿有效性标志 |
| `time_feat` | 2 | 27–28 | 1 | 否（passthrough） | `[sin(2πφ), cos(2πφ)]`，φ = 归一化时间 |

> 消融 A4 关闭 `slam_input.enabled` 后，`slam_pose` 与 `slam_valid` 两块被移除，`d_s = 24`。
> 列选择由 `src/datasets/schema.py:state_column_selector` 统一决定，训练脚本不得硬编码列号。

## 3. processed 分片（每字段一个 `.npy`）

`data/processed/{split}.{field}.npy`，`field ∈ {images, state, action_chunk, mask, episode_index, t0, timestamp}`。

- **为什么不用 npz**：`np.load(..., mmap_mode="r")` 对 npz 无效（zip 成员每次访问都要整体解压），
  实测会把训练拖成「CPU 100% / GPU 空转」；`.npy` 才能真 mmap 懒加载。
- `state` 与 `action_chunk` 在构建阶段就已按 `data/stats/normalization.json` 归一化并**固化**，
  `WindowDataset.__getitem__` 不再做任何依赖全局统计的归一化。
- `mask[i]=0` 表示动作块末尾 padding（episode 结束），损失与时间集成都必须乘掩码。

## 4. 时间戳语义

- **采集时钟 `t_capture`**：与 episode `timestamp` 同一时间轴（MOCK 回放为仿真时间，起点 0）。
  SLAM 位姿流对齐到策略时钟用的就是它。
- **单调时钟 `t_mono`**：`time.monotonic()`，用于在线 watchdog 与延迟测量，不参与数据集对齐。
- 对齐容差：`configs/slam.yaml:sync.max_align_tolerance_s = 0.10`；超容差样本必须标 `valid=0`，
  禁止强行取最近值当有效数据。

## 5. 版本与复现

- `data/manifest.json` 记录：来源（MOCK/REAL）、episode 数、总帧数、fps、schema 版本、
  config hash、过滤统计、各文件 sha256、生成命令。
- 同 seed 重跑 `scripts/11_build_dataset.py` 必须得到相同划分与相同哈希（S3 门禁）。
- `source=MOCK` 的数据不得用于任何"真实机器人/真实场地"结论。
