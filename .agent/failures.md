# failures

> 记录"失败 → 定位 → 修复 → 验证"，失败本身也是报告素材。

1. **单 batch 过拟合门禁 FAIL（recon 0.4216 不下降）**
   - 定位：`logs/21_overfit_gate.json` 显示 recon 完全不动、KL 从 24.7 缓慢下降。
   - 根因有两个：(a) 先验/后验输出层前接了 `LayerNorm`，强制 `mu≈1, log_var≈±1`，
     初始 KL 高达 20+ nats 压制重建；(b) `LambdaLR` 构造时立即执行 `lr_lambda(0)`，
     把基准 LR 1e-3 变成 2e-5，门禁在极低 LR 下"假失败"。
   - 修复：输出层零初始化（初始 KL=0）+ 门禁显式恢复基准 LR。
   - 验证：门禁 PASS，loss 0.278 → 0.0446（相对下降 83.9%）。

2. **训练被 npz 拖成 CPU 100% / GPU 空转**
   - 定位：进程 CPU 106%、GPU 9%，日志停在 step 1 数分钟。
   - 根因：`np.load(npz, mmap_mode="r")` 对 zip 成员无效，每次取样本都解压整个数组。
   - 修复：processed 改为每字段一个 `.npy`（真 mmap）。
   - 验证：同样 20 步从"数分钟未完成"降到 3.5 s（含启动）。

3. **消融训练用错归一化统计**
   - 定位：`no_chunk` 评测报 `统计哈希不一致：checkpoint=a206… 当前=48a7…`（被 checkpoint 校验拦下）。
   - 根因：训练侧硬编码读 `stats/normalization.json`，而消融数据写的是 `stats/ablation/<name>_normalization.json`。
   - 修复：统计路径统一走 `stats_path_for(paths, run_name)`。
   - 验证：消融 `no_chunk` 评测通过，指标进入 `outputs/ablation/summary.csv`。

4. **A4 消融数据构建被 schema 判为非法**
   - 定位：60 条 episode 全部报 `state shape (120,29) 应为 [120,24]`。
   - 根因：`validate_episode` 按"当前配置维度"校验原始数据，而原始数据是自描述全量布局（29 维），
     选列发生在窗口构建阶段。
   - 修复：按原始块布局校验维度，并额外检查"配置所需块是否都存在"。
   - 验证：base/no_slam_input/chunk_len_H4/no_chunk 四种配置下校验均通过。

5. **高倍速离线回放时位姿流几乎全丢**
   - 定位：1200 帧回放只收到 34 个 TF 样本（0.28 Hz），ATE 无法计算（`logs/42_...` 报 BLOCKED）。
   - 根因：`cartographer_offline_node` 以远高于实时的速度处理数据并按仿真时间高频发布 TF，
     rclpy 订阅侧来不及消费。
   - 修复：新增 `--mode pbstream_query`，用 `trajectory_query` 服务从 pbstream 取优化后轨迹。
   - 验证：得到 1037 个节点位姿，ATE 评估通过。

6. **ATE/航向误差一度被算错（2.7 cm 误报为 1.96 m）**
   - 定位：对齐后的 ATE 反而比"未对齐"更大，航向误差 0.59 rad 明显不合理。
   - 根因：Umeyama 变换方向写反（把 est→ref 写成了 ref→est），且航向误差没有扣除坐标系固定旋转。
   - 修复：按 `est→ref` 求变换、显式把旋转角加到估计航向上、并单独报告"未对齐 ATE"作为坐标系差异。
   - 验证：ATE 0.0275 m、RPE 0.0360 m、航向误差 0.0024 rad，且"未对齐 ATE=1.48 m"与之自洽。

7. **Bag 写出来是空的（`No topics were listed in metadata`）**
   - 定位：offline_node 报 `message_count: 0`。
   - 根因：`rosbag2_py.TopicMetadata` 用了关键字传参 → TypeError → bag 里没有 topic 定义。
   - 修复：按位置传参 `TopicMetadata(id, name, type, serialization_format)`。

8. **`ros2 bag record` 直接失败（`AMENT_PREFIX_PATH is not set`）**
   - 定位：RoboStack 环境未激活时 CLI 找不到插件。
   - 修复：脚本内显式导出 AMENT/CMAKE/COLCON/LD_LIBRARY/PYTHONPATH；
     并且不依赖残留的 `CONDA_PREFIX`（实测会被解析到 unitree_rt 导致 rosbag2 插件加载失败）。
