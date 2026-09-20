# 实验登记表（自动生成，禁止手改）

> 生成时间：2026-09-20T23:22:48；数据来源：MOCK（合成世界）。
> 每行对应一次真实运行；config hash 用于证明「除消融变量外其余配置一致」。

| 编号 | 变量 | run | config hash (data+model) | seed | commit | 步数 | 最优 val/total | 结论限定 |
|---|---|---|---|---|---|---|---|---|
| 2 | base | base | 6b5ee4759573 | 0 | fc152c06 | 456 | 0.06945 | 基准配置 |
| 3 | chunk_len_H16 | chunk_len_H16 | da8033046699 | 0 | fc152c06 | 480 | 0.1085 | H=16 |
| 4 | chunk_len_H32 | chunk_len_H32 | ffc847e80688 | 0 | fc152c06 | 528 | 0.1444 | H=32 |
| 5 | chunk_len_H4 | chunk_len_H4 | 34765f2fab85 | 0 | fc152c06 | 432 | 0.04843 | H=4 |
| 6 | no_chunk | no_chunk | b13351fdbac4 | 0 | fc152c06 | 750 | 0.03268 | H=1 |
| 7 | no_cvae | no_cvae | 2df9906b486b | 0 | fc152c06 | 576 | 0.0609 | 无潜变量（确定性回归） |
| 8 | no_slam_input | no_slam_input | 6cec8c1ee17a | 0 | fc152c06 | 432 | 0.07069 | 无 SLAM 输入 |

## 结论限定条件（必须一起读）

1. 所有数据为 MOCK：合成 2D 世界 + 64×64 合成图像 + 解析射线投射激光；
2. 无真机：执行循环默认 dry-run，动作只写日志；
3. 消融运行与 SLAM 全局优化曾存在 CPU 竞争，`duration_s` 与设备占比可能偏高，但模型指标（val/total、recon、抖动、延迟）不受影响；
4. 推理侧消融（时间集成开关/权重）复用 base checkpoint，因此它们与 base 的 config hash 相同；
5. 未跑项：A7（观测历史帧数 K 扫描）标 NOT_RUN。

## 原始产物索引

| 产物 | 路径 |
|---|---|
| 训练指标 | `logs/train_metrics.jsonl`、`logs/val_metrics.jsonl` |
| 设备分工 | `logs/20_device_usage.json` |
| 评测指标 | `outputs/eval/*.csv`、`logs/infer_eval.json` |
| 消融汇总 | `outputs/ablation/summary.csv` |
| SLAM 轨迹 | `outputs/slam/trajectory_metrics.json`、`*.png` |
| 证据索引 | `evidence/index.md` |
