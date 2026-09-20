# 算法笔记：CVAE / 动作分块 / 时间集成

> 逐条标注实现文件；公式与代码必须一致，不一致时以代码为准并修正本文件。

## 1. CVAE 策略

### 1.1 生成模型与推断模型

给定观测历史 `o`（K 帧图像）、机器人状态 `s`（29 维）、未来动作块 `a_{t:t+H}`：

```
先验      p(z | o, s)      = N( μ_p(o,s),  diag(σ_p²(o,s)) )      src/models/encoder.py:PriorNetwork
后验      q(z | o, s, a)   = N( μ_q(o,s,a), diag(σ_q²(o,s,a)) )   src/models/encoder.py:PosteriorEncoder
解码器    π(a | o, s, z)                                        src/models/cvae_policy.py:ActionDecoder
重参数化  z = μ + σ ⊙ ε,  ε ~ N(0, I)                            src/models/layers.py:reparameterize
```

**teacher forcing 边界**：训练时后验吃真实动作块（`forward(..., mode="train")` 必须传 `action_chunk`，
否则报错）；推理时 `predict_chunk` 只走先验，且 `mode="infer"` 下传入 `action_chunk` **直接抛错**
（`src/models/cvae_policy.py:forward`），由 `tests/test_cvae_shapes.py` 双重拦截。

### 1.2 损失

```
L_recon = Σ_h m_h · w_h · ℓ(pred_h, gt_h) / Σ_h m_h w_h        ℓ ∈ {L1, MSE, Huber}
L_kl    = 0.5 · Σ_d ( σ_q²/σ_p² + (μ_p-μ_q)²/σ_p² - 1 + log σ_p² - log σ_q² )
L_smooth= 掩码加权的一阶/二阶差分均值
L_tc    = 相邻窗口重叠部分预测的一致性（可选）
L_total = w_recon·L_recon + β(t)·L_kl + λ_smooth·L_smooth + λ_tc·L_tc
```

实现位置：`src/models/losses.py`（公式）、`src/train/losses.py`（权重装配）、
`src/models/cvae_policy.py:compute_loss`（装配调用）。

### 1.3 数值稳定化的两个关键设计（都有实测依据）

1. **输出层零初始化**：`PriorNetwork` / `PosteriorEncoder` 的最后一层 Linear 权重与偏置置零
   （`src/models/encoder.py:_zero_init_head`）。
   * 原因：早期版本在该输出上加 `LayerNorm`，强制 `μ≈1, log_var≈±1`，初始 KL 高达 20–25 nats，
     直接把重建项压住（单 batch 过拟合门禁 FAIL：recon 0.42 不下降）。零初始化后初始 KL=0。
     证据：`logs/21_overfit_gate.json`（修复前 FAIL / 修复后 PASS 0.278→0.045）。
   * `log_var` 始终 clamp 到 `configs/model.yaml:latent.logvar_clamp = [-6, 4]`。
2. **free-bits + 小 β**：`configs/train.yaml` 取 `loss.kl.weight = 0.05`、`free_bits = 0.02/维`。
   * 原因：归一化动作上的 Huber 重建量级 0.03–0.4，而潜变量一旦被使用 KL 就是数个 nats，
     β=1.0 会让后验立刻塌缩（KL→0，退化为无条件 VAE，失去多模态能力）。
   * 实测：β=1.0 时 `val/kl` 恒为 0；改 0.05+free-bits 后 `val/kl ≈ 0.35`，`val/sample_diversity ≈ 0.78`
     （`logs/val_metrics.jsonl`）。

### 1.4 推理纯度

- `predict_chunk(mode="mean"|"sample", n_samples=k)`：`mean` 取先验均值（闭环推荐），
  `sample` 从先验采样，`n_samples>1` 返回 `[k,B,H,d_a]` 用于展示多模态。
- 推理链路（`src/infer/rollout_policy.py`）只调用 `predict_chunk`，全文件不出现 posterior。

## 2. 动作分块（Action Chunking）

### 2.1 切片语义（唯一实现）

`src/utils/chunking.py:build_chunk(actions, t, H, d_a)`：取 `action[t : t+H]`，
越过 episode 末尾的步**以 0 填充并置 mask=0**。
`src/datasets/window_builder.py` 与 `src/models/action_chunker.py` 都调用它，
`tests/test_action_chunker.py:test_build_chunk_matches_window_builder` 做交叉验证。

### 2.2 调度

`ChunkScheduler(H, n_exec)`（`src/models/action_chunker.py`）：每 `n_exec` 步重新预测；
尾部（`H - n_exec`）默认显式丢弃（`tail()` 返回空数组），不做隐式复用。

### 2.3 H 的取舍（A5 消融要回答的问题）

| | H 小（4） | H 大（32） |
|---|---|---|
| 单次推理摊销 | 每 4 步推理一次 → 推理频率高、单步延迟占比大 | 每 32 步推理一次 → GPU/CPU 推理开销摊薄 |
| 响应滞后 | 小（新观测很快进入动作） | 大（一次预测要执行 0.8 s @10Hz） |
| 平滑性 | 逐步噪声更容易体现 | 块内一致性好，但块边界可能不连续（需时间集成） |
| 训练难度 | 长程依赖弱 | 需要模型具备更长程的一致性 |

实测数据见 `outputs/ablation/summary.csv` 的 `chunk_len_H*` 行与
`outputs/figures/30_per_step_error_*.png`（误差随 chunk 内步数增长）。

## 3. 时间集成（Temporal Ensembling）

### 3.1 权重

维护 `deque[(chunk, t0_pred, n_exec, meta, mask)]`，对时刻 `t`：

```
action(t) = Σ_j w_j · a_j(t) / Σ_j w_j      只对满足 t0_j ≤ t < t0_j + H 且 mask>0 的预测求和
uniform      w = 1
exponential  w = decay^(t - t0_j)            age 按步计，decay ∈ (0,1]
inverse_age  w = 1 / (1 + (t - t0_j))
```

实现：`src/models/temporal_ensemble.py`。

### 3.2 硬约束（都是踩过的坑）

- **冷启动**：只有 1 条预测时直接返回该预测（权重归一化不会除零）；
- **过期丢弃**：`t0 + H ≤ t` 的预测在 `update` 时就被移除（`stats()['n_expired']` 计数）；
- **禁止补零**：没有任何贡献者时 `action_at(t)` 返回 `None`，**不返回 0 动作**
  （补零会系统性把动作拉向原点）；
- **逐步掩码**：`mask` 必须是 `[H]` 数组；标量掩码直接抛错，防止静默语义错误；
- **不回改已执行动作**：融合只作用于调用方请求的 `t`，调用方按时间单调推进。

### 3.3 延迟与稳定性

时间集成用**平滑性换一部分滞后**：`decay` 越小越信任最新预测（响应快、抖动大），
`decay → 1` 等价于所有历史预测等权（最平滑、滞后最大）。
实测（`outputs/eval/base_test_metrics.csv`，3 条 test episode 闭环）：

| 指标 | 无集成 | 有集成（decay=0.5） | 相对变化 |
|---|---|---|---|
| 二阶差分方差（抖动） | 1.208e-06 | 1.156e-06 | **−4.3%** |
| 推理延迟 p50 | 5.09 ms | 5.09 ms | 0%（融合在 CPU 侧，成本可忽略） |

> 说明：本实验的 `latent.mode=mean` 使预测本身已相当确定，因此集成的抖动收益有限（−4.3%）；
> 若改用 `latent.mode=sample`（随机策略），集成收益会显著变大——这是"在什么条件下成立"的关键限定。

## 4. 多帧视觉

`observation.num_frames = 2`（K=2）、`frame_stride = 1`、聚合方式 `gru`
（`configs/model.yaml:aggregation.mode`）。多帧编码路径：
`[B,K,C,H,W] --共享权重 CNN--> [B,K,d_e] --GRU--> [B,d_h] --投影--> [B,d_v]`
（`src/models/vision_backbone.py`）。K 的影响属于 A7 消融（本实验未跑，标 NOT_RUN）。

## 5. 与 Cartographer 的算法耦合点

1. SLAM 位姿进入状态向量的形式：**相对 episode 起点的 (dx, dy, sinΔyaw, cosΔyaw)** +
   `valid` 标志（角度用 sin/cos 编码，禁止直接线性归一化原始弧度）；
2. 位姿既作为策略输入，也用于轨迹一致性评估（ATE/RPE）与 A4 消融对照；
3. 时间对齐容差 0.10 s，超容差样本 `valid=0`；
4. 具体实现与实测见 `docs/cartographer_integration.md`。
