"""滚动推理核心：读观测 → 组成 (o, s) → predict_chunk → 分块调度 + 时间集成 → 输出动作序列。

支持开环（预测一次执行整段）与闭环（每 n_exec 步用新观测重新预测）。
每一次输出动作都记录「来源（第几次预测 / 融合权重）」，便于复盘与报告。

约束：
    * 推理路径不得访问后验（本文件只调用 CVAEPolicy.predict_chunk）；
    * 时间集成只作用于尚未执行的步，不回改已执行动作；
    * 缺步必须显式裁剪，禁止补零。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..datasets.transforms import load_stats
from ..models.action_chunker import ActionChunker
from ..models.cvae_policy import CVAEPolicy
from ..models.temporal_ensemble import TemporalEnsembler
from ..utils.config import get
from ..utils.metrics import jitter


@dataclass
class StepRecord:
    """单步执行记录（动作来源必须可追溯，便于审查是否走过时间集成）。"""

    t: int
    action_norm: np.ndarray
    action_phys: np.ndarray
    n_prediction: int
    ensemble_used: bool
    weight_sum: float
    contributors: int
    latency_ms: float
    gt_action: np.ndarray | None = None
    mask: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": int(self.t),
            "action": [float(v) for v in self.action_phys],
            "action_norm": [float(v) for v in self.action_norm],
            "gt_action": None if self.gt_action is None else [float(v) for v in self.gt_action],
            "n_prediction": int(self.n_prediction),
            "ensemble_used": bool(self.ensemble_used),
            "weight_sum": float(self.weight_sum),
            "contributors": int(self.contributors),
            "latency_ms": float(self.latency_ms),
            "mask": float(self.mask),
        }


@dataclass
class RolloutResult:
    """一次滚动的完整结果。"""

    steps: list[StepRecord] = field(default_factory=list)
    chunk_sources: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    ensemble_stats: dict[str, Any] = field(default_factory=dict)
    n_predictions: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def actions_phys(self) -> np.ndarray:
        return np.asarray([s.action_phys for s in self.steps], dtype=np.float32)

    def actions_norm(self) -> np.ndarray:
        return np.asarray([s.action_norm for s in self.steps], dtype=np.float32)

    def gt_actions(self) -> np.ndarray:
        return np.asarray([s.gt_action if s.gt_action is not None else np.full_like(s.action_phys, np.nan)
                           for s in self.steps], dtype=np.float32)

    def masks(self) -> np.ndarray:
        return np.asarray([s.mask for s in self.steps], dtype=np.float32)

    def summary(self) -> dict[str, Any]:
        actions = self.actions_phys()
        lat = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "n_steps": len(self.steps),
            "n_predictions": self.n_predictions,
            "ensemble_used_steps": int(sum(1 for s in self.steps if s.ensemble_used)),
            "jitter_order2": float(jitter(actions, order=2)) if actions.size else 0.0,
            "jitter_order1": float(jitter(actions, order=1)) if actions.size else 0.0,
            "latency_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
            "latency_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
            "ensemble": self.ensemble_stats,
        }


class RolloutRunner:
    """闭环 / 开环滚动推理执行器（离线数据集回放口径）。"""

    def __init__(self, model: CVAEPolicy, data_cfg: Mapping[str, Any], infer_cfg: Mapping[str, Any],
                 stats: Mapping[str, Any] | None = None, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.data_cfg = data_cfg
        self.infer_cfg = infer_cfg
        self.device = torch.device(device)
        self.stats = stats or load_stats(data_cfg=data_cfg)
        self.chunker = ActionChunker.from_config(data_cfg, infer_cfg)
        self.ensemble: TemporalEnsembler | None = (
            TemporalEnsembler.from_config(infer_cfg)
            if bool(get(infer_cfg, "temporal_ensemble.enabled", False)) else None
        )
        self.latent_mode = str(get(infer_cfg, "latent.mode", "mean"))
        self.n_samples = int(get(infer_cfg, "latent.n_samples", 1))

    # ----------------------------------------------------------------------------------
    # 单次预测
    # ----------------------------------------------------------------------------------
    def predict(self, images: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, float]:
        """输入单帧观测与状态，返回（归一化动作块, 推理耗时 ms）。"""
        img_t = torch.from_numpy(np.asarray(images, dtype=np.float32)[None]).to(self.device)
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)[None]).to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            chunk = self.model.predict_chunk(img_t, state_t, mode=self.latent_mode, n_samples=1)
        latency = (time.perf_counter() - started) * 1000.0
        return chunk[0].cpu().numpy(), latency

    def predict_multimodal(self, images: np.ndarray, state: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        """从先验多次采样，返回 [N, H, d_a]（用于展示多模态；闭环控制不使用）。"""
        img_t = torch.from_numpy(np.asarray(images, dtype=np.float32)[None]).to(self.device)
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)[None]).to(self.device)
        with torch.no_grad():
            chunks = self.model.predict_chunk(img_t, state_t, mode="sample",
                                              n_samples=n_samples or self.n_samples)
        return chunks[:, 0].cpu().numpy()

    # ----------------------------------------------------------------------------------
    # 滚动推理
    # ----------------------------------------------------------------------------------
    def run_episode(self, episode: Mapping[str, np.ndarray], obs_indices: Sequence[np.ndarray],
                    actions_norm: np.ndarray | None = None, max_steps: int | None = None,
                    masks: np.ndarray | None = None) -> RolloutResult:
        """在一条 episode 上滚动推理。

        episode 必须包含 `images`（[T,K,C,H,W]，已按训练同源方式归一化）与 `state`（[T,d_s]，已归一化）；
        `actions_norm` 为归一化后的真实动作（用于逐步对比），可为 None（纯闭环）。
        """
        # images 必须是**已按时间对齐**的观测张量 [T, K, C, H, W]：
        # 即 images[t] 就是时刻 t 的历史窗口，禁止再用 obs_indices 二次索引
        # （二次索引会得到 [K,K,C,H,W]，被视觉主干误判为多相机输入）。
        images = np.asarray(episode["images"], dtype=np.float32)
        if images.ndim != 5:
            raise ValueError(f"episode['images'] 应为 [T,K,C,H,W]，实际 {images.shape}")
        states = np.asarray(episode["state"], dtype=np.float32)
        length = states.shape[0]
        limit = int(max_steps or get(self.infer_cfg, "episode.max_steps", length))
        limit = min(limit, length, len(obs_indices))
        n_exec = self.chunker.n_exec
        mode = str(get(self.infer_cfg, "chunk_exec.mode", "closed_loop"))
        if mode == "open_loop":
            n_exec = self.chunker.horizon  # 开环：一次预测执行整段
            self.chunker.scheduler.n_exec = n_exec

        result = RolloutResult(meta={"length": length, "n_exec": n_exec, "mode": mode,
                                     "ensemble": self.ensemble.mode if self.ensemble else None})
        if self.ensemble:
            self.ensemble.reset()
        t = 0
        pending: list[np.ndarray] = []
        while t < limit:
            if not pending:
                obs = images[t]
                state = states[t]
                chunk, latency = self.predict(obs, state)
                result.n_predictions += 1
                result.chunk_sources.append({"t": t, "n_prediction": result.n_predictions,
                                             "latency_ms": latency,
                                             "chunk_head": chunk[0].tolist()})
                if self.ensemble is not None:
                    # 集成器需要逐步掩码 [H]；episode 级掩码在此广播到 chunk 的每一步
                    mask = None if masks is None else np.full(chunk.shape[0], float(masks[t]), dtype=np.float32)
                    self.ensemble.update(chunk, t0=t, n_exec=n_exec,
                                         meta={"n_prediction": result.n_predictions}, mask=mask)
                pending = list(self.chunker.steps(chunk))
                result.latency_ms.append(latency)
                if not pending:
                    break
            action_norm = pending.pop(0)
            action = action_norm.astype(np.float32)
            weight_sum = 1.0
            contributors = 1
            used_ensemble = False
            if self.ensemble is not None:
                fused = self.ensemble.action_at(t)
                if fused is not None:
                    action = fused
                    used_ensemble = True
                    pairs = self.ensemble.weights_at(t)
                    weight_sum = float(sum(w for _, w in pairs))
                    contributors = len(pairs)
            gt = None
            mask_val = 1.0
            if actions_norm is not None and t < actions_norm.shape[0]:
                gt = actions_norm[t]
            if masks is not None and t < masks.shape[0]:
                mask_val = float(masks[t])
            result.steps.append(
                StepRecord(
                    t=t, action_norm=action, action_phys=self._denorm(action),
                    n_prediction=result.n_predictions, ensemble_used=used_ensemble,
                    weight_sum=weight_sum, contributors=contributors,
                    latency_ms=result.latency_ms[-1] if result.latency_ms else 0.0,
                    gt_action=None if gt is None else self._denorm(np.asarray(gt, dtype=np.float32)),
                    mask=mask_val,
                )
            )
            t += 1
        if self.ensemble is not None:
            result.ensemble_stats = self.ensemble.stats()
        return result

    # ----------------------------------------------------------------------------------
    def _denorm(self, action_norm: np.ndarray) -> np.ndarray:
        from ..datasets.transforms import inverse_action

        return inverse_action(np.asarray(action_norm, dtype=np.float32)[None], self.stats)[0]

    def denormalize_actions(self, actions_norm: np.ndarray) -> np.ndarray:
        from ..datasets.transforms import inverse_action

        return inverse_action(np.asarray(actions_norm, dtype=np.float32), self.stats)

    def describe(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker.describe(),
            "latent_mode": self.latent_mode,
            "n_samples": self.n_samples,
            "ensemble": None if self.ensemble is None else self.ensemble.stats(),
            "device": str(self.device),
        }
