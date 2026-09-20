"""动作块切片语义的唯一底层实现。

放在 utils（最底层）是为了让 datasets 与 models 都能依赖它，
同时满足依赖方向约束（models 不得 import datasets）。

语义（与 configs/data.yaml、dataset_schema.json 严格一致）：
    在时刻 t 取未来 H 步动作 action[t : t+H]；
    越过 episode 末尾的步以 0 填充，并由 mask=0 显式标记为无效步
    （禁止用 0 当作真实动作参与损失与时间集成）。
"""

from __future__ import annotations

import numpy as np


def chunk_slice(t: int, horizon: int) -> slice:
    """动作块在时间轴上的切片 [t, t+H)。"""
    if horizon < 1:
        raise ValueError(f"horizon 必须 ≥ 1，实际 {horizon}")
    if t < 0:
        raise ValueError(f"t 必须 ≥ 0，实际 {t}")
    return slice(t, t + horizon)


def build_chunk(actions: np.ndarray, t: int, horizon: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """构造动作块 [H, d_a] 与掩码 [H]。"""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions 应为 [T, d_a]，实际 {arr.shape}")
    if arr.shape[1] != dim:
        raise ValueError(f"actions 维度 {arr.shape[1]} 与配置 d_a={dim} 不一致")
    out = np.zeros((horizon, dim), dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.float32)
    available = max(0, min(horizon, arr.shape[0] - t))
    if available > 0:
        out[:available] = arr[t : t + available]
        mask[:available] = 1.0
    return out, mask


def apply_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按掩码置零（仅用于统计聚合，不改变原始语义）。"""
    m = np.asarray(mask, dtype=np.float32)
    while m.ndim < np.asarray(values).ndim:
        m = m[..., None]
    return np.asarray(values) * m


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """掩码加权均值；掩码和为 0 时返回 NaN（而不是静默返回 0）。"""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    while m.ndim < v.ndim:
        m = m[..., None]
    denom = m.sum(axis=axis)
    num = (v * m).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, num / np.maximum(denom, 1e-12), np.nan)
