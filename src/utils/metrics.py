"""指标聚合与导出：滑动平均、分位数、抖动指标、CSV/JSON 落盘。

供训练曲线、评测表、消融表统一使用，避免各处重复实现统计口径。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io_utils import atomic_write_json, ensure_dir


def moving_average(values: Sequence[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="valid")


def quantiles(values: Sequence[float], qs: Iterable[float] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.percentile(arr, q * 100)) for q in qs}


def jitter(values: np.ndarray, order: int = 2) -> float:
    """抖动指标：沿时间轴 order 阶差分的方差（越大越抖）。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < order + 1:
        return 0.0
    for _ in range(order):
        arr = np.diff(arr, axis=0)
    return float(np.var(arr))


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    diff = p - t
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        while m.ndim < diff.ndim:
            m = m[..., None]
        denom = m.sum()
        if denom == 0:
            return float("nan")
        return float(np.sqrt((diff**2 * m).sum() / denom))
    return float(np.sqrt(np.mean(diff**2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    diff = np.abs(p - t)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        while m.ndim < diff.ndim:
            m = m[..., None]
        denom = m.sum()
        if denom == 0:
            return float("nan")
        return float((diff * m).sum() / denom)
    return float(np.mean(diff))


def summarize_latency(samples_s: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples_s, dtype=np.float64) * 1000.0  # ms
    if arr.size == 0:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "mean_ms": float("nan")}
    q = quantiles(arr, (0.5, 0.95))
    return {"p50_ms": q["p50"], "p95_ms": q["p95"], "mean_ms": float(arr.mean())}


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """写 CSV（原子性由调用方按需保证；表头顺序显式声明以便报告引用）。"""
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})
    return p


def _fmt(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(_fmt(v)) for v in value) + "]"
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """样本量小时的均值置信区间（消融表需要给出波动范围，而非单点数字）。"""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "lo": float(arr[0]), "hi": float(arr[0]), "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(arr.size),
    }
