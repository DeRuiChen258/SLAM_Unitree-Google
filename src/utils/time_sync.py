"""多源时间对齐：最近邻 / 线性插值 / 时钟偏移估计 / gap 检测与有效性掩码。

被 datasets（图像-状态-动作-位姿对齐）与 slam（位姿流）共用。
硬约束：超过容差的样本必须被显式标记（valid=0），禁止强行取最近值后当作有效数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class AlignReport:
    """对齐质量报告（写入 logs/12_data_check.json）。"""

    n_query: int = 0
    n_matched: int = 0
    n_out_of_tolerance: int = 0
    n_extrapolated: int = 0
    offsets: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)

    @property
    def out_of_tolerance_ratio(self) -> float:
        return self.n_out_of_tolerance / self.n_query if self.n_query else 0.0

    def summary(self) -> dict:
        offsets = np.asarray(self.offsets, dtype=np.float64) if self.offsets else np.zeros(1)
        gaps = np.asarray(self.gaps, dtype=np.float64) if self.gaps else np.zeros(1)
        return {
            "n_query": int(self.n_query),
            "n_matched": int(self.n_matched),
            "n_out_of_tolerance": int(self.n_out_of_tolerance),
            "n_extrapolated": int(self.n_extrapolated),
            "out_of_tolerance_ratio": float(self.out_of_tolerance_ratio),
            "offset_mean_s": float(offsets.mean()),
            "offset_p95_s": float(np.percentile(np.abs(offsets), 95)),
            "offset_max_s": float(np.abs(offsets).max()),
            "gap_mean_s": float(gaps.mean()),
            "gap_max_s": float(gaps.max()),
        }


def align_nearest(src_t: Sequence[float], query_t: Sequence[float], tol: float,
                  report: AlignReport | None = None) -> tuple[np.ndarray, np.ndarray]:
    """最近邻对齐。

    返回 (index, valid)：index 为 src 中的下标（未命中时为最近的，但 valid=0）；
    valid=0 的样本禁止参与训练与评测。
    """
    src = np.asarray(src_t, dtype=np.float64)
    query = np.asarray(query_t, dtype=np.float64)
    idx = np.zeros(query.shape[0], dtype=np.int64)
    valid = np.zeros(query.shape[0], dtype=bool)
    if src.size == 0:
        if report is not None:
            report.n_query += int(query.shape[0])
            report.n_out_of_tolerance += int(query.shape[0])
        return idx, valid
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    pos = np.searchsorted(src_sorted, query)
    for i, p in enumerate(pos):
        cand: list[int] = []
        if p < src_sorted.size:
            cand.append(int(p))
        if p > 0:
            cand.append(int(p - 1))
        best = min(cand, key=lambda c: abs(src_sorted[c] - query[i]))
        offset = float(src_sorted[best] - query[i])
        idx[i] = order[best]
        ok = abs(offset) <= tol
        valid[i] = ok
        if report is not None:
            report.n_query += 1
            report.offsets.append(offset)
            if ok:
                report.n_matched += 1
            else:
                report.n_out_of_tolerance += 1
    return idx, valid


def align_linear(src_t: Sequence[float], src_v: np.ndarray, query_t: Sequence[float],
                 tol: float, report: AlignReport | None = None) -> tuple[np.ndarray, np.ndarray]:
    """线性插值对齐（用于连续量位姿/状态）。

    超出 src 时间范围的点做最近端外推，但仅在 |offset| ≤ tol 时标记 valid=1；
    外推样本计入 report.n_extrapolated。
    """
    src = np.asarray(src_t, dtype=np.float64)
    values = np.asarray(src_v, dtype=np.float64)
    query = np.asarray(query_t, dtype=np.float64)
    out = np.zeros((query.shape[0],) + values.shape[1:], dtype=np.float64)
    valid = np.zeros(query.shape[0], dtype=bool)
    if src.size == 0:
        if report is not None:
            report.n_query += int(query.shape[0])
            report.n_out_of_tolerance += int(query.shape[0])
        return out, valid
    for i, t in enumerate(query):
        if t <= src[0]:
            out[i] = values[0]
            offset = float(src[0] - t)
            extrapolated = True
        elif t >= src[-1]:
            out[i] = values[-1]
            offset = float(t - src[-1])
            extrapolated = True
        else:
            hi = int(np.searchsorted(src, t))
            lo = hi - 1
            span = src[hi] - src[lo]
            w = 0.0 if span <= 0 else (t - src[lo]) / span
            out[i] = (1.0 - w) * values[lo] + w * values[hi]
            offset = 0.0
            extrapolated = False
        ok = abs(offset) <= tol
        valid[i] = ok
        if report is not None:
            report.n_query += 1
            report.offsets.append(offset)
            report.n_matched += int(ok)
            report.n_out_of_tolerance += int(not ok)
            report.n_extrapolated += int(extrapolated)
    return out, valid


def detect_gaps(t: Sequence[float], expected_dt: float, gap_factor: float = 2.0) -> np.ndarray:
    """返回 gap 掩码：True 表示该处存在超过 gap_factor*expected_dt 的时间跳变。"""
    arr = np.asarray(t, dtype=np.float64)
    mask = np.zeros(arr.shape[0], dtype=bool)
    if arr.size < 2:
        return mask
    diff = np.diff(arr)
    mask[1:] = diff > gap_factor * expected_dt
    return mask


def estimate_clock_offset(stream_a: Sequence[float], stream_b: Sequence[float]) -> float:
    """用中位数差估计两个时钟域的偏移（a - b）。"""
    a = np.asarray(stream_a, dtype=np.float64)
    b = np.asarray(stream_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.median(a) - np.median(b))


def validity_mask_from_report(report: AlignReport) -> np.ndarray:
    """由 report 生成逐样本有效性掩码（供下游显式丢弃无效样本）。"""
    return np.asarray([abs(off) <= 0 for off in report.offsets], dtype=bool)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m src.utils.time_sync --report <path>`：汇总并打印对齐报告。"""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="时间对齐报告汇总")
    parser.add_argument("--report", required=True, help="对齐报告 JSON 路径")
    args = parser.parse_args(argv)
    path = Path(args.report)
    if not path.exists():
        print(json.dumps({"status": "MISSING", "path": str(path)}, ensure_ascii=False))
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "status": "OK",
        "path": str(path),
        "out_of_tolerance_ratio": data.get("out_of_tolerance_ratio"),
        "offset_p95_s": data.get("offset_p95_s"),
        "n_out_of_tolerance": data.get("n_out_of_tolerance"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
