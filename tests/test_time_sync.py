"""时间对齐：容差内对齐、超容差标记、gap 掩码、时钟偏移估计、有效性掩码。"""

from __future__ import annotations

import numpy as np

from src.utils.time_sync import (
    AlignReport,
    align_linear,
    align_nearest,
    detect_gaps,
    estimate_clock_offset,
)


def test_nearest_within_tolerance():
    src = [0.0, 0.1, 0.2, 0.3]
    query = [0.02, 0.11, 0.29]
    idx, valid = align_nearest(src, query, tol=0.05)
    assert valid.all()
    assert idx.tolist() == [0, 1, 3]


def test_nearest_out_of_tolerance_marked():
    src = [0.0, 1.0]
    query = [0.5]
    report = AlignReport()
    idx, valid = align_nearest(src, query, tol=0.1, report=report)
    assert not valid[0]
    assert report.out_of_tolerance_ratio == 1.0
    assert report.summary()["n_out_of_tolerance"] == 1


def test_linear_interpolation():
    src_t = [0.0, 1.0, 2.0]
    src_v = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    out, valid = align_linear(src_t, src_v, [0.5, 1.5], tol=0.01)
    assert valid.all()
    assert np.allclose(out, [[0.5, 1.0], [1.5, 3.0]])


def test_linear_extrapolation_flagged():
    src_t = [0.0, 1.0]
    src_v = np.array([[0.0], [1.0]])
    report = AlignReport()
    out, valid = align_linear(src_t, src_v, [-0.5, 1.5], tol=0.1, report=report)
    assert not valid.any()
    assert report.n_extrapolated == 2


def test_detect_gaps():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    mask = detect_gaps(t, expected_dt=0.1)
    assert mask.tolist() == [False, False, False, True, False]


def test_clock_offset():
    a = np.array([10.0, 10.1, 10.2])
    b = np.array([0.0, 0.1, 0.2])
    assert estimate_clock_offset(a, b) == 10.0


def test_empty_source_is_handled():
    report = AlignReport()
    idx, valid = align_nearest([], [1.0, 2.0], tol=0.1, report=report)
    assert not valid.any()
    assert report.out_of_tolerance_ratio == 1.0
