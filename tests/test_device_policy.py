"""CPU/GPU 分工策略：worker 自动推导、DataLoader 参数、耗时比例判定、线程上限。"""

from __future__ import annotations

import os

from src.utils.device_profile import DevicePolicy, StepTiming


def test_auto_worker_derivation():
    cfg = {
        "device": "cuda",
        "batch_size": 128,
        "dataloader": {"num_workers": None, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4},
        "device_policy": {"max_workers": 8, "worker_cpu_divisor": 4, "torch_threads": None,
                          "non_blocking": True, "gpu_idle_warn_ratio": 0.3, "timing_every": 10},
    }
    policy = DevicePolicy.from_config(cfg)
    expected = max(1, min(8, (os.cpu_count() or 1) // 4))
    assert policy.num_workers == expected
    assert policy.persistent_workers is True
    assert policy.non_blocking is True
    assert policy.torch_threads >= 1


def test_small_batch_reduces_workers():
    cfg = {
        "device": "cuda", "batch_size": 16,
        "dataloader": {"num_workers": None},
        "device_policy": {"max_workers": 8, "worker_cpu_divisor": 4},
    }
    small = DevicePolicy.from_config(cfg)
    cfg["batch_size"] = 256
    big = DevicePolicy.from_config(cfg)
    assert small.num_workers <= big.num_workers


def test_explicit_workers_win():
    policy = DevicePolicy.from_config({"device": "cpu", "batch_size": 64,
                                       "dataloader": {"num_workers": 2},
                                       "device_policy": {"max_workers": 8}})
    assert policy.num_workers == 2


def test_dataloader_kwargs_drop_persistent_when_zero_workers():
    policy = DevicePolicy.from_config({"device": "cpu", "batch_size": 64,
                                       "dataloader": {"num_workers": 0},
                                       "device_policy": {}})
    kwargs = policy.dataloader_kwargs(is_train=True)
    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs      # torch 在 num_workers=0 时不允许该参数
    assert "prefetch_factor" not in kwargs


def test_timing_summary_detects_cpu_bound():
    timing = StepTiming()
    for _ in range(10):
        timing.add(data_ms=50.0, h2d_ms=1.0, compute_ms=10.0)
    summary = timing.summary(warn_ratio=0.3)
    assert summary["verdict"] == "CPU_BOUND"
    assert summary["data_ratio"] > 0.7
    assert "num_workers" in summary["advice"]


def test_timing_summary_balanced_and_gpu_bound():
    balanced = StepTiming()
    for _ in range(10):
        balanced.add(data_ms=8.0, h2d_ms=0.5, compute_ms=21.0)
    assert balanced.summary(0.3)["verdict"] == "BALANCED"

    gpu = StepTiming()
    for _ in range(10):
        gpu.add(data_ms=0.2, h2d_ms=0.2, compute_ms=100.0)
    assert gpu.summary(0.3)["verdict"] == "GPU_BOUND"


def test_timing_summary_empty():
    assert StepTiming().summary()["n_steps"] == 0


def test_placement_documentation_is_explicit():
    policy = DevicePolicy.from_config({"device": "cuda", "batch_size": 64,
                                       "dataloader": {}, "device_policy": {}})
    desc = policy.describe()
    assert "gpu" in desc["placement"] and "cpu" in desc["placement"]
    assert desc["placement"]["worker_processes"] == policy.num_workers
