"""CUDA 启用核查测试。

设计原则：CUDA 不可用的机器上这些用例必须 **skip 而不是 fail**，
但一旦配置声明 device=cuda 却不可用，训练入口必须硬报错（test_cuda_required_is_hard_error）。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not _cuda_available(), reason="本机无可用 CUDA 设备")
def test_cuda_report_passes():
    from src.utils.cuda_check import cuda_report

    report = cuda_report()
    assert report["available"] is True
    assert report["is_cuda_build"] is True
    assert report["matmul_ok"] is True, report.get("errors")
    assert report["conv_ok"] is True, report.get("errors")
    assert report["status"] == "PASS"
    # 算力必须被 torch 的 arch_list 覆盖，否则 kernel 可能缺失
    assert report["arch_supported"] is True, f"capability={report.get('capability')} arch={report.get('arch_list')}"


@pytest.mark.skipif(not _cuda_available(), reason="本机无可用 CUDA 设备")
def test_training_config_requests_cuda():
    """配置必须显式声明 cuda（本项目要求 GPU 训练，禁止隐式 auto）。"""
    from src.utils.config import load_config

    assert load_config("train")["device"] == "cuda"
    assert load_config("infer")["device"] == "cuda"


def test_resolve_device_rejects_missing_cuda(monkeypatch):
    """声明 cuda 但设备不可用时必须抛错，而不是静默退回 CPU。"""
    from src.train import train_cvae

    monkeypatch.setattr(train_cvae.torch.cuda, "is_available", lambda: False)
    with pytest.raises(Exception):
        train_cvae.resolve_device("cuda")


def test_resolve_device_cpu_is_explicit():
    from src.train.train_cvae import resolve_device

    assert str(resolve_device("cpu")) == "cpu"
