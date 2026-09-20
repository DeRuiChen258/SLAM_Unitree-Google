"""CUDA 启用状态核查（训练/推理入口启动时调用，结果写入日志与 manifest）。

目的：把"是否真的在用 GPU"变成可核查的事实，而不是靠配置文件里写个 cuda 就算数：
    * 检查 torch 是否为 CUDA 构建、设备是否可见、算力是否在 torch 的 arch_list 内；
    * 做一次真实 GPU 矩阵乘 + 一次本工程典型卷积前向，确认 kernel 能跑而不是"能看见不能用"；
    * 输出结构化结论，供 logs/00_env_check.txt 与训练 summary 引用。
"""

from __future__ import annotations

import time
from typing import Any


def cuda_report(verbose: bool = False) -> dict[str, Any]:
    """返回 CUDA 可用性与实测性能摘要（不抛异常，失败项写入 errors）。"""
    report: dict[str, Any] = {"requested": True, "available": False, "errors": []}
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        report["errors"].append(f"import torch 失败: {exc}")
        return report

    report["torch_version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["is_cuda_build"] = bool(torch.version.cuda)
    try:
        report["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        report["errors"].append(f"device_count 失败: {exc}")
        return report

    if not torch.cuda.is_available():
        report["errors"].append("torch.cuda.is_available() == False")
        return report

    report["available"] = True
    report["device_name"] = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    report["capability"] = list(capability)
    arch_list = torch.cuda.get_arch_list()
    report["arch_list"] = arch_list
    sm = f"sm_{capability[0]}{capability[1]}"
    report["sm"] = sm
    report["arch_supported"] = sm in arch_list
    if not report["arch_supported"]:
        report["errors"].append(f"设备算力 {sm} 不在 torch arch_list {arch_list} 内，kernel 可能不可用")

    # 1) 真实 GPU 矩阵乘（确认 kernel 可执行，不是"看得见跑不动"）
    try:
        torch.cuda.synchronize()
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        for _ in range(5):          # 预热：冷启动时 cuBLAS 还在选算法，直接计时会低估数倍
            _ = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        flops = 2 * 2048**3 * 10
        report["matmul_gflops"] = flops / dt / 1e9
        report["matmul_ok"] = bool(torch.isfinite(c).all().item())
    except Exception as exc:
        report["errors"].append(f"GPU 矩阵乘失败: {exc}")

    # 2) 本工程典型卷积前向（与 vision_backbone 同形状）
    try:
        import torch.nn as nn

        net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
        ).cuda()
        x = torch.randn(64, 3, 64, 64, device="cuda")
        for _ in range(5):          # 同理：cudnn 首次前向包含算法选择开销
            _ = net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = net(x)
        torch.cuda.synchronize()
        report["conv_forward_ms"] = (time.perf_counter() - t0) / 20 * 1000
        report["conv_ok"] = tuple(y.shape) == (64, 64, 16, 16)
    except Exception as exc:
        report["errors"].append(f"卷积前向失败: {exc}")

    report["memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
    report["status"] = "PASS" if not report["errors"] else "PARTIAL"
    if verbose:
        print(report)
    return report


def assert_cuda_available(device: str) -> dict[str, Any]:
    """训练/推理入口用：device='cuda' 时若不可用立即抛错，禁止静默降级。"""
    if device != "cuda":
        return {"requested": False}
    report = cuda_report()
    if not report.get("available"):
        raise RuntimeError(
            "配置要求 device=cuda，但 CUDA 不可用：" + "; ".join(report.get("errors", [])) or "unknown"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI：`python -m src.utils.cuda_check [--json]`（被 scripts/00_env_check.sh 调用）。"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CUDA 启用状态核查")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args(argv)
    report = cuda_report(verbose=not args.json)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("available") and not report.get("errors") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
