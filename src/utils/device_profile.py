"""CPU/GPU 分工策略与每步耗时拆分（"协调比例"的可度量实现）。

设计原则（不偏科：既不让 GPU 空转等数据，也不把本该 CPU 干的活塞进 GPU）：
    * GPU 只做张量计算：视觉主干前向、CVAE 前向/反向、策略解码、时间集成的向量化部分；
    * CPU 做数据与 I/O：npz/npy 读取、图像归一化、窗口切片、指标聚合、画图、CSV/JSON 落盘；
    * CPU→GPU 只传 batch（非阻塞拷贝 + pin_memory），不做逐样本拷贝；
    * DataLoader worker 数按 CPU 核数与 batch 规模自动推导，避免"单线程喂 GPU"这种隐性瓶颈。

本模块只提供策略与度量，不负责建模型（保持单一职责）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .config import get


@dataclass
class DevicePolicy:
    """一次运行的数据/计算放置策略（全部可从配置覆盖，无隐藏魔数）。"""

    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    non_blocking: bool = True
    torch_threads: int = 8
    gpu_idle_warn_ratio: float = 0.30
    timing_every: int = 10
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DevicePolicy":
        """从 train.yaml / infer.yaml 读取；未声明的字段用保守默认值。"""
        cpu_count = os.cpu_count() or 1
        batch = int(get(cfg, "batch_size", 64) or 64)
        raw_workers = get(cfg, "dataloader.num_workers")
        max_workers = int(get(cfg, "device_policy.max_workers", 8) or 8)
        divisor = max(1, int(get(cfg, "device_policy.worker_cpu_divisor", 4) or 4))
        if raw_workers is None:
            # 自动推导：每个 worker 至少分到 divisor 个核；batch 很小时再砍一半（避免小 batch 的调度开销）
            auto = max(1, min(max_workers, cpu_count // divisor))
            if batch < 32:
                auto = max(1, auto // 2)
            num_workers = auto
        else:
            num_workers = int(raw_workers)
        threads = get(cfg, "device_policy.torch_threads")
        if threads is None:
            # 主进程的 CPU 线程数：留出核给 DataLoader worker，避免 32 核机器上互相抢核
            threads = max(1, min(8, max(1, cpu_count // 4)))
        return cls(
            device=str(get(cfg, "device", "cuda")),
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=bool(get(cfg, "dataloader.pin_memory", True)),
            persistent_workers=bool(get(cfg, "dataloader.persistent_workers", True)) and num_workers > 0,
            prefetch_factor=int(get(cfg, "dataloader.prefetch_factor", 4) or 4),
            non_blocking=bool(get(cfg, "device_policy.non_blocking", True)),
            torch_threads=int(threads),
            gpu_idle_warn_ratio=float(get(cfg, "device_policy.gpu_idle_warn_ratio", 0.30) or 0.30),
            timing_every=max(1, int(get(cfg, "device_policy.timing_every", 10) or 10)),
            cpu_count=cpu_count,
        )

    def dataloader_kwargs(self, is_train: bool) -> dict[str, Any]:
        """传给 torch.utils.data.DataLoader 的放置相关参数。"""
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = bool(self.persistent_workers and is_train)
            kwargs["prefetch_factor"] = max(1, self.prefetch_factor)
        return kwargs

    def to_device(self, tensor, device):
        """统一的 H2D 拷贝：pin_memory + non_blocking，避免在 CPU 上等拷贝完成。"""
        import torch

        if not torch.is_tensor(tensor):
            return tensor
        return tensor.to(device, non_blocking=bool(self.non_blocking) and device.type == "cuda")

    def apply_thread_limits(self) -> None:
        """限制主进程 CPU 线程数（在 GPU 训练时防止 CPU 侧过度并行抢核）。"""
        try:
            import torch

            torch.set_num_threads(int(self.torch_threads))
        except Exception:
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "non_blocking": self.non_blocking,
            "torch_threads": self.torch_threads,
            "timing_every": self.timing_every,
            "cpu_count": self.cpu_count,
            "placement": {
                "gpu": ["视觉主干前向", "CVAE 前向/反向", "策略解码", "时间集成向量化"],
                "cpu": ["npy/npz 读取与 mmap", "图像归一化与切片", "指标聚合", "画图与 CSV/JSON 落盘"],
                "worker_processes": self.num_workers,
            },
        }


@dataclass
class StepTiming:
    """逐步耗时拆分：数据等待 / H2D 拷贝 / 计算（前向+反向+优化器）。"""

    data_ms: list[float] = field(default_factory=list)
    h2d_ms: list[float] = field(default_factory=list)
    compute_ms: list[float] = field(default_factory=list)

    def add(self, data_ms: float, h2d_ms: float, compute_ms: float) -> None:
        self.data_ms.append(float(data_ms))
        self.h2d_ms.append(float(h2d_ms))
        self.compute_ms.append(float(compute_ms))

    @property
    def n(self) -> int:
        return len(self.compute_ms)

    def summary(self, warn_ratio: float = 0.30) -> dict[str, Any]:
        if self.n == 0:
            return {"n_steps": 0}
        data = np.asarray(self.data_ms)
        h2d = np.asarray(self.h2d_ms)
        comp = np.asarray(self.compute_ms)
        total = data + h2d + comp
        total_sum = float(total.sum()) or 1.0
        data_ratio = float(data.sum() / total_sum)
        h2d_ratio = float(h2d.sum() / total_sum)
        compute_ratio = float(comp.sum() / total_sum)
        if data_ratio > warn_ratio:
            verdict, advice = "CPU_BOUND", (
                f"CPU 数据供给占 {data_ratio:.0%}：增大 dataloader.num_workers 或 device_policy.prefetch_factor，"
                "或把预处理（归一化/切片）提前固化到 processed 数据"
            )
        elif h2d_ratio > warn_ratio:
            verdict, advice = "TRANSFER_BOUND", (
                f"H2D 拷贝占 {h2d_ratio:.0%}：检查 pin_memory/non_blocking，或降低单 batch 图像尺寸/帧数"
            )
        elif compute_ratio > 0.95:
            verdict, advice = "GPU_BOUND", (
                f"GPU 计算占 {compute_ratio:.1%}（已接近饱和）：主算力已在 CUDA 上，"
                "继续提速只能靠减小模型/输入分辨率、或增大 batch 摊薄固定开销，"
                "再加 DataLoader worker 已无收益"
            )
        else:
            verdict, advice = "BALANCED", "CPU 供给与 GPU 计算比例均衡（GPU 有效利用率高）"
        return {
            "n_steps": self.n,
            "data_ms_mean": float(data.mean()),
            "h2d_ms_mean": float(h2d.mean()),
            "compute_ms_mean": float(comp.mean()),
            "total_ms_mean": float(total.mean()),
            "data_ratio": data_ratio,
            "h2d_ratio": h2d_ratio,
            "compute_ratio": compute_ratio,
            "gpu_busy_ratio": compute_ratio,
            "cpu_feed_ratio": data_ratio + h2d_ratio,
            "verdict": verdict,
            "advice": advice,
            "warn_ratio": float(warn_ratio),
        }

    def series(self) -> dict[str, list[float]]:
        return {"data_ms": self.data_ms, "h2d_ms": self.h2d_ms, "compute_ms": self.compute_ms}
