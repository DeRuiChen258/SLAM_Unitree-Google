"""全局随机种子与 DataLoader worker 种子。

确定性说明：`deterministic=True` 会调用 torch.use_deterministic_algorithms(True)，
在 GPU 上可能显著降速（部分算子回退到确定性实现），默认关闭并在此显式标注代价。
"""

from __future__ import annotations

import os
import random
from typing import Any


def set_seed(seed: int, deterministic: bool = False, torch_module: Any | None = None) -> dict:
    """设置 python / numpy / torch / cuda 全局种子，返回状态摘要。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    state = {"seed": seed, "deterministic": bool(deterministic)}

    try:
        import numpy as np

        np.random.seed(seed)
        state["numpy"] = True
    except Exception:  # pragma: no cover - numpy 缺失属环境错误
        state["numpy"] = False

    torch = torch_module
    if torch is None:
        try:
            import torch as _torch

            torch = _torch
        except Exception:
            torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 固定输入尺寸的卷积网络：让 cudnn 做算法挑选可以显著提速；
            # 只有要求确定性时才关闭（见下面的 deterministic 分支）。
            if not deterministic:
                torch.backends.cudnn.benchmark = True
        if deterministic:
            # 代价：部分 CUDA 算子无确定性实现，会抛错或回退；训练吞吐可能下降 10%~40%
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        state["torch"] = torch.__version__
        state["cuda_available"] = bool(torch.cuda.is_available())
    return state


def make_worker_init(seed: int):
    """DataLoader worker_init_fn：worker 内固定随机种子，保证多 worker 可复现。"""

    def _init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**31 - 1)
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except Exception:
            pass

    return _init
