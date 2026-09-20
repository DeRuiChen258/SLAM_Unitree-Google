"""torch Dataset：读取 data/splits + data/processed，返回训练样本 dict。

样本字段：
    images       [K, C, H, W] float32（按 configs/data.yaml 归一化）
    state        [d_s]        float32（构建数据集时已固化归一化）
    action_chunk [H, d_a]     float32（同上）
    mask         [H]          0/1
    meta         dict(episode_id / t0 / timestamp / source)

约束：`__getitem__` 内不做需要全局统计的归一化（统计在构建数据集时固化）；
worker 内固定随机种子（见 src/utils/seed.make_worker_init）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..utils.config import get, load_paths
from ..utils.io_utils import read_json
from .transforms import build_transform


class ProcessedSplit:
    """已构建好的 processed 分片：每字段一个 .npy，用 mmap 真正懒加载。

    为什么不存 npz：`np.load(..., mmap_mode="r")` 对 npz **无效**（zip 成员必须整体解压），
    每次 `handle["images"]` 都会解压整段数组，训练会被拖到 CPU 100% 而 GPU 空转。
    """

    FIELDS = ("images", "state", "action_chunk", "mask", "episode_index", "t0", "timestamp")

    def __init__(self, stem: str | Path) -> None:
        self.stem = Path(stem)
        self.paths = {f: self.stem.with_name(f"{self.stem.name}.{f}.npy") for f in self.FIELDS}
        missing = [str(p) for p in self.paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少 processed 分片字段文件: {missing[:3]}…（先运行 scripts/11_build_dataset.py）"
            )
        self._arrays = {f: np.load(p, allow_pickle=False, mmap_mode="r") for f, p in self.paths.items()}
        self.length = int(self._arrays["state"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        a = self._arrays
        return {
            "images": np.asarray(a["images"][index]),
            "state": np.asarray(a["state"][index], dtype=np.float32),
            "action_chunk": np.asarray(a["action_chunk"][index], dtype=np.float32),
            "mask": np.asarray(a["mask"][index], dtype=np.float32),
            "episode_index": int(a["episode_index"][index]),
            "t0": int(a["t0"][index]),
            "timestamp": float(a["timestamp"][index]),
        }

    def meta_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(self._arrays[k]) for k in ("episode_index", "t0", "timestamp")}

    def close(self) -> None:
        for arr in self._arrays.values():
            if hasattr(arr, "_mmap") and arr._mmap is not None:
                arr._mmap.close()
        self._arrays.clear()


class WindowDataset:
    """轻量 Dataset（不依赖 torch 也能做纯 numpy 遍历，便于测试）。

    CUDA 为主的数据契约：默认返回**原始 uint8 图像**，归一化在设备侧由模型完成
    （见 `src/models/vision_backbone.py:normalize_images`）。
    `preprocess_on_cpu=True` 仅用于离线可视化/数据体检等需要在 CPU 上看图的场景。
    """

    def __init__(self, split_stem: str | Path, data_cfg: Mapping[str, Any],
                 preprocess_on_cpu: bool = False) -> None:
        """`split_stem` 形如 <processed_dir>/train（不带扩展名）。"""
        self.split = ProcessedSplit(split_stem)
        self.data_cfg = data_cfg
        self.transform = build_transform(data_cfg) if preprocess_on_cpu else None
        self.episode_ids: list[str] = []
        meta_path = Path(split_stem).with_suffix(".meta.json")
        if meta_path.exists():
            self.episode_ids = json.loads(meta_path.read_text(encoding="utf-8")).get("episode_ids", [])

    def __len__(self) -> int:
        return self.split.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.split[index]
        images = raw["images"]
        if self.transform is not None:
            images = self.transform(images)
        # else：保持 uint8 原样交给 DataLoader（归一化在 GPU 侧完成）
        episode_index = raw["episode_index"]
        episode_id = self.episode_ids[episode_index] if episode_index < len(self.episode_ids) else str(episode_index)
        return {
            "images": images,
            "state": raw["state"],
            "action_chunk": raw["action_chunk"],
            "mask": raw["mask"],
            "meta": {
                "episode_id": episode_id,
                "t0": raw["t0"],
                "timestamp": raw["timestamp"],
                "split": self.split.stem.name,
            },
        }

    def close(self) -> None:
        self.split.close()


def collate_fn(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    """把样本列表拼成 batch；meta 保持 list 结构（含变长字段）。"""
    import torch

    stacked = np.stack([np.asarray(b["images"]) for b in batch], axis=0)
    # uint8 保持原类型：显存带宽比 float32 省 4 倍，归一化交给模型在设备侧做
    images = torch.from_numpy(stacked) if stacked.dtype == np.uint8 else torch.from_numpy(stacked).float()
    state = torch.from_numpy(np.stack([np.asarray(b["state"]) for b in batch], axis=0)).float()
    action = torch.from_numpy(np.stack([np.asarray(b["action_chunk"]) for b in batch], axis=0)).float()
    mask = torch.from_numpy(np.stack([np.asarray(b["mask"]) for b in batch], axis=0)).float()
    return {
        "images": images,
        "state": state,
        "action_chunk": action,
        "mask": mask,
        "meta": [dict(b.get("meta", {})) for b in batch],
    }


def make_torch_dataset(split_stem: str | Path, data_cfg: Mapping[str, Any]) -> Any:
    """构造 torch.utils.data.Dataset（torch 为可选依赖，只在真正训练时需要）。"""
    import torch

    class _TorchDataset(torch.utils.data.Dataset):
        def __init__(self, inner: WindowDataset) -> None:
            self.inner = inner

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.inner[index]

    return _TorchDataset(WindowDataset(split_stem, data_cfg))


def split_paths(split: str, paths: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
    """返回 (processed 分片 stem, splits json) 两个路径。"""
    paths = paths or load_paths()
    return (
        Path(paths["processed_dir"]) / split,
        Path(paths["splits_dir"]) / f"{split}.json",
    )


def load_split_manifest(split: str, paths: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _, json_path = split_paths(split, paths)
    if not json_path.exists():
        raise FileNotFoundError(f"缺少划分文件: {json_path}")
    return read_json(json_path)


def dataset_shapes(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """给日志与 README 用的形状摘要。"""
    from .schema import observation_spec, state_spec

    obs = observation_spec(data_cfg)
    st = state_spec(data_cfg)
    return {
        "images": list(obs.image_shape),
        "state_dim": st.dim,
        "chunk": [int(get(data_cfg, "chunk.H", 8)), int(get(data_cfg, "action.dim", 7))],
        "num_frames_K": obs.num_frames,
    }
