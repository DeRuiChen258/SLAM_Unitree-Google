"""数据契约的数据结构与校验函数。

读取 data/schema/dataset_schema.json 作为机器可读契约，对 episode npz 做全字段校验：
字段存在性 / shape / dtype / 时间戳单调性与间隔一致性 / 取值域 / NaN/Inf。
接口：`validate_episode(path) -> ValidateReport`，被 scripts/12_data_check.py 与 tests 复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.config import get, project_root, state_block_layout


class SchemaError(ValueError):
    """schema 校验失败（硬错误，必须修复数据而不是绕过）。"""


@dataclass
class ObservationSpec:
    num_frames: int
    channels: int
    height: int
    width: int
    num_cameras: int = 1

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return (self.num_frames, self.channels, self.height, self.width)


@dataclass
class StateSpec:
    blocks: list[tuple[str, int, int]] = field(default_factory=list)
    normalize_blocks: list[str] = field(default_factory=list)
    passthrough_blocks: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return self.blocks[-1][2] if self.blocks else 0

    def slice_of(self, name: str) -> slice:
        for block_name, start, end in self.blocks:
            if block_name == name:
                return slice(start, end)
        raise SchemaError(f"状态块 {name!r} 不存在，当前块: {[b[0] for b in self.blocks]}")


@dataclass
class ActionChunkSpec:
    dim: int
    horizon: int
    layout: list[str]


@dataclass
class EpisodeMeta:
    episode_id: str
    path: str
    length: int
    fps: float
    source: str
    success: int
    state_dim: int
    action_dim: int


@dataclass
class ValidateReport:
    """校验报告：errors 非空即视为数据不可用。"""

    path: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def load_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(schema_path) if schema_path else project_root() / "data" / "schema" / "dataset_schema.json"
    if not path.exists():
        raise SchemaError(f"缺少 schema 文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def observation_spec(data_cfg: Mapping[str, Any]) -> ObservationSpec:
    return ObservationSpec(
        num_frames=int(get(data_cfg, "observation.num_frames", 1)),
        channels=int(get(data_cfg, "image.channels", 3)),
        height=int(get(data_cfg, "image.height", 64)),
        width=int(get(data_cfg, "image.width", 64)),
        num_cameras=int(get(data_cfg, "observation.num_cameras", 1)),
    )


def state_spec(data_cfg: Mapping[str, Any]) -> StateSpec:
    return StateSpec(
        blocks=[(name, start, end) for name, start, end in state_block_layout(data_cfg)],
        normalize_blocks=list(get(data_cfg, "state.normalize_blocks", []) or []),
        passthrough_blocks=list(get(data_cfg, "state.passthrough_blocks", []) or []),
    )


def action_chunk_spec(data_cfg: Mapping[str, Any]) -> ActionChunkSpec:
    return ActionChunkSpec(
        dim=int(get(data_cfg, "action.dim", 7)),
        horizon=int(get(data_cfg, "chunk.H", 8)),
        layout=list(get(data_cfg, "action.layout", []) or []),
    )


def assert_finite(array: np.ndarray, name: str, report: ValidateReport | None = None) -> bool:
    """NaN/Inf 检查；返回 True 表示健康。"""
    if array.dtype.kind in "fc":
        bad = ~np.isfinite(array)
        if bad.any():
            message = f"{name}: 含 NaN/Inf（{int(bad.sum())}/{bad.size} 个元素）"
            if report is not None:
                report.add_error(message)
            return False
    return True


def validate_episode(path: str | Path, data_cfg: Mapping[str, Any] | None = None,
                     schema: Mapping[str, Any] | None = None) -> ValidateReport:
    """校验单个 episode npz，返回结构化报告（不抛异常，由调用方决定是否中止）。"""
    from ..utils.config import load_config

    data_cfg = data_cfg or load_config("data")
    schema = schema or load_schema()
    report = ValidateReport(path=str(path))
    p = Path(path)
    if not p.exists():
        report.add_error(f"文件不存在: {p}")
        return report

    obs = observation_spec(data_cfg)
    st = state_spec(data_cfg)
    a_dim = int(get(data_cfg, "action.dim", 7))

    with np.load(p, allow_pickle=False) as data:
        required = schema["episode"]["required"]
        present = set(data.files)
        alias = {"episode_id": {"episode_id", "_episode_id"}, "source": {"source", "_source"}}
        for name in required:
            if name in alias:
                if not (alias[name] & present):
                    report.add_error(f"缺少必需字段: {name}")
            elif name not in present:
                report.add_error(f"缺少必需字段: {name}")
        if report.errors:
            return report

        images = data["images"]
        state = data["state"]
        action = data["action"]
        ts = data["timestamp"]
        base_pose = data["base_pose"]
        slam_pose = data["slam_pose"]
        slam_valid = data["slam_valid"]
        length = int(images.shape[0])

        if images.ndim != 4 or tuple(images.shape[1:]) != (obs.channels, obs.height, obs.width):
            report.add_error(
                f"images shape {tuple(images.shape)} 与配置 [T,{obs.channels},{obs.height},{obs.width}] 不一致"
            )
        # 原始 episode 是**自描述**的：state 列布局由文件自带的 state_block_names/dims 决定，
        # 而"按当前配置选列"发生在窗口构建阶段（src/datasets/schema.py:state_column_selector）。
        # 因此这里必须按原始布局校验维度，而不是按配置维度——
        # 否则任何改变状态维度的消融（如 A4 no_slam_input）都会把原始数据判为非法。
        if "state_block_names" in present and "state_block_dims" in present:
            names = [str(v) for v in np.asarray(data["state_block_names"]).tolist()]
            dims = [int(v) for v in np.asarray(data["state_block_dims"]).tolist()]
            raw_dim = int(sum(dims))
            raw_layout = raw_block_layout(names, dims)
            missing = [b[0] for b in st.blocks if b[0] not in raw_layout]
            if missing:
                report.add_error(
                    f"当前配置需要状态块 {missing}，但原始数据只提供 {names}（无法通过选列得到目标布局）"
                )
        else:
            raw_dim = st.dim
        if tuple(state.shape) != (length, raw_dim):
            report.add_error(
                f"state shape {tuple(state.shape)} 应为 [{length},{raw_dim}]"
                f"（原始块布局 {names if 'names' in dir() else '未声明'}；配置要求的块顺序见 configs/data.yaml）"
            )
        if tuple(action.shape) != (length, a_dim):
            report.add_error(f"action shape {tuple(action.shape)} 应为 [{length},{a_dim}]")
        for name, arr, shape in (
            ("timestamp", ts, (length,)),
            ("base_pose", base_pose, (length, 3)),
            ("slam_pose", slam_pose, (length, 3)),
            ("slam_valid", slam_valid, (length,)),
        ):
            if tuple(arr.shape) != shape:
                report.add_error(f"{name} shape {tuple(arr.shape)} 应为 {list(shape)}")
        if report.errors:
            return report

        if images.dtype != np.uint8:
            report.add_warning(f"images dtype={images.dtype}，建议 uint8（内存中按配置归一化）")
        if not np.issubdtype(state.dtype, np.floating):
            report.add_error(f"state dtype={state.dtype} 必须为浮点")
        if not np.issubdtype(action.dtype, np.floating):
            report.add_error(f"action dtype={action.dtype} 必须为浮点")
        if slam_valid.dtype.kind not in "iu":
            report.add_error(f"slam_valid dtype={slam_valid.dtype} 必须为整数")
        elif np.any((slam_valid != 0) & (slam_valid != 1)):
            report.add_error("slam_valid 只能取 0/1")

        assert_finite(state, "state", report)
        assert_finite(action, "action", report)
        assert_finite(ts.astype(np.float64), "timestamp", report)
        assert_finite(base_pose, "base_pose", report)
        assert_finite(slam_pose, "slam_pose", report)

        if ts.size > 1:
            diff = np.diff(ts.astype(np.float64))
            if np.any(diff <= 0):
                report.add_error(f"timestamp 非严格单调递增（最小间隔 {float(diff.min()):.6g} s）")
            max_gap = float(get(data_cfg, "filters.max_gap_s", 0.35))
            gaps = diff[diff > max_gap]
            if gaps.size:
                report.add_warning(f"存在 {gaps.size} 处时间跳变 > {max_gap}s（最大 {float(gaps.max()):.3f}s）")
            report.stats["dt_median_s"] = float(np.median(diff))
            report.stats["dt_expected_s"] = 1.0 / float(get(data_cfg, "control.hz", 10.0))

        trans_lim = float(get(data_cfg, "action.delta_translation_limit", 0.02)) * float(
            get(data_cfg, "filters.action_bound_sigma", 6.0)
        )
        rot_lim = float(get(data_cfg, "action.delta_rotation_limit", 0.1)) * float(
            get(data_cfg, "filters.action_bound_sigma", 6.0)
        )
        if a_dim >= 6:
            over_t = np.abs(action[:, :3]) > trans_lim
            over_r = np.abs(action[:, 3:6]) > rot_lim
            if over_t.any() or over_r.any():
                report.add_warning(f"动作越界：平移 {int(over_t.sum())} 个、旋转 {int(over_r.sum())} 个")
                report.stats["action_out_of_bound"] = int(over_t.sum() + over_r.sum())
        g_lo, g_hi = get(data_cfg, "action.gripper_range", [-1.0, 1.0])
        if a_dim >= 7 and (action[:, 6].min() < g_lo - 1e-6 or action[:, 6].max() > g_hi + 1e-6):
            report.add_warning("夹爪动作超出 gripper_range")

        report.stats.update(
            {
                "length": length,
                "state_dim": int(state.shape[1]),
                "action_dim": int(action.shape[1]),
                "image_shape": list(images.shape[1:]),
                "slam_valid_ratio": float(slam_valid.mean()),
                "action_abs_mean": float(np.abs(action).mean()),
                "action_abs_max": float(np.abs(action).max()),
            }
        )
    return report


def validate_sample(sample: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> None:
    """校验单个训练样本（Dataset __getitem__ 输出），失败即抛 SchemaError。"""
    obs = observation_spec(data_cfg)
    st = state_spec(data_cfg)
    chunk = action_chunk_spec(data_cfg)

    images = np.asarray(sample["images"])
    state = np.asarray(sample["state"])
    actions = np.asarray(sample["action_chunk"])
    mask = np.asarray(sample["mask"])

    if images.shape != obs.image_shape:
        raise SchemaError(f"images shape {images.shape} != {obs.image_shape}")
    if state.shape != (st.dim,):
        raise SchemaError(f"state shape {state.shape} != ({st.dim},)")
    if actions.shape != (chunk.horizon, chunk.dim):
        raise SchemaError(f"action_chunk shape {actions.shape} != ({chunk.horizon},{chunk.dim})")
    if mask.shape != (chunk.horizon,):
        raise SchemaError(f"mask shape {mask.shape} != ({chunk.horizon},)")
    if not np.isin(mask, [0, 1]).all():
        raise SchemaError("mask 只能取 0/1")
    for name, arr in (("images", images), ("state", state), ("action_chunk", actions)):
        if not np.isfinite(arr.astype(np.float32)).all():
            raise SchemaError(f"{name} 含 NaN/Inf")


def episode_summary(paths: Sequence[str | Path], data_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """批量汇总：episode 数、总帧数、fps、字段统计（供 manifest 使用）。"""
    from ..utils.config import load_config

    data_cfg = data_cfg or load_config("data")
    lengths: list[int] = []
    sources: set[str] = set()
    actions: list[np.ndarray] = []
    slams: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            lengths.append(int(data["images"].shape[0]))
            if "source" in data.files:
                sources.add(str(data["source"]))
            elif "_source" in data.files:
                sources.add(str(data["_source"]))
            actions.append(np.asarray(data["action"], dtype=np.float32))
            slams.append(np.asarray(data["slam_valid"], dtype=np.float32))
    a = np.concatenate(actions, axis=0) if actions else np.zeros((0, 1), dtype=np.float32)
    s = np.concatenate(slams, axis=0) if slams else np.zeros((0,), dtype=np.float32)
    hz = float(get(data_cfg, "control.hz", 10.0))
    return {
        "num_episodes": len(lengths),
        "num_frames": int(sum(lengths)),
        "episode_lengths": lengths,
        "fps": hz,
        "duration_s": float(sum(lengths) / hz) if hz else 0.0,
        "sources": sorted(sources),
        "action_mean": a.mean(axis=0).tolist() if a.size else [],
        "action_std": a.std(axis=0).tolist() if a.size else [],
        "slam_valid_ratio": float(s.mean()) if s.size else 0.0,
    }


def raw_block_layout(block_names: Sequence[str], block_dims: Sequence[int]) -> dict[str, slice]:
    """由原始 npz 自带的块清单还原切片表（raw 数据自描述，不依赖当前配置）。"""
    layout: dict[str, slice] = {}
    cursor = 0
    for name, dim in zip(block_names, block_dims):
        layout[str(name)] = slice(cursor, cursor + int(dim))
        cursor += int(dim)
    return layout


def state_column_selector(raw_layout: Mapping[str, slice], data_cfg: Mapping[str, Any]) -> np.ndarray:
    """把「原始全量状态列」映射到「当前配置需要的列」。

    消融 A4（no_slam_input）通过 configs/data.yaml 的 slam_input.enabled=false
    直接改变目标布局，因此数据侧的列选择必须由本函数统一决定，
    避免训练脚本各自硬编码列号。
    """
    target_blocks = state_block_layout(data_cfg)
    columns: list[int] = []
    for name, start, end in target_blocks:
        if name not in raw_layout:
            raise SchemaError(f"配置要求状态块 {name!r}，但原始数据没有该块（raw 块: {sorted(raw_layout)}）")
        raw = raw_layout[name]
        if raw.stop - raw.start != end - start:
            raise SchemaError(
                f"状态块 {name!r} 维度不一致：原始 {raw.stop - raw.start} vs 配置 {end - start}"
            )
        columns.extend(range(raw.start, raw.stop))
    return np.asarray(columns, dtype=np.int64)
