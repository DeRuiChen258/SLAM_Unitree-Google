"""配置加载、字段级校验、路径解析与 config hash。

优先级（低 → 高）：
    默认值 → 主配置 YAML → 消融覆盖 YAML → CLI --override k=v → 环境变量（仅外部路径）

硬约束：
    * `paths.yaml:project_root` 固定，不允许被 CLI/环境变量覆盖；
    * 未知字段必须报错（禁止静默忽略拼错的超参）；
    * config hash 用于 checkpoint 与实验登记，必须跨进程稳定。
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .io_utils import sha256_json


class ConfigError(ValueError):
    """配置错误：字段缺失、类型不符、越界、未知字段。"""


# --------------------------------------------------------------------------------------
# 校验规格 DSL
#   ("float", lo, hi) / ("int", lo, hi) / ("bool",) / ("str", choices|None)
#   ("list", spec|None) / ("dict", {..}|None) / ("any",) / ("nullable_int", lo, hi)
# --------------------------------------------------------------------------------------
F_ANY = ("any",)
F_BOOL = ("bool",)
F_STR = ("str", None)


def _f(lo: float | None = None, hi: float | None = None) -> tuple:
    return ("float", lo, hi)


def _i(lo: int | None = None, hi: int | None = None) -> tuple:
    return ("int", lo, hi)


def _s(*choices: str) -> tuple:
    return ("str", list(choices) if choices else None)


FLOAT_01 = _f(0.0, 1.0)

_PATHS_SCHEMA: dict[str, Any] = {
    "project_root": F_STR,
    "data_dir": F_STR, "raw_dir": F_STR, "mock_dir": F_STR, "processed_dir": F_STR,
    "splits_dir": F_STR, "stats_dir": F_STR, "schema_dir": F_STR, "slam_data_dir": F_STR,
    "bag_dir_placeholder": F_STR, "configs_dir": F_STR, "scripts_dir": F_STR, "src_dir": F_STR,
    "tests_dir": F_STR, "docs_dir": F_STR, "evidence_dir": F_STR, "logs_dir": F_STR,
    "checkpoints_dir": F_STR, "outputs_dir": F_STR, "figures_dir": F_STR, "eval_dir": F_STR,
    "ablation_dir": F_STR, "slam_out_dir": F_STR, "infer_samples_dir": F_STR, "third_party_dir": F_STR,
    "env_root": F_STR,
    "train_env_name": F_STR, "train_python": F_STR, "conda_base": F_STR, "unitree_workspace": F_STR,
    "ros2_env_script": F_STR, "ros2_setup_bash": F_STR, "cartographer_env_name": F_STR,
    "cartographer_prefix": F_STR, "cartographer_core_prefix": F_STR, "external_bag_dir": F_STR,
    "override_env": ("dict", None),
}

_DATA_SCHEMA: dict[str, Any] = {
    "schema_version": F_STR,
    "image": {
        "height": _i(8, 4096), "width": _i(8, 4096), "channels": _i(1, 4),
        "color_space": _s("RGB", "BGR", "GRAY"), "resize": ("list", _i(1, 4096)),
        "crop": F_ANY,
        "normalize": {"mean": ("list", _f()), "std": ("list", _f(1e-9, None))},
        "transform_impl": F_STR,
    },
    "observation": {
        "num_frames": _i(1, 64), "frame_stride": _i(1, 64), "num_cameras": _i(1, 8),
        "multi_camera_mode": _s("concat", "stack"),
    },
    "state": {
        "blocks": ("list", {"name": F_STR, "dim": _i(1, 4096), "requires": F_STR}),
        "normalize_blocks": ("list", F_STR),
        "passthrough_blocks": ("list", F_STR),
    },
    "slam_input": {
        "enabled": F_BOOL, "origin": _s("episode_start", "map_origin", "first_valid"),
        "fields": ("list", F_STR), "encode_angle": _s("sincos", "raw"),
        "normalize": F_BOOL,
    },
    "action": {
        "dim": _i(1, 64), "layout": ("list", F_STR), "frame": _s("base", "ee", "world"),
        "is_delta": F_BOOL, "gripper_range": ("list", _f()),
        "delta_translation_limit": _f(1e-6, 1.0), "delta_rotation_limit": _f(1e-6, 3.15),
    },
    "chunk": {"H": _i(1, 256), "stride": _i(1, 256), "pad_mode": _s("mask", "crop", "error")},
    "control": {"hz": _f(1e-3, 1000.0), "dt": _f(1e-6, 10.0)},
    "splits": {"train": FLOAT_01, "val": FLOAT_01, "test": FLOAT_01, "seed": _i(0, 2**31 - 1)},
    "filters": {
        "max_gap_s": _f(1e-6, 60.0), "min_motion": _f(0.0, 1.0), "action_bound_sigma": _f(0.1, 100.0),
        "max_slam_gap_s": _f(1e-6, 60.0), "reject_invalid_slam": F_BOOL, "black_frame_std": _f(0.0, 1.0),
        "max_reject_ratio": FLOAT_01,
    },
    "stats": {"clip_sigma": _f(0.1, 100.0), "min_std": _f(1e-12, 1.0)},
    "mock": {
        "seed": _i(0, 2**31 - 1), "num_episodes": _i(1, 100000), "episode_len": _i(8, 100000),
        "multimodal_prob": FLOAT_01, "action_noise_std": _f(0.0, 1.0),
        "observation_delay_frames": _i(0, 64), "drop_frame_prob": FLOAT_01,
        "slam_drift_std": _f(0.0, 1.0), "slam_dropout_prob": FLOAT_01,
        "world": {"size": ("list", _f(0.1, 1000.0)), "num_obstacles": _i(0, 1000), "resolution": _f(0.005, 1.0)},
        "laser": {"beams": _i(8, 4096), "range_min": _f(0.01, 10.0), "range_max": _f(0.02, 100.0), "noise_std": _f(0.0, 1.0)},
    },
}

_MODEL_SCHEMA: dict[str, Any] = {
    "vision": {
        "backbone": _s("small_cnn"), "in_channels": _i(1, 8), "feature_dim": _i(4, 4096),
        "pretrained": F_BOOL, "freeze": F_BOOL, "norm": _s("group", "batch", "none"),
    },
    "aggregation": {"mode": _s("last", "mean", "gru"), "hidden": _i(4, 4096)},
    "state_encoder": {"hidden": ("list", _i(1, 8192)), "activation": _s("relu", "gelu", "tanh"), "dropout": FLOAT_01},
    "latent": {
        "enabled": F_BOOL, "dim": _i(1, 512), "prior_hidden": ("list", _i(1, 8192)),
        "posterior_hidden": ("list", _i(1, 8192)), "logvar_clamp": ("list", _f()),
    },
    "decoder": {
        "hidden": ("list", _i(1, 8192)), "activation": _s("relu", "gelu", "tanh"),
        "dropout": FLOAT_01, "output_mode": _s("chunk_full", "stepwise"),
    },
    "fusion": {"mode": _s("concat", "add"), "action_head_hidden": ("list", _i(1, 8192))},
}

_TRAIN_SCHEMA: dict[str, Any] = {
    "seed": _i(0, 2**31 - 1), "device": _s("auto", "cpu", "cuda"), "epochs": _i(1, 100000),
    "batch_size": _i(1, 100000), "max_steps": ("nullable_int", 1, 10**9),
    "optimizer": {
        "name": _s("adamw", "adam", "sgd"), "lr": _f(1e-8, 10.0),
        "weight_decay": _f(0.0, 1.0), "momentum": _f(0.0, 1.0),
    },
    "scheduler": {"name": _s("none", "cosine", "step"), "warmup_steps": _i(0, 10**7), "min_lr_ratio": FLOAT_01},
    "grad_clip": _f(0.0, 1e6), "amp": F_BOOL, "num_workers": _i(0, 64), "deterministic": F_BOOL,
    "loss": {
        "recon": {
            "type": _s("l1", "mse", "huber"), "huber_delta": _f(1e-6, 1e3), "weight": _f(0.0, 1e4),
            "step_weight_mode": _s("uniform", "decay"), "step_weight_decay": _f(0.0, 1.0),
        },
        "kl": {
            "enabled": F_BOOL, "weight": _f(0.0, 1e4), "free_bits": _f(0.0, 1e3),
            "anneal": {"scheme": _s("linear", "cosine", "cyclical", "none"), "warmup_steps": _i(0, 10**8),
                       "start": _f(0.0, 1e4), "end": _f(0.0, 1e4)},
        },
        "smooth": {"enabled": F_BOOL, "weight": _f(0.0, 1e4), "order": _i(1, 2)},
        "temporal_consistency": {"enabled": F_BOOL, "weight": _f(0.0, 1e4), "shift": _i(1, 64)},
    },
    "validation": {"every_n_epochs": _i(1, 10000), "batch_limit": _i(1, 100000)},
    "checkpoint": {"every_n_epochs": _i(1, 10000), "monitor": F_STR, "mode": _s("min", "max"),
                   "early_stop_patience": _i(1, 100000)},
    "overfit_gate": {"enabled": F_BOOL, "steps": _i(1, 10**7), "batch_size": _i(1, 100000),
                     "loss_drop_threshold": FLOAT_01, "log_every": _i(1, 10**6)},
    "dataloader": {
        "drop_last": F_BOOL, "pin_memory": F_BOOL, "num_workers": ("nullable_int", 0, 64),
        "persistent_workers": F_BOOL, "prefetch_factor": _i(1, 64),
    },
    "device_policy": {
        "max_workers": _i(0, 128), "worker_cpu_divisor": _i(1, 128),
        "torch_threads": ("nullable_int", 1, 4096), "non_blocking": F_BOOL,
        "gpu_idle_warn_ratio": FLOAT_01, "timing_every": _i(1, 10000),
    },
}

_INFER_SCHEMA: dict[str, Any] = {
    "checkpoint": F_STR, "split": _s("train", "val", "test"), "device": _s("auto", "cpu", "cuda"),
    "latent": {"mode": _s("mean", "sample"), "n_samples": _i(1, 64), "sample_seed": _i(0, 2**31 - 1)},
    "chunk_exec": {"n_exec": _i(1, 256), "mode": _s("closed_loop", "open_loop")},
    "temporal_ensemble": {
        "enabled": F_BOOL, "mode": _s("uniform", "exponential", "inverse_age"),
        "decay": _f(1e-6, 1.0), "window": _i(1, 4096), "min_weight": _f(0.0, 1.0), "normalize_weights": F_BOOL,
    },
    "limits": {
        "max_translation_per_step": _f(1e-6, 1.0), "max_rotation_per_step": _f(1e-6, 3.15),
        "max_gripper_rate": _f(1e-6, 10.0), "max_action_delta": _f(1e-6, 10.0),
    },
    "safety": {
        "allow_command_publish": F_BOOL, "require_estop_reachable": F_BOOL, "require_watchdog": F_BOOL,
        "watchdog_timeout_s": _f(1e-3, 600.0), "max_clip_events": _i(0, 10**7), "fail_fast_on_nan": F_BOOL,
    },
    "episode": {"max_steps": _i(1, 10**7), "stop_on_success_flag": F_BOOL},
    "logging": {"save_executed_actions": F_BOOL, "save_weight_sources": F_BOOL},
    "dataloader": {
        "num_workers": ("nullable_int", 0, 64), "pin_memory": F_BOOL,
        "persistent_workers": F_BOOL, "prefetch_factor": _i(1, 64),
    },
    "device_policy": {
        "max_workers": _i(0, 128), "worker_cpu_divisor": _i(1, 128),
        "torch_threads": ("nullable_int", 1, 4096), "non_blocking": F_BOOL,
        "gpu_idle_warn_ratio": FLOAT_01, "timing_every": _i(1, 10000),
    },
}

_SLAM_SCHEMA: dict[str, Any] = {
    "mode": _s("online", "replay", "mock", "dry_run"),
    "degradation_level": _s("auto", "a_source_build", "b_container", "c_bag_replay", "d_mock"),
    "source": {"kind": _s("tf", "tracked_pose_topic"), "pose_topic": F_STR, "pose_topic_type": F_STR},
    "frames": {
        "map_frame": F_STR, "tracking_frame": F_STR, "odom_frame": F_STR, "published_frame": F_STR,
        "sensor_frame": F_STR, "provide_odom_frame": F_BOOL,
    },
    "tf": {"lookup_timeout_s": _f(0.0, 60.0), "max_transform_age_s": _f(0.0, 60.0), "require_map_to_base_link": F_BOOL},
    "pose_stream": {
        "transport": _s("jsonl", "udp"), "jsonl_path": F_STR, "udp_host": F_STR,
        "udp_port": _i(1, 65535), "flush_every": _i(1, 100000),
    },
    "sync": {"query_clock": _s("capture", "mono"),
             "max_align_tolerance_s": _f(1e-6, 60.0), "max_extrapolation_s": _f(0.0, 60.0),
             "report_path": F_STR},
    "health": {
        "min_rate_hz": _f(0.0, 100000.0), "max_drop_ratio": FLOAT_01,
        "jump_translation_threshold": _f(1e-6, 1e3), "jump_rotation_threshold": _f(1e-6, 1e3),
        "buffer_size": _i(1, 10**7),
    },
    "fallback": {
        "odom_source": F_STR, "mark_source": F_STR, "unavailable_marker": F_STR, "never_publish_zero_pose": F_BOOL,
    },
    "state_input": {"use_slam_input": F_BOOL, "origin": _s("episode_start", "map_origin", "first_valid"),
                    "velocity_from_diff": F_BOOL, "invalid_policy": _s("keep_last", "mark_invalid")},
    "replay": {
        "scan_file": F_STR, "scan_topic": F_STR, "scan_topic_type": F_STR, "clock_topic": F_STR,
        "publish_rate_hz": _f(0.1, 1000.0), "use_sim_time": F_BOOL, "start_delay_s": _f(0.0, 600.0),
    },
    "cartographer": {
        "config_basename": F_STR, "configuration_directory": F_STR, "pbstream_path": F_STR,
        "map_output_prefix": F_STR, "trajectory_topic": F_STR, "submap_topic": F_STR,
    },
    "metrics": {"reference": _s("ground_truth", "odometry", "mock"), "output_dir": F_STR},
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "paths": _PATHS_SCHEMA,
    "data": _DATA_SCHEMA,
    "model": _MODEL_SCHEMA,
    "train": _TRAIN_SCHEMA,
    "infer": _INFER_SCHEMA,
    "slam": _SLAM_SCHEMA,
}

_CACHE: dict[tuple[str, str], dict] = {}


# --------------------------------------------------------------------------------------
# 基础读写
# --------------------------------------------------------------------------------------
def _resolve_placeholders(node: Any, context: Mapping[str, Any], depth: int = 0) -> Any:
    """解析 ${key} 占位符（最多 8 层，防止循环引用）。

    取值顺序：先查配置内已有键（自引用），未定义时回退到同名环境变量
    （例如 ${HOME}）——这样 paths.yaml 不必把本机绝对路径写进仓库。
    """
    if depth > 8:
        raise ConfigError("路径占位符解析超过 8 层，疑似循环引用")
    if isinstance(node, dict):
        return {k: _resolve_placeholders(v, context, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_placeholders(v, context, depth + 1) for v in node]
    if isinstance(node, str):
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
        out = node
        for _ in range(8):
            match = pattern.search(out)
            if not match:
                break
            key = match.group(1)
            if key in context:
                value = str(context[key])
            else:
                value = os.environ.get(key, "")
                if not value:
                    raise ConfigError(f"路径占位符 ${{{key}}} 未定义（来源: {node}）")
            out = out[: match.start()] + value + out[match.end():]
        if pattern.search(out):
            raise ConfigError(f"占位符未完全解析: {node}")
        return out
    return node


def project_root() -> Path:
    """返回强制部署根目录（本文件位于 <root>/src/utils/config.py）。"""
    return Path(__file__).resolve().parents[2]


def load_paths(use_cache: bool = True) -> dict:
    """加载 paths.yaml；project_root 只允许显式传参或由本文件位置推导。"""
    cache_key = ("paths", "base")
    if use_cache and cache_key in _CACHE:
        return copy.deepcopy(_CACHE[cache_key])
    path = project_root() / "configs" / "paths.yaml"
    if not path.exists():
        raise ConfigError(f"缺少路径配置: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("paths.yaml 顶层必须是映射")
    declared_root = str(raw.get("project_root", "")).rstrip("/")
    derived_root = str(project_root())
    # 哨兵值：项目根由代码位置自动推导，仓库里不出现任何本机绝对路径
    if declared_root in ("auto", "."):
        declared_root = derived_root
        raw["project_root"] = derived_root
    if declared_root != derived_root:
        raise ConfigError(
            f"paths.yaml:project_root 与代码实际部署位置不一致：配置={declared_root} 实际={derived_root}。"
            "该字段禁止被覆盖，请修正部署位置或配置。"
        )
    cfg = _resolve_placeholders(raw, raw)
    # 仅外部依赖路径允许环境变量覆盖
    for key, env_name in (cfg.get("override_env") or {}).items():
        if env_name in os.environ and os.environ[env_name]:
            cfg[key] = os.environ[env_name]
    validate_config("paths", cfg)
    _CACHE[cache_key] = copy.deepcopy(cfg)
    return cfg


def load_config(name: str, ablation: str | None = None, overrides: Iterable[str] | None = None,
                use_cache: bool = False) -> dict:
    """加载主配置（可选叠加消融覆盖与 CLI override）并做字段级校验。"""
    if name not in SCHEMAS:
        raise ConfigError(f"未知配置名 {name!r}，可选：{sorted(SCHEMAS)}")
    cache_key = (name, ablation or "")
    if use_cache and cache_key in _CACHE:
        cfg = copy.deepcopy(_CACHE[cache_key])
    else:
        cfg = _load_base(name)
        if ablation:
            cfg = _apply_ablation(name, ablation, cfg)
        if use_cache:
            _CACHE[cache_key] = copy.deepcopy(cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate_config(name, cfg)
    return cfg


def _load_base(name: str) -> dict:
    path = project_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"缺少配置文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 顶层必须是映射")
    ctx = dict(load_paths())
    ctx.update(raw)
    return _resolve_placeholders(raw, ctx)


def _apply_ablation(name: str, ablation: str, base: dict) -> dict:
    """消融文件按 configs/ablation/<name>.yaml 查找，只写相对 base 的差异。"""
    path = project_root() / "configs" / "ablation" / f"{ablation}.yaml"
    if not path.exists():
        raise ConfigError(f"缺少消融配置: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if name not in raw:
        # 消融文件可只覆盖部分配置文件；未涉及者保持原样
        return base
    delta = raw[name]
    if not isinstance(delta, dict):
        raise ConfigError(f"{path} 中 {name} 段必须是映射")
    merged = deep_merge(copy.deepcopy(base), delta)
    return merged


def deep_merge(base: dict, delta: Mapping[str, Any]) -> dict:
    for key, value in delta.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


_OVERRIDE_RE = re.compile(r"^([A-Za-z0-9_.]+)=(.*)$", re.S)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """应用 `a.b.c=value` 形式的 CLI 覆盖；value 用 YAML 解析以支持 list/bool/float。"""
    out = copy.deepcopy(cfg)
    for item in overrides:
        match = _OVERRIDE_RE.match(item.strip())
        if not match:
            raise ConfigError(f"--override 格式错误（应为 key=value）: {item!r}")
        key_path, raw_value = match.group(1), match.group(2)
        value = yaml.safe_load(raw_value)
        node = out
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"--override 路径不存在: {key_path!r}（在 {part!r} 处中断）")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"--override 未知字段: {key_path!r}")
        node[parts[-1]] = value
    return out


def get(cfg: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key_path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------------------
def validate_config(name: str, cfg: Mapping[str, Any]) -> None:
    if name not in SCHEMAS:
        raise ConfigError(f"未知配置名 {name!r}")
    errors: list[str] = []
    _validate_node(cfg, SCHEMAS[name], name, errors)
    if name == "data":
        _validate_data_cross_fields(cfg, errors)
    if name == "infer":
        _validate_infer_cross_fields(cfg, errors)
    if name == "train":
        _validate_train_cross_fields(cfg, errors)
    if errors:
        raise ConfigError("配置校验失败:\n  - " + "\n  - ".join(errors))


def _validate_node(node: Any, spec: Any, path: str, errors: list[str]) -> None:
    if isinstance(spec, dict):
        if not isinstance(node, Mapping):
            errors.append(f"{path}: 期望映射，实际 {type(node).__name__}")
            return
        unknown = [k for k in node if k not in spec]
        if unknown:
            errors.append(f"{path}: 未知字段 {unknown}（拼写错误必须显式报错）")
        for key, sub in spec.items():
            if key in node:
                _validate_node(node[key], sub, f"{path}.{key}", errors)
        return
    kind = spec[0]
    if kind == "any":
        return
    if kind == "bool":
        if not isinstance(node, bool):
            errors.append(f"{path}: 期望 bool，实际 {type(node).__name__}")
        return
    if kind == "str":
        choices = spec[1]
        if not isinstance(node, str):
            errors.append(f"{path}: 期望 str，实际 {type(node).__name__}")
        elif choices and node not in choices:
            errors.append(f"{path}: {node!r} 不在允许取值 {choices} 内")
        return
    if kind in ("int", "float"):
        lo, hi = spec[1], spec[2]
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            errors.append(f"{path}: 期望 {kind}，实际 {type(node).__name__}")
            return
        if kind == "int" and not float(node).is_integer():
            errors.append(f"{path}: 期望整数，实际 {node}")
            return
        if lo is not None and node < lo:
            errors.append(f"{path}: {node} 小于下限 {lo}")
        if hi is not None and node > hi:
            errors.append(f"{path}: {node} 大于上限 {hi}")
        return
    if kind == "nullable_int":
        if node is None:
            return
        if isinstance(node, bool) or not isinstance(node, int):
            errors.append(f"{path}: 期望 int 或 null，实际 {type(node).__name__}")
            return
        lo, hi = spec[1], spec[2]
        if lo is not None and node < lo:
            errors.append(f"{path}: {node} 小于下限 {lo}")
        if hi is not None and node > hi:
            errors.append(f"{path}: {node} 大于上限 {hi}")
        return
    if kind == "list":
        if not isinstance(node, list):
            errors.append(f"{path}: 期望列表，实际 {type(node).__name__}")
            return
        sub = spec[1]
        if sub is not None:
            for idx, item in enumerate(node):
                _validate_node(item, sub, f"{path}[{idx}]", errors)
        return
    if kind == "dict":
        if not isinstance(node, Mapping):
            errors.append(f"{path}: 期望映射，实际 {type(node).__name__}")
        return
    errors.append(f"{path}: 未支持的校验规格 {spec!r}")


def _validate_data_cross_fields(cfg: Mapping[str, Any], errors: list[str]) -> None:
    blocks = get(cfg, "state.blocks", []) or []
    names = [b.get("name") for b in blocks]
    if len(names) != len(set(names)):
        errors.append("data.state.blocks: 块名重复")
    for name in get(cfg, "state.normalize_blocks", []) or []:
        if name not in names:
            errors.append(f"data.state.normalize_blocks: {name!r} 不在 state.blocks 中")
    for name in get(cfg, "state.passthrough_blocks", []) or []:
        if name not in names:
            errors.append(f"data.state.passthrough_blocks: {name!r} 不在 state.blocks 中")
    if get(cfg, "slam_input.enabled", False) and "slam_pose" not in names:
        errors.append("data.slam_input.enabled=true 但 state.blocks 缺少 slam_pose 块")
    layout = get(cfg, "action.layout", []) or []
    if layout and len(layout) != get(cfg, "action.dim", 0):
        errors.append(f"data.action.layout 长度 {len(layout)} 与 action.dim {get(cfg, 'action.dim')} 不一致")
    split_sum = sum(float(get(cfg, f"splits.{k}", 0.0)) for k in ("train", "val", "test"))
    if abs(split_sum - 1.0) > 1e-6:
        errors.append(f"data.splits 之和 {split_sum} 必须等于 1.0")
    resize = get(cfg, "image.resize")
    if resize and list(resize) != [get(cfg, "image.height"), get(cfg, "image.width")]:
        errors.append("data.image.resize 必须等于 [height, width]")
    mean, std = get(cfg, "image.normalize.mean"), get(cfg, "image.normalize.std")
    if mean is not None and len(mean) != get(cfg, "image.channels"):
        errors.append("data.image.normalize.mean 长度必须等于 channels")
    if std is not None and len(std) != get(cfg, "image.channels"):
        errors.append("data.image.normalize.std 长度必须等于 channels")
    clipped = get(cfg, "stats.clip_sigma")
    if clipped is not None and clipped <= 0:
        errors.append("data.stats.clip_sigma 必须为正")


def _validate_infer_cross_fields(cfg: Mapping[str, Any], errors: list[str]) -> None:
    n_exec = get(cfg, "chunk_exec.n_exec")
    if n_exec is not None and n_exec < 1:
        errors.append("infer.chunk_exec.n_exec 必须 ≥ 1")
    mode = get(cfg, "temporal_ensemble.mode")
    if mode == "exponential" and get(cfg, "temporal_ensemble.decay") is None:
        errors.append("infer.temporal_ensemble.mode=exponential 必须提供 decay")
    safety = cfg.get("safety", {}) if isinstance(cfg, Mapping) else {}
    if safety.get("allow_command_publish") and not (safety.get("require_estop_reachable") and safety.get("require_watchdog")):
        errors.append("infer.safety: 打开 allow_command_publish 必须同时启用 estop 与 watchdog 门禁")


def _validate_train_cross_fields(cfg: Mapping[str, Any], errors: list[str]) -> None:
    kl = get(cfg, "loss.kl", {}) or {}
    if kl.get("enabled") and float(kl.get("weight", 0.0)) == 0.0 and float(get(cfg, "loss.kl.anneal.end", 0.0)) > 0.0:
        errors.append("train.loss.kl: weight=0 但 anneal.end>0，语义冲突（请显式关闭 kl.enabled）")
    if get(cfg, "loss.temporal_consistency.enabled") and float(get(cfg, "loss.temporal_consistency.weight", 0.0)) == 0.0:
        errors.append("train.loss.temporal_consistency: enabled=true 但 weight=0")


# --------------------------------------------------------------------------------------
# hash
# --------------------------------------------------------------------------------------
def config_hash(*configs: Mapping[str, Any]) -> str:
    """多份配置的联合哈希（前 12 位十六进制），用于 checkpoint 与实验登记。"""
    return sha256_json([dict(c) for c in configs])[:12]


def state_dim(data_cfg: Mapping[str, Any]) -> int:
    """由 configs/data.yaml 推出状态向量维度（与 schema 同源）。"""
    total = 0
    for block in get(data_cfg, "state.blocks", []) or []:
        requires = block.get("requires")
        if requires and not get(data_cfg, f"{requires}.enabled", False):
            continue
        total += int(block["dim"])
    return total


def state_block_layout(data_cfg: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """返回 [(block_name, start, end)]，供状态切片与可视化标注。"""
    layout: list[tuple[str, int, int]] = []
    cursor = 0
    for block in get(data_cfg, "state.blocks", []) or []:
        requires = block.get("requires")
        if requires and not get(data_cfg, f"{requires}.enabled", False):
            continue
        end = cursor + int(block["dim"])
        layout.append((str(block["name"]), cursor, end))
        cursor = end
    return layout


def load_all(base: str | None = None, overrides: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """一次性加载全部配置（scripts 与入口使用）。"""
    overrides = overrides or {}
    return {name: load_config(name, ablation=base, overrides=overrides.get(name)) for name in SCHEMAS}


def load_runtime(ablation: str | None = None, overrides: list[str] | None = None) -> tuple[dict, dict, dict, dict]:
    """推理/执行入口常用组合：返回 (data, model, infer, slam) 四份已校验配置。"""
    overrides = overrides or []
    return (
        load_config("data", ablation=ablation, overrides=overrides),
        load_config("model", ablation=ablation, overrides=overrides),
        load_config("infer", ablation=ablation, overrides=overrides),
        load_config("slam", overrides=overrides),
    )


def describe(cfg: Mapping[str, Any], keys: Iterable[str]) -> str:
    return ", ".join(f"{k}={get(cfg, k)}" for k in keys)
