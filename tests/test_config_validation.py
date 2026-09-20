"""配置校验：未知字段、越界值、互斥项必须报错（禁止静默忽略拼错的超参）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (
    ConfigError,
    apply_overrides,
    config_hash,
    get,
    load_config,
    load_paths,
    state_block_layout,
    state_dim,
)


def test_project_root_is_fixed():
    paths = load_paths()
    from src.utils.config import project_root as derive_root

    # 项目根 = 本仓库所在目录（代码实际位置），必须与配置声明一致
    assert paths["project_root"] == str(derive_root())
    # 与克隆位置、机器用户名无关：仓库里不得出现写死的本机路径
    assert Path(paths["project_root"]).is_absolute()
    # 环境与工具根是**外部依赖**（conda 锁定文件 / Cartographer 构建脚本与产物），允许被环境变量覆盖
    assert Path(paths["env_root"]).name == "SLAM"
    assert (paths.get("override_env") or {}).get("env_root") == "CVSLAM_ENV_ROOT"
    # 不允许任何覆盖入口改动 project_root：环境变量也不生效（无对应 override_env 项）
    assert "project_root" not in (paths.get("override_env") or {})


def test_all_configs_load():
    for name in ("paths", "data", "model", "train", "infer", "slam"):
        assert isinstance(load_config(name), dict)


def test_unknown_field_rejected():
    with pytest.raises(ConfigError):
        load_config("train", overrides=["optimizer.lerning_rate=1e-3"])   # 拼写错误必须报错


def test_out_of_range_rejected():
    with pytest.raises(ConfigError):
        load_config("data", overrides=["chunk.H=0"])
    with pytest.raises(ConfigError):
        load_config("train", overrides=["batch_size=-1"])


def test_type_error_rejected():
    with pytest.raises(ConfigError):
        load_config("train", overrides=["amp=not_a_bool"])


def test_splits_must_sum_to_one():
    with pytest.raises(ConfigError):
        load_config("data", overrides=["splits.train=0.5", "splits.val=0.1", "splits.test=0.1"])


def test_unknown_override_path_rejected():
    with pytest.raises(ConfigError):
        apply_overrides(load_config("train"), ["does.not.exist=1"])


def test_override_format_checked():
    with pytest.raises(ConfigError):
        apply_overrides(load_config("train"), ["just_a_key_without_equals"])


def test_ablation_override_applies():
    base = load_config("data")
    reduced = load_config("data", ablation="no_slam_input")
    assert get(base, "slam_input.enabled") is True
    assert get(reduced, "slam_input.enabled") is False
    assert state_dim(reduced) == state_dim(base) - 5      # slam_pose(4) + slam_valid(1)
    assert [b[0] for b in state_block_layout(reduced)] == [
        "joint_pos", "joint_vel", "gripper", "ee_pose", "time_feat"
    ]


def test_chunk_length_ablation_changes_horizon():
    for h in (4, 16, 32):
        cfg = load_config("data", ablation=f"chunk_len_H{h}")
        assert get(cfg, "chunk.H") == h


def test_infer_safety_mutex():
    """allow_command_publish=true 必须同时启用 estop 与 watchdog，否则拒绝加载。"""
    with pytest.raises(ConfigError):
        load_config("infer", overrides=[
            "safety.allow_command_publish=true",
            "safety.require_estop_reachable=false",
        ])


def test_config_hash_is_stable_and_sensitive():
    a = load_config("data")
    b = load_config("data")
    assert config_hash(a) == config_hash(b)
    c = load_config("data", overrides=["chunk.H=8"])
    assert config_hash(a) == config_hash(c)          # 与 base 相同 → 同哈希
    d = load_config("data", ablation="chunk_len_H32")
    assert config_hash(a) != config_hash(d)
