"""合成演示数据生成器（无真实数据时的 MOCK 路径，所有产物带 source="MOCK"）。

设计目标（对应提示词【五】5.4 与【十三】关键提示）：
  (a) 隐式策略：由图像中的可控视觉线索（目标块方位）与机器人状态（含 SLAM 位姿）决定目标动作；
      目标块在世界系中固定于 dock + T_OFFSET，因此「目标在机体系的位置」
          g(t) = R(-Δyaw) · (T_OFFSET - Δp_base)
      完全由 SLAM 相对位姿给出；图像只提供方位（有限视场、无尺度线索）。
      于是 A4（no_slam_input）会真实地变差，而不是人为制造差异。
  (b) 多模态：每条 episode 采样一个不可观测的接近方向 bias ∈ {-1,+1}，
      它在轨迹早期显著改变动作块、在后期衰减到同一目标 → 同一观测对应 ≥2 种合理动作。
  (c) 可控噪声：动作噪声、观测延迟、丢帧、SLAM 漂移与丢帧，用于鲁棒性与时间同步验证。

同时导出与真值位姿严格同步的激光扫描流（data/slam/scans.npz），
供真正的 Google Cartographer 离线回放使用（见 src/slam/ros2_scan_player.py）。

坐标系：世界系为右手系（x 前，y 左）；角度 rad；长度 m；图像为机器人中心俯视图。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..utils.config import get, load_config, load_paths
from ..utils.io_utils import atomic_write_json, ensure_dir, save_npz, sha256_json

# 全量状态块顺序（原始数据永远按这套布局存储；训练侧按配置选列，见 build_dataset）
FULL_BLOCKS: list[tuple[str, int]] = [
    ("joint_pos", 7),
    ("joint_vel", 7),
    ("gripper", 1),
    ("ee_pose", 7),
    ("slam_pose", 4),
    ("slam_valid", 1),
    ("time_feat", 2),
]
FULL_LAYOUT: dict[str, slice] = {}
_cursor = 0
for _name, _dim in FULL_BLOCKS:
    FULL_LAYOUT[_name] = slice(_cursor, _cursor + _dim)
    _cursor += _dim
FULL_STATE_DIM = _cursor

T_OFFSET = np.array([1.60, 0.90], dtype=np.float64)      # 目标块相对 dock 的世界偏移
DISTRACTOR_OFFSET = np.array([-1.45, 1.55], dtype=np.float64)
VIEW_HALF_EXTENT = 1.6                                    # 图像视野半宽（m）


# ======================================================================================
# 世界模型与射线投射
# ======================================================================================
@dataclass
class World:
    """2D 世界：矩形场地边界 + 若干矩形障碍物。"""

    lower: np.ndarray
    upper: np.ndarray
    boxes: np.ndarray      # [N, 4] = (xmin, ymin, xmax, ymax)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "boxes": self.boxes.tolist(),
            "goal_world": (self.lower * 0 + np.array([3.0, 3.0]) + T_OFFSET).tolist(),
            "distractor_world": (np.array([3.0, 3.0]) + DISTRACTOR_OFFSET).tolist(),
        }


def build_world(size: tuple[float, float], num_obstacles: int, rng: np.random.Generator,
                keep_clear: np.ndarray | None = None, clear_radius: float = 0.55) -> World:
    """随机生成障碍物，并保证不压到机器人轨迹（keep_clear: [T,2] 轨迹点）。"""
    lower = np.array([0.0, 0.0])
    upper = np.array([size[0], size[1]], dtype=np.float64)
    boxes: list[np.ndarray] = []
    attempts = 0
    while len(boxes) < num_obstacles and attempts < num_obstacles * 200:
        attempts += 1
        half = rng.uniform(0.15, 0.45, size=2)
        center = rng.uniform(lower + half + 0.4, upper - half - 0.4)
        box = np.array([center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]])
        goal = np.array([3.0, 3.0]) + T_OFFSET
        goal_world = np.array([3.0, 3.0]) + DISTRACTOR_OFFSET
        if np.linalg.norm(box[:2] - goal) < 0.5 or np.linalg.norm(box[:2] - goal_world) < 0.5:
            continue
        if keep_clear is not None:
            inside = (
                (keep_clear[:, 0] > box[0] - clear_radius)
                & (keep_clear[:, 0] < box[2] + clear_radius)
                & (keep_clear[:, 1] > box[1] - clear_radius)
                & (keep_clear[:, 1] < box[3] + clear_radius)
            )
            if inside.any():
                continue
        boxes.append(box)
    return World(lower=lower, upper=upper, boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4))


def _slab(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """轴对齐 slab 法，返回 (t_enter, t_exit)；无交点时 t_enter > t_exit。"""
    eps = 1e-9
    safe = np.where(np.abs(dirs) < eps, eps * np.sign(np.where(dirs == 0.0, 1.0, dirs)), dirs)
    t1 = (lo - origin) / safe
    t2 = (hi - origin) / safe
    t_lo = np.minimum(t1, t2)
    t_hi = np.maximum(t1, t2)
    t_enter = t_lo.max(axis=-1)
    t_exit = t_hi.min(axis=-1)
    return t_enter, t_exit


def cast_rays(world: World, pose: np.ndarray, angles: np.ndarray, range_min: float, range_max: float,
              noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """从位姿 (x, y, yaw) 向 angles 方向投射激光，返回距离 [B]（无命中为 inf）。"""
    origin = np.asarray(pose, dtype=np.float64)[:2]
    dirs = np.stack([np.cos(angles + pose[2]), np.sin(angles + pose[2])], axis=-1)  # [B,2]
    _, t_exit_wall = _slab(origin, dirs, world.lower, world.upper)
    ranges = t_exit_wall
    for box in world.boxes:
        t_enter, t_exit = _slab(origin, dirs, box[:2], box[2:])
        hit = (t_enter > 0.0) & (t_enter <= t_exit)
        ranges = np.where(hit & (t_enter < ranges), t_enter, ranges)
    ranges = ranges + rng.normal(0.0, noise_std, size=ranges.shape)
    ranges = np.where(ranges < range_min, np.inf, np.clip(ranges, range_min, range_max))
    return ranges


# ======================================================================================
# 渲染（机器人中心俯视图）
# ======================================================================================
def _to_base_frame(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """世界点 → 机体系 (x 前, y 左)。"""
    rel = np.asarray(points_world, dtype=np.float64) - np.asarray(pose, dtype=np.float64)[:2]
    c, s = math.cos(-pose[2]), math.sin(-pose[2])
    x = c * rel[..., 0] - s * rel[..., 1]
    y = s * rel[..., 0] + c * rel[..., 1]
    return np.stack([x, y], axis=-1)


def _draw_disc(img: np.ndarray, row: float, col: float, radius: float, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    r = int(math.ceil(radius))
    r0, r1 = max(0, int(row) - r), min(h, int(row) + r + 1)
    c0, c1 = max(0, int(col) - r), min(w, int(col) + r + 1)
    if r0 >= r1 or c0 >= c1:
        return
    rr, cc = np.mgrid[r0:r1, c0:c1]
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    for ch in range(3):
        img[r0:r1, c0:c1, ch][mask] = color[ch]


def render_image(world: World, base_pose: np.ndarray, ee_pose: np.ndarray, gripper: float,
                 image_size: int, goal_world: np.ndarray, distractor_world: np.ndarray) -> np.ndarray:
    """渲染单帧 RGB 俯视图，返回 CHW uint8（与 dataset_schema.json 的 [T,C,H,W] 一致）。"""
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img[..., :] = (18, 18, 24)                       # 场地底色（视野外）
    scale = image_size / (2.0 * VIEW_HALF_EXTENT)
    cx = cy = image_size / 2.0

    # 场地地面：按格子填充，形成可辨识纹理
    grid = np.mgrid[0:image_size, 0:image_size]
    rows, cols = grid[0], grid[1]
    bx = (cy - rows) / scale
    by = (cx - cols) / scale
    c, s = math.cos(base_pose[2]), math.sin(base_pose[2])
    wx = base_pose[0] + c * bx - s * by
    wy = base_pose[1] + s * bx + c * by
    inside = (wx >= world.lower[0]) & (wx <= world.upper[0]) & (wy >= world.lower[1]) & (wy <= world.upper[1])
    img[inside] = (46, 48, 58)
    checker = ((np.floor(wx).astype(np.int32) + np.floor(wy).astype(np.int32)) % 2 == 0) & inside
    img[checker] = (56, 58, 70)

    # 障碍物
    for box in world.boxes:
        corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        base_pts = _to_base_frame(corners, base_pose)
        cols_ = cx - scale * base_pts[:, 1]
        rows_ = cy - scale * base_pts[:, 0]
        x0, x1 = int(np.clip(cols_.min(), 0, image_size - 1)), int(np.clip(cols_.max(), 0, image_size - 1))
        y0, y1 = int(np.clip(rows_.min(), 0, image_size - 1)), int(np.clip(rows_.max(), 0, image_size - 1))
        if x1 > x0 and y1 > y0:
            img[y0 : y1 + 1, x0 : x1 + 1] = (120, 122, 130)

    # 目标块（粉）与干扰块（青）：只在视野内绘制，且不提供尺度线索（固定像素半径）
    for world_point, color in ((goal_world, (255, 105, 180)), (distractor_world, (60, 200, 255))):
        base_pt = _to_base_frame(np.asarray(world_point)[None, :], base_pose)[0]
        if np.max(np.abs(base_pt)) <= VIEW_HALF_EXTENT * 1.05:
            _draw_disc(img, cy - scale * base_pt[0], cx - scale * base_pt[1], 2.0, color)

    # 机器人底盘（白三角，始终朝向图像上方）
    _draw_disc(img, cy, cx - 3, 1.5, (240, 240, 240))
    _draw_disc(img, cy, cx + 3, 1.5, (240, 240, 240))
    _draw_disc(img, cy - 3, cx, 1.5, (240, 240, 240))

    # 末端执行器（绿）与其在图像中的位置（深度由 SLAM 位姿补足，图像只给方位）
    ee_base = np.asarray(ee_pose, dtype=np.float64)[:2]
    ee_row = cy - scale * ee_base[0]
    ee_col = cx - scale * ee_base[1]
    _draw_disc(img, np.clip(ee_row, 0, image_size - 1), np.clip(ee_col, 0, image_size - 1), 2.0, (40, 230, 90))

    # 夹爪开合度：在图像底部画一条长度随开合变化的亮条
    bar_len = int(np.clip((float(gripper) + 1.0) / 2.0, 0.0, 1.0) * (image_size - 8))
    img[image_size - 2 : image_size, 4 : 4 + bar_len] = (250, 210, 60)
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


# ======================================================================================
# 生成
# ======================================================================================
def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def generate_episodes(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """生成全部 episode、激光扫描流与世界定义，返回 manifest 片段。"""
    paths = load_paths()
    mock = cfg["mock"]
    hz = float(get(cfg, "control.hz", 10.0))
    dt = 1.0 / hz
    num_episodes = int(mock["num_episodes"])
    length = int(mock["episode_len"])
    total = num_episodes * length
    rng = np.random.default_rng(int(mock["seed"]))

    # --- 连续的世界轨迹（所有 episode 是同一次长程运行的切片）---
    steps = np.arange(total, dtype=np.float64)
    w = steps * dt
    dock = np.array([3.0, 3.0])
    bx = dock[0] + 1.15 * np.sin(0.55 * w + 0.30) + 0.35 * np.sin(0.21 * w)
    by = dock[1] + 0.95 * np.sin(0.43 * w + 1.20) + 0.30 * np.sin(0.29 * w + 0.40)
    byaw = 0.70 * np.sin(0.17 * w + 0.90) + 0.25 * np.sin(0.41 * w)
    base_pose = np.stack([bx, by, byaw], axis=1)

    world = build_world(
        size=tuple(mock["world"]["size"]),
        num_obstacles=int(mock["world"]["num_obstacles"]),
        rng=rng,
        keep_clear=base_pose[:, :2],
    )
    goal_world = dock + T_OFFSET
    distractor_world = dock + DISTRACTOR_OFFSET

    # --- 激光扫描流（与真值位姿同频同步，供 Cartographer 回放）---
    laser = mock["laser"]
    angles = np.linspace(-math.pi, math.pi, int(laser["beams"]), endpoint=False)
    scans = np.zeros((total, angles.size), dtype=np.float32)
    for i in range(total):
        scans[i] = cast_rays(world, base_pose[i], angles, float(laser["range_min"]),
                             float(laser["range_max"]), float(laser["noise_std"]), rng)

    # --- 每 episode 的时序语义 ---
    obs_delay = int(mock["observation_delay_frames"])
    drop_prob = float(mock["drop_frame_prob"])
    act_noise = float(mock["action_noise_std"])
    multi_prob = float(mock["multimodal_prob"])
    drift_std = float(mock["slam_drift_std"])
    dropout_prob = float(mock["slam_dropout_prob"])

    joint_map = rng.normal(0.0, 0.35, size=(7, 3))
    joint_bias = rng.uniform(-0.2, 0.2, size=7)
    image_size = int(get(cfg, "image.height", 64))
    episodes: list[dict[str, Any]] = []
    multimodal_flags: list[int] = []
    bias_labels: list[int] = []

    for e in range(num_episodes):
        t0 = e * length
        idx = slice(t0, t0 + length)
        base = base_pose[idx].copy()
        i = np.arange(length, dtype=np.float64)

        # --- 目标在机体系的位置 g(t)：只由 SLAM 相对位姿决定 ---
        d_rel = base[:, :2] - base[0, :2]
        yaw_rel = base[:, 2] - base[0, 2]
        vx = T_OFFSET[0] - d_rel[:, 0]
        vy = T_OFFSET[1] - d_rel[:, 1]
        c, s = np.cos(yaw_rel), np.sin(yaw_rel)
        gx = c * vx + s * vy
        gy = -s * vx + c * vy

        # --- 不可观测的多模态接近方向 ---
        if rng.random() < multi_prob:
            bias = 1.0 if rng.random() < 0.5 else -1.0
        else:
            bias = 1.0
        bias_labels.append(int(bias))
        basis = np.exp(-2.5 * i / length)

        # 目标方位 θ（机体系）：由 SLAM 相对位姿给出的 g(t) 直接决定。
        # 用 atan2 而不是 tanh 饱和映射：基座持续运动时 θ 持续变化，
        # 演示轨迹因此始终有激励（tanh 版本在大部分时间饱和，导致名义动作退化为噪声）。
        theta = np.arctan2(gy, np.maximum(gx, 0.25))
        ee_target = np.stack(
            [
                0.30 + 0.06 * np.cos(theta),
                0.30 * np.sin(theta) + 0.05 * bias * basis,
                0.42 + 0.04 * np.cos(theta),
            ],
            axis=1,
        )
        rz_target = np.clip(0.8 * theta, -0.9, 0.9)
        # 绕 x/y 的小幅姿态调整（跟随目标方位时的自然微调），避免这两维退化为纯噪声
        rx_target = 0.06 * np.sin(theta)
        ry_target = 0.04 * np.cos(theta)
        rpy_target = np.stack([rx_target, ry_target, rz_target], axis=1)

        # --- 一阶动力学 + 噪声 ---
        # 一阶动力学：alpha=0.15（时间常数约 6.7 步 = 0.67s@10Hz）。
        # 取值依据：alpha 越大跟踪越紧、名义步长越小，动作会被噪声盖住；
        # 0.15 使跟踪误差与名义步长保持可观测（见 logs/12_data_check.json 的 action 统计）。
        alpha = 0.15
        pos = np.zeros((length, 3))
        rpy = np.zeros((length, 3))
        pos[0] = np.array([0.30, 0.0, 0.40])
        for t in range(1, length):
            pos[t] = pos[t - 1] + alpha * (ee_target[t - 1] - pos[t - 1]) + rng.normal(0, act_noise, 3)
            rpy[t] = rpy[t - 1] + alpha * (rpy_target[t - 1] - rpy[t - 1]) + rng.normal(0, act_noise, 3)

        gripper = 0.9 - 1.6 / (1.0 + np.exp(-(i / length - 0.6) * 8.0))
        quat = _quat_from_rpy(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        ee_pose = np.concatenate([pos, quat], axis=1)                      # [L,7] x,y,z,qw,qx,qy,qz

        # --- 动作 = 状态差分（含夹爪）---
        full = np.concatenate([pos, rpy, gripper[:, None]], axis=1).astype(np.float32)
        action = np.zeros_like(full)
        action[:-1] = full[1:] - full[:-1]

        # --- 关节量与关节速度（由末端位置线性映射得到，非真实 IK）---
        joint_pos = pos @ joint_map.T + joint_bias
        joint_vel = np.gradient(joint_pos, dt, axis=0)

        # --- SLAM 位姿（相对 episode 起点）+ 漂移 + 丢帧 ---
        drift = np.cumsum(rng.normal(0.0, drift_std / 10.0, size=(length, 2)), axis=0)
        drift += 0.01 * np.sin(0.7 * i / hz)[:, None]
        drift_yaw = np.cumsum(rng.normal(0.0, drift_std / 20.0, size=length))
        slam_xy = d_rel + drift
        slam_yaw = yaw_rel + drift_yaw
        slam_pose_abs = np.stack([slam_xy[:, 0], slam_xy[:, 1], slam_yaw], axis=1)
        slam_valid = (rng.random(length) > dropout_prob).astype(np.uint8)

        # --- 状态矩阵（全量块布局；训练侧再按配置选列）---
        state = np.zeros((length, FULL_STATE_DIM), dtype=np.float32)
        state[:, FULL_LAYOUT["joint_pos"]] = joint_pos
        state[:, FULL_LAYOUT["joint_vel"]] = joint_vel
        state[:, FULL_LAYOUT["gripper"]] = gripper[:, None]
        state[:, FULL_LAYOUT["ee_pose"]] = ee_pose
        state[:, FULL_LAYOUT["slam_pose"]] = np.stack(
            [slam_xy[:, 0], slam_xy[:, 1], np.sin(slam_yaw), np.cos(slam_yaw)], axis=1
        )
        state[:, FULL_LAYOUT["slam_valid"]] = slam_valid[:, None]
        phase = 2.0 * math.pi * i / max(1.0, length)
        state[:, FULL_LAYOUT["time_feat"]] = np.stack([np.sin(phase), np.cos(phase)], axis=1)

        # --- 图像：按观测延迟渲染，并允许丢帧（重复上一帧）---
        images = np.zeros((length, 3, image_size, image_size), dtype=np.uint8)
        last = None
        for t in range(length):
            td = max(0, t - obs_delay)
            if last is not None and rng.random() < drop_prob:
                images[t] = last
                continue
            frame = render_image(world, base_pose[t0 + td], ee_pose[td], float(gripper[td]),
                                 image_size, goal_world, distractor_world)
            images[t] = frame
            last = frame

        timestamps = (t0 + np.arange(length, dtype=np.float64)) * dt
        dist_to_goal = float(np.linalg.norm(pos[-1] - ee_target[-1]))
        success = int(dist_to_goal < 0.06)
        ep_id = f"mock_{e:04d}"
        path = ensure_dir(paths["mock_dir"]) / f"{ep_id}.npz"
        save_npz(
            path,
            images=images,
            state=state,
            action=action,
            timestamp=timestamps,
            base_pose=base.astype(np.float32),
            slam_pose=slam_pose_abs.astype(np.float32),
            slam_valid=slam_valid,
            episode_id=np.array(ep_id),
            source=np.array("MOCK"),
            success=np.array(success, dtype=np.uint8),
            state_block_names=np.array([n for n, _ in FULL_BLOCKS]),
            state_block_dims=np.array([d for _, d in FULL_BLOCKS], dtype=np.int32),
        )
        episodes.append(
            {
                "episode_id": ep_id,
                "path": str(path),
                "length": length,
                "success": success,
                "final_goal_distance": dist_to_goal,
                "approach_bias": int(bias),
                "t_start": float(timestamps[0]),
                "t_end": float(timestamps[-1]),
                "sha256": sha256_json({"episode_id": ep_id, "seed": int(mock["seed"]), "bias": int(bias)}),
            }
        )
        multimodal_flags.append(1 if abs(bias) > 0 and multi_prob > 0 else 0)

    # --- 保存激光扫描流与世界定义（供真正的 Cartographer 回放）---
    slam_dir = ensure_dir(paths["slam_data_dir"])
    scan_path = slam_dir / "scans.npz"
    save_npz(
        scan_path,
        angles=angles.astype(np.float32),
        ranges=scans.astype(np.float32),
        timestamps=(np.arange(total, dtype=np.float64) * dt),
        ground_truth_pose=base_pose.astype(np.float32),
        episode_index=np.repeat(np.arange(num_episodes, dtype=np.int32), length),
        range_min=np.array(float(laser["range_min"]), dtype=np.float32),
        range_max=np.array(float(laser["range_max"]), dtype=np.float32),
        source=np.array("MOCK"),
    )
    atomic_write_json(slam_dir / "world.json", world.to_dict())
    atomic_write_json(
        slam_dir / "ground_truth.json",
        {
            "source": "MOCK",
            "num_frames": total,
            "fps": hz,
            "trajectory": base_pose.tolist(),
            "episode_index": np.repeat(np.arange(num_episodes, dtype=np.int32), length).tolist(),
            "dock": dock.tolist(),
            "goal_world": goal_world.tolist(),
        },
    )

    manifest = {
        "source": "MOCK",
        "generator": "src/datasets/mock_generator.py",
        "seed": int(mock["seed"]),
        "num_episodes": num_episodes,
        "episode_len": length,
        "total_frames": total,
        "fps": hz,
        "duration_s": total / hz,
        "multimodal_episodes": int(sum(multimodal_flags)),
        "bias_histogram": {str(b): int(bias_labels.count(b)) for b in (-1, 1)},
        "world": world.to_dict(),
        "scan_file": str(scan_path),
        "state_layout": {name: [FULL_LAYOUT[name].start, FULL_LAYOUT[name].stop] for name, _ in FULL_BLOCKS},
        "state_dim_full": FULL_STATE_DIM,
        "episodes": episodes,
        "notes": [
            "MOCK 数据：由规则化隐式策略生成，不代表真实机器人动力学与真实相机成像。",
            "图像为 64x64 俯视示意图，只提供目标方位，不提供尺度线索。",
            "激光扫描由同一世界模型解析射线投射得到，用于驱动真实 Cartographer。",
        ],
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 MOCK 演示数据（含激光扫描流）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 configs/data.yaml 的 mock.seed")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖 mock.num_episodes")
    parser.add_argument("--length", type=int, default=None, help="覆盖 mock.episode_len")
    parser.add_argument("--manifest-out", type=str, default=None, help="manifest 落盘路径")
    args = parser.parse_args(argv)

    overrides = []
    if args.seed is not None:
        overrides.append(f"mock.seed={args.seed}")
    if args.episodes is not None:
        overrides.append(f"mock.num_episodes={args.episodes}")
    if args.length is not None:
        overrides.append(f"mock.episode_len={args.length}")
    cfg = load_config("data", overrides=overrides)
    manifest = generate_episodes(cfg)
    if args.manifest_out:
        atomic_write_json(args.manifest_out, manifest)
    print(
        json.dumps(
            {
                "status": "OK",
                "source": "MOCK",
                "num_episodes": manifest["num_episodes"],
                "total_frames": manifest["total_frames"],
                "multimodal_episodes": manifest["multimodal_episodes"],
                "scan_file": manifest["scan_file"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
