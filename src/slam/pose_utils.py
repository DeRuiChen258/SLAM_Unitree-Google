"""位姿数学与特征化（坐标系约定见 src/slam/__init__.py）。

包含：四元数 ↔ 旋转矩阵、RPY ↔ 四元数、SE(3) 复合/求逆、2D 与 6D/7D 位姿互转、
差分求线/角速度、角度 sin/cos 编码、以 episode 起点为原点的相对位姿。

所有函数必须满足往返一致性（由 tests/test_pose_utils.py 覆盖）。
"""

from __future__ import annotations

import numpy as np


def angle_wrap(angle: np.ndarray | float) -> np.ndarray:
    """把角度规整到 (-π, π]。"""
    arr = np.asarray(angle, dtype=np.float64)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped if isinstance(angle, np.ndarray) else float(wrapped)


def sincos_yaw(yaw: np.ndarray | float) -> np.ndarray:
    """角度特征化：返回 [..., sin(yaw), cos(yaw)]（禁止直接线性归一化原始弧度）。"""
    arr = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.sin(arr), np.cos(arr)], axis=-1)


def yaw_from_sincos(features: np.ndarray) -> np.ndarray:
    """由 [sin, cos] 还原 yaw（用于反归一化与可视化）。"""
    arr = np.asarray(features, dtype=np.float64)
    return np.arctan2(arr[..., 0], arr[..., 1])


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """四元数 [.., (w,x,y,z)] → 旋转矩阵 [.., 3, 3]。"""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"四元数最后一维必须为 4（w,x,y,z），实际 {q.shape}")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("四元数范数为 0，无法归一化")
    w, x, y, z = (q / n)[..., 0], (q / n)[..., 1], (q / n)[..., 2], (q / n)[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 [.., 3, 3] → 四元数 [.., (w,x,y,z)]（Shepperd 分支法）。"""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"旋转矩阵形状应为 [..,3,3]，实际 {m.shape}")
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for i, r in enumerate(flat):
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            out[i] = [s / 4.0, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            out[i] = [(r[2, 1] - r[1, 2]) / s, s / 4.0, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            out[i] = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4.0, (r[1, 2] + r[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            out[i] = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4.0]
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    out[np.sign(out[:, 0]) < 0] *= -1.0   # 统一 w ≥ 0，消除双覆盖歧义
    return out.reshape(m.shape[:-2] + (4,))


def rpy_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ZYX 内旋 RPY → 四元数 [.., (w,x,y,z)]。"""
    r, p, y = np.asarray(roll), np.asarray(pitch), np.asarray(yaw)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """四元数 → ZYX RPY (roll, pitch, yaw)。"""
    m = quat_to_matrix(quat)
    pitch = np.arcsin(np.clip(-m[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    yaw = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def se3_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SE(3) 复合：a ∘ b（先 b 后 a），输入输出均为 4×4。"""
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def se3_inverse(transform: np.ndarray) -> np.ndarray:
    """SE(3) 求逆（利用旋转正交性，比通用求逆稳定）。"""
    t = np.asarray(transform, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def pose2d_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(x, y, yaw) → 4×4 齐次矩阵。"""
    x, y, yaw = (float(v) for v in np.asarray(pose, dtype=np.float64)[:3])
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = np.array([x, y, 0.0])
    return m


def matrix_to_pose2d(matrix: np.ndarray) -> np.ndarray:
    """4×4 齐次矩阵 → (x, y, yaw)（只取平面分量）。"""
    m = np.asarray(matrix, dtype=np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])], dtype=np.float64)


def pose7d_to_matrix(pose: np.ndarray) -> np.ndarray:
    """(x,y,z,qw,qx,qy,qz) → 4×4。"""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape[-1] != 7:
        raise ValueError(f"7D 位姿形状应为 [..,7]，实际 {p.shape}")
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(p[..., 3:])
    m[:3, 3] = p[..., :3]
    return m


def matrix_to_pose7d(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([m[:3, 3], matrix_to_quat(m[:3, :3])])


def relative_pose_2d(poses: np.ndarray, origin_index: int = 0) -> np.ndarray:
    """以 origin_index 为原点求相对位姿 (Δx, Δy, Δyaw)，返回 [T,3]。

    这是 SLAM 位姿进入状态向量的标准形式（提示词 6.5）：相对 episode 起点。
    """
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"poses 应为 [T,3+]，实际 {arr.shape}")
    origin = arr[origin_index]
    c, s = np.cos(-origin[2]), np.sin(-origin[2])
    rot = np.array([[c, -s], [s, c]])
    xy = (arr[:, :2] - origin[:2]) @ rot.T
    return np.concatenate([xy, angle_wrap(arr[:, 2] - origin[2])[:, None]], axis=1)


def velocities_from_poses(poses: np.ndarray, dt: float) -> np.ndarray:
    """由相邻位姿差分求 (vx, vy, omega)（机体系线速度 + 角速度）。"""
    arr = np.asarray(poses, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.zeros((arr.shape[0], 3), dtype=np.float64)
    d = np.gradient(arr[:, :2], dt, axis=0)
    yaw = angle_wrap(arr[:, 2])
    omega = np.gradient(np.unwrap(yaw), dt)
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * d[:, 0] + s * d[:, 1]
    vy = -s * d[:, 0] + c * d[:, 1]
    return np.stack([vx, vy, omega], axis=1)


def slam_features(pose: np.ndarray) -> np.ndarray:
    """把一帧相对 SLAM 位姿变成状态向量特征块：[dx, dy, sin(yaw), cos(yaw)]。"""
    p = np.asarray(pose, dtype=np.float64)
    return np.array([p[0], p[1], np.sin(p[2]), np.cos(p[2])], dtype=np.float64)


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool = False) -> np.ndarray:
    """求把 source 对齐到 target 的刚体变换（3×3 齐次，用于 ATE 去漂移对齐）。"""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("source 与 target 形状必须一致且为 [N, D]")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(src.shape[1])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    scale = 1.0
    if with_scale:
        var_src = float((src_c**2).sum() / src.shape[0])
        scale = float(np.trace(np.diag(d) @ s) / var_src) if var_src > 0 else 1.0
    rot = u @ s @ vt
    trans = mu_dst - scale * rot @ mu_src
    out = np.eye(src.shape[1] + 1)
    out[: src.shape[1], : src.shape[1]] = scale * rot
    out[: src.shape[1], -1] = trans
    return out


def apply_transform2d(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """把 [N,2] 点集按 3×3 齐次变换。"""
    pts = np.asarray(points, dtype=np.float64)
    t = np.asarray(transform, dtype=np.float64)
    return pts @ t[:2, :2].T + t[:2, 2]
