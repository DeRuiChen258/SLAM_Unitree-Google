"""位姿往返一致性：四元数↔矩阵、RPY、SE(3) 复合/求逆、角度编码、相对位姿。"""

from __future__ import annotations

import numpy as np
import pytest

from src.slam.pose_utils import (
    angle_wrap,
    apply_transform2d,
    matrix_to_pose2d,
    matrix_to_pose7d,
    matrix_to_quat,
    pose2d_to_matrix,
    pose7d_to_matrix,
    quat_to_matrix,
    quat_to_rpy,
    relative_pose_2d,
    rpy_to_quat,
    se3_compose,
    se3_inverse,
    sincos_yaw,
    slam_features,
    umeyama_alignment,
    velocities_from_poses,
    yaw_from_sincos,
)


def test_angle_wrap_range():
    angles = np.array([0.0, np.pi, -np.pi, 3 * np.pi, 10.0, -10.0])
    wrapped = angle_wrap(angles)
    assert np.all(wrapped > -np.pi - 1e-9)
    assert np.all(wrapped <= np.pi + 1e-9)


def test_sincos_roundtrip():
    yaw = np.linspace(-np.pi, np.pi, 17)
    # 比较"角度差"而不是原始数值：±π 的表示不唯一（+π 与 -π 是同一方向）
    recovered = yaw_from_sincos(sincos_yaw(yaw))
    assert np.allclose(angle_wrap(recovered - yaw), np.zeros_like(yaw), atol=1e-9)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(20, 4))
    matrices = quat_to_matrix(quats)
    back = matrix_to_quat(matrices)
    # 四元数有双重覆盖（q 与 -q），比较旋转矩阵才是严格判据
    assert np.allclose(quat_to_matrix(back), matrices, atol=1e-9)


def test_rpy_roundtrip():
    rpy = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.3, 0.15, 2.9]])
    q = rpy_to_quat(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    back = quat_to_rpy(q)
    assert np.allclose(back, rpy, atol=1e-9)


def test_se3_inverse_and_compose():
    a = pose2d_to_matrix([1.0, 2.0, 0.4])
    b = pose2d_to_matrix([-0.5, 0.25, -0.9])
    assert np.allclose(se3_inverse(a) @ a, np.eye(4), atol=1e-9)
    composed = se3_compose(a, b)
    assert np.allclose(se3_inverse(composed), se3_inverse(b) @ se3_inverse(a), atol=1e-9)
    # 复合的几何含义：先 b 后 a，等于把 b 的位移用 a 变换后叠加
    assert np.allclose(matrix_to_pose2d(composed)[:2], (a[:3, :3] @ b[:3, 3])[:2] + a[:3, 3][:2], atol=1e-9)


def test_pose7d_roundtrip():
    rpy = np.array([0.2, -0.1, 0.7])
    q = rpy_to_quat(rpy[0], rpy[1], rpy[2])
    pose7 = np.concatenate([[0.3, -0.4, 0.9], q])
    assert np.allclose(matrix_to_pose7d(pose7d_to_matrix(pose7)), pose7, atol=1e-9)


def test_relative_pose_origin():
    poses = np.array([[1.0, 2.0, 0.5], [1.5, 2.5, 0.7], [0.5, 1.5, 0.3]])
    rel = relative_pose_2d(poses, origin_index=0)
    assert np.allclose(rel[0], [0.0, 0.0, 0.0], atol=1e-9)
    # 期望值：把世界系位移 (0.5, 0.5) 用原点朝向 0.5 rad 旋转到机体系
    theta = 0.5
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    expected_xy = rot @ np.array([0.5, 0.5])
    assert np.allclose(rel[1][:2], expected_xy, atol=1e-9)
    assert rel[1][2] == pytest.approx(0.2, abs=1e-9)


def test_velocities_from_poses():
    dt = 0.1
    t = np.arange(50) * dt
    poses = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)   # 沿 x 以 1 m/s 匀速
    v = velocities_from_poses(poses, dt)
    assert np.allclose(v[1:-1, 0], 1.0, atol=1e-6)
    assert np.allclose(v[1:-1, 2], 0.0, atol=1e-6)


def test_slam_features_layout():
    f = slam_features(np.array([0.5, -0.25, np.pi / 2]))
    assert f.shape == (4,)
    assert f[0] == pytest.approx(0.5)
    assert f[1] == pytest.approx(-0.25)
    assert f[2] == pytest.approx(1.0, abs=1e-9)     # sin(pi/2)
    assert f[3] == pytest.approx(0.0, abs=1e-9)     # cos(pi/2)


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 2))
    theta = 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = src @ rot.T + np.array([1.0, -2.0])
    transform = umeyama_alignment(src, dst, with_scale=False)
    assert np.allclose(apply_transform2d(src, transform), dst, atol=1e-9)
