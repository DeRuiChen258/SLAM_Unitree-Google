"""Cartographer → 策略的桥接节点（**ROS2 侧，必须用带 rclpy 的解释器运行**）。

运行解释器：`$HOME/Workspace/miniconda/envs/cartographer_ros/bin/python`
（RoboStack 独立环境，自带 rclpy + cartographer_ros；禁止在 conda `unitree_rt` 里装 rclpy）

职责：
  (1) 通过 TF 查询 `map → odom → base_link`（frame 名以 configs/slam.yaml 与本机实测为准），
      或订阅 Cartographer 发布的位姿话题（`source.kind=tracked_pose_topic`）；
  (2) 每个位姿打「采集时间戳 + 单调时钟」双时间戳，缓存最近 N 条；
  (3) 检测 TF 超时、丢帧率、跳变（相邻帧位移/角度超阈值）并计数；
  (4) 通过 pose stream（JSONL / UDP）对外发布；
  (5) 降级：Cartographer 不可用时切里程计位姿并把 source 标记为 `odom_fallback`；
      彻底无数据时标记 `unavailable`，**绝不发 0 位姿冒充有效数据**。

模式：online（在线 ROS2）/ replay（bag 或扫描回放）/ mock（确定性合成位姿）/ dry_run（只打印）。
本文件禁止 import torch（ROS2 侧不安装 torch）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..utils.config import get, load_config, load_paths
from ..utils.io_utils import ensure_dir
from ..utils.logging_utils import StageLogger
from .pose_stream import PoseSample, PoseStreamWriter


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """四元数 → yaw（假设 roll/pitch 较小，2D SLAM 场景成立）。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _mat2d(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


@dataclass
class TfEdge:
    parent: str
    child: str
    matrix: np.ndarray
    stamp: float
    is_static: bool = False


class TfGraph:
    """轻量 2D TF 图：支持 map→odom→base_link 链式查询与逆变换。

    为什么不直接用 tf2_ros：本机 RoboStack 环境未提供 `tf2_ros` 的 Python 绑定
    （只有 C++ 包），因此这里用 rclpy 订阅 /tf 与 /tf_static 自行维护等价缓冲，
    并把这一事实写入 docs/cartographer_integration.md（零虚构：不声称用了 tf2_ros）。
    """

    def __init__(self, buffer_size: int = 512) -> None:
        self.edges: dict[tuple[str, str], deque[TfEdge]] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.static: dict[tuple[str, str], TfEdge] = {}
        self.frames: set[str] = set()
        self.n_updates = 0

    def add(self, parent: str, child: str, x: float, y: float, yaw: float, stamp: float,
            is_static: bool = False) -> None:
        edge = TfEdge(parent=parent, child=child, matrix=_mat2d(x, y, yaw), stamp=stamp, is_static=is_static)
        if is_static:
            self.static[(parent, child)] = edge
        else:
            self.edges[(parent, child)].append(edge)
        self.frames.update((parent, child))
        self.n_updates += 1

    def _edge_at(self, parent: str, child: str, t: float | None) -> TfEdge | None:
        if (parent, child) in self.static:
            return self.static[(parent, child)]
        history = self.edges.get((parent, child))
        if not history:
            return None
        if t is None:
            return history[-1]
        best = min(history, key=lambda e: abs(e.stamp - t))
        return best

    def lookup(self, target: str, source: str, t: float | None = None) -> np.ndarray | None:
        """返回把 source 系下的点变换到 target 系的 3×3 矩阵；失败返回 None。"""
        if target == source:
            return np.eye(3)
        front: deque[tuple[str, np.ndarray]] = deque([(source, np.eye(3))])
        visited = {source}
        while front:
            node, t_node_source = front.popleft()
            for (parent, child) in list(self.static.keys()) + list(self.edges.keys()):
                if parent == node and child not in visited:
                    edge = self._edge_at(parent, child, t)
                    if edge is None:
                        continue
                    # child 在 parent 下的位姿 = edge.matrix；需要 T_child_source
                    t_child_source = np.linalg.inv(edge.matrix) @ t_node_source
                    if child == target:
                        return t_child_source
                    visited.add(child)
                    front.append((child, t_child_source))
                elif child == node and parent not in visited:
                    edge = self._edge_at(parent, child, t)
                    if edge is None:
                        continue
                    t_parent_source = edge.matrix @ t_node_source
                    if parent == target:
                        return t_parent_source
                    visited.add(parent)
                    front.append((parent, t_parent_source))
        return None

    def stats(self) -> dict[str, object]:
        return {
            "n_updates": self.n_updates,
            "frames": sorted(self.frames),
            "n_dynamic_edges": len(self.edges),
            "n_static_edges": len(self.static),
        }


class HealthMonitor:
    """丢帧率、频率与跳变检测（阈值来自 configs/slam.yaml:health）。"""

    def __init__(self, slam_cfg: Mapping[str, object]) -> None:
        self.min_rate = float(get(slam_cfg, "health.min_rate_hz", 5.0))
        self.max_drop_ratio = float(get(slam_cfg, "health.max_drop_ratio", 0.1))
        self.jump_t = float(get(slam_cfg, "health.jump_translation_threshold", 0.3))
        self.jump_r = float(get(slam_cfg, "health.jump_rotation_threshold", 0.5))
        self.times: deque[float] = deque(maxlen=200)
        self.last_pose: np.ndarray | None = None
        self.jump_events = 0
        self.n_samples = 0
        self.n_missing = 0

    def observe(self, pose: np.ndarray | None, t: float) -> str | None:
        """返回告警字符串（无告警返回 None）。"""
        self.times.append(t)
        if pose is None:
            self.n_missing += 1
            return "pose_missing"
        self.n_samples += 1
        if self.last_pose is not None:
            d = float(np.linalg.norm(pose[:2] - self.last_pose[:2]))
            dyaw = abs(float((pose[2] - self.last_pose[2] + math.pi) % (2 * math.pi) - math.pi))
            if d > self.jump_t or dyaw > self.jump_r:
                self.jump_events += 1
                self.last_pose = pose
                return f"jump(d={d:.3f}, dyaw={dyaw:.3f})"
        self.last_pose = pose
        return None

    def rate_hz(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

    def drop_ratio(self) -> float:
        total = self.n_samples + self.n_missing
        return self.n_missing / total if total else 0.0

    def degraded(self) -> bool:
        return self.drop_ratio() > self.max_drop_ratio or (
            self.rate_hz() < self.min_rate and len(self.times) > 20
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rate_hz": self.rate_hz(),
            "drop_ratio": self.drop_ratio(),
            "jump_events": self.jump_events,
            "n_samples": self.n_samples,
            "n_missing": self.n_missing,
            "degraded": self.degraded(),
        }


def mock_pose_stream(paths: Mapping[str, object], slam_cfg: Mapping[str, object], duration_s: float,
                     rate_hz: float = 20.0, seed: int = 0) -> dict[str, object]:
    """确定性 mock 位姿流（降级链 d 级）：显式标 source=mock，绝不冒充真实 SLAM。"""
    writer = PoseStreamWriter(
        transport=str(get(slam_cfg, "pose_stream.transport", "jsonl")),
        jsonl_path=get(slam_cfg, "pose_stream.jsonl_path"),
        flush_every=int(get(slam_cfg, "pose_stream.flush_every", 1)),
    )
    n = int(duration_s * rate_hz)
    t0 = time.monotonic()
    for i in range(n):
        t = i / rate_hz
        x = 0.6 * math.sin(0.5 * t)
        y = 0.4 * math.sin(0.9 * t + 0.7)
        yaw = 0.3 * math.sin(0.3 * t)
        writer.write(PoseSample(t_capture=t, t_mono=t0 + t, x=x, y=y, yaw=yaw,
                                source="mock", valid=1, frame_id="map", child_frame_id="base_link"))
    writer.close()
    return {"mode": "mock", "n_samples": n, "path": str(get(slam_cfg, "pose_stream.jsonl_path"))}


def run_bridge(args: argparse.Namespace) -> int:
    """启动 ROS2 桥接节点（需要 rclpy；失败时明确报错而不是静默降级）。"""
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage

    paths = load_paths()
    slam_cfg = load_config("slam", overrides=args.override)
    log = StageLogger("cartographer_bridge", paths["logs_dir"], stage="40")
    stamp_origin: dict[str, float] = {"value": -1.0}
    stream = PoseStreamWriter(
        transport=str(get(slam_cfg, "pose_stream.transport", "jsonl")),
        jsonl_path=get(slam_cfg, "pose_stream.jsonl_path"),
        flush_every=int(get(slam_cfg, "pose_stream.flush_every", 1)),
    )
    graph = TfGraph(buffer_size=int(get(slam_cfg, "health.buffer_size", 512)))
    health = HealthMonitor(slam_cfg)
    map_frame = str(get(slam_cfg, "frames.map_frame", "map"))
    odom_frame = str(get(slam_cfg, "frames.odom_frame", "odom"))
    tracking = str(get(slam_cfg, "frames.tracking_frame", "base_link"))
    marker = str(get(slam_cfg, "fallback.unavailable_marker", "unavailable"))
    fallback_source = str(get(slam_cfg, "fallback.mark_source", "odom_fallback"))

    class Bridge(Node):
        def __init__(self) -> None:
            # 消息驱动（replay）模式下用 TF 消息自带的仿真时间戳，不需要 /clock；
            # 定时（online）模式才依赖仿真时钟。
            use_sim_time = bool(get(slam_cfg, "replay.use_sim_time", True)) and args.trigger == "timer"
            super().__init__("cvae_slam_bridge", parameter_overrides=[])
            self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=use_sim_time)])
            self.create_subscription(TFMessage, "/tf", self.on_tf, 100)
            self.create_subscription(TFMessage, "/tf_static", self.on_tf_static, 100)
            self.create_subscription(PoseStamped, str(get(slam_cfg, "source.pose_topic", "/cartographer/tracked_pose")),
                                     self.on_pose, 50)
            rate = float(args.rate_hz)
            self.create_timer(1.0 / rate, self.tick)
            self.n_since_report = 0
            self.last_event: str | None = None
            # 消息驱动模式（replay）下的节流：按**仿真时间**间隔写样本，
            # 而不是按 wall-clock 定时采样——离线回放会以远高于实时的速度跑完，
            # 定时采样会系统性丢帧（这是回放模式下最容易犯的对齐错误）。
            self.min_period = float(args.min_period)
            self.last_emit_stamp = -1.0
            self.last_stamp_seen = 0.0

        # --- 订阅回调 ---------------------------------------------------------
        def _ingest(self, msg: TransformStamped, static: bool) -> None:
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            t = msg.transform.translation
            q = msg.transform.rotation
            yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
            graph.add(msg.header.frame_id, msg.child_frame_id, t.x, t.y, yaw, stamp, is_static=static)

        def on_tf(self, msg: TFMessage) -> None:
            for tr in msg.transforms:
                self._ingest(tr, static=False)
                stamp = float(tr.header.stamp.sec) + float(tr.header.stamp.nanosec) * 1e-9
                self.last_stamp_seen = max(self.last_stamp_seen, stamp)
            if args.trigger == "tf" and "base_link" in {tr.child_frame_id for tr in msg.transforms}:
                self._emit_from_tf(self.last_stamp_seen)

        def on_tf_static(self, msg: TFMessage) -> None:
            for tr in msg.transforms:
                self._ingest(tr, static=True)

        def on_pose(self, msg: PoseStamped) -> None:
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            yaw = _yaw_from_quaternion(msg.pose.orientation.x, msg.pose.orientation.y,
                                       msg.pose.orientation.z, msg.pose.orientation.w)
            sample = PoseSample(t_capture=stamp, t_mono=time.monotonic(),
                                x=msg.pose.position.x, y=msg.pose.position.y, yaw=yaw,
                                z=msg.pose.position.z, source="cartographer", valid=1)
            self._emit(sample)

        # --- 定时查询 TF ------------------------------------------------------
        def _emit_from_tf(self, stamp: float) -> None:
            """按 TF 消息驱动写位姿样本（replay 模式，不丢帧）。"""
            if stamp <= 0.0:
                return
            if stamp - self.last_emit_stamp < self.min_period:
                return
            matrix = graph.lookup(map_frame, tracking, t=stamp)
            source = "cartographer"
            if matrix is None:
                matrix = graph.lookup(odom_frame, tracking, t=stamp)
                source = fallback_source if matrix is not None else marker
            if matrix is None:
                return
            self.last_emit_stamp = stamp
            x, y = float(matrix[0, 2]), float(matrix[1, 2])
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
            # t_capture 直接使用消息的采集时钟（回放=仿真时间，起点与 episode 时间戳同一时间轴），
            # 不做原点平移：这样位姿流可被数据集构建/轨迹评估直接按时间对齐。
            sample = PoseSample(t_capture=stamp, t_mono=time.monotonic(),
                                x=x, y=y, yaw=yaw, source=source, valid=1)
            event = health.observe(sample.pose3, time.monotonic())
            if event and event != self.last_event:
                log.warning(f"SLAM 健康告警：{event}", health=health.to_dict())
                self.last_event = event
            self._emit(sample)
            self.n_since_report += 1
            if self.n_since_report >= int(args.report_every):
                self.n_since_report = 0
                log.metric_now(message="bridge stats", **health.to_dict(), **graph.stats())

        def tick(self) -> None:
            if args.trigger == "tf":
                return
            now = time.monotonic()
            # use_sim_time=true 时，clock 由扫描回放节点发布的 /clock 驱动；
            # t_capture = 采集时钟本身（与 episode 时间戳同一时间轴）。
            clock_now = self.get_clock().now().nanoseconds * 1e-9
            matrix = graph.lookup(map_frame, tracking, t=None)
            if matrix is None:
                # 退化 1：只拿到 odom→base_link
                matrix = graph.lookup(odom_frame, tracking, t=None)
                source = fallback_source if matrix is not None else marker
            else:
                source = "cartographer"
            if matrix is None:
                sample = PoseSample.unavailable(t_capture=clock_now, t_mono=now, marker=marker)
            else:
                x, y = float(matrix[0, 2]), float(matrix[1, 2])
                yaw = math.atan2(matrix[1, 0], matrix[0, 0])
                sample = PoseSample(t_capture=clock_now, t_mono=now, x=x, y=y, yaw=yaw,
                                    source=source, valid=1)
            event = health.observe(None if not sample.valid else sample.pose3, now)
            if event and event != self.last_event:
                log.warning(f"SLAM 健康告警：{event}", health=health.to_dict())
                self.last_event = event
            self._emit(sample)
            self.n_since_report += 1
            if self.n_since_report >= int(args.report_every):
                self.n_since_report = 0
                log.metric_now(message="bridge stats", **health.to_dict(), **graph.stats())

        def _emit(self, sample: PoseSample) -> None:
            if args.mode == "dry_run":
                if health.n_samples % 20 == 0:
                    print(json.dumps(sample.to_json(), ensure_ascii=False))
                return
            stream.write(sample)

        def shutdown(self) -> None:
            stream.close()

    rclpy.init()
    if args.reset_stream and str(get(slam_cfg, "pose_stream.transport", "jsonl")) == "jsonl":
        stream_path = Path(str(get(slam_cfg, "pose_stream.jsonl_path")))
        ensure_dir(stream_path.parent)
        stream_path.write_text("", encoding="utf-8")   # 清空旧位姿流，避免跨实验混样
    node = Bridge()
    log.info(f"桥接节点启动：mode={args.mode} map={map_frame} tracking={tracking} rate={args.rate_hz}Hz",
             pose_stream=str(get(slam_cfg, "pose_stream.jsonl_path")))
    started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration and time.monotonic() - started > args.duration:
                log.info(f"达到 --duration={args.duration}s，停止桥接")
                break
            if args.max_samples and health.n_samples + health.n_missing >= args.max_samples:
                break
    except KeyboardInterrupt:
        log.warning("收到中断：退出桥接节点")
    finally:
        node.shutdown()
        log.info(f"桥接结束：{json.dumps({**health.to_dict(), **graph.stats()}, ensure_ascii=False)}")
        log.flush()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # 信号处理路径可能已经 shutdown 过，重复调用会抛 RCLError；这不是错误，静默忽略即可
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cartographer → 策略位姿桥接（ROS2 侧）")
    parser.add_argument("--mode", default="replay",
                        choices=["online", "replay", "mock", "dry_run", "pbstream_query"])
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数（0=不限）")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--trigger", default="timer", choices=["timer", "tf"],
                        help="timer=定时采样（在线）；tf=消息驱动（离线回放，不丢帧）")
    parser.add_argument("--min-period", type=float, default=0.05,
                        help="消息驱动模式下的最小写样间隔（仿真秒），0.05 对应 20Hz")
    parser.add_argument("--reset-stream", action="store_true", help="启动前清空 JSONL 位姿流")
    parser.add_argument("--trajectory-id", type=int, default=0, help="pbstream_query 模式下的 trajectory id")
    parser.add_argument("--service-timeout", type=float, default=60.0,
                        help="等待 trajectory_query 服务的秒数")
    parser.add_argument("--mock-duration", type=float, default=30.0)
    parser.add_argument("--override", action="append", default=[])
    return parser


def run_pbstream_query(args: argparse.Namespace) -> int:
    """从**已建好的 pbstream** 取优化后的完整轨迹（走 cartographer 的 trajectory_query 服务）。

    为什么需要它：离线回放时 `cartographer_offline_node` 以远高于实时的速度处理数据，
    TF 按仿真时间高频发布，订阅侧（rclpy）会大量丢包——实测 1200 帧只收到 34 个 TF 样本。
    对"轨迹精度评估"这种必须完整采样的用途，正确做法是从 pbstream 查询，
    而不是靠订阅实时 TF 采样。

    前置：另起 `cartographer_offline_node -load_state_filename=<pbstream> -keep_running=true`
    作为服务端（见 scripts/44_query_pbstream_trajectory.sh）。
    """
    import rclpy
    from cartographer_ros_msgs.srv import TrajectoryQuery

    paths = load_paths()
    slam_cfg = load_config("slam", overrides=args.override)
    log = StageLogger("pbstream_query", paths["logs_dir"], stage="44")
    stream_path = Path(str(get(slam_cfg, "pose_stream.jsonl_path")))
    stream = PoseStreamWriter(
        transport="jsonl", jsonl_path=stream_path,
        flush_every=int(get(slam_cfg, "pose_stream.flush_every", 1)),
    )
    if args.reset_stream:
        ensure_dir(stream_path.parent)
        stream_path.write_text("", encoding="utf-8")

    rclpy.init()
    node = rclpy.create_node("cvae_pbstream_query")
    client = node.create_client(TrajectoryQuery, "/trajectory_query")
    log.info(f"等待 /trajectory_query 服务（最多 {args.service_timeout}s）…")
    if not client.wait_for_service(timeout_sec=float(args.service_timeout)):
        log.error("未等到 trajectory_query 服务：请用 -keep_running=true 启动 cartographer_offline_node")
        node.destroy_node()
        rclpy.shutdown()
        return 4

    request = TrajectoryQuery.Request()
    request.trajectory_id = int(args.trajectory_id)
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=float(args.service_timeout))
    response = future.result()
    if response is None:
        log.error("trajectory_query 调用超时")
        node.destroy_node()
        rclpy.shutdown()
        return 5

    # 本机构建（cartographer_ros 2.0.9003）的 TrajectoryQuery.srv 返回
    # `geometry_msgs/PoseStamped[] trajectory`（**数组**，不是 nav_msgs/Path）。
    # 为兼容两种定义，这里统一取出 pose 列表。
    trajectory = response.trajectory
    poses = list(trajectory.poses) if hasattr(trajectory, "poses") else list(trajectory)
    n_total = len(poses)
    n_written = 0
    for pose_stamped in poses:
        q = pose_stamped.pose.orientation
        if not (q.x or q.y or q.z or q.w):
            continue    # 四元数全零表示空节点：跳过，绝不补 0 位姿
        stamp = float(pose_stamped.header.stamp.sec) + float(pose_stamped.header.stamp.nanosec) * 1e-9
        stream.write(PoseSample(
            t_capture=stamp, t_mono=time.monotonic(),
            x=pose_stamped.pose.position.x, y=pose_stamped.pose.position.y,
            yaw=_yaw_from_quaternion(q.x, q.y, q.z, q.w), z=pose_stamped.pose.position.z,
            source="cartographer_pbstream", valid=1,
            extra={"frame_id": pose_stamped.header.frame_id, "trajectory_id": int(args.trajectory_id)},
        ))
        n_written += 1
    stream.close()
    log.info(f"trajectory_id={request.trajectory_id}：服务返回 {n_total} 个节点，写入 {n_written} 个有效位姿"
             f" → {stream_path}；status={getattr(response.status, 'message', '')}")
    log.flush()
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps({"status": "OK", "n_poses": n_written, "n_nodes": n_total,
                      "trajectory_id": int(args.trajectory_id), "pose_stream": str(stream_path)},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "mock":
        paths = load_paths()
        slam_cfg = load_config("slam", overrides=args.override)
        info = mock_pose_stream(paths, slam_cfg, args.mock_duration)
        print(json.dumps({"status": "OK", "mode": "mock", **info, "evidence": "MOCK（非真实 SLAM）"},
                         ensure_ascii=False, indent=2))
        return 0
    if args.mode == "pbstream_query":
        try:
            return run_pbstream_query(args)
        except ImportError as exc:
            print(json.dumps({"status": "ROS2_UNAVAILABLE", "message": str(exc)}, ensure_ascii=False))
            return 3
    try:
        return run_bridge(args)
    except ImportError as exc:
        print(json.dumps({"status": "ROS2_UNAVAILABLE", "message": str(exc),
                          "hint": "请用 conda 环境 cartographer_ros 的 python 运行本文件"}, ensure_ascii=False))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
