"""ROS2 侧激光回放节点：把 data/slam/scans.npz 按原始时间轴发布为 /scan + /clock + 静态 TF。

运行解释器：conda 环境 `cartographer_ros`（RoboStack Jazzy，自带 rclpy 与 sensor_msgs）。
为什么需要它：本实验没有真实激光雷达，而 Cartographer 必须吃真实的消息流；
因此用与世界模型严格同步的扫描流驱动**真正的 cartographer_node**，
而不是绕过 Cartographer 直接伪造位姿（提示词【十】降级链 c 级）。

发布内容：
    /clock            sensor 时间轴（use_sim_time=true 时驱动 Cartographer 的时钟）
    /scan             sensor_msgs/LaserScan，frame_id=laser
    /tf_static        base_link → laser（恒等外参，来自 configs/slam.yaml:frames）

参数全部来自 configs/slam.yaml:replay，代码内不硬编码话题名与频率。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..utils.config import get, load_config, load_paths
from ..utils.logging_utils import StageLogger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把合成的激光扫描流发布为 ROS2 消息（驱动真 Cartographer）")
    parser.add_argument("--max-frames", type=int, default=0, help="0=全部")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速（>1 更快）")
    parser.add_argument("--start-delay", type=float, default=None, help="发布前等待秒数（等 Cartographer 起好）")
    parser.add_argument("--record-bag", default=None,
                        help="同时把消息写入 ROS2 bag（sqlite3）。用 rosbag2_py 直接写，"
                             "避免依赖 ros2 CLI 的 AMENT_PREFIX_PATH 等激活变量")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage

    paths = load_paths()
    slam_cfg = load_config("slam", overrides=args.override)
    log = StageLogger("ros2_scan_player", paths["logs_dir"], stage="41")

    scan_file = Path(str(get(slam_cfg, "replay.scan_file")))
    if not scan_file.exists():
        log.error(f"扫描文件不存在: {scan_file}（先运行 scripts/10_gen_mock_data.py）")
        return 3
    with np.load(scan_file, allow_pickle=False) as data:
        angles = np.asarray(data["angles"], dtype=np.float64)
        ranges = np.asarray(data["ranges"], dtype=np.float32)
        stamps = np.asarray(data["timestamps"], dtype=np.float64)
        gt_pose = np.asarray(data["ground_truth_pose"], dtype=np.float32)
        range_min = float(data["range_min"])
        range_max = float(data["range_max"])

    n_frames = int(args.max_frames) if args.max_frames else ranges.shape[0]
    n_frames = min(n_frames, ranges.shape[0])
    sensor_frame = str(get(slam_cfg, "frames.sensor_frame", "laser"))
    tracking_frame = str(get(slam_cfg, "frames.tracking_frame", "base_link"))
    scan_topic = str(get(slam_cfg, "replay.scan_topic", "/scan"))
    clock_topic = str(get(slam_cfg, "replay.clock_topic", "/clock"))
    rate = float(get(slam_cfg, "replay.publish_rate_hz", 10.0))
    speed = max(1e-3, float(args.speed))
    start_delay = float(args.start_delay if args.start_delay is not None
                        else get(slam_cfg, "replay.start_delay_s", 2.0))

    rclpy.init()
    node = rclpy.create_node("cvae_scan_player")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=False)])
    scan_pub = node.create_publisher(LaserScan, scan_topic, 50)
    clock_pub = node.create_publisher(Clock, clock_topic, 50)
    tf_static_pub = node.create_publisher(TFMessage, "/tf_static", 10)

    # 静态外参 base_link → laser（恒等；外参来自配置，不是"顺手写死"）
    static = TFMessage()
    tr = TransformStamped()
    tr.header.frame_id = tracking_frame
    tr.child_frame_id = sensor_frame
    tr.transform.rotation.w = 1.0
    static.transforms.append(tr)
    # 静态 TF 用 transient_local 语义：ROS2 里由 /tf_static 的 QoS 保证，这里周期性重发以兼容订阅者
    for _ in range(5):
        tf_static_pub.publish(static)
        rclpy.spin_once(node, timeout_sec=0.05)

    log.info(f"扫描回放启动：{n_frames} 帧 / {rate}Hz × {speed}x，topic={scan_topic} frame={sensor_frame}")

    bag_writer = None
    bag_serialize = None
    if args.record_bag:
        from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
        from rclpy.serialization import serialize_message

        bag_serialize = serialize_message
        bag_path = Path(str(args.record_bag))
        if bag_path.exists():
            import shutil

            shutil.rmtree(bag_path)
        bag_writer = SequentialWriter()
        bag_writer.open(StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
                        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
        # 注意构造签名：TopicMetadata(id, name, type, serialization_format, ...) 必须按位置传参，
        # 关键字传参会抛 TypeError 并导致 bag 里没有 topic（本机实测踩过，表现为
        # offline_node 报 "No topics were listed in metadata"）。
        bag_writer.create_topic(TopicMetadata(0, scan_topic, "sensor_msgs/msg/LaserScan", "cdr"))
        bag_writer.create_topic(TopicMetadata(1, "/tf_static", "tf2_msgs/msg/TFMessage", "cdr"))
        bag_writer.create_topic(TopicMetadata(2, clock_topic, "rosgraph_msgs/msg/Clock", "cdr"))
        log.info(f"同时写入 ROS2 bag：{bag_path}")
    if start_delay > 0:
        log.info(f"等待 {start_delay:.1f}s 让 Cartographer 完成初始化")
        deadline = time.monotonic() + start_delay
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

    t_wall0 = time.monotonic()
    stamp0 = float(stamps[0])
    published = 0
    try:
        for i in range(n_frames):
            if not rclpy.ok():
                break
            sim_t = float(stamps[i])
            target_wall = t_wall0 + (sim_t - stamp0) / speed
            now = time.monotonic()
            if target_wall > now:
                time.sleep(min(0.5, target_wall - now))

            sec = int(sim_t)
            nanosec = int(round((sim_t - sec) * 1e9))
            clock_msg = Clock()
            clock_msg.clock.sec = sec
            clock_msg.clock.nanosec = nanosec
            clock_pub.publish(clock_msg)
            if bag_writer is not None:
                stamp_ns = sec * 10**9 + nanosec
                bag_writer.write(clock_topic, bag_serialize(clock_msg), stamp_ns)
                bag_writer.write("/tf_static", bag_serialize(static), stamp_ns)

            scan = LaserScan()
            scan.header.stamp = clock_msg.clock
            scan.header.frame_id = sensor_frame
            scan.angle_min = float(angles[0])
            scan.angle_max = float(angles[-1])
            scan.angle_increment = float(angles[1] - angles[0])
            scan.time_increment = 0.0
            scan.scan_time = 1.0 / rate
            scan.range_min = range_min
            scan.range_max = range_max
            r = np.asarray(ranges[i], dtype=np.float32)
            # inf 在 LaserScan 中的语义就是"无回波"，直接保留（不要替换成大数）
            scan.ranges = [float(v) if np.isfinite(v) else float("inf") for v in r]
            scan_pub.publish(scan)
            if bag_writer is not None:
                bag_writer.write(scan_topic, bag_serialize(scan), sec * 10**9 + nanosec)
            published += 1
            if published % 100 == 0:
                log.metric_now(message="scan published", frames=published, sim_time=sim_t,
                               gt_x=float(gt_pose[i, 0]), gt_y=float(gt_pose[i, 1]))
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        log.warning("回放被中断")
    finally:
        if bag_writer is not None:
            bag_writer.close()
        log.info(f"回放结束：发布 {published} 帧（模拟时间 {float(stamps[min(published, n_frames-1)]):.2f}s）")
        log.flush()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
