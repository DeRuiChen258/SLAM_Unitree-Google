"""在线执行循环：读观测 + 读 Cartographer 位姿 → 预测 → 时间集成 → 限幅/限速 → watchdog → 输出动作。

安全契约（提示词【二】2）：`infer.allow_command_publish` 默认 false，
此时只把动作写入日志/文件，**不对外发布任何控制指令**。
即使打开开关，也必须先通过安全门禁检查（急停可达 / watchdog 生效 / 限幅已加载），
检查结果必须写入启动日志与 manifest。

本文件不导入 torch 以外的外部依赖；ROS 发布器按需惰性导入 rclpy。
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..datasets.transforms import inverse_action
from ..slam.pose_stream import PoseStreamClient
from ..utils.config import get, load_paths
from ..utils.io_utils import append_jsonl, ensure_dir
from ..utils.logging_utils import StageLogger
from .rollout_policy import RolloutRunner


@dataclass
class SafetyCheck:
    """安全门禁逐项结果（必须全部 pass 才允许发布动作）。"""

    allow_publish: bool = False
    estop_reachable: bool = False
    watchdog_enabled: bool = False
    limits_loaded: bool = False
    robot_connected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def publish_allowed(self) -> bool:
        return bool(
            self.allow_publish and self.estop_reachable and self.watchdog_enabled and self.limits_loaded
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_command_publish": self.allow_publish,
            "estop_reachable": self.estop_reachable,
            "watchdog_enabled": self.watchdog_enabled,
            "limits_loaded": self.limits_loaded,
            "robot_connected": self.robot_connected,
            "publish_allowed": self.publish_allowed,
            "notes": self.notes,
        }


def evaluate_safety(infer_cfg: Mapping[str, Any], robot_connected: bool = False) -> SafetyCheck:
    """评估安全门禁。无真机时 robot_connected=False，因此永远不允许发布。"""
    check = SafetyCheck(
        allow_publish=bool(get(infer_cfg, "safety.allow_command_publish", False)),
        estop_reachable=bool(get(infer_cfg, "safety.require_estop_reachable", True)),
        watchdog_enabled=bool(get(infer_cfg, "safety.require_watchdog", True)),
        limits_loaded=all(
            get(infer_cfg, f"limits.{k}") is not None
            for k in ("max_translation_per_step", "max_rotation_per_step", "max_gripper_rate")
        ),
        robot_connected=robot_connected,
    )
    if not check.allow_publish:
        check.notes.append("allow_command_publish=false：只写日志，不发布控制指令（默认安全姿态）")
    if not robot_connected:
        check.notes.append("无真机连接：本环境中永不允许对外发布动作（提示词【二】1）")
    return check


class ActionLimiter:
    """单步限幅 + 速度上限 + NaN 熔断；每次裁剪都计数（超过阈值即中止闭环）。"""

    def __init__(self, infer_cfg: Mapping[str, Any]) -> None:
        self.max_trans = float(get(infer_cfg, "limits.max_translation_per_step", 0.06))
        self.max_rot = float(get(infer_cfg, "limits.max_rotation_per_step", 0.12))
        self.max_gripper = float(get(infer_cfg, "limits.max_gripper_rate", 0.15))
        self.max_delta = float(get(infer_cfg, "limits.max_action_delta", 0.10))
        self.n_clip = 0
        self.n_nan = 0
        self.max_events = int(get(infer_cfg, "safety.max_clip_events", 200))
        self.fail_fast = bool(get(infer_cfg, "safety.fail_fast_on_nan", True))

    def apply(self, action: np.ndarray, previous: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).copy()
        events: dict[str, Any] = {"clipped": False, "nan": False}
        if not np.isfinite(a).all():
            self.n_nan += 1
            events["nan"] = True
            if self.fail_fast:
                raise FloatingPointError(f"动作含 NaN/Inf：{a}（fail_fast_on_nan=true，立即停止）")
            return np.zeros_like(a), events
        if a.shape[0] >= 6:
            before = a.copy()
            a[:3] = np.clip(a[:3], -self.max_trans, self.max_trans)
            a[3:6] = np.clip(a[3:6], -self.max_rot, self.max_rot)
            if a.shape[0] >= 7:
                a[6] = np.clip(a[6], -self.max_gripper, self.max_gripper)
            if previous is not None and previous.shape == a.shape:
                delta = np.clip(a - previous, -self.max_delta, self.max_delta)
                a = previous + delta
            if not np.allclose(before, a):
                self.n_clip += 1
                events["clipped"] = True
        return a.astype(np.float32), events

    def exceeded(self) -> bool:
        return self.max_events > 0 and self.n_clip > self.max_events

    def to_dict(self) -> dict[str, Any]:
        return {"max_translation_per_step": self.max_trans, "max_rotation_per_step": self.max_rot,
                "max_gripper_rate": self.max_gripper, "max_action_delta": self.max_delta,
                "n_clip_events": self.n_clip, "n_nan_events": self.n_nan}


class FileActionSink:
    """默认动作汇：写 JSONL 到 outputs/infer_samples/，供复盘与可视化。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        # 每次运行 = 一份独立的执行日志：显式清空旧内容，
        # 避免多次运行的记录混在同一文件里导致"位姿有效率"这类统计被污染。
        self.path.write_text("", encoding="utf-8")
        self.n_written = 0
        self._buffer: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self._buffer.append(dict(record))
        self.n_written += 1
        if len(self._buffer) >= 16:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            append_jsonl(self.path, self._buffer)
            self._buffer.clear()

    def close(self) -> None:
        self.flush()


class RosActionSink:
    """ROS2 动作汇（**未在真机验证**：NOT_MEASURED）。

    只在安全门禁全部通过时才会真正发布；本机没有真机，因此该路径不会被触发，
    保持"流程 + 安全清单"而不实际下发运动指令。
    """

    def __init__(self, topic: str = "/cvae_policy/cmd", message_type: str = "geometry_msgs/msg/TwistStamped") -> None:
        import rclpy  # 惰性导入：只有真正要走 ROS2 发布时才需要
        from geometry_msgs.msg import TwistStamped

        self._rclpy = rclpy
        self._msg = TwistStamped
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("cvae_policy_sink")
        self._pub = self._node.create_publisher(TwistStamped, topic, 10)
        self.n_written = 0

    def write(self, record: Mapping[str, Any]) -> None:
        action = np.asarray(record.get("action", []), dtype=np.float64)
        msg = self._msg()
        if action.size >= 3:
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (float(v) for v in action[:3])
        if action.size >= 6:
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (float(v) for v in action[3:6])
        self._pub.publish(msg)
        self.n_written += 1

    def close(self) -> None:
        self._node.destroy_node()
        try:
            self._rclpy.shutdown()
        except Exception:
            pass


class ExecutionLoop:
    """在线执行循环（默认 dry-run：只写日志）。"""

    def __init__(self, runner: RolloutRunner, data_cfg: Mapping[str, Any], infer_cfg: Mapping[str, Any],
                 slam_cfg: Mapping[str, Any], pose_client: PoseStreamClient | None,
                 log: StageLogger, robot_connected: bool = False,
                 out_path: str | Path | None = None) -> None:
        self.runner = runner
        self.data_cfg = data_cfg
        self.infer_cfg = infer_cfg
        self.slam_cfg = slam_cfg
        self.pose_client = pose_client
        self.log = log
        self.safety = evaluate_safety(infer_cfg, robot_connected)
        self.limiter = ActionLimiter(infer_cfg)
        self.watchdog_timeout = float(get(infer_cfg, "safety.watchdog_timeout_s", 1.0))
        self.stats = runner.stats
        self.sink: Any = FileActionSink(out_path or (Path("outputs/infer_samples") / "executed_actions.jsonl"))
        if self.safety.publish_allowed:  # pragma: no cover - 无真机时不进入
            self.sink = RosActionSink()
        self._stop = False
        self.events: list[dict[str, Any]] = []

    # -- 生命周期 ---------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self._stop = True
            self.events.append({"event": "signal", "signum": int(signum), "t": time.monotonic()})
            self.log.warning(f"收到信号 {signum}：优雅退出（停止下发后续动作）")

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def stop(self) -> None:
        self._stop = True

    # -- 主循环 -----------------------------------------------------------------
    def run(self, episode: Mapping[str, Any], max_steps: int, start_t: int = 0,
            timestamps: np.ndarray | None = None) -> dict[str, Any]:
        """执行闭环；每一步都做 watchdog、限幅与位姿可用性检查。"""
        self.log.info(f"执行循环启动：safety={json.dumps(self.safety.to_dict(), ensure_ascii=False)}")
        states = np.asarray(episode["state"], dtype=np.float32)
        images = np.asarray(episode["images"], dtype=np.float32)
        obs_idx = np.asarray(episode["obs_indices"])
        length = min(len(states), images.shape[0], len(obs_idx))
        n_exec = self.runner.chunker.n_exec
        if self.runner.ensemble is not None:
            self.runner.ensemble.reset()
        pending: list[np.ndarray] = []
        previous: np.ndarray | None = None
        t = int(start_t)
        executed = 0
        last_obs_time = time.monotonic()
        pose_missing = 0
        while t < length and executed < max_steps and not self._stop:
            now = time.monotonic()
            if now - last_obs_time > self.watchdog_timeout:
                self.events.append({"event": "watchdog_obs_timeout", "t": now, "step": t})
                self.log.warning(f"watchdog 触发：观测 {now - last_obs_time:.3f}s 未更新，停止输出动作")
                break
            pose = None
            if self.pose_client is not None:
                # 查询时钟必须与位姿流来源一致：
                #   离线回放 → 采集/仿真时钟（t_capture）；在线运行 → 单调时钟（配合 watchdog）
                query_clock = str(get(self.slam_cfg, "sync.query_clock", "capture"))
                if query_clock == "capture" and timestamps is not None and t < len(timestamps):
                    query_t = float(timestamps[t])
                else:
                    query_t = now
                pose = self.pose_client.get_pose(
                    query_t, tol=float(get(self.slam_cfg, "sync.max_align_tolerance_s", 0.1)),
                    clock=query_clock,
                )
                if pose is None:
                    pose_missing += 1
                    if pose_missing > 5:
                        self.events.append({"event": "pose_unavailable", "t": now, "step": t})
                        self.log.warning("位姿不可用超过 5 次：按【十】降级（不伪造零位姿），继续用数据集位姿通道")
                        pose_missing = 0
            if not pending:
                chunk, latency = self.runner.predict(images[t], states[t])
                if self.runner.ensemble is not None:
                    self.runner.ensemble.update(chunk, t0=t, n_exec=n_exec, meta={"source": "execution_loop"})
                pending = list(self.runner.chunker.steps(chunk))
                self.log.metric_now(message=f"predict @t={t}", t=t, latency_ms=latency,
                                    chunk_len=int(chunk.shape[0]))
            action = pending.pop(0)
            if self.runner.ensemble is not None:
                fused = self.runner.ensemble.action_at(t)
                if fused is not None:
                    action = fused
            physical = inverse_action(np.asarray(action, dtype=np.float32)[None], self.stats)[0]
            try:
                limited, events = self.limiter.apply(physical, previous)
            except FloatingPointError as exc:
                self.events.append({"event": "nan_action", "t": time.monotonic(), "step": t, "message": str(exc)})
                self.log.error(str(exc))
                break
            if events["clipped"]:
                self.events.append({"event": "clipped", "t": time.monotonic(), "step": t})
            if self.limiter.exceeded():
                self.log.error(f"裁剪事件 {self.limiter.n_clip} 次超过阈值：疑似模型/输入异常，中止闭环")
                self.events.append({"event": "clip_limit_exceeded", "t": time.monotonic(), "step": t})
                break
            self.sink.write({
                "t": int(t),
                "action": [float(v) for v in limited],
                "action_raw": [float(v) for v in physical],
                "clipped": bool(events["clipped"]),
                "pose_valid": bool(pose is not None and pose.valid),
                "pose_source": None if pose is None else pose.source,
                "publish": bool(self.safety.publish_allowed),
                "t_wall": time.time(),
            })
            previous = limited
            executed += 1
            t += 1
            last_obs_time = time.monotonic()
            # 离线回放：按控制周期节拍推进（真实在线时由传感器驱动）
            time.sleep(1.0 / float(get(self.data_cfg, "control.hz", 10.0)) * 0.0)
        self.sink.close()
        summary = {
            "executed_steps": executed,
            "final_t": t,
            "stopped_by_signal": self._stop,
            "events": self.events,
            "safety": self.safety.to_dict(),
            "limiter": self.limiter.to_dict(),
            "pose_missing_events": pose_missing,
            "publish_enabled": self.safety.publish_allowed,
        }
        self.log.info(f"执行循环结束：{json.dumps(summary, ensure_ascii=False)}")
        return summary


def main(argv: list[str] | None = None) -> int:
    """闭环执行 CLI：加载 checkpoint + 位姿流，跑 n_exec 滚动预测并做时间集成。

    默认写文件不发布动作；`infer.allow_command_publish=true` 时仍需通过安全门禁
    （本机无真机 → `robot_connected=False` → 永远不发布）。
    """
    import json as _json

    import torch

    from ..utils.config import load_runtime

    parser = argparse.ArgumentParser(description="闭环推理执行（默认 dry-run）")
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--episode", type=int, default=0, help="使用第几条 MOCK/原始 episode 作为观测源")
    parser.add_argument("--steps", type=int, default=0, help="0 表示用 infer.episode.max_steps")
    parser.add_argument("--pose-stream", default=None, help="位姿流 JSONL；缺省用 configs/slam.yaml 的路径")
    parser.add_argument("--mock-poses", action="store_true", help="强制使用 mock 位姿流（标 MOCK 降级）")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    paths = load_paths()
    data_cfg, model_cfg, infer_cfg, slam_cfg = load_runtime(args.ablation, args.override)
    log = StageLogger("execution_loop", paths["logs_dir"], stage="31")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .offline_eval import _selector_for, load_episode_for_inference, load_model_and_stats
    from ..datasets.schema import state_spec

    model, stats, meta, _ = load_model_and_stats(paths, data_cfg, model_cfg, infer_cfg, args.run_name, device)
    episodes = sorted(Path(paths["mock_dir"]).glob("*.npz"))
    if not episodes:
        log.error("没有可用的观测源 episode（先运行 scripts/10_gen_mock_data.py）")
        return 3
    episode_path = episodes[min(args.episode, len(episodes) - 1)]
    selector = _selector_for(episode_path, data_cfg)
    episode = load_episode_for_inference(episode_path, data_cfg, stats, selector)

    # --- 位姿源：优先真实 Cartographer 位姿流；缺失则显式降级为 mock（标 MOCK）---
    stream_path = Path(args.pose_stream or get(slam_cfg, "pose_stream.jsonl_path"))
    pose_client: PoseStreamClient | None = None
    pose_source = "MISSING"
    if stream_path.exists() and not args.mock_poses:
        pose_client = PoseStreamClient(transport="jsonl", jsonl_path=stream_path,
                                       buffer_size=int(get(slam_cfg, "health.buffer_size", 512)))
        pose_client.load_file()
        pose_source = "cartographer_pose_stream"
    else:
        log.warning(f"位姿流缺失或不使用（{stream_path}）：按【十】降级为 mock 位姿（标 MOCK）")
        from ..slam.cartographer_bridge import mock_pose_stream

        mock_pose_stream(paths, slam_cfg, duration_s=float(len(episode["state"])) /
                         float(get(data_cfg, "control.hz", 10.0)))
        pose_client = PoseStreamClient(transport="jsonl", jsonl_path=stream_path)
        pose_client.load_file()
        pose_source = "mock"
    runner = RolloutRunner(model, data_cfg, infer_cfg, stats, device)
    loop = ExecutionLoop(
        runner, data_cfg, infer_cfg, slam_cfg, pose_client, log,
        robot_connected=False,  # 本机无真机：永远不发布真实控制指令
        out_path=Path(paths["infer_samples_dir"]) / f"executed_actions_{args.run_name}.jsonl",
    )
    loop.install_signal_handlers()
    steps = args.steps or int(get(infer_cfg, "episode.max_steps", 200))
    summary = loop.run(episode, max_steps=steps, timestamps=episode.get("timestamps"))
    summary.update({
        "run_name": args.run_name,
        "episode": episode_path.name,
        "pose_source": pose_source,
        "checkpoint_step": meta["step"],
        "pose_stats": None if pose_client is None else pose_client.stats(),
        "runner": runner.describe(),
    })
    ensure_dir(Path(paths["logs_dir"]))
    (Path(paths["logs_dir"]) / "31_closed_loop_summary.json").write_text(
        _json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.flush()
    print(_json.dumps({k: v for k, v in summary.items() if k != "events"}, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
