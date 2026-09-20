"""Cartographer 集成：位姿数学、位姿流、ROS2 桥接、轨迹指标、回放驱动。

坐标系约定（全工程统一，禁止各自假设）：
    * 右手系；`map` 为 Cartographer 的全局建图坐标系，`odom` 为连续里程计坐标系；
    * `base_link` 为机器人基座（tracking_frame）；`laser` 为激光雷达（2D 扫描平面内）；
    * 角度单位一律为弧度（rad），长度单位为米（m），时间单位为秒（s）；
    * 位姿以 (x, y, yaw) 表示 2D 情形（z 轴向上），3D 扩展时使用 7 维 (x,y,z,qw,qx,qy,qz)。
"""

from .pose_stream import PoseSample, PoseStreamClient, PoseStreamWriter
from .pose_utils import angle_wrap, relative_pose_2d, sincos_yaw

__all__ = [
    "PoseSample",
    "PoseStreamClient",
    "PoseStreamWriter",
    "angle_wrap",
    "sincos_yaw",
    "relative_pose_2d",
]
