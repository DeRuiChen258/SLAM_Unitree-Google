"""推理侧：滚动推理、在线执行循环、离线评测、回放对比。

依赖方向：可以 import src/models、src/datasets、src/slam、src/utils；不得反向被依赖。
"""

from .rollout_policy import RolloutResult, RolloutRunner, StepRecord

__all__ = ["RolloutRunner", "RolloutResult", "StepRecord"]
