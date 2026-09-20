"""算法核心：多帧视觉编码、CVAE 策略、动作分块与时间集成。

依赖约束：本包不得 import src/datasets、src/train、src/infer。
所有张量契约见 docs/algorithm_notes.md 与 configs/data.yaml。
"""

from .action_chunker import ActionChunker, ActionSpec, ChunkScheduler
from .cvae_policy import CVAEPolicy
from .temporal_ensemble import TemporalEnsembler

__all__ = ["CVAEPolicy", "ActionChunker", "ActionSpec", "ChunkScheduler", "TemporalEnsembler"]
