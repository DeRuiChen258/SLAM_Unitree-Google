"""CVAE + Action Chunking + Temporal Ensemble × Cartographer SLAM 实验工程。

包边界（依赖方向，违反即返工）：
    utils  ← datasets ← models ← train
      ↑                    ↑
      └── slam ── pose_stream ── infer

本文件只提供版本号，不放业务逻辑。
"""

__version__ = "1.0.0"
