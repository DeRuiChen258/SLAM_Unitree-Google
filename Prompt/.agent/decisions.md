# decisions

1. 路由到 `unitree` workflow：任务对象是 Unitree 工作区内的视觉操控 + 激光 SLAM 实验。
2. 交付物仅提示词正文（用户明确「最后提交提示词即可」），不生成工程代码。
3. 工程根目录定为 `a_Visual_experiment/SLAM/cvae_slam/`：与同级 `a_Visual_experiment/ACT/` 保持一致的"一个实验一个目录"约定；Prompt/ 保留提示词与任务台账。
4. ROS2 侧与训练侧 Python 隔离：ROS2/Cartographer 桥接用系统 Python，训练/推理用 conda `unitree_rt`，跨解释器用 pose stream（UDP/JSONL）传递位姿，避免在系统 Python 里装 torch。
5. Cartographer 不假设可用：写入「探测 → 构建/容器 → bag 离线回放 → mock 位姿」降级链，每级显式标注证据等级。
6. 真实数据缺失时以 `mock_generator.py` 合成多模态演示数据，并强制打 MOCK 标记与可学习性自检（单 batch 过拟合门禁）。
