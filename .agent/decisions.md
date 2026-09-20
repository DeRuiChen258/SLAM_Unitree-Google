# decisions

1. **按 unitree workflow 执行**：任务对象是 Unitree 工作区内的视觉操控 + 激光 SLAM 实验，
   走 `unitree` 主 workflow；跨域技能（rl-infra 等）只在训练/推理阶段使用并在本文件登记。
2. **训练/推理与 ROS2 双解释器隔离**：`unitree_rt`（torch）与 `cartographer_ros`（rclpy）互不安装对方依赖，
   跨解释器只通过 `pose_stream.jsonl` 传递位姿——避免"在系统 Python 装 torch / 在 conda 装 rclpy"这类污染。
3. **Cartographer 走 conda 二进制分发 + 离线回放**（而非源码构建）：
   理由是 a 级源码构建被 `sudo` 交互密码阻断，而 b 级容器会引入额外复杂度；
   c 级已能得到真实的 Cartographer 地图与轨迹（PASS），符合【十】降级链且成本最低。
4. **配置文件自包含生成**：官方 Lua 示例在本机构建上无法 `include`，因此把官方默认值内联生成单一配置，
   并把每个来源文件的 sha256 写进生成文件头——保证"参数名有据可查"且"可复现生成"。
5. **processed 数据存 `.npy` 而不是 npz**：只有 `.npy` 支持真正的 mmap 懒加载；
   npz 每次访问成员都要整体解压，实测会把训练拖成 CPU 100% / GPU 空转。
6. **CUDA 显式启用 + 主算力放 GPU**：`device: cuda`（不可用即报错，禁止静默降级），
   图像归一化迁到设备侧（uint8 上显存，H2D 数据量降 4 倍），并启用 AMP。
7. **CPU/GPU 分工可度量**：新增 `device_policy` 与每步耗时拆分（data wait / H2D / compute），
   输出判定与调参建议，避免"某一侧空转"或"过度依赖单一设备"。
8. **轨迹评估用 `trajectory_query` 服务而非订阅 TF**：高倍速离线回放时订阅侧丢包严重（34/1200），
   服务查询能拿到 pbstream 里优化后的完整节点位姿（1037 个），评估口径才站得住。
9. **消融只改一个变量**：数据侧消融（H、状态维度）重建 processed，推理侧消融（集成开关/权重）
   复用 base checkpoint，保证同一 splits、同一 seed、同一模型结构。
10. **不做 git 大文件提交**：`data/mock`、`data/processed`、`checkpoints/*.pt`、`outputs/slam/*.pbstream`
    被 `.gitignore` 忽略，来源与哈希登记在 manifest 与文档中；仓库只保留可复现的代码、配置、文档与图。
11. **环境/工具与实验代码分仓目录（用户指令）**：
    环境与工具根（conda 锁定、环境激活脚本、Cartographer 源码构建脚本与产物）留在
    `<环境与工具根>`；实验工程迁到
    `<项目根>`。
    两者关系用 `configs/paths.yaml:env_root` 表达（外部依赖，可环境变量覆盖），
    项目内不再保留环境文件副本，避免"两份事实源"；
    `scripts/00_env_check.sh` 作为 S0 阶段同时校验两个根的存在性与内容。
12. **可视化分两层**：可复现的静态图为 `outputs/figures/*.png`（由脚本生成）；
    面向"在对话窗口里看结果"的内联看板为 `outputs/figures/results-dashboard.html`
    （数据由 `scripts/91_export_viz_data.py` 从真实产物导出后内联，禁止手工填数）。
13. **公开快照的发布方式（用户指令：去个人信息 + 推到 GitHub）**：
    仓库内不得出现本机绝对路径与个人邮箱，因此发布提交**不含**原始开发历史（原始 3 个提交含
    `/home/<user>/…` 路径与个人邮箱）——做法是：
    * 本地保留备份分支 `backup/local-history-pre-publish`；
    * 以远程 `Initial commit` 为父提交创建**单个干净提交**（含 sanitize 后的完整快照 + 保留 MIT LICENSE），
      `git push` 为 fast-forward，**不使用强推**；
    * 提交作者用 GitHub noreply 身份（`<id>+<login>@users.noreply.github.com`），不写个人邮箱；
    * 本地 `main` 指到该干净提交，保证"本地 == 远程"，避免后续 push 出现分叉。
14. **可移植化路径的两条规则**：`configs/paths.yaml` 的 `project_root` 用哨兵值 `auto`
    （由 `src/utils/config.py` 的代码位置推导，仍禁止被环境变量/CLI 覆盖）；
    外部依赖路径用 `${HOME}/…`，占位符解析顺序为"配置内键 → 同名环境变量"。
    这样克隆到任意目录、任意用户名下都能直接跑，同时保持"外部路径可覆盖"的原设计。
15. **产物不得内嵌个人路径**：图页脚、JSON 指标、生成的 Lua 注释、HTML 看板一律用
    相对项目根的路径或占位符（`_portable()` / 相对 `cartographer_prefix`），
    避免"删了仓库里的路径、却把路径画进 PNG"这种漏网。
