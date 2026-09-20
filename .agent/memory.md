# memory

- **用户偏好**：最终回复用简体中文；要求"可运行、可复现、可验证"，明确禁止伪代码与补数；
  中途会追加算力相关指令（"启用 cuda 版本" → "协调 CPU/GPU 比例" → "主要使用 CUDA"），
  这类指令要落到**配置 + 实测证据**上，而不是只改一行开关。
- **工程约定**：实验目录内保留 `Prompt/`（提示词 + TASK.md + .agent/）；一个实验一个目录。
- **踩坑速查**：
  1. npz + mmap_mode 无效 → 大数据一律 `.npy`；
  2. LambdaLR 构造即 step(一次 warmup) → 门禁类脚本要显式恢复基准 LR；
  3. VAE 输出层不要接 LayerNorm，零初始化才对（初始 KL=0）；
  4. β 要按重建损失量级定（本例 0.05 + free-bits 0.02），否则后验必塌缩；
  5. RoboStack 的 `ros2` CLI 必须显式导出 AMENT_PREFIX_PATH 等，且不能残留别的 CONDA_PREFIX；
  6. `rosbag2_py.TopicMetadata` 必须位置传参；
  7. Cartographer 该构建不能 `include`，必须自包含配置；
  8. 离线高倍速回放不要靠订阅 TF 采样轨迹，要用 `trajectory_query`；
  9. Umeyama 对齐方向与航向偏移必须一起处理，否则 ATE/航向误差都会离谱。
- **环境事实**：本机 `sudo` 需交互密码（无 NOPASSWD）→ 需系统包的任务要预判 BLOCKED；
  conda base 在 `$HOME/Workspace/miniconda`；ROS2 rootless 在 unitree_workspace/ros2/ros2-linux。
- **开源前去个人信息化清单**（本次实操，按此顺序扫）：
  1. 文本：`git ls-files | xargs grep -n "/home/<user>\|邮箱\|账号"`（含 `.agent/`、`Prompt/` 与文档）；
  2. 配置：`project_root` 用 `auto` 哨兵 + 外部路径用 `${HOME}`/环境变量，避免写死本机路径；
  3. **图片**：图页脚是重灾区（PNG 里的绝对路径 grep 不到）→ 要改生成代码后**重新出图**；
  4. 生成物：JSON 指标、生成的 Lua 注释、HTML 看板都要改成相对路径，否则下次 `make` 又写回去；
  5. 历史：旧提交里的路径/邮箱删不掉 → 用一个干净提交覆盖发布分支（父提交指向远程已有提交，
     fast-forward 推送，不强推），旧历史留在本地备份分支；
  6. 身份：`git -c user.name=… -c user.email=<id>+<login>@users.noreply.github.com commit`，
     不把个人邮箱写进公开历史。
