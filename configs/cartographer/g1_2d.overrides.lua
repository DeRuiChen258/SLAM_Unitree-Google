-- configs/cartographer/g1_2d.overrides.lua
-- 作用：本实验对 Cartographer 官方默认配置的**唯一人工可编辑覆盖层**。
--       由 scripts/43_gen_cartographer_config.py 与官方默认值拼接成自包含的 g1_2d.lua。
--
-- 为什么要生成自包含文件：本机 RoboStack `ros-jazzy-cartographer-ros 2.0.9003` 的
--   Lua `include` 解析存在缺陷（实测：连官方自带 backpack_2d.lua 也会因 include 而
--   抛 `basic_filebuf::underflow ... Is a directory`），因此不能依赖 include，
--   必须把官方默认值内联后用同一份 override 覆盖。证据见 logs/40_slam_bringup.log。
--
-- 参数名核对来源（禁止凭记忆拼参数名）：
--   $CONDA_PREFIX/share/cartographer/configuration_files/{map_builder,pose_graph,trajectory_builder,trajectory_builder_2d}.lua
--   $CONDA_PREFIX/share/cartographer_ros/configuration_files/backpack_2d.lua
--   官方文档：https://google-cartographer.readthedocs.io/en/latest/configuration.html
--   核对日期：2026-09-20

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- 坐标系（与 configs/slam.yaml:frames 严格一致；改一处必须改两处）
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = true,           -- 本实验无外部里程计，由 Cartographer 提供 odom→base_link
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,

  -- 传感器使用项
  use_odometry = false,                -- 不引入第二个位姿源，避免污染 SLAM 评估
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,                 -- 单线激光 /scan
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1, -- 10Hz 单线雷达无需按时间细分
  num_point_clouds = 0,

  -- 时间与发布节奏
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

-- ---------------------------------------------------------------------------
-- 2D 轨迹构建
-- ---------------------------------------------------------------------------
MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 4

TRAJECTORY_BUILDER_2D.use_imu_data = false            -- 本实验无 IMU；有 IMU 时必须置 true 并核对外参
TRAJECTORY_BUILDER_2D.min_range = 0.15                 -- 与 data/slam/scans.npz 的 range_min 一致
TRAJECTORY_BUILDER_2D.max_range = 8.0                  -- 与 range_max 一致（场地 6m×6m）
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1    -- 单线雷达逐帧处理
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025

-- 实时相关扫描匹配（给 Ceres 提供初值）
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1      -- m
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1e-1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- Ceres 扫描匹配（精细位姿优化）
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 1.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 20
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 1

-- 运动滤波：位姿变化过小时不插入新节点
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.5
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.05    -- 本实验步长约 0.06m，取 0.05 保证节点密度
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.5)

-- 子图
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90                 -- 每子图约 9s@10Hz
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05   -- 与 data.yaml mock.world.resolution 一致

-- ---------------------------------------------------------------------------
-- 位姿图优化
-- ---------------------------------------------------------------------------
POSE_GRAPH.optimize_every_n_nodes = 30
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.optimization_problem.huber_scale = 1e1

-- 调参提示（改这里之前先读 docs/cartographer_integration.md）：
--   * 轨迹抖动大 → 降低 ceres_scan_matcher.translation_weight / rotation_weight；
--   * 跟不上快速运动 → 增大 real_time_correlative_scan_matcher.linear_search_window；
--   * 子图边界重影 → 减小 submaps.num_range_data 或提高位姿图优化频率；
--   * 有回环的场地漂移大 → 降低 constraint_builder.min_score（注意误回环风险）。

return options
