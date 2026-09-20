-- ============================================================================
-- configs/cartographer/g1_2d.lua
-- 【自动生成，禁止手改】编辑 configs/cartographer/g1_2d.overrides.lua 后重新运行：
--     python scripts/43_gen_cartographer_config.py
--
-- 为什么自包含：本机 cartographer_ros 2.0.9003 的 Lua include 解析有缺陷
-- （官方 backpack_2d.lua 也会失败：basic_filebuf::underflow, Is a directory），
-- 因此把官方默认值内联，仅覆盖本实验需要的参数。
--
-- 内联来源（官方 Cartographer 默认配置，只读）：
--     share/cartographer/configuration_files/pose_graph.lua  (sha256:2eb6515948a7fe92)
--     share/cartographer/configuration_files/trajectory_builder_2d.lua  (sha256:c3142599308a80fd)
--     share/cartographer/configuration_files/trajectory_builder_3d.lua  (sha256:2ff3cea0a6770f08)
--     share/cartographer/configuration_files/trajectory_builder.lua  (sha256:4b8fb7053c0be2ac)
--     share/cartographer/configuration_files/map_builder.lua  (sha256:f58f56ef01ddb427)
-- >>>>>>>>>> BEGIN pose_graph.lua （官方原文，已移除 include 行） >>>>>>>>>>
-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

POSE_GRAPH = {
  optimize_every_n_nodes = 90,
  constraint_builder = {
    sampling_ratio = 0.3,
    max_constraint_distance = 15.,
    min_score = 0.55,
    global_localization_min_score = 0.6,
    loop_closure_translation_weight = 1.1e4,
    loop_closure_rotation_weight = 1e5,
    log_matches = true,
    fast_correlative_scan_matcher = {
      linear_search_window = 7.,
      angular_search_window = math.rad(30.),
      branch_and_bound_depth = 7,
    },
    ceres_scan_matcher = {
      occupied_space_weight = 20.,
      translation_weight = 10.,
      rotation_weight = 1.,
      ceres_solver_options = {
        use_nonmonotonic_steps = true,
        max_num_iterations = 10,
        num_threads = 1,
      },
    },
    fast_correlative_scan_matcher_3d = {
      branch_and_bound_depth = 8,
      full_resolution_depth = 3,
      min_rotational_score = 0.77,
      min_low_resolution_score = 0.55,
      linear_xy_search_window = 5.,
      linear_z_search_window = 1.,
      angular_search_window = math.rad(15.),
    },
    ceres_scan_matcher_3d = {
      occupied_space_weight_0 = 5.,
      occupied_space_weight_1 = 30.,
      translation_weight = 10.,
      rotation_weight = 1.,
      only_optimize_yaw = false,
      ceres_solver_options = {
        use_nonmonotonic_steps = false,
        max_num_iterations = 10,
        num_threads = 1,
      },
    },
  },
  matcher_translation_weight = 5e2,
  matcher_rotation_weight = 1.6e3,
  optimization_problem = {
    huber_scale = 1e1,
    acceleration_weight = 1.1e2,
    rotation_weight = 1.6e4,
    local_slam_pose_translation_weight = 1e5,
    local_slam_pose_rotation_weight = 1e5,
    odometry_translation_weight = 1e5,
    odometry_rotation_weight = 1e5,
    fixed_frame_pose_translation_weight = 1e1,
    fixed_frame_pose_rotation_weight = 1e2,
    fixed_frame_pose_use_tolerant_loss = false,
    fixed_frame_pose_tolerant_loss_param_a = 1,
    fixed_frame_pose_tolerant_loss_param_b = 1,
    log_solver_summary = false,
    use_online_imu_extrinsics_in_3d = true,
    fix_z_in_3d = false,
    ceres_solver_options = {
      use_nonmonotonic_steps = false,
      max_num_iterations = 50,
      num_threads = 7,
    },
  },
  max_num_final_iterations = 200,
  global_sampling_ratio = 0.003,
  log_residual_histograms = true,
  global_constraint_search_after_n_seconds = 10.,
  --  overlapping_submaps_trimmer_2d = {
  --    fresh_submaps_count = 1,
  --    min_covered_area = 2,
  --    min_added_submaps_count = 5,
  --  },
}
-- <<<<<<<<<< END pose_graph.lua <<<<<<<<<<

-- >>>>>>>>>> BEGIN trajectory_builder_2d.lua （官方原文，已移除 include 行） >>>>>>>>>>
-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

TRAJECTORY_BUILDER_2D = {
  use_imu_data = true,
  min_range = 0.,
  max_range = 30.,
  min_z = -0.8,
  max_z = 2.,
  missing_data_ray_length = 5.,
  num_accumulated_range_data = 1,
  voxel_filter_size = 0.025,

  adaptive_voxel_filter = {
    max_length = 0.5,
    min_num_points = 200,
    max_range = 50.,
  },

  loop_closure_adaptive_voxel_filter = {
    max_length = 0.9,
    min_num_points = 100,
    max_range = 50.,
  },

  use_online_correlative_scan_matching = false,
  real_time_correlative_scan_matcher = {
    linear_search_window = 0.1,
    angular_search_window = math.rad(20.),
    translation_delta_cost_weight = 1e-1,
    rotation_delta_cost_weight = 1e-1,
  },

  ceres_scan_matcher = {
    occupied_space_weight = 1.,
    translation_weight = 10.,
    rotation_weight = 40.,
    ceres_solver_options = {
      use_nonmonotonic_steps = false,
      max_num_iterations = 20,
      num_threads = 1,
    },
  },

  motion_filter = {
    max_time_seconds = 5.,
    max_distance_meters = 0.2,
    max_angle_radians = math.rad(1.),
  },

  -- TODO(schwoere,wohe): Remove this constant. This is only kept for ROS.
  imu_gravity_time_constant = 10.,
  pose_extrapolator = {
    use_imu_based = false,
    constant_velocity = {
      imu_gravity_time_constant = 10.,
      pose_queue_duration = 0.001,
    },
    imu_based = {
      pose_queue_duration = 5.,
      gravity_constant = 9.806,
      pose_translation_weight = 1.,
      pose_rotation_weight = 1.,
      imu_acceleration_weight = 1.,
      imu_rotation_weight = 1.,
      odometry_translation_weight = 1.,
      odometry_rotation_weight = 1.,
      solver_options = {
        use_nonmonotonic_steps = false;
        max_num_iterations = 10;
        num_threads = 1;
      },
    },
  },

  submaps = {
    num_range_data = 90,
    grid_options_2d = {
      grid_type = "PROBABILITY_GRID",
      resolution = 0.05,
    },
    range_data_inserter = {
      range_data_inserter_type = "PROBABILITY_GRID_INSERTER_2D",
      probability_grid_range_data_inserter = {
        insert_free_space = true,
        hit_probability = 0.55,
        miss_probability = 0.49,
      },
      tsdf_range_data_inserter = {
        truncation_distance = 0.3,
        maximum_weight = 10.,
        update_free_space = false,
        normal_estimation_options = {
          num_normal_samples = 4,
          sample_radius = 0.5,
        },
        project_sdf_distance_to_scan_normal = true,
        update_weight_range_exponent = 0,
        update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0.5,
        update_weight_distance_cell_to_hit_kernel_bandwidth = 0.5,
      },
    },
  },
}
-- <<<<<<<<<< END trajectory_builder_2d.lua <<<<<<<<<<

-- >>>>>>>>>> BEGIN trajectory_builder_3d.lua （官方原文，已移除 include 行） >>>>>>>>>>
-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

MAX_3D_RANGE = 60.
INTENSITY_THRESHOLD = 40

TRAJECTORY_BUILDER_3D = {
  min_range = 1.,
  max_range = MAX_3D_RANGE,
  num_accumulated_range_data = 1,
  voxel_filter_size = 0.15,

  high_resolution_adaptive_voxel_filter = {
    max_length = 2.,
    min_num_points = 150,
    max_range = 15.,
  },

  low_resolution_adaptive_voxel_filter = {
    max_length = 4.,
    min_num_points = 200,
    max_range = MAX_3D_RANGE,
  },

  use_online_correlative_scan_matching = false,
  real_time_correlative_scan_matcher = {
    linear_search_window = 0.15,
    angular_search_window = math.rad(1.),
    translation_delta_cost_weight = 1e-1,
    rotation_delta_cost_weight = 1e-1,
  },

  ceres_scan_matcher = {
    occupied_space_weight_0 = 1.,
    occupied_space_weight_1 = 6.,
    intensity_cost_function_options_0 = {
        weight = 0.5,
        huber_scale = 0.3,
        intensity_threshold = INTENSITY_THRESHOLD,
    },
    translation_weight = 5.,
    rotation_weight = 4e2,
    only_optimize_yaw = false,
    ceres_solver_options = {
      use_nonmonotonic_steps = false,
      max_num_iterations = 12,
      num_threads = 1,
    },
  },

  motion_filter = {
    max_time_seconds = 0.5,
    max_distance_meters = 0.1,
    max_angle_radians = 0.004,
  },

  rotational_histogram_size = 120,

  -- TODO(schwoere,wohe): Remove this constant. This is only kept for ROS.
  imu_gravity_time_constant = 10.,
  pose_extrapolator = {
    use_imu_based = false,
    constant_velocity = {
      imu_gravity_time_constant = 10.,
      pose_queue_duration = 0.001,
    },
    -- TODO(wohe): Tune these parameters on the example datasets.
    imu_based = {
      pose_queue_duration = 5.,
      gravity_constant = 9.806,
      pose_translation_weight = 1.,
      pose_rotation_weight = 1.,
      imu_acceleration_weight = 1.,
      imu_rotation_weight = 1.,
      odometry_translation_weight = 1.,
      odometry_rotation_weight = 1.,
      solver_options = {
        use_nonmonotonic_steps = false;
        max_num_iterations = 10;
        num_threads = 1;
      },
    },
  },

  submaps = {
    high_resolution = 0.10,
    high_resolution_max_range = 20.,
    low_resolution = 0.45,
    num_range_data = 160,
    range_data_inserter = {
      hit_probability = 0.55,
      miss_probability = 0.49,
      num_free_space_voxels = 2,
      intensity_threshold = INTENSITY_THRESHOLD,
    },
  },

  -- When setting use_intensites to true, the intensity_cost_function_options_0
  -- parameter in ceres_scan_matcher has to be set up as well or otherwise
  -- CeresScanMatcher will CHECK-fail.
  use_intensities = false,
}
-- <<<<<<<<<< END trajectory_builder_3d.lua <<<<<<<<<<

-- >>>>>>>>>> BEGIN trajectory_builder.lua （官方原文，已移除 include 行） >>>>>>>>>>
-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


TRAJECTORY_BUILDER = {
  trajectory_builder_2d = TRAJECTORY_BUILDER_2D,
  trajectory_builder_3d = TRAJECTORY_BUILDER_3D,
--  pure_localization_trimmer = {
--    max_submaps_to_keep = 3,
--  },
  collate_fixed_frame = true,
  collate_landmarks = false,
}
-- <<<<<<<<<< END trajectory_builder.lua <<<<<<<<<<

-- >>>>>>>>>> BEGIN map_builder.lua （官方原文，已移除 include 行） >>>>>>>>>>
-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


MAP_BUILDER = {
  use_trajectory_builder_2d = false,
  use_trajectory_builder_3d = false,
  num_background_threads = 4,
  pose_graph = POSE_GRAPH,
  collate_by_trajectory = false,
}
-- <<<<<<<<<< END map_builder.lua <<<<<<<<<<

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
