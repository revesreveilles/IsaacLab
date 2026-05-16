"""Mobile manipulator RL env: projection-driven progress + direct Bezier preview.

The centerline (Bezier path) provides coarse guidance via projection-driven
progress: s_proj → uniform preview sampling on Bezier → q1=s_ref.
No obstacle-aware filtering is applied; the RL policy is fully responsible
for obstacle avoidance.
Path-ahead clearance modulates an adaptive contouring tube deadband.
Single path / single goal / episodic task.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp as mdp

# fmt: off
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mm_env_cfg import (  # noqa: E501
    MobileManipulatorBaseEnvCfg,
)
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (  # noqa: E501
    UR_ARM_JOINT_NAMES,
    UR_ARM_JOINT_CFG,
    WHEEL_DRIVE_JOINTS,
    WHEEL_PIVOT_JOINTS,
    PI,
    NUM_DYN_TOTAL,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    DYN_ALL_NAMES,
    DYN_ALL_SIZES,
    CUBOID_SIZES,
    CYLINDER_SIZES,
    CUBOID_HALF_EXTENTS,
    CYLINDER_PARAMS,
)
# fmt: on

# ── Shared tilt angle constant (termination threshold) ──
TIPPED_OVER_LIMIT_ANGLE = 0.25


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    ee_traj = mdp.EETrajectoryCommandCfg(
            asset_name="robot",
            body_name="ur_wrist_3_link",
            # Prevent auto-resample within episode
            resampling_time_range=(1e6, 1e6),
            sample_uniform_orientation=False,
            arm_base_offset=(0.3, 0.0, 0.52),
            ranges_pos={
                "rho_xy": (1.0, 10.0),
                "yaw": (-1.0, 1.0),
            },
            z_range=(0.60, 0.80),
            delta_euler_ranges={
                "roll":  (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw":   (-0.1, 0.1),
            },
            collision_box_lower=(-0.55, -0.4, 0.0),
            collision_box_upper=(0.55, 0.4, 0.60),
            check_obstacle_collision=True,
            # Bezier path parameters
            num_control_points=1,
            control_point_lateral_range=(-0.3, 0.3),
            control_point_vertical_range=(-0.15, 0.15),
            num_path_cache_samples=64,
            num_bezier_dense_samples=256,
            # Local preview stack
            num_preview_points=4,
            preview_spacing_s_base=0.1,
            preview_spacing_s_min=0.06,
            preview_spacing_s_max=0.15,
            preview_spacing_error_gain=1.0,
            # Path-ahead clearance (static only)
            path_clearance_num=3,
            path_clearance_probe_radius=0.05,
            path_clearance_safe=0.50,
            # Adaptive tube deadband
            tube_deadband_min=0.15,
            tube_deadband_max=0.35,
            tube_deadband_gain=1.0,
            min_clearance_update_interval=4,
            # Virtual progress (s_hat) – MPCC-style lag/contouring gates
            # Design intent: s_hat guides EE toward the goal, NOT for path
            # adherence.  Lag error is the primary brake; contouring error
            # only mildly slows s_hat so obstacle detours don't stall it.
            s_hat_initial_offset_s=0.05,
            s_hat_nominal_speed_mps=0.70,
            s_hat_speed_max_mps=3.0,
            s_hat_filter_tau=0.08,
            s_hat_catchup_tau=0.06,
            # Lag gate (primary brake): generous deadband, moderate sigma
            s_hat_lag_deadband_m=0.30,
            s_hat_lag_sigma_m=0.45,
            # Contouring gate (mild influence): wide deadband + sigma + low power
            s_hat_contouring_deadband_m=0.25,
            s_hat_contouring_sigma_m=0.50,
            s_hat_contouring_gate_power=0.15,
            # Tail release (末段虚拟进度释放)
            s_hat_tail_release_start_s=0.90,
            s_hat_tail_release_start_dist_m=0.60,
            s_hat_tail_nominal_floor_ratio=0.50,
            # Reward reference (s_ref)
            ori_weight_switch_distance=0.40,
            ori_weight_k=12.0,
            # Base reachability (one-sided adaptive base participation)
            arm_workspace_radius=0.45,
            base_switch_sigma_m=0.08,
            # Visualization
            debug_vis=False,
            vis_update_interval=5,
            num_trajectory_samples=32,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    base_4ws = mdp.FourWheelFourSteerActionCfg(
        asset_name="robot",
        pivot_joint_names=WHEEL_PIVOT_JOINTS,
        drive_joint_names=WHEEL_DRIVE_JOINTS,
        use_world_frame=False,
        scale=(1.0, 1.0, 1.0),
        offset=(0.0, 0.0, 0.0),
        wheelbase=0.6,
        track_width=0.6,
        wheel_radii=(0.1, 0.1, 0.1, 0.1),
        max_linear_velocity=1.5,
        max_angular_velocity=3.0,
        max_steering_angle=1.57,
    )

    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        scale={
            ".*shoulder_pan_joint": 1.57,
            ".*shoulder_lift_joint": 1.57,
            ".*elbow_joint": 1.57,
            ".*wrist_1_joint": 0.8,
            ".*wrist_2_joint": 0.8,
            ".*wrist_3_joint": 0.5,
        },
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-2.0, 2.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy network."""

        # Base state (6 dim)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=3,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=3,
        )

        # Arm joint state (12 dim)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=3,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=3,
            clip=(-10.0, 10.0),
        )

        # Actions (9 dim)
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=3,
            clip=(-10.0, 10.0),
        )

        # EE current pose (7 dim)
        ee_pose_b = ObsTerm(
            func=mdp.ee_pose_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="ur_wrist_3_link"
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        # Commands: q1 base-frame full pose (7 dim)
        ee_traj_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_traj"},
            history_length=1,
        )

        # Final goal pose in base frame (7 dim): [pos_b(3), quat_b(4)]
        final_goal_pose_b = ObsTerm(
            func=mdp.ee_traj_final_goal_pose_b,
            params={
                "command_name": "ee_traj",
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        # Future refs q2..qN, relative pose chain (num_points * 6 dim)
        # Each link: delta_pos(3) + delta_axis_angle(3)
        preview_points = ObsTerm(
            func=mdp.ee_traj_preview_points_b,
            params={
                "command_name": "ee_traj",
                "asset_cfg": SceneEntityCfg("robot"),
                "num_points": 3,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        # Collision sphere to point cloud distance (N_spheres * 4 dim)
        sphere_distances = ObsTerm(
            func=mdp.sphere_pointcloud_distance,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "asset_cfg": SceneEntityCfg("robot"),
                "ground_clearance": 0.1,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=1,
        )

        # Raw lidar range scan (num_rays dim, e.g. 144)
        # lidar_scan = ObsTerm(
        #     func=mdp.lidar_range_scan_flat,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("lidar"),
        #         "ground_clearance": 0.1,
        #         "ray_subsample": 1,
        #         "normalize": True,
        #     },
        #     noise=Unoise(n_min=-0.02, n_max=0.02),
        #     history_length=1,
        # )

        # K nearest dynamic obstacles — 10D features (K*10 dim)
        dyn_obstacles = ObsTerm(
            func=mdp.dynamic_obstacles,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_asset_names": DYN_ALL_NAMES,
                "obstacle_sizes": DYN_ALL_SIZES,
                "cuboid_names": DYN_CUBOID_NAMES,
                "cylinder_names": DYN_CYLINDER_NAMES,
                "cuboid_half_extents": CUBOID_HALF_EXTENTS,
                "cylinder_params": CYLINDER_PARAMS,
                "top_k": 5,
                "max_range": 5.0,
                "max_vel": 1.5,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        # Projected gravity in body frame (3 dim) — tilt awareness
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic network only.

        With runner obs_groups:
            "policy": ["policy"],
            "critic": ["policy", "critic"],

        the critic already receives all PolicyCfg observations. Therefore this
        group should only contain extra privileged terms that the actor does not
        receive, instead of duplicating PolicyCfg terms.
        """

        # Path progress features (3 dim, privileged): [s_hat, speed, 1-s_hat]
        progress_features = ObsTerm(
            func=mdp.ee_traj_progress_features,
            params={
                "command_name": "ee_traj",
            },
            history_length=1,
        )

        # Path pose at geometric projection s_proj (7 dim, privileged)
        path_pose_at_s_proj = ObsTerm(
            func=mdp.ee_traj_path_pose_at_s_proj_b,
            params={
                "command_name": "ee_traj",
                "asset_cfg": SceneEntityCfg("robot"),
            },
            history_length=1,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Reward structure (prioritized base-arm + tube penalties):
      r = r_task + r_tube(s_proj) + r_obs + r_smooth + r_success

      r_task = r_base + (1 - w_base) * r_arm

      w_base = 1 - exp(-(relu(rho_base - arm_workspace_radius) / sigma)^2)
      → one-sided adaptive base participation: w_base = 0 inside the
        arm comfort handoff radius, and rises smoothly only outside it.
      rho_base = ||p_ref_xy(s_ref) - p_base_xy||

      r_base contains a w_base-gated delta approach term plus a positive
      membership bonus exp(-relu(rho_base - radius) / std_base), which is 1
      inside the radius and smoothly decays outside.

      r_arm  = alpha_p * delta_d_p  +  beta_p * gate * phi_p(d_p)
             + w_o(d_p) * (alpha_o * delta_d_o  +  beta_o * gate * phi_o(d_o))

      w_o = sigmoid(k * (d_switch - d_p))   (position-first gating)

      gate = max(clamp(Δs_proj / δs_norm, 0, 1),
                 clamp(Δd_final / δg_norm, 0, 1))
      → tracking rewards are only paid during real progress;
        standing still yields gate ≈ 0 → no free tracking bonus.

    Tube penalties allow small path deviations (obstacle avoidance)
    but constrain the robot from straying too far from the path tube.
    Tube deadband is adaptive based on path-ahead static clearance.
    """

    # ===== Global path progress (Δs_proj) — PRIMARY completion signal =====

    path_progress = RewTerm(
        func=mdp.path_progress_reward,
        weight=4.0,
        params={
            "command_name": "ee_traj",
            "max_delta_s": 0.05,
        },
    )

    # ===== Reference-based local tracking (s_ref=q1) — r_arm =====
    # approach = improvement-only delta (safe, zero at stall)
    # tracking = progress-gated exp shaping (zero at stall)

    reference_position_approach = RewTerm(
        func=mdp.reference_position_approach_reward,
        weight=4.0,
        params={
            "command_name": "ee_traj",
            "max_delta": 0.05,
        },
    )

    reference_position_tracking = RewTerm(
        func=mdp.reference_position_tracking_reward,
        weight=8.0,
        params={
            "command_name": "ee_traj",
            "std": 0.12,
            "delta_s_norm": 0.01,
            "delta_g_norm": 0.01,
        },
    )

    reference_orientation_approach = RewTerm(
        func=mdp.reference_orientation_approach_reward,
        weight=1.0,
        params={
            "command_name": "ee_traj",
            "max_delta": 0.10,
        },
    )

    reference_orientation_tracking = RewTerm(
        func=mdp.reference_orientation_tracking_reward,
        weight=3.0,
        params={
            "command_name": "ee_traj",
            "std": 0.35,
            "delta_s_norm": 0.01,
            "delta_g_norm": 0.01,
        },
    )

    # ===== Base reachability — r_base =====

    base_reachability_approach = RewTerm(
        func=mdp.base_reachability_approach_reward,
        weight=3.0,
        params={
            "command_name": "ee_traj",
            "max_delta": 0.05,
        },
    )

    base_reachability_tracking = RewTerm(
        func=mdp.base_reachability_tracking_reward,
        weight=2.0,
        params={
            "command_name": "ee_traj",
            "arm_workspace_radius": 0.45,
            "std_base": 0.10,
        },
    )

    # ===== Final goal closure (always-active dense signal) =====

    final_goal_approach = RewTerm(
        func=mdp.final_goal_approach_reward,
        weight=1.0,
        params={
            "command_name": "ee_traj",
            "max_delta": 0.05,
        },
    )

    # ===== Tube penalties (s_proj based, adaptive deadband) =====

    contouring_tube = RewTerm(
        func=mdp.adaptive_contouring_tube_penalty,
        weight=1.5,
        params={
            "command_name": "ee_traj",
        },
    )

    orientation_tube = RewTerm(
        func=mdp.orientation_tube_penalty,
        weight=1.5,
        params={
            "command_name": "ee_traj",
            "deadband": 0.40,
        },
    )

    # ===== Time penalty =====

    time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=-0.10,
    )

    # ===== Action Regularization =====

    arm_action_rate = RewTerm(
        func=mdp.arm_action_rate_l2,
        weight=-0.08,
    )

    base_action_rate = RewTerm(
        func=mdp.base_action_rate_l2,
        weight=-0.10,
    )
    arm_action_l2 = RewTerm(
        func=mdp.arm_action_l2,
        weight=-0.005,
    )
    base_action_l2 = RewTerm(
        func=mdp.base_action_l2,
        weight=-0.02,
    )

    # Base Stability — penalize aggressive motion and non-flat orientation
    base_motion = RewTerm(
        func=mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ===== Terminated rewards =====

    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=150.0,
        params={
            "command_name": "ee_traj",
            "pos_tolerance": 0.15,
            "ori_tolerance": 0.30,
        },
    )

    tipped_over_penalty = RewTerm(
        func=mdp.terminated_event_reward,
        weight=-200.0,
        params={"term_keys": ["tipped_over"]},
    )

    # penalty on geometric obstacle collision (lidar + collision spheres)
    obstacle_collision_penalty = RewTerm(
        func=mdp.terminated_event_reward,
        weight=-80.0,
        params={"term_keys": ["obstacle_collision"]},
    )

    # ===== Timeout penalties  =====
    # NOTE: is_terminated_term CANNOT be used for timeouts — it explicitly
    #       filters out time_out via (~time_outs), always returning 0.
    #       Use timeout_terminal_penalty which directly returns the flag.
    timeout_terminal = RewTerm(
        func=mdp.timeout_terminal_penalty,
        weight=-10.0,
    )

    # ===== Collision Avoidance =====

    collision_avoidance = RewTerm(
        func=mdp.collision_avoidance_reward,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "asset_cfg": SceneEntityCfg("robot"),
            "d_safe": 0.06,
            "d_repulsion": 0.30,
            "repulsion_weight": 3.5,
            "hard_penalty": -2.0,
            "ground_clearance": 0.1,
        },
    )

    dynamic_obstacle_avoidance = RewTerm(
        func=mdp.dynamic_obstacle_cbf_reward,
        weight=10.0,
        params={
            "obstacle_asset_names": DYN_ALL_NAMES,
            "cuboid_names": DYN_CUBOID_NAMES,
            "cylinder_names": DYN_CYLINDER_NAMES,
            "cuboid_half_extents": CUBOID_HALF_EXTENTS,
            "cylinder_params": CYLINDER_PARAMS,
            "d_safe": 0.10,
            "d_trigger": 0.60,
            "t_react": 0.30,
            "barrier_delta": 0.06,
            "barrier_mu": 0.05,
            "max_penalty": 2.0,
        },
    )
    # base-line Reward
    # collision_avoidance = RewTerm(
    #     func=mdp.static_full_body_log_distance_reward,
    #     weight=10.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("lidar"),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "d_trigger": 0.60,
    #         "eps": 1e-3,
    #         "max_penalty": 2.0,
    #         "ground_clearance": 0.1,
    #         "aggregation": "max",
    #     },
    # )

    # dynamic_obstacle_avoidance = RewTerm(
    #     func=mdp.dynamic_full_body_log_distance_reward,
    #     weight=10.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "obstacle_asset_names": DYN_ALL_NAMES,
    #         "cuboid_names": DYN_CUBOID_NAMES,
    #         "cylinder_names": DYN_CYLINDER_NAMES,
    #         "cuboid_half_extents": CUBOID_HALF_EXTENTS,
    #         "cylinder_params": CYLINDER_PARAMS,
    #         "d_trigger": 0.70,
    #         "eps": 1e-3,
    #         "max_penalty": 2.0,
    #         "aggregation": "max",
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Success: ``goal_reached`` — EE position (and optionally orientation)
             within tolerance of the **global sampled goal / path endpoint**
             (world-fixed target sampled at episode start).
             This is **not** a local body-frame target, **not** an
             intermediate waypoint, and **not** survival-until-timeout.
    Failures: tipping, collision, out_of_bounds.
    time_out: episode length exceeded (not counted as success).

    Note: contouring/lag violations are NOT registered as active
    terminations. Temporarily deviating from the path during
    obstacle avoidance should not end the episode.
    """

    time_out = DoneTerm(
        func=mdp.proportional_time_out,
        time_out=True,
        params={
            "command_name": "ee_traj",
            "nominal_speed": 1.0,
            "safety_factor": 4.0,
            "min_timeout_s": 20.0,
        },
    )

    # Success: EE reached the global sampled goal / path endpoint.
    goal_reached = DoneTerm(
        func=mdp.ee_pose_goal_reached,
        time_out=False,
        params={
            "command_name": "ee_traj",
            "pos_tolerance": 0.15,
            "ori_tolerance": 0.30,
        },
    )

    tipped_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": TIPPED_OVER_LIMIT_ANGLE},
        time_out=False,
    )

    obstacle_collision = DoneTerm(
        func=mdp.obstacle_collision,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "asset_cfg": SceneEntityCfg("robot"),
            "static_collision_margin": 0.05,
            "dynamic_collision_margin": 0.05,
            "ground_clearance": 0.1,
            "cuboid_names": DYN_CUBOID_NAMES,
            "cylinder_names": DYN_CYLINDER_NAMES,
            "cuboid_half_extents": CUBOID_HALF_EXTENTS,
            "cylinder_params": CYLINDER_PARAMS,
        },
        time_out=False,
    )

    static_obstacle_collision = DoneTerm(
        func=mdp.static_obstacle_collision,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "asset_cfg": SceneEntityCfg("robot"),
            "static_collision_margin": 0.05,
            "ground_clearance": 0.1,
        },
        time_out=False,
    )

    dynamic_obstacle_collision = DoneTerm(
        func=mdp.dynamic_obstacle_collision,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "dynamic_collision_margin": 0.05,
            "cuboid_names": DYN_CUBOID_NAMES,
            "cylinder_names": DYN_CYLINDER_NAMES,
            "cuboid_half_extents": CUBOID_HALF_EXTENTS,
            "cylinder_params": CYLINDER_PARAMS,
        },
        time_out=False,
    )


@configclass
class EventCfg:
    """Event terms for randomization and resets."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_arm_joint = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": UR_ARM_JOINT_CFG,
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_dynamic_obstacles = EventTerm(
        func=mdp.reset_dynamic_obstacles_navrl_style,
        mode="reset",
        params={
            "safe_radius": 2.0,
            "cuboid_hover_range": (0.7, 1.0),
            "vel_range": (0.5, 1.5),
            "local_range": (5.0, 5.0, 1.0),
            "write_to_sim": False,
        },
    )

    step_kinematic_obstacles = EventTerm(
        func=mdp.step_kinematic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "local_range": (5.0, 5.0, 1.0),
            "goal_reach_threshold": 0.5,
            "vel_resample_interval": 2.0,
            "vel_range": (0.5, 1.5),
            "cuboid_hover_range": (0.7, 1.0),
            "goal_check_interval_s": 0.10,
            "goal_timeout_s": 4.0,
            "write_to_sim": False,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum learning configuration.

    All curricula use unified success + quality semantics:

    - **success**: ``goal_reached`` — EE position (and optionally
      orientation) within tolerance of the **global sampled goal / path
      endpoint** for the current episode.  This is the world-fixed target
      sampled at episode start.  It is **not** a local body-frame reference,
      **not** an intermediate waypoint, and **not** survival-until-timeout.

    Dynamic obstacle curriculum is driven by success/collision/tip rates:
    - success_rate: goal_reached episodes
    - collision_rate: obstacle_collision termination episodes
    - tip_rate: tipped_over termination episodes

    Quality-rate (contouring/orientation) is NOT used for obstacle difficulty
    progression.  The path serves only as coarse guidance.
    """

    # =========================================================================
    # Dynamic Obstacle Curriculum (quantized active-count, NOT
    # a fixed stage list).  Upper bound = NUM_DYN_TOTAL.
    # Difficulty changes by ±difficulty_step per evaluation window.
    # Driven by success/collision/tip rates (NOT quality-rate).
    # =========================================================================
    # dynamic_obstacles = CurrTerm(
    #     func=mdp.dynamic_obstacle_curriculum,
    #     params={
    #         "max_active": NUM_DYN_TOTAL,
    #         "difficulty_step": 5,
    #         "command_name": "ee_traj",
    #         # Success = EE reached global sampled goal
    #         "goal_pos_tolerance": 0.15,
    #         "goal_ori_tolerance": 0.30,
    #         # Upgrade thresholds
    #         "move_up_success_rate": 0.85,
    #         "move_up_collision_rate": 0.13,
    #         "move_up_tip_rate": 0.01,
    #         # Downgrade thresholds
    #         "move_down_success_rate": 0.75,
    #         "move_down_collision_rate": 0.20,
    #         "move_down_tip_rate": 0.03,
    #         # Ring buffer
    #         "window_size": 8000,
    #         # Warmup
    #         "warmup_min_samples": 2000,
    #         "warmup_min_adjust_samples": 4000,
    #         # Evaluation interval (samples)
    #         "adjust_every_samples": 8000,
    #         # Hysteresis: consecutive windows needed
    #         "up_hold_windows": 3,
    #         "down_hold_windows": 3,
    #     },
    # )


@configclass
class MMEeTrajTrackingEnvCfg(MobileManipulatorBaseEnvCfg):
    """RL environment for mobile manipulator EE path tracking.

    Single path / single goal / episodic task with path-progress parameterization.
    """

    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class MMEeTrajTrackingEnvCfg_PLAY(MMEeTrajTrackingEnvCfg):
    """Play/Evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()

        # ===== Scene Settings =====
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.5

        # ===== Episode Settings =====
        self.episode_length_s = 60.0

        # ===== Observation Settings =====
        # Disable noise in evaluation
        self.observations.policy.enable_corruption = False

        # ===== Curriculum Settings =====
        # Disable all curricula in PLAY mode (use fixed ranges/std/difficulty)
        self.curriculum.dynamic_obstacles = None

        # Sync dynamic obstacle buffers to RigidObjectCollections for PLAY visualization.
        self.events.reset_dynamic_obstacles.params["write_to_sim"] = True
        self.events.step_kinematic_obstacles.params["write_to_sim"] = True

        # ===== Command Settings =====
        self.commands.ee_traj.ranges_pos = {
            "rho_xy": (10.0, 15.0),
            "yaw": (-3 * PI / 5, 3 * PI / 5),
        }
        self.commands.ee_traj.z_range = (0.6, 0.8)
        self.commands.ee_traj.collision_box_lower = (-0.55, -0.4, 0.0)
        self.commands.ee_traj.collision_box_upper = (0.55, 0.4, 0.60)
        self.commands.ee_traj.arm_base_offset = (0.3, 0.0, 0.52)
        self.commands.ee_traj.sample_uniform_orientation = False
        self.commands.ee_traj.delta_euler_ranges = {
            "roll":  (-0.1, 0.1),
            "pitch": (-0.2, 0.2),
            "yaw":   (-0.2, 0.2),
        }
        # ==== Visualization settings ====
        self.scene.lidar.debug_vis = True
        self.commands.ee_traj.debug_vis = True
        self.commands.ee_traj.vis_update_interval = 1
        self.commands.ee_traj.num_trajectory_samples = 32
