"""Mobile manipulator RL env: end-effector reaching task.

This config defines an RL task where:
- The robot (4WD4WS mobile base + UR3 arm) learns full-body coordination
- Task: Navigate and manipulate to reach 3D spatial targets
- Training: DAgger with privileged teacher and corrupted student policy
- Networks: Separated base and arm control heads for modular learning

Architecture:
    - Base: 3 DOF velocity control (vx, vy, wz)
    - Arm: 6 DOF joint position control (UR3)
    - Observations: 370-dim policy (37x10 history) + 10-dim privileged
    - Rewards: Split base and arm rewards with exponential kernels
"""

from __future__ import annotations
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp as mdp

# Import base configuration
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mm_env_cfg import MobileManipulatorBaseEnvCfg

# Import constants
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    UR_ARM_JOINT_NAMES,
    UR_ARM_JOINT_CFG,
    WHEEL_DRIVE_JOINTS,
    WHEEL_PIVOT_JOINTS,
    PI
)

##
# Task-specific: EE Pose Command Ranges
##

EE_POSE_RANGES_INIT = mdp.EEPoseCommandCfg.Ranges(
    pos_x=(0.8, 1.0),
    pos_y=(-0.3, 0.3),
    pos_z=(0.4, 0.7),    # world frame not base frame
    roll=(0.0, 0.0),    # depend on end-effector axis
    pitch=(-PI / 18, PI / 18),   # ±10 degrees initial
    yaw=(-PI / 18, PI / 18),     # ±10 degrees initial
)

EE_POSE_RANGES = mdp.EEPoseCommandCfg.Ranges(
            pos_x=(0.7, 1.5),
            pos_y=(-0.35, 0.35),
            pos_z=(0.4, 1.0),
            roll=(0.0, 0.0),
            pitch=(-PI / 9, PI / 9),
            yaw=(-PI / 9, PI / 9),
)

EE_POSE_RANGES_FINAL = mdp.EEPoseCommandCfg.Ranges(
    pos_x=(0.8, 1.5),
    pos_y=(-0.35, 0.35),
    pos_z=(0.4, 1.0),
    roll=(0.0, 0.0),
    pitch=(-PI / 9, PI / 9),
    yaw=(-PI / 9, PI / 9),
)

##
# Task-specific MDP components
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # End-effector pose command with curriculum learning
    ee_pose = mdp.EEPoseCommandCfg(
        asset_name="robot",
        body_name="robotiq_85_base_link",
        resampling_time_range=(6.0, 8.0),
        debug_vis=True,
        chassis_sphere_radius=0.45,
        curriculum_coeff=1000,
        ranges=EE_POSE_RANGES,    # Nominal range (Not used during training)
        ranges_init=EE_POSE_RANGES_INIT,
        ranges_final=EE_POSE_RANGES_FINAL,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 4WD4WS base velocity control (3 DOF: vx, vy, wz)
    base_4ws = mdp.FourWheelFourSteerActionCfg(
        asset_name="robot",
        pivot_joint_names=WHEEL_PIVOT_JOINTS,
        drive_joint_names=WHEEL_DRIVE_JOINTS,
        use_world_frame=True,
        scale=(1.0, 1.0, 1.0),
        offset=(0.0, 0.0, 0.0),
        wheelbase=0.6,
        track_width=0.6,
        wheel_radii=(0.1, 0.1, 0.1, 0.1),
        max_linear_velocity=2.0,
        max_angular_velocity=2.0,
        max_steering_angle=1.57,
    )

    # UR3 arm joint position control (6 DOF)
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy (actor) network."""

        # Base state (6 dim)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=10,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=10,
        )

        # Arm joint state (12 dim)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=10,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=10,
        )

        # Proprioception (9 dim)
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=10,
        )

        # Commands (7 dim)
        ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
            history_length=10,
        )

        # Total policy obs: 3+3+6+6+9+7 = 34 (×10 history = 340 dim)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic (value) network - includes privileged info."""

        # === Include all policy observations (without noise for critic) ===
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=None,  # Critic gets clean observations
            history_length=10,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=None,
            history_length=10,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=None,
            history_length=10,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            noise=None,
            history_length=10,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=10,
        )
        ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
            history_length=10,
        )

        # === Privileged observations (only for critic) ===
        # World-frame velocity (3 dim, no history)
        base_lin_vel_world = ObsTerm(func=mdp.root_lin_vel_w)

        # Arm joint torques (6 dim, no history)
        arm_joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
        )

        # End-effector pose error (1 dim, no history)
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_command_error_obs,
            params={"command_name": "ee_pose"},
        )

        # Total critic obs: 340 (same as policy) + 3 + 6 + 1 = 350 dim

        def __post_init__(self):
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # Observation groups (CRITICAL: must have "policy" and "critic")
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ===== ARM REWARDS (prefix: end_effector_) =====
    ee_position_tracking = RewTerm(
        func=mdp.ee_position_command_error_exp,
        weight=3.0,
        params={"command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="robotiq_85_base_link"),
                "std": 0.2,
                },
    )

    # Orientation tracking (exponential kernel, same as position)
    ee_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_exp,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="robotiq_85_base_link"),
            "command_name": "ee_pose",
            "std": 0.5,  # ~28 degrees for exp(-1) = 0.37
            },
    )

    # ========== Action Regularization ==========
    arm_action_rate = RewTerm(
        func=mdp.arm_action_rate_l2,
        weight=-0.001,
    )
    base_action_rate = RewTerm(
        func=mdp.base_action_rate_l2,
        weight=-0.01,
    )
    # ========== Base Stability Rewards ==========
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate when tipped over
    tipped_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.3},    # radians (~17 degrees)
    )
    arm_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ur.*_link"),
                "threshold": 1.0,
            },
    )


@configclass
class EventCfg:
    """Event terms for randomization and resets."""

    # Reset base pose and velocity
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Reset arm joints
    reset_arm_joint = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": UR_ARM_JOINT_CFG,
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum schedule.
    
    Note: Orientation tracking now uses exponential kernel and is enabled from start.
    Command ranges still use curriculum (ranges_init -> ranges_final).
    """
    # Gradually turn off base flatness penalty
    base_flat_ori_modify = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "base_flat_orientation",
            "weight": 0.0,
            "num_steps": 48000,  # iterations * num_steps_per_env
        },
    )


@configclass
class MMEeTrackingEnvCfg(MobileManipulatorBaseEnvCfg):
    """RL environment for mobile manipulator EE tracking task.
    
    Inherits scene, sim, basic observations/actions/events from base.
    Adds task-specific commands, rewards, terminations, curriculum.
    """

    # Override observations to include EE command
    observations: ObservationsCfg = ObservationsCfg()

    # Task-specific components
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        """Post initialization - inherits base settings."""
        # Call parent's __post_init__
        super().__post_init__()


@configclass
class MMEeTrackingEnvCfg_PLAY(MMEeTrackingEnvCfg):
    """Play/Evaluation configuration for mobile manipulator EE tracking task.
    
    Differences from training config:
    - Fewer environments for visualization
    - No observation noise (corruption disabled)
    - Longer episode length for evaluation
    - Debug visualization enabled
    """

    def __post_init__(self):
        """Post initialization for play configuration."""
        # Call parent's __post_init__
        super().__post_init__()
        
        # Reduce number of environments for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable observation noise for evaluation
        self.observations.policy.enable_corruption = False
        
        # Longer episode for evaluation
        self.episode_length_s = 30.0
        
        # Ensure debug visualization is enabled
        self.commands.ee_pose.debug_vis = True

        play_range = EE_POSE_RANGES
        self.commands.ee_pose.ranges = play_range
        self.commands.ee_pose.ranges_init = play_range
        self.commands.ee_pose.ranges_final = play_range

        self.commands.ee_pose.resampling_time_range = (10.0, 10.0)
