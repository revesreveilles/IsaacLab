"""Mobile manipulator RL env: end-effector trajectory tracking task.

Uses Lie group SE(3) interpolation for trajectory generation:
- Cubic polynomial interpolation on SE(3) manifold
- Uniform sampling in spherical workspace around robot
- Unified exponential reward: exp(-||T_cmd ⊖ T||)
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
)
# fmt: on


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    ee_traj = mdp.EETrajectoryCommandCfg(
            asset_name="robot",
            body_name="robotiq_85_base_link",
            trajectory_time_range=(1.0, 3.0),
            sample_uniform_orientation=True,
            arm_length_min=0.25,         # UR3 最小臂长
            arm_length_max=0.45,         # UR3 最大臂长
            chassis_radius=0.6,          # 底盘半径
            arm_base_offset=(0.3, 0.0, 0.45),
            min_height=0.3,             # 最小高度（避免碰地面）
            max_height=1.0,              # 最大高度（合理的工作范围）
            max_resample_attempts=10,    # 最大重采样次数
            debug_vis=False,
            vis_update_interval=10,
            num_trajectory_samples=15,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
        )

        # Actions (9 dim)
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=3,
        )

        # EE current pose (7 dim)
        ee_position = ObsTerm(
            func=mdp.body_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="robotiq_85_base_link"
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=2,
        )
        ee_orientation = ObsTerm(
            func=mdp.body_quat_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="robotiq_85_base_link"
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=2,
        )

        # Commands (7 dim)
        ee_traj_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_traj"},
            history_length=1,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic network."""

        # Policy observations (no noise)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=3)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            history_length=3,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": UR_ARM_JOINT_CFG},
            history_length=3,
        )

        actions = ObsTerm(func=mdp.last_action, history_length=3)

        ee_position = ObsTerm(
            func=mdp.body_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="robotiq_85_base_link"
                )
            },
            history_length=2,
        )
        ee_orientation = ObsTerm(
            func=mdp.body_quat_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="robotiq_85_base_link"
                )
            },
            history_length=2,
        )

        ee_traj_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_traj"},
            history_length=1,
        )

        # Privileged observations
        base_lin_vel_world = ObsTerm(func=mdp.root_lin_vel_w)
        base_ang_vel_world = ObsTerm(func=mdp.root_ang_vel_w)

        arm_joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": UR_ARM_JOINT_CFG}
        )

        ee_pose_error = ObsTerm(
            func=mdp.ee_command_error_obs,
            params={
                "command_name": "ee_traj",
                "std_pos": 0.3,
                "std_rot": 0.5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Unified SE(3) tracking reward: exp(-||T_cmd ⊖ T||)
    ee_traj_tracking = RewTerm(
        func=mdp.ee_traj_pose_error_exp,
        weight=20.0,
        params={
            "command_name": "ee_traj",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="robotiq_85_base_link"
            ),
            "std_pos": 0.2,
            "std_rot": 0.3,
        },
    )

    # Action Regularization
    arm_action_rate = RewTerm(
        func=mdp.arm_action_rate_l2,
        weight=-0.015,
    )

    base_action_rate = RewTerm(
        func=mdp.base_action_rate_l2,
        weight=-0.02,
    )

    # Base Stability
    base_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    tipped_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.3}
    )

    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="ur.*_link"
            ),
            "threshold": 1.0,
        },
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
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
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


@configclass
class MMEeTrajTrackingEnvCfg(MobileManipulatorBaseEnvCfg):
    """RL environment for mobile manipulator EE trajectory tracking.

    Uses Lie group SE(3) interpolation with exponential reward.
    """

    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class MMEeTrajTrackingEnvCfg_PLAY(MMEeTrajTrackingEnvCfg):
    """Play/Evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()

        # ===== Scene Settings =====
        self.scene.num_envs = 4  # 减少环境数量，避免渲染卡顿
        self.scene.env_spacing = 3.5  # 增大间距，便于观察

        # ===== Episode Settings =====
        self.episode_length_s = 30.0  # 延长 episode，便于观察

        # ===== Observation Settings =====
        self.observations.policy.enable_corruption = False  # 关闭噪声

        # ===== Command Settings =====
        self.commands.ee_traj.trajectory_time_range = (0.5, 1)

        # Change workspace constraints (optional)
        # self.commands.ee_traj.arm_length_max = 0.6

        # ==== Visualization settings ====
        self.commands.ee_traj.debug_vis = True
        self.commands.ee_traj.vis_update_interval = 30  # 降低更新频率
        self.commands.ee_traj.num_trajectory_samples = 10  # 减少可视化样本

        # 保持姿态采样
        self.commands.ee_traj.sample_uniform_orientation = True
