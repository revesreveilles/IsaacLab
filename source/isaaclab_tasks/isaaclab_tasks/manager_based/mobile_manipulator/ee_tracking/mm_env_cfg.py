# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mobile manipulator environment configuration for teleoperation and testing.

NOTE: This config is for manual control and testing only.
For RL training, use mm_rl_env_cfg.py instead.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg,RayCasterCfg,patterns

import isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots import MOBILE_MANIPULATOR_CFG  # isort:skip

from .constants import (
    UR_ARM_JOINT_NAMES,
    UR_ARM_JOINT_CFG,
    WHEEL_PIVOT_JOINTS,
    WHEEL_DRIVE_JOINTS,
    PI,
)
##
# Scene definition
##


@configclass
class MobileManipulatorSceneCfg(InteractiveSceneCfg):
    """Scene configuration for mobile manipulator."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Mobile manipulator robot
    robot: ArticulationCfg = MOBILE_MANIPULATOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # Contact sensors for collision detection
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_forward_link",
        update_period=0.1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),   # 垂直视场角
            horizontal_fov_range=(-180.0, 180.0),   # 水平 360 度
            horizontal_res=1.0,     # 水平分辨率 1 度
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        ),
    )


##
# Common MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for mobile manipulator."""

    # 4WD4WS base velocity control
    base_4ws = mdp.FourWheelFourSteerActionCfg(
        asset_name="robot",
        # ========== Joint order MUST be [FL, FR, BL, BR] ==========
        pivot_joint_names=WHEEL_PIVOT_JOINTS,
        drive_joint_names=WHEEL_DRIVE_JOINTS,
        use_world_frame=True,
        scale=(1.0, 1.0, 1.0),
        offset=(0.0, 0.0, 0.0),
        wheelbase=0.6,
        track_width=0.6,
        wheel_radii=(0.1, 0.1, 0.1, 0.1),
        max_linear_velocity=2.0,
        max_angular_velocity=4.0,
        max_steering_angle=1.57,
    )

    # UR3 arm joint position control
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        scale=0.3,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Base observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (deployable)."""

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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations with privileged info."""

        # Policy observations (clean, no noise)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None, history_length =10)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None, history_length =10)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": UR_ARM_JOINT_CFG}, noise=None, history_length=10)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": UR_ARM_JOINT_CFG}, noise=None, history_length=10)
        actions = ObsTerm(func=mdp.last_action, history_length=10)

        # Privileged observations (no history)
        base_lin_vel_world = ObsTerm(func=mdp.root_lin_vel_w)
        arm_joint_torques = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": UR_ARM_JOINT_CFG})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Common event terms for resets."""

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
                "yaw": (-1.0, 1.0),
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


##
# Base Environment Configuration
##

@configclass
class MobileManipulatorBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for mobile manipulator RL environments.
    
    Provides common scene, simulation, observations, actions, and events.
    Task-specific configs should inherit and add commands, rewards, terminations, curriculum.
    """

    # Scene
    scene: MobileManipulatorSceneCfg = MobileManipulatorSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    # Basic MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization - common settings."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # Simulation settings
        self.sim.dt = 0.005  # 200Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15