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

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import CylinderCfg, CuboidCfg
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.terrain.terrain_cfg import MobileManipulatorTerrainCfg

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
    NUM_DYN_CUBOIDS,
    NUM_DYN_CYLINDERS,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    CUBOID_SIZES,
    CYLINDER_SIZES,
    _CUBOID_COLORS,
    _CYLINDER_COLORS,
)
##
# Scene definition
##


@configclass
class MobileManipulatorSceneCfg(InteractiveSceneCfg):
    """Scene configuration for mobile manipulator."""

    # Ground plane (used when terrain is disabled)
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(),
    # )

    # Terrain (curriculum-based, replaces ground plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MobileManipulatorTerrainCfg(),
        max_init_terrain_level=0,  # 单块地形，所有机器人在同一块
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # ── Dynamic obstacles ──
    # Pre-allocated as two collections.
    # Uses constants from constants.py:
    #   NUM_DYN_CUBOIDS cuboids + NUM_DYN_CYLINDERS cylinders
    # All start hidden at z=-100, activated by curriculum.
    dynamic_cuboids = RigidObjectCollectionCfg(
        rigid_objects={
            DYN_CUBOID_NAMES[i]: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/DynCuboid{i}",
                spawn=CuboidCfg(
                    size=CUBOID_SIZES[i],
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_CUBOID_COLORS[i % len(_CUBOID_COLORS)]
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,
                        solver_position_iteration_count=0,
                        solver_velocity_iteration_count=0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100.0)),
            )
            for i in range(NUM_DYN_CUBOIDS)
        }
    )

    dynamic_cylinders = RigidObjectCollectionCfg(
        rigid_objects={
            DYN_CYLINDER_NAMES[i]: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/DynCylinder{i}",
                spawn=CylinderCfg(
                    radius=CYLINDER_SIZES[i][0] / 2.0,
                    height=CYLINDER_SIZES[i][2],
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_CYLINDER_COLORS[i % len(_CYLINDER_COLORS)]
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,
                        solver_position_iteration_count=0,
                        solver_velocity_iteration_count=0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100.0)),
            )
            for i in range(NUM_DYN_CYLINDERS)
        }
    )

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_forward_link",
        update_period=0.1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=4,
            vertical_fov_range=(-10.0, 20.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10,
        ),
        max_distance=5.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Mobile manipulator robot
    robot: ArticulationCfg = MOBILE_MANIPULATOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # Contact sensors for collision detection (arm links only)
    # Only track ur_* links since rewards only use body_names="ur.*_link"
    # Avoids unnecessary PhysX contact reporting for base/wheel bodies
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur_.*",
        history_length=3,
        track_air_time=False,
    )
    base_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=3,
        track_air_time=False,
    )
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        ),
    )

    def __post_init__(self):
        """Scene post-init hook."""
        pass


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
        scale={
            ".*shoulder_pan_joint": 2.1,
            ".*shoulder_lift_joint": 1.0,
            ".*elbow_joint": 0.8,
            ".*wrist_1_joint": 0.4,
            ".*wrist_2_joint": 0.4,
            ".*wrist_3_joint": 0.2,
        },
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

    reset_dynamic_obstacles = EventTerm(
        func=mdp.reset_dynamic_obstacles_navrl_style,
        mode="reset",
        params={
            "vel_range": (0.5, 2.0),
            "local_range": (3.0, 3.0, 1.0),
            "write_to_sim": False,
        },
    )

    # Kinematic obstacle position integration (interval event)
    # Goal-based random walk: obstacles move toward sampled goals,
    # resample velocity magnitude periodically,
    # and pick new goals when reaching the current one.
    # Direction is recomputed every step (pos → goal).
    step_kinematic_obstacles = EventTerm(
        func=mdp.step_kinematic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every step
        params={
            "local_range": (3.0, 3.0, 1.0),
            "goal_reach_threshold": 0.5,
            "vel_resample_interval": 2.0,
            "vel_range": (0.5, 2.0),
            "cuboid_hover_range": (0.5, 1.0),
            "goal_check_interval_s": 0.10,
            "goal_timeout_s": 4.0,
            "write_to_sim": False,
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
        num_envs=16384,
        env_spacing=2.5,
    )

    # Basic MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization - common settings."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 40.0
        # Simulation settings
        self.sim.dt = 0.008  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**27

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.max_depenetration_velocity = 10.0
