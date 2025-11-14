"""Configuration for a mobile manipulator with a UR arm and a 4WS/4WD base.

This asset assumes the base has:
- 4 steering pivot joints (position-controlled), named like ``*_pivot_joint`` in order: left_forward, right_forward, left_back, right_back.
- 4 wheel drive joints (velocity-controlled), named like ``*_drive_joint`` in the same order.

And the arm has 6 UR joints (position-controlled), named like ``ur_*_joint``.

Exported config:
- ``MOBILE_MANIPULATOR_CFG``

Notes
- You can use the FourWheelSteerAdapter to command the base with [vx, vy, wz].
- Gains are conservative defaults; tune stiffness/damping to your robot.
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

MOBILE_MANIPULATOR_CFG = ArticulationCfg(
    # Spawn from a USD with the expected joint names.
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/revesreveilles/Isaac_lab_ws/data/Robots/mobile_manipulator/mm_ur3.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        activate_contact_sensors=False,
    ),
    # Initial joint states
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            # base steering pivots (position control)
            "left_forward_pivot_joint": 0.0,
            "right_forward_pivot_joint": 0.0,
            "left_back_pivot_joint": 0.0,
            "right_back_pivot_joint": 0.0,
            # UR arm (position control)
            "ur_shoulder_pan_joint": 1.712,
            "ur_shoulder_lift_joint": -1.712,
            "ur_elbow_joint": 0.0,
            "ur_wrist_1_joint": 0.0,
            "ur_wrist_2_joint": 0.0,
            "ur_wrist_3_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Group actuators by control mode. The action application decides whether to set pos or vel targets.
    actuators={
        # Base steering pivots: position-controlled PD with per-joint tuning
        # Drive type: force
        "left_forward_pivot": ImplicitActuatorCfg(
            joint_names_expr=["left_forward_pivot_joint"],
            effort_limit_sim=5.0,
            stiffness=1377.16003,
            damping=0.55086,
        ),
        "left_back_pivot": ImplicitActuatorCfg(
            joint_names_expr=["left_back_pivot_joint"],
            effort_limit_sim=5.0,
            stiffness=253.95029,
            damping=0.10158,
        ),
        "right_forward_pivot": ImplicitActuatorCfg(
            joint_names_expr=["right_forward_pivot_joint"],
            effort_limit_sim=5.0,
            stiffness=1753.51636,
            damping=0.70141,
        ),
        "right_back_pivot": ImplicitActuatorCfg(
            joint_names_expr=["right_back_pivot_joint"],
            effort_limit_sim=5.0,
            stiffness=717.0954,
            damping=0.28684,
        ),
        # Base wheel drives: velocity-controlled; keep stiffness 0 to avoid position pulling.
        "base_drives": ImplicitActuatorCfg(
            joint_names_expr=[r".*_drive_joint$"],
            effort_limit_sim=50000.0,
            stiffness=1.0,
            damping=1000.0,         # viscous term for velocity servo behavior
        ),
        # UR arm: position-controlled PD with per-joint tuning
        # Drive type: force
        "ur_shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["ur_shoulder_pan_joint"],
            effort_limit_sim=3360.0,
            stiffness=205.97929,
            damping=3.29567,
        ),
        "ur_shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["ur_shoulder_lift_joint"],
            effort_limit_sim=3360.0,
            stiffness=205.97929,
            damping=0.75645,
        ),
        "ur_elbow": ImplicitActuatorCfg(
            joint_names_expr=["ur_elbow_joint"],
            effort_limit_sim=1680.0,
            stiffness=205.97929,
            damping=3.29567,
        ),
        "ur_wrist_1": ImplicitActuatorCfg(
            joint_names_expr=["ur_wrist_1_joint"],
            effort_limit_sim=720.0,
            stiffness=104.67349,
            damping=1.67478,
        ),
        "ur_wrist_2": ImplicitActuatorCfg(
            joint_names_expr=["ur_wrist_2_joint"],
            effort_limit_sim=720.0,
            stiffness=104.67349,
            damping=0.16116,
        ),
        "ur_wrist_3": ImplicitActuatorCfg(
            joint_names_expr=["ur_wrist_3_joint"],
            effort_limit_sim=720.0,
            stiffness=104.67349,
            damping=1.67478,
        ),
    },
)
"""Mobile manipulator with UR arm and a 4WS/4WD base using implicit actuator models.

Control semantics:
- Base pivots: position targets (steering angles)
- Base drives: velocity targets (wheel angular speed)
- UR3 arm: position targets

Make sure your USD uses the expected joint naming patterns. If names differ, update joint_names_expr accordingly.
"""
