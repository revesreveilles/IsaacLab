# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Mobile manipulator EE tracking task registration.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Mobile manipulator: EE trajectory tracking (Training)
gym.register(
    id="Isaac-MM-UR3-EeTrajTracking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ee_traj_tracking_env_cfg:MMEeTrajTrackingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:MobileManipulatorPPORunnerCfg"
        ),
    },
)

# Mobile manipulator: EE trajectory tracking (Play/Evaluation)
gym.register(
    id="Isaac-MM-UR3-EeTrajTracking-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ee_traj_tracking_env_cfg:MMEeTrajTrackingEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:MobileManipulatorPPORunnerCfg"
        ),
    },
)
