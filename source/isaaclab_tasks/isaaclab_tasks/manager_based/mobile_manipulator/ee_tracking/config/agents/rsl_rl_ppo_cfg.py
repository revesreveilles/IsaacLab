# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for mobile manipulator EE tracking task."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MobileManipulatorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for mobile manipulator.

    Uses asymmetric actor-critic:
    - Actor: policy observations (deployable).
      Dim varies with history and sensor config.
    - Critic: policy + privileged observations.

    Note: RslRlVecEnvWrapper automatically detects observation groups
    from env.observation_manager and enables asymmetric training.
    Observation dimensions are determined dynamically.
    """

    # ========== Runner Settings ==========
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "mm_ee_tracking"
    empirical_normalization = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }

    # ========== Policy Network ==========
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,

        # Actor network
        actor_hidden_dims=[512, 256, 128],
        activation="elu",

        # Critic network
        critic_hidden_dims=[512, 256, 128],
    )

    # ========== PPO Algorithm ==========
    algorithm = RslRlPpoAlgorithmCfg(
        # PPO hyperparameters
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Learning rate
        learning_rate=5e-4,
        schedule="adaptive",
        # GAE parameters
        gamma=0.99,
        lam=0.95,
        # KL divergence
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
