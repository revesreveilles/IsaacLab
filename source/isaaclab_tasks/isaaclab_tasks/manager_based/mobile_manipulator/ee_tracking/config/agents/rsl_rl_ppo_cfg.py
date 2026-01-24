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
    - Actor: 340-dim policy observations (deployable)
    - Critic: 350-dim (policy + privileged observations)

    Note: RslRlVecEnvWrapper automatically detects observation groups
    from env.observation_manager and enables asymmetric training.
    """

    # ========== Runner Settings ==========
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 1000
    experiment_name = "mm_ee_tracking"
    empirical_normalization = False

    # ========== Policy Network ==========
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,

        # Actor network (uses policy observations: 340 dim)
        actor_hidden_dims=[256, 128, 64],
        activation="elu",

        # Critic network (uses policy + privileged: 350 dim)
        critic_hidden_dims=[256, 256, 128],
    )

    # ========== PPO Algorithm ==========
    algorithm = RslRlPpoAlgorithmCfg(
        # PPO hyperparameters
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Learning rate
        learning_rate=1e-3,
        schedule="adaptive",
        # GAE parameters
        gamma=0.99,
        lam=0.95,
        # KL divergence
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def __post_init__(self):
    """Post initialization to set observation groups."""
    if not hasattr(self.policy, 'obs_groups') or self.policy.obs_groups is None:
        self.policy.obs_groups = {
            "policy": ["policy"],           # Actor用env的"policy"组
            "critic": ["policy", "critic"], # Critic用env的"policy"+"critic"组
        }