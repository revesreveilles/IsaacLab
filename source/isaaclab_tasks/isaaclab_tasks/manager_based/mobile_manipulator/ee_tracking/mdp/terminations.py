from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_tolerance: float = 0.1,
    ori_tolerance: float | None = None,
) -> torch.Tensor:
    """Terminate when the end-effector reaches the target pose.

    Uses the position_error (and optionally orientation_error) metrics 
    already computed by the pose command.

    Args:
        env: The environment.
        command_name: Name of the pose command term.
        pos_tolerance: Position distance threshold (meters).
        ori_tolerance: Orientation error threshold (radians). 
                      If None, only check position. Defaults to None.

    Returns:
        A boolean tensor [num_envs] indicating which envs reached the goal.
    """
    # Get the command term
    command_term = env.command_manager.get_term(command_name)
    
    # Check position error
    pos_error = command_term.metrics["position_error"]
    position_reached = pos_error < pos_tolerance
    
    # If orientation tracking is required
    if ori_tolerance is not None:
        ori_error = command_term.metrics["orientation_error"]
        orientation_reached = ori_error < ori_tolerance
        return position_reached & orientation_reached
    
    # Fixed: return the pre-computed variable
    return position_reached