# Copyright (c) 2025, The Nav-Suite Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.utils.math import quat_apply


class RobotCollisionBubbles:
    """
    Calculates collision bubbles for a mobile manipulator robot.
    """
    def __init__(self, env):
        """
        Args:
            env: ManagerBasedRLEnv environment instance, used to access the robot (Articulation).
        """
        # 1. 获取机器人实例
        # 注意：这里假设在 env_cfg 中机器人的名称为 "robot"
        try:
            self.robot = env.scene["robot"]
        except KeyError:
            raise KeyError("RobotCollisionBubbles: Could not find asset 'robot' in env.scene. Please check your Asset configuration.")

        self.device = self.robot.device
        self.num_envs = self.robot.num_instances

        # --- Configuration section ---
        # Format: 'link_name': [[x, y, z, radius], ...]
        # Coordinate frame: Relative to the link frame
        self.bubbles_cfg = {
            "base_link": [
                [0.28, 0.15, 0.0, 0.3],
                [0.0, 0.15, 0.0, 0.3],
                [-0.28, 0.15, 0.0, 0.3],
                [0.28, -0.15, 0.0, 0.3],
                [0.0, -0.15, 0.0, 0.3],
                [-0.28, -0.15, 0.0, 0.3],
            ],
            "ur_shoulder_link": [
                [0.0, 0.0, 0.0, 0.06],
                # [0.0, 0.0, -0.11, 0.06]
            ],
            "ur_upper_arm_link": [
                # [0.0, 0.0, 0.12, 0.05],
                [-0.0, 0.0, 0.12, 0.06],
                [-0.11, 0.0, 0.12, 0.06],
                [-0.22, 0.0, 0.12, 0.06]
            ],
            "ur_forearm_link": [
                [0.0, 0.0, 0.02, 0.06],
                [-0.11, 0.0, 0.02, 0.06],
                [-0.22, 0.0, 0.02, 0.06]
            ],
            "ur_wrist_1_link": [
                [0.0, 0.0, 0.0, 0.050]
            ],
            "ur_wrist_2_link": [
                [0.0, -0.02, 0.01, 0.050]
            ],
            "ur_wrist_3_link": [
                [0.0, 0.0, 0.0, 0.05],
                [0.0, 0.0, 0.10, 0.065]
                # [0.0, 0.0, 0.04, 0.12]
            ],
            # 如果你有抓夹，可以在这里添加，例如 "robotiq_85_base_link"
        }

        # --- 预处理：构建索引映射 ---
        all_body_names = self.robot.body_names

        self.body_indices = []
        self.local_positions = []
        self.radii = []

        for link_name, spheres in self.bubbles_cfg.items():
            if link_name in all_body_names:
                body_idx = all_body_names.index(link_name)
                for s in spheres:
                    self.body_indices.append(body_idx)
                    self.local_positions.append(s[:3])
                    self.radii.append(s[3])
            else:
                print(f"[Warning] RobotCollisionBubbles: Link '{link_name}' not found in robot body names! Available names: {all_body_names[:5]}...")

        if len(self.body_indices) == 0:
            raise ValueError("RobotCollisionBubbles: No valid bubbles were configured! Check your link names.")

        # 转为 Tensor 并移动到 GPU
        self.body_indices = torch.tensor(self.body_indices, device=self.device, dtype=torch.long)
        self.local_positions = torch.tensor(self.local_positions, device=self.device, dtype=torch.float32)
        self.radii = torch.tensor(self.radii, device=self.device, dtype=torch.float32)

        self.num_bubbles = len(self.radii)

        data_cls = type(self.robot.data)
        if (
            hasattr(data_cls, "body_com_vel_w")
            and hasattr(data_cls, "body_com_pos_b")
            and hasattr(data_cls, "body_link_quat_w")
        ):
            self._vel_source = "body_com_vel"
        elif hasattr(data_cls, "body_link_lin_vel_w") and hasattr(data_cls, "body_link_ang_vel_w"):
            self._vel_source = "body_link_vel"
        elif hasattr(data_cls, "body_link_state_w"):
            self._vel_source = "body_link_state"
        elif hasattr(data_cls, "body_state_w"):
            self._vel_source = "body_state"
        else:
            self._vel_source = "finite_difference"

        # Step-based cache for get_world_spheres
        self._env = env
        self._ws_cached_step = -1
        self._ws_cached_centers = None
        self._ws_cached_offsets = None
        self._ws_cached_link_quat = None
        self._wv_cached_step = -1
        self._wv_cached_velocities = None
        self._fd_prev_step = -1
        self._fd_prev_centers = None

    def get_world_spheres(self):
        """
        计算当前时刻所有碰撞球的世界坐标。

        Step-based 缓存: 同一步内 obs + reward 共享结果,
        避免重复 quat_apply 计算。

        Returns:
            centers_w (torch.Tensor): [num_envs, num_bubbles, 3]
            radii (torch.Tensor): [num_bubbles]
        """
        step = self._env.common_step_counter
        if step == self._ws_cached_step and self._ws_cached_centers is not None:
            return self._ws_cached_centers, self.radii
        # 1. 获取对应 Link 的世界坐标姿态
        # self.robot.data.body_pos_w: [num_envs, num_bodies, 3]
        # self.robot.data.body_quat_w: [num_envs, num_bodies, 4]

        # 即使在 reset 后，robot.data 也会由 sim 自动更新，因此这里总是能取到最新的一帧
        link_pos_w = self.robot.data.body_pos_w[:, self.body_indices, :]
        link_quat_w = self.robot.data.body_quat_w[:, self.body_indices, :]
        link_quat_w = link_quat_w / link_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # 2. 将局部偏移转换到世界坐标系
        # 扩展 local_positions 以匹配 batch size: [num_envs, num_bubbles, 3]
        local_pos_batch = self.local_positions.unsqueeze(0).expand(self.num_envs, -1, -1)

        # 应用旋转: q * v
        offset_w = quat_apply(link_quat_w, local_pos_batch)

        # 加上平移: P_world = P_link + R_link * P_local
        centers_w = link_pos_w + offset_w

        # 缓存结果
        self._ws_cached_step = step
        self._ws_cached_centers = centers_w
        self._ws_cached_offsets = offset_w
        self._ws_cached_link_quat = link_quat_w

        return centers_w, self.radii

    def get_world_sphere_velocities(self):
        """
        计算当前时刻所有碰撞球中心的世界坐标速度。

        首选 COM velocity + omega × COM-to-sphere offset 的点速度，
        避免触发 IsaacLab 对全 body 的 link-velocity 懒转换。
        若当前 IsaacLab 版本缺少 link velocity 属性，则依次回退到
        body_link_state_w、body_state_w，最后使用 collision sphere center
        的有限差分。

        Returns:
            sphere_vel_w (torch.Tensor): [num_envs, num_bubbles, 3]
        """
        step = self._env.common_step_counter
        if step == self._wv_cached_step and self._wv_cached_velocities is not None:
            return self._wv_cached_velocities

        if step == self._ws_cached_step and self._ws_cached_offsets is not None:
            offset_w = self._ws_cached_offsets
        else:
            link_quat_w = self.robot.data.body_quat_w[:, self.body_indices, :]
            link_quat_w = link_quat_w / link_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            local_pos_batch = self.local_positions.unsqueeze(0).expand(self.num_envs, -1, -1)
            offset_w = quat_apply(link_quat_w, local_pos_batch)

        link_lin_vel_w = None
        link_ang_vel_w = None
        sphere_offset_w = offset_w

        if self._vel_source == "body_com_vel":
            body_com_vel_w = self.robot.data.body_com_vel_w[:, self.body_indices, :]
            link_lin_vel_w = body_com_vel_w[..., :3]
            link_ang_vel_w = body_com_vel_w[..., 3:6]
            if step == self._ws_cached_step and self._ws_cached_link_quat is not None:
                link_quat_w = self._ws_cached_link_quat
            else:
                link_quat_w = self.robot.data.body_quat_w[:, self.body_indices, :]
                link_quat_w = link_quat_w / link_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            local_pos_batch = self.local_positions.unsqueeze(0).expand(self.num_envs, -1, -1)
            com_pos_b = self.robot.data.body_com_pos_b[:, self.body_indices, :]
            sphere_offset_w = quat_apply(link_quat_w, local_pos_batch - com_pos_b)
        elif self._vel_source == "body_link_vel":
            link_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.body_indices, :]
            link_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self.body_indices, :]
        elif self._vel_source == "body_link_state":
            link_state_w = self.robot.data.body_link_state_w[:, self.body_indices, :]
            link_lin_vel_w = link_state_w[..., 7:10]
            link_ang_vel_w = link_state_w[..., 10:13]
        elif self._vel_source == "body_state":
            body_state_w = self.robot.data.body_state_w[:, self.body_indices, :]
            link_lin_vel_w = body_state_w[..., 7:10]
            link_ang_vel_w = body_state_w[..., 10:13]

        if link_lin_vel_w is not None and link_ang_vel_w is not None:
            sphere_vel_w = link_lin_vel_w + torch.cross(link_ang_vel_w, sphere_offset_w, dim=-1)
        else:
            centers_w, _ = self.get_world_spheres()
            if self._fd_prev_centers is None or self._fd_prev_step < 0 or step <= self._fd_prev_step:
                sphere_vel_w = torch.zeros_like(centers_w)
            else:
                step_dt = float(getattr(self._env, "step_dt", getattr(self._env, "physics_dt", 1.0)))
                dt = max((step - self._fd_prev_step) * step_dt, 1e-6)
                sphere_vel_w = (centers_w - self._fd_prev_centers) / dt
            self._fd_prev_step = step
            self._fd_prev_centers = centers_w.detach().clone()

        self._wv_cached_step = step
        self._wv_cached_velocities = sphere_vel_w
        return sphere_vel_w
