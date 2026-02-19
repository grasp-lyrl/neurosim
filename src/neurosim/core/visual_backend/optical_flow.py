"""
GPU-accelerated Optical Flow Computation for Neurosim.

Computes ground-truth optical flow from depth maps and known camera
ego-motion. Assumes a static scene with only camera/drone motion.

The approach:
    1. Back-project each pixel to 3D using the current depth map.
    2. Transform the 3D points from the current camera frame to the
       previous camera frame using the known relative camera motion.
    3. Re-project to the previous image to obtain per-pixel displacement.

Camera conventions follow Habitat-Sim / OpenGL:
    - Camera looks along -Z
    - +X right, +Y up
    - Depth is positive (planar depth from camera plane)
"""

import torch
import logging
import numpy as np
import quaternion as quat_lib

logger = logging.getLogger(__name__)


class OpticalFlowComputer:
    """Computes ground-truth optical flow from depth + ego-motion on GPU.

    Like event simulators, this class maintains internal state (previous camera pose)
    and computes flow incrementally as new observations arrive.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        hfov: Horizontal field of view in degrees.
        device: CUDA device string.
        sensor_local_pose: Optional tuple of (p_local, R_local) for sensor offset from agent.
    """

    def __init__(
        self,
        width: int,
        height: int,
        hfov: float,
        device: str = "cuda:0",
        sensor_local_pose: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.width = width
        self.height = height
        self.device = device

        # Compute camera intrinsics from horizontal field of view
        hfov_rad = hfov * np.pi / 180.0
        self.fx = width / (2.0 * np.tan(hfov_rad / 2.0))
        self.fy = self.fx  # Square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Pre-compute pixel coordinate grids on GPU
        v_coords, u_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Pre-compute normalized back-projection coordinates
        # (u - cx) / fx  and  (cy - v) / fy
        self._x_norm = ((u_coords - self.cx) / self.fx).reshape(-1)  # (N,)
        self._y_norm = ((self.cy - v_coords) / self.fy).reshape(-1)  # (N,)
        self._u_flat = u_coords.reshape(-1)  # (N,)
        self._v_flat = v_coords.reshape(-1)  # (N,)

        # State: sensor local pose and previous camera world pose
        if sensor_local_pose is not None:
            self._p_local, self._R_local = sensor_local_pose
        else:
            self._p_local = np.zeros(3, dtype=np.float32)
            self._R_local = np.eye(3, dtype=np.float32)

        self._prev_R_cam: np.ndarray | None = None
        self._prev_t_cam: np.ndarray | None = None

        logger.info(
            f"OpticalFlowComputer initialized: {width}x{height}, "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, device={device}"
        )

    @torch.no_grad()
    def compute(
        self,
        depth: torch.Tensor,
        R_rel: torch.Tensor,
        t_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optical flow from depth and relative camera motion.

        Computes backward flow: for each pixel in the current frame, where was
        that 3D point located in the previous frame's image.

        Math (OpenGL camera, -Z forward):
            1. Back-project pixel (u,v) with depth d:
               P = [(u-cx)/fx * d, (cy-v)/fy * d, -d]
            2. Transform to previous camera frame:
               P' = R_rel @ P + t_rel
            3. Re-project:
               u' = fx * P'_x / (-P'_z) + cx
               v' = cy - fy * P'_y / (-P'_z)
            4. Flow = [u' - u, v' - v]

        Args:
            depth: (H, W) depth tensor on GPU (planar depth, positive values).
            R_rel: (3, 3) rotation matrix from current to previous camera frame.
            t_rel: (3,) translation vector from current to previous camera frame.

        Returns:
            (H, W, 2) flow tensor on GPU. flow[..., 0] = du, flow[..., 1] = dv.
        """
        depth_flat = depth.reshape(-1)  # (N,)

        # Back-project to 3D in current camera frame
        # OpenGL: camera looks along -Z, so Z = -depth
        X = self._x_norm * depth_flat  # (N,)
        Y = self._y_norm * depth_flat  # (N,)
        Z = -depth_flat  # (N,)

        # Stack to [3, N]
        points = torch.stack([X, Y, Z], dim=0)  # (3, N)

        # Transform to previous camera frame: P_prev = R_rel @ P + t_rel
        points_prev = R_rel @ points + t_rel.unsqueeze(1)  # (3, N)

        # Depth in previous camera frame (positive = in front of camera)
        d_prev = -points_prev[2]  # (N,)

        # Mask invalid pixels: behind camera, too close, or invalid depth
        valid = (d_prev > 1e-6) & (depth_flat > 1e-6)

        # Safe division for re-projection
        d_prev_safe = torch.where(valid, d_prev, torch.ones_like(d_prev))

        # Re-project to previous image coordinates
        u_prev = self.fx * points_prev[0] / d_prev_safe + self.cx
        v_prev = self.cy - self.fy * points_prev[1] / d_prev_safe

        # Compute flow displacement
        flow_u = torch.where(valid, u_prev - self._u_flat, torch.zeros_like(u_prev))
        flow_v = torch.where(valid, v_prev - self._v_flat, torch.zeros_like(v_prev))

        # Reshape to (H, W, 2)
        flow = torch.stack(
            [
                flow_u.reshape(self.height, self.width),
                flow_v.reshape(self.height, self.width),
            ],
            dim=-1,
        )

        return flow

    def compute_flow(
        self,
        depth: torch.Tensor,
        agent_pos: np.ndarray,
        agent_rot: np.ndarray,
    ) -> torch.Tensor:
        """Compute optical flow from depth and current agent pose.

        This is the main entry point. Maintains internal state of previous camera pose.

        Args:
            depth: (H, W) depth tensor on GPU.
            agent_pos: Agent position in world frame (3,).
            agent_rot: Agent rotation as quaternion (np.quaternion or 4-element ndarray).

        Returns:
            (H, W, 2) flow tensor on GPU. Returns zeros on first call.
        """
        # Convert quaternion to rotation matrix if needed
        if isinstance(agent_rot, np.ndarray):
            agent_rot = np.quaternion(*agent_rot)  # Assume [w, x, y, z] format
        elif not isinstance(agent_rot, np.quaternion):
            raise ValueError("agent_rot must be a quaternion or 4-element ndarray")

        R_agent = quat_lib.as_rotation_matrix(agent_rot)

        R_cam = R_agent @ self._R_local
        t_cam = R_agent @ self._p_local + agent_pos

        # First call: store pose and return zeros
        if self._prev_R_cam is None:
            self._prev_R_cam = R_cam
            self._prev_t_cam = t_cam
            return torch.zeros(
                (self.height, self.width, 2),
                device=self.device,
                dtype=torch.float32,
            )

        # Compute relative transform: current camera frame -> previous camera frame
        # We want: P_prev_frame = R_rel @ P_curr_frame + t_rel
        # Where P_curr_frame = R_cam.T @ (P_world - t_cam)
        # And P_prev_frame = R_prev.T @ (P_world - t_prev)
        # Substituting: R_prev.T @ (P_world - t_prev) = R_rel @ R_cam.T @ (P_world - t_cam) + t_rel
        # Solving: R_rel = R_prev.T @ R_cam, t_rel = R_prev.T @ (t_cam - t_prev)
        R_rel = self._prev_R_cam.T @ R_cam
        t_rel = self._prev_R_cam.T @ (self._prev_t_cam - t_cam)

        R_rel_gpu = torch.from_numpy(R_rel.astype(np.float32)).to(self.device)
        t_rel_gpu = torch.from_numpy(t_rel.astype(np.float32)).to(self.device)

        flow = self.compute(depth, R_rel_gpu, t_rel_gpu)  # (H, W, 2)

        # Update previous pose
        self._prev_R_cam = R_cam
        self._prev_t_cam = t_cam

        return flow
