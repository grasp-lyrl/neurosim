"""
Unit tests for optical flow computation.

Tests the OpticalFlowComputer with synthetic depth maps and known camera
transformations to verify correctness of the flow computation.
"""

import torch
import pytest
import numpy as np
import quaternion as quat_lib

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurosim.core.visual_backend.optical_flow import OpticalFlowComputer


class TestOpticalFlowComputer:
    """Test suite for optical flow computation."""

    @pytest.fixture
    def flow_computer(self):
        """Create a basic optical flow computer for testing."""
        return OpticalFlowComputer(
            width=640,
            height=480,
            hfov=90.0,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    def test_initialization(self, flow_computer):
        """Test that optical flow computer initializes correctly."""
        assert flow_computer.width == 640
        assert flow_computer.height == 480
        assert flow_computer.fx > 0
        assert flow_computer.fy > 0
        assert flow_computer._prev_R_cam is None
        assert flow_computer._prev_t_cam is None

        # Check that all arrays are float32
        assert flow_computer._p_local.dtype == np.float32
        assert flow_computer._R_local.dtype == np.float32

    def test_first_call_returns_zeros(self, flow_computer):
        """Test that first call returns zero flow."""
        depth = torch.ones((480, 640), device=flow_computer.device)
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)  # Identity rotation

        flow = flow_computer.compute_flow(depth, agent_pos, agent_rot)

        assert flow.shape == (480, 640, 2)
        assert torch.allclose(flow, torch.zeros_like(flow))
        assert flow_computer._prev_R_cam is not None
        assert flow_computer._prev_t_cam is not None

    def test_no_motion_gives_zero_flow(self, flow_computer):
        """Test that no camera motion produces zero flow."""
        depth = torch.ones((480, 640), device=flow_computer.device)
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        # First call (initializes state)
        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Second call with same pose
        flow = flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Relax tolerance for float32 precision
        assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-4)

    def test_pure_translation_forward(self, flow_computer):
        """Test pure forward translation produces outward radial flow."""
        device = flow_computer.device
        depth = torch.ones((480, 640), device=device) * 5.0  # 5m depth

        # Start at origin
        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        # Initialize
        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Move forward by 0.1m (in -Z direction in camera frame)
        # In Habitat/OpenGL, camera looks along -Z, so forward is -Z
        agent_pos_new = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        flow = flow_computer.compute_flow(depth, agent_pos_new, agent_rot)

        # For forward motion, flow should be outward from center
        # Center pixel should have near-zero flow
        center_flow = flow[240, 320]
        assert torch.allclose(center_flow, torch.zeros(2, device=device), atol=1e-3)

        # Top-left corner should have negative flow (moving away from center)
        top_left_flow = flow[10, 10]
        assert top_left_flow[0] < -0.5  # u direction
        assert top_left_flow[1] < -0.5  # v direction

        # Bottom-right corner should have positive flow
        bottom_right_flow = flow[470, 630]
        assert bottom_right_flow[0] > 0.5  # u direction
        assert bottom_right_flow[1] > 0.5  # v direction

    def test_pure_translation_sideways(self, flow_computer):
        """Test pure sideways translation produces uniform horizontal flow."""
        device = flow_computer.device
        depth = torch.ones((480, 640), device=device) * 5.0

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Move right by 0.1m (+X direction)
        agent_pos_new = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        flow = flow_computer.compute_flow(depth, agent_pos_new, agent_rot)

        # For sideways motion, flow should be entirely horizontal (zero vertical component)
        assert flow[..., 1].abs().max() < 1e-3  # Vertical flow should be zero

        # Horizontal flow should be significant and negative everywhere (leftward)
        assert flow[..., 0].mean() < -1.0  # Mean horizontal flow is negative
        assert (flow[..., 0] < 0).all()  # All pixels have negative horizontal flow

    def test_pure_rotation_yaw(self, flow_computer):
        """Test pure yaw rotation produces flow independent of depth."""
        device = flow_computer.device

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        # Test with two different depths
        depth_near = torch.ones((480, 640), device=device) * 2.0
        depth_far = torch.ones((480, 640), device=device) * 10.0

        # Rotate by 10 degrees around Y axis (yaw)
        angle_rad = np.deg2rad(10.0)
        agent_rot_new = quat_lib.from_rotation_vector(np.array([0, angle_rad, 0]))

        # Compute flow for near depth
        flow_computer._prev_R_cam = None
        flow_computer._prev_t_cam = None
        flow_computer.compute_flow(depth_near, agent_pos, agent_rot)
        flow_near = flow_computer.compute_flow(depth_near, agent_pos, agent_rot_new)

        # Compute flow for far depth
        flow_computer._prev_R_cam = None
        flow_computer._prev_t_cam = None
        flow_computer.compute_flow(depth_far, agent_pos, agent_rot)
        flow_far = flow_computer.compute_flow(depth_far, agent_pos, agent_rot_new)

        # For pure rotation, flow should be INDEPENDENT of depth
        # Flow fields should be nearly identical
        flow_diff = (flow_near - flow_far).abs()
        assert flow_diff.max() < 1.0  # Flows should be almost identical

        # For yaw rotation, flow magnitude should be higher at left/right edges than center
        # (pixels farther from rotation axis have larger flow)
        center_flow_mag = torch.norm(flow_near[240, 320], dim=0)
        left_edge_flow_mag = torch.norm(flow_near[240, 10], dim=0)
        right_edge_flow_mag = torch.norm(flow_near[240, 630], dim=0)
        assert left_edge_flow_mag > center_flow_mag
        assert right_edge_flow_mag > center_flow_mag

    def test_pure_rotation_roll(self, flow_computer):
        """Test pure roll rotation produces flow independent of depth."""
        device = flow_computer.device

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        # Test with two different depths
        depth_near = torch.ones((480, 640), device=device) * 2.0
        depth_far = torch.ones((480, 640), device=device) * 10.0

        # Rotate by 10 degrees around Z axis (roll)
        angle_rad = np.deg2rad(10.0)
        agent_rot_new = quat_lib.from_rotation_vector(np.array([0, 0, angle_rad]))

        # Compute flow for near depth
        flow_computer._prev_R_cam = None
        flow_computer._prev_t_cam = None
        flow_computer.compute_flow(depth_near, agent_pos, agent_rot)
        flow_near = flow_computer.compute_flow(depth_near, agent_pos, agent_rot_new)

        # Compute flow for far depth
        flow_computer._prev_R_cam = None
        flow_computer._prev_t_cam = None
        flow_computer.compute_flow(depth_far, agent_pos, agent_rot)
        flow_far = flow_computer.compute_flow(depth_far, agent_pos, agent_rot_new)

        # For pure rotation, flow should be INDEPENDENT of depth
        flow_diff = (flow_near - flow_far).abs()
        assert flow_diff.max() < 1e-3  # Flows should be almost identical

        # For pure roll, the center pixel should have zero flow
        center_flow = flow_near[240, 320]
        assert torch.norm(center_flow, dim=0) < 1e-3

        # For roll, both horizontal and vertical components should be present
        assert flow_near[..., 0].abs().mean() > 5.0
        assert flow_near[..., 1].abs().mean() > 5.0

    def test_varying_depth(self, flow_computer):
        """Test that varying depth produces correct flow magnitudes."""
        device = flow_computer.device

        # Create depth gradient: near at top, far at bottom
        depth = (
            torch.linspace(1.0, 10.0, 480, device=device).unsqueeze(1).expand(480, 640)
        )

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Move forward
        agent_pos_new = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        flow = flow_computer.compute_flow(depth, agent_pos_new, agent_rot)

        # Closer objects (top) should have larger flow magnitude
        top_flow_mag = torch.norm(flow[50, 320], dim=0)
        bottom_flow_mag = torch.norm(flow[430, 320], dim=0)
        assert top_flow_mag > bottom_flow_mag * 2

    def test_invalid_depth_handling(self, flow_computer):
        """Test that invalid/zero depth is handled correctly."""
        device = flow_computer.device

        # Create depth with some invalid pixels
        depth = torch.ones((480, 640), device=device) * 5.0
        depth[100:200, 200:300] = 0.0  # Invalid region

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        agent_pos_new = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        flow = flow_computer.compute_flow(depth, agent_pos_new, agent_rot)

        # Invalid region should have zero flow
        invalid_flow = flow[100:200, 200:300]
        assert torch.allclose(invalid_flow, torch.zeros_like(invalid_flow))

        # Valid region should have non-zero flow
        valid_flow = flow[300:400, 400:500]
        assert valid_flow.abs().mean() > 0.5

    def test_known_transformation_case(self):
        """Test with a known transformation and manually computed expected flow.

        Simplified case: flat plane at z=-5, camera moves right by 0.1m.
        Center pixel (u=320, v=240) should map to 3D point (0, 0, -5).
        After right translation, point is at (-0.1, 0, -5) in new camera frame.
        Re-projection: u' = fx * (-0.1) / 5 + cx ≈ 320 - 12.8 = 307.2
        Flow at center: du ≈ -12.8 pixels
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        width, height = 640, 480
        hfov = 90.0

        flow_computer = OpticalFlowComputer(
            width=width, height=height, hfov=hfov, device=device
        )

        # Flat depth
        depth = torch.ones((height, width), device=device) * 5.0

        agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        agent_rot = np.quaternion(1, 0, 0, 0)

        flow_computer.compute_flow(depth, agent_pos, agent_rot)

        # Move right by 0.1m
        agent_pos_new = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        flow = flow_computer.compute_flow(depth, agent_pos_new, agent_rot)

        # Expected flow at center
        fx = width / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
        expected_du = -fx * 0.1 / 5.0  # ≈ -12.8 pixels
        expected_dv = 0.0

        center_flow = flow[240, 320].cpu().numpy()

        # Allow 10% tolerance
        assert np.abs(center_flow[0] - expected_du) < np.abs(expected_du) * 0.1
        assert np.abs(center_flow[1] - expected_dv) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
