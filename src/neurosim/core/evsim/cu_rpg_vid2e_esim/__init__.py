"""
Code taken from https://github.com/uzh-rpg/rpg_vid2e

This module provides a wrapper around the VID2E event simulator to match
the common interface used by other event simulators in neurosim.
"""

import torch
from typing import Optional, Tuple
from .src.esim_torch import EventSimulator_torch


class EventSimulatorVID2E_ESIM:
    """Wrapper for VID2E event simulator with unified interface.

    This adapter wraps the RPG VID2E event simulator to match the interface
    expected by the neurosim event simulator factory.
    """

    def __init__(
        self,
        W: int = 640,
        H: int = 480,
        start_time: int = 0,
        first_image: Optional[torch.Tensor] = None,
        contrast_threshold_neg: float = 0.2,
        contrast_threshold_pos: float = 0.2,
        refractory_period_ns: int = 0,
        device: str = "cuda",
        **kwargs,
    ):
        """Initialize the VID2E event simulator.

        Args:
            W: Image width
            H: Image height
            start_time: Initial timestamp in microseconds
            first_image: Optional first image to initialize the simulator
            contrast_threshold_neg: Negative contrast threshold
            contrast_threshold_pos: Positive contrast threshold
            refractory_period_ns: Refractory period in nanoseconds
            device: Device to use (cuda or cpu)
        """
        self.W = W
        self.H = H
        self.device = device
        self.last_time = int(start_time)

        # Create the underlying VID2E simulator
        self._sim = EventSimulator_torch(
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            refractory_period_ns=refractory_period_ns,
        )

        # Initialize with first image if provided
        if first_image is not None:
            self._init_with_image(first_image, start_time)

    def _init_with_image(self, image: torch.Tensor, time: int) -> None:
        """Initialize the simulator with the first image."""
        if not image.is_cuda:
            image = image.to(self.device)

        # VID2E expects log images and int64 timestamps
        log_image = torch.log(image.float() + 1e-6)
        timestamp = torch.tensor([time], dtype=torch.int64, device=self.device)

        # Call forward to initialize
        self._sim.forward(log_image, timestamp)
        self.last_time = time

        print(
            f"[evsim-vid2e] Initialized event camera sim with sensor size: {image.shape}"
        )

    def image_callback(
        self, new_image: torch.Tensor, new_time: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process a new image and generate events.

        Args:
            new_image: The new image to process (H, W), values in [0, 1]
            new_time: Timestamp in microseconds

        Returns:
            Tuple of (x, y, t, p) event tensors or None if no events
        """
        if not new_image.is_cuda:
            new_image = new_image.to(self.device)

        # VID2E expects log images and int64 timestamps
        log_image = torch.log(new_image.float() + 1e-6)
        timestamp = torch.tensor([new_time], dtype=torch.int64, device=self.device)

        # Call the underlying simulator
        result = self._sim.forward(log_image, timestamp)

        self.last_time = new_time

        if result is None:
            return None

        # Convert from dict format to tuple format (x, y, t, p)
        x = result["x"].to(torch.uint16)
        y = result["y"].to(torch.uint16)
        t = result["t"].to(torch.uint64)
        p = result["p"].to(torch.uint8)

        return (x, y, t, p)

    def reset(self, first_image: Optional[torch.Tensor] = None) -> None:
        """Reset the simulator state."""
        self._sim.reset()
        if first_image is not None:
            self._init_with_image(first_image, self.last_time)

    def get_performance_info(self) -> dict:
        """Get performance-related information."""
        return {
            "backend": "vid2e",
            "device": self.device,
            "contrast_threshold_neg": self._sim.contrast_threshold_neg,
            "contrast_threshold_pos": self._sim.contrast_threshold_pos,
            "refractory_period_ns": self._sim.refractory_period_ns,
        }
