"""
Event Simulator Factory and Registry.

This module provides a factory pattern for creating event camera simulators.
It supports four backends:
- cuda: Custom CUDA kernel (recommended, fastest)
- torch: PyTorch implementation (fallback, GPU-accelerated)
- airsim: AirSim/Numba implementation (CPU-based, slower)
- vid2e: RPG VID2E CUDA implementation from UZH

Usage:
    from neurosim.utils.evsim import create_event_simulator, EventSimulatorType

    # Create using enum
    sim = create_event_simulator(
        EventSimulatorType.CUDA,
        width=640,
        height=480,
        start_time=0,
    )

    # Or create using string
    sim = create_event_simulator(
        "cuda",
        width=640,
        height=480,
        start_time=0,
    )
"""

from enum import Enum
from typing import Union, Optional, Protocol, Any


class EventSimulatorType(Enum):
    """Supported event simulator backends."""

    AUTO = "auto"
    CUDA = "cuda"
    TORCH = "torch"
    AIRSIM = "airsim"
    VID2E = "vid2e"


class EventSimulatorProtocol(Protocol):
    """Protocol defining the interface for event simulators."""

    def image_callback(
        self, new_image: Any, new_time: int
    ) -> Optional[tuple[Any, ...]]:
        """Process a new image and generate events.

        Args:
            new_image: The new image to process (H, W)
            new_time: Timestamp in microseconds

        Returns:
            Tuple of event data (x, y, t, p) or None if no events
        """
        ...

    def reset(self, first_image: Optional[Any] = None) -> None:
        """Reset the simulator state."""
        ...


def _import_cuda_simulator():
    """Lazily import CUDA simulator."""
    try:
        from .cu_evsim import EventSimulatorCUDA

        return EventSimulatorCUDA
    except ImportError as e:
        raise ImportError(
            f"CUDA event simulator not available. "
            f"Please install cu_evsim with: cd src/neurosim/utils/cu_evsim && pip install -e . "
            f"Original error: {e}"
        )


def _import_torch_simulator():
    """Lazily import Torch simulator."""
    try:
        from .py_evsim import EventSimulatorTorch

        return EventSimulatorTorch
    except ImportError as e:
        raise ImportError(f"Torch event simulator not available. Original error: {e}")


def _import_airsim_simulator():
    """Lazily import AirSim simulator."""
    try:
        from .py_evsim import EventSimulatorAirsim

        return EventSimulatorAirsim
    except ImportError as e:
        raise ImportError(
            f"AirSim event simulator not available. "
            f"Please install numba: pip install numba "
            f"Original error: {e}"
        )


def _import_vid2e_simulator():
    """Lazily import VID2E (RPG) simulator."""
    try:
        from .cu_rpg_vid2e_esim import EventSimulatorVID2E_ESIM

        return EventSimulatorVID2E_ESIM
    except ImportError as e:
        raise ImportError(
            f"VID2E event simulator not available. "
            f"Please install esim_cuda with: cd src/neurosim/utils/evsim/cu_rpg_vid2e_esim && pip install -e . "
            f"Original error: {e}"
        )


def get_available_backends() -> list:
    """Get list of available event simulator backends.

    Returns:
        List of available EventSimulatorType values.
    """
    available = []

    try:
        _import_cuda_simulator()
        available.append(EventSimulatorType.CUDA)
    except ImportError:
        pass

    try:
        _import_torch_simulator()
        available.append(EventSimulatorType.TORCH)
    except ImportError:
        pass

    try:
        _import_airsim_simulator()
        available.append(EventSimulatorType.AIRSIM)
    except ImportError:
        pass

    try:
        _import_vid2e_simulator()
        available.append(EventSimulatorType.VID2E)
    except ImportError:
        pass

    return available


def get_best_available_backend() -> EventSimulatorType:
    """Get the best available backend (preference: CUDA > Torch > AirSim).

    Returns:
        The best available EventSimulatorType.

    Raises:
        RuntimeError: If no backends are available.
    """
    available = get_available_backends()

    if not available:
        raise RuntimeError(
            "No event simulator backends available. "
            "Please install at least one: cu_evsim (CUDA), torch, or numba (AirSim)."
        )

    # Priority order
    for backend in [
        EventSimulatorType.CUDA,
        EventSimulatorType.VID2E,
        EventSimulatorType.TORCH,
        EventSimulatorType.AIRSIM,
    ]:
        if backend in available:
            return backend

    return available[0]


def create_event_simulator(
    backend: Union[EventSimulatorType, str],
    width: int,
    height: int,
    start_time: int = 0,
    first_image: Optional[Any] = None,
    contrast_threshold_neg: float = 0.35,
    contrast_threshold_pos: float = 0.35,
    **kwargs,
) -> EventSimulatorProtocol:
    """Create an event simulator with the specified backend.

    Args:
        backend: The backend to use (EventSimulatorType or string: "cuda", "torch", "airsim")
        width: Image width
        height: Image height
        start_time: Initial timestamp in microseconds
        first_image: Optional first image to initialize the simulator
        contrast_threshold_neg: Negative contrast threshold (default: 0.35)
        contrast_threshold_pos: Positive contrast threshold (default: 0.35)
        **kwargs: Additional backend-specific arguments

    Returns:
        An event simulator instance.

    Raises:
        ValueError: If the backend is not recognized.
        ImportError: If the backend is not available.
    """
    # Convert string to enum if necessary
    if isinstance(backend, str):
        try:
            backend = EventSimulatorType(backend.lower())
        except ValueError:
            valid_backends = [b.value for b in EventSimulatorType]
            raise ValueError(
                f"Unknown backend: '{backend}'. Valid backends: {valid_backends}"
            )

    if backend == EventSimulatorType.AUTO:
        backend = get_best_available_backend()

    if backend == EventSimulatorType.CUDA:
        SimClass = _import_cuda_simulator()
        return SimClass(
            W=width,
            H=height,
            start_time=start_time,
            first_image=first_image,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            **kwargs,
        )

    elif backend == EventSimulatorType.TORCH:
        SimClass = _import_torch_simulator()
        return SimClass(
            W=width,
            H=height,
            start_time=start_time,
            first_image=first_image,
            **kwargs,
        )

    elif backend == EventSimulatorType.AIRSIM:
        SimClass = _import_airsim_simulator()
        return SimClass(
            W=width,
            H=height,
            first_image=first_image,
            first_time=start_time,
            **kwargs,
        )

    elif backend == EventSimulatorType.VID2E:
        SimClass = _import_vid2e_simulator()
        return SimClass(
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


# For backward compatibility, export commonly used items
__all__ = [
    "EventSimulatorType",
    "EventSimulatorProtocol",
    "create_event_simulator",
    "get_available_backends",
    "get_best_available_backend",
]
