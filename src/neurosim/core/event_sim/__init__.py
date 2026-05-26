"""
Event Simulator Factory and Registry.

This module provides a factory pattern for creating event camera simulators.
It supports five backends:
- cuda: cu_esim ``EventSimulator`` (modes: "single" / "multi"); fastest, default
- voltmeter: cu_esim ``DVSVoltmeterSimulator`` (stochastic DVS-Voltmeter model)
- torch: PyTorch implementation (fallback, GPU-accelerated)
- airsim: AirSim/Numba implementation (CPU-based, slower)
- vid2e: RPG VID2E CUDA implementation from UZH

The factory takes the YAML sensor config dict directly — each backend branch
reads the keys it cares about, so backend-specific knobs (contrast thresholds,
voltmeter k-params, etc.) never appear in the wrapper.

Usage:
    from neurosim.core.event_sim import create_event_simulator

    sim = create_event_simulator(
        sensor_cfg={
            "backend": "cuda",
            "mode": "single",
            "width": 640,
            "height": 480,
            "contrast_threshold_neg": 0.15,
            "contrast_threshold_pos": 0.15,
        },
        device="cuda:0",
    )
"""

from enum import Enum
from typing import Optional, Protocol, Any

from .types import Events


class EventSimulatorType(Enum):
    """Supported event simulator backends."""

    AUTO = "auto"
    CUDA = "cuda"
    VOLTMETER = "voltmeter"
    TORCH = "torch"
    AIRSIM = "airsim"
    VID2E = "vid2e"


class EventSimulatorProtocol(Protocol):
    """Protocol defining the interface for event simulators."""

    def __call__(self, image: Any, timestamp_us: int) -> Optional[Events]:
        """Process a new image and generate events.

        Args:
            image: The new image to process (H, W), positive values
            timestamp_us: Timestamp in microseconds

        Returns:
            Named tuple Events(x, y, t, p) or None if no events
        """
        ...

    def init(self, first_image: Any) -> None:
        """Initialize internal state with the first image.

        Args:
            first_image: First grayscale frame (H, W), positive values
        """
        ...

    def reset(self, first_image: Optional[Any] = None) -> None:
        """Reset the simulator state."""
        ...


def _import_cuda_simulator():
    """Lazily import CUDA simulator from neurosim_cu_esim package."""
    try:
        from neurosim_cu_esim import EventSimulator

        return EventSimulator
    except ImportError as e:
        raise ImportError(
            f"CUDA event simulator not available. "
            f"Please install neurosim_cu_esim with: "
            f"pip install git+https://github.com/grasp-lyrl/neurosim_cu_esim.git "
            f"Original error: {e}"
        )


def _import_voltmeter_simulator():
    """Lazily import DVS-Voltmeter simulator from neurosim_cu_esim package."""
    try:
        from neurosim_cu_esim import DVSVoltmeterSimulator

        return DVSVoltmeterSimulator
    except ImportError as e:
        raise ImportError(
            f"DVS-Voltmeter simulator not available. "
            f"Please install neurosim_cu_esim with: "
            f"pip install git+https://github.com/grasp-lyrl/neurosim_cu_esim.git "
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
        _import_voltmeter_simulator()
        available.append(EventSimulatorType.VOLTMETER)
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
            "Please install at least one: neurosim_cu_esim (CUDA), torch, or numba (AirSim)."
        )

    # Priority order
    for backend in [
        EventSimulatorType.CUDA,
        EventSimulatorType.VOLTMETER,
        EventSimulatorType.VID2E,
        EventSimulatorType.TORCH,
        EventSimulatorType.AIRSIM,
    ]:
        if backend in available:
            return backend

    return available[0]


def create_event_simulator(
    sensor_cfg: dict,
    *,
    device: str,
    first_image: Optional[Any] = None,
) -> EventSimulatorProtocol:
    """Create an event simulator from a sensor config dict.

    The factory dispatches on ``sensor_cfg["backend"]`` (default ``"auto"``) and
    each branch reads the keys relevant to that backend. The wrapper passes the
    raw YAML dict in; this function is the single source of truth for the
    sensor-YAML → simulator-kwargs mapping.

    Common keys (read by all backends that use them):
        backend: "auto" | "cuda" | "voltmeter" | "torch" | "airsim" | "vid2e"
        width, height: sensor resolution (required)
        start_time: initial timestamp in microseconds (torch / airsim / vid2e); default 0

    Backend-specific keys:
        cuda: mode ("single"|"multi"), contrast_threshold_neg/pos,
              max_events_per_pixel (multi only; default 16)
        voltmeter: camera_type ("DVS346"|"DVS240"), leak_scale, randomize_phase,
                   seed, max_events_per_pixel (default 16), optional k=[k1..k6]
        vid2e: contrast_threshold_neg/pos
        torch / airsim: no extra keys

    Args:
        sensor_cfg: YAML sensor config dict.
        device: Torch device string (e.g. "cuda:0"). AirSim ignores it.
        first_image: Optional first frame to initialise the simulator.

    Returns:
        An event simulator instance following EventSimulatorProtocol.
    """
    backend_str = sensor_cfg.get("backend", "auto")
    try:
        backend = EventSimulatorType(backend_str.lower())
    except ValueError:
        valid_backends = [b.value for b in EventSimulatorType]
        raise ValueError(
            f"Unknown backend: {backend_str!r}. Valid backends: {valid_backends}"
        )

    if backend == EventSimulatorType.AUTO:
        backend = get_best_available_backend()

    width = sensor_cfg["width"]
    height = sensor_cfg["height"]
    start_time = sensor_cfg.get("start_time", 0)

    NEUROSIM_ERR = (
        "CUDA event simulator import failed. "
        "To restore CUDA, reinstall neurosim_cu_esim against your current "
        "PyTorch/CUDA environment (e.g. `pip uninstall -y neurosim_cu_esim "
        "&& pip install --no-build-isolation --force-reinstall "
        "./deps/neurosim_cu_esim`). "
    )

    if backend == EventSimulatorType.CUDA:
        try:
            SimClass = _import_cuda_simulator()
        except ImportError as cuda_error:
            raise ImportError(
                NEUROSIM_ERR + f"Original CUDA import error: {cuda_error}",
            )
        mode = sensor_cfg.get("mode", "single")
        cuda_kwargs: dict[str, Any] = {
            "mode": mode,
            "contrast_threshold_neg": sensor_cfg.get("contrast_threshold_neg", 0.15),
            "contrast_threshold_pos": sensor_cfg.get("contrast_threshold_pos", 0.15),
        }
        if mode == "multi":
            cuda_kwargs["max_events"] = (
                width * height * int(sensor_cfg.get("max_events_per_pixel", 16))
            )
        sim = SimClass(width=width, height=height, device=device, **cuda_kwargs)
        if first_image is not None:
            sim.init(first_image)
        return sim

    elif backend == EventSimulatorType.VOLTMETER:
        try:
            SimClass = _import_voltmeter_simulator()
        except ImportError as voltmeter_error:
            raise ImportError(
                NEUROSIM_ERR + f"Original voltmeter import error: {voltmeter_error}",
            )
        # neurosim's render pipeline emits [0, 1] linear intensity; the
        # voltmeter k-params are calibrated to 0-255, so let the simulator
        # rescale internally.
        assert "camera_type" in sensor_cfg or "k" in sensor_cfg, (
            "voltmeter backend requires 'camera_type' or 'k' params"
        )

        volt_kwargs: dict[str, Any] = {
            "camera_type": sensor_cfg.get("camera_type", "DVS346"),
            "leak_scale": float(sensor_cfg.get("leak_scale", 1.0)),
            "randomize_phase": bool(sensor_cfg.get("randomize_phase", True)),
            "seed": int(sensor_cfg.get("seed", 67)),
            "max_events": width
            * height
            * int(sensor_cfg.get("max_events_per_pixel", 16)),
            "input_normalized": True,
        }
        if "k" in sensor_cfg:
            volt_kwargs["k"] = list(sensor_cfg["k"])

        sim = SimClass(width=width, height=height, device=device, **volt_kwargs)
        if first_image is not None:
            sim.init(first_image)
        return sim

    elif backend == EventSimulatorType.TORCH:
        SimClass = _import_torch_simulator()
        return SimClass(
            W=width,
            H=height,
            start_time=start_time,
            first_image=first_image,
            device=device,
        )

    elif backend == EventSimulatorType.AIRSIM:
        #! TODO: Add the kwargs correctly.
        SimClass = _import_airsim_simulator()
        return SimClass(
            W=width,
            H=height,
            first_image=first_image,
            first_time=start_time,
        )

    elif backend == EventSimulatorType.VID2E:
        SimClass = _import_vid2e_simulator()
        return SimClass(
            W=width,
            H=height,
            start_time=start_time,
            first_image=first_image,
            contrast_threshold_neg=sensor_cfg.get("contrast_threshold_neg", 0.15),
            contrast_threshold_pos=sensor_cfg.get("contrast_threshold_pos", 0.15),
            device=device,
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
