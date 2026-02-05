"""
CARLA Wrapper for Neurosim.

This module provides a wrapper around the CARLA simulator for outdoor scenes
with event camera simulation support. It mirrors the HabitatWrapper interface
to allow seamless switching between indoor (Habitat) and outdoor (CARLA) scenes.

Requirements:
    - CARLA server running (typically on localhost:2000)
    - carla Python package installed

Performance Notes:
    - CARLA uses a client-server architecture which adds network latency
    - For highest throughput (1000+ Hz), use synchronous mode with no_rendering_mode=False
    - GPU-to-GPU transfer is optimized when client runs on same machine as server
    - Consider using CARLA's benchmark mode for consistent frame timing
"""

import logging
import queue
from typing import Any

import numpy as np
import torch

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None

from neurosim.core.utils import color2intensity
from neurosim.core.event_sim import create_event_simulator
from neurosim.core.visual_backend.base import VisualBackendProtocol

logger = logging.getLogger(__name__)


class CarlaWrapper(VisualBackendProtocol):
    """Wrapper around CARLA simulator for outdoor scenes with event camera support.

    This wrapper provides the same interface as HabitatWrapper, allowing the
    Neurosim simulator to use CARLA for outdoor driving/flying scenes.

    Key Features:
        - Connects to a running CARLA server
        - Supports RGB and depth cameras
        - Event camera simulation via color sensor + event simulator
        - Synchronous rendering mode for deterministic simulation
        - Weather and time-of-day control for outdoor realism

    Args:
        settings: A dictionary containing simulator settings.
    """

    def __init__(self, settings: dict[str, Any]):
        if not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA Python package not found. Install it with:\n"
                "pip install carla\n"
                "Or see: https://carla.readthedocs.io/en/latest/start_quickstart/"
            )
        self.settings = settings

        # Connection settings
        self._host = settings["host"]
        self._port = settings["port"]
        self._timeout = settings["timeout"]

        # Simulation settings
        self._synchronous_mode = settings.get("synchronous_mode", True)
        self._fixed_delta_seconds = settings.get("fixed_delta_seconds", 0.001)
        self._no_rendering_mode = settings.get("no_rendering_mode", False)

        # Connect to CARLA server
        try:
            self._client = carla.Client(self._host, self._port)
            self._client.set_timeout(self._timeout)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to CARLA server at {self._host}:{self._port}.\n"
                f"Make sure the CARLA server is running:\n"
                f"  Linux:   ./CarlaUnreal.sh -RenderOffScreen\n"
                f"Error: {e}"
            ) from e

        # Load map/world
        map_name = settings.get("map", None)
        if map_name:
            available_maps = self._client.get_available_maps()
            logger.debug(f"Available maps: {available_maps}")

            # Check if the requested map is available (handle full path or short name)
            map_found = any(
                map_name in m or m.endswith(map_name) for m in available_maps
            )

            if not map_found:
                logger.warning(
                    f"Requested map '{map_name}' not found in available maps. "
                    f"Available: {available_maps}. Will try to load anyway..."
                )

            self._world = self._client.load_world(map_name)
        else:
            self._world = self._client.get_world()

        # Configure world settings
        self._configure_world_settings()

        # Set weather
        weather_preset = settings["weather"]
        self._set_weather(weather_preset)

        # Spawn the "ego" actor (spectator or vehicle depending on use case)
        self._spectator = self._world.get_spectator()
        self._sensors: dict[str, Any] = {}  # carla.Sensor
        self._sensor_queues: dict[str, queue.Queue] = {}
        self._latest_data: dict[str, Any] = {}
        self._event_simulators: dict[str, Any] = {}  # Software event simulators

        # Create sensors from settings
        self._create_sensors()

        # Set initial spawn point
        spawn_point = settings.get("spawn_point", [0.0, 0.0, 10.0])
        spawn_rotation = settings.get("spawn_rotation", [0.0, 0.0, 0.0])
        self._set_spectator_transform(spawn_point, spawn_rotation)

        # Get map name safely
        try:
            current_map = self._world.get_map()
            map_name_str = current_map.name if current_map else "Unknown"
        except Exception:
            map_name_str = "Unknown"

        logger.info("════════════════════════════════════════════════════════════════")
        logger.info(
            f"✅ CARLA simulator initialized - Server: {self._host}:{self._port}"
        )
        logger.info(f"   Map: {map_name_str}")
        logger.info(f"   Synchronous mode: {self._synchronous_mode}")
        logger.info(
            f"   Fixed delta: {self._fixed_delta_seconds}s ({1 / self._fixed_delta_seconds:.0f} Hz)"
        )
        logger.info("════════════════════════════════════════════════════════════════")

    def _configure_world_settings(self) -> None:
        """Configure CARLA world settings for high-speed simulation."""
        settings = self._world.get_settings()
        settings.synchronous_mode = self._synchronous_mode
        settings.fixed_delta_seconds = self._fixed_delta_seconds
        settings.no_rendering_mode = self._no_rendering_mode

        # Substepping for physics stability at high frame rates
        # settings.substepping = True
        # settings.max_substep_delta_time = 0.01
        # settings.max_substeps = 10

        self._world.apply_settings(settings)

    def get_sensor_config(self, uuid: str) -> dict[str, Any]:
        """Get configuration for a specific sensor.

        Args:
            uuid: Sensor unique identifier.

        Returns:
            Sensor configuration dictionary.
        """
        return self.settings.get("sensors", {}).get(uuid, {})

    def _set_weather(self, preset: str) -> None:
        """Set weather from a preset name.

        Args:
            preset: Weather preset name (e.g., "ClearNoon", "CloudySunset", etc.)
        """
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
            "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
            "MidRainSunset": carla.WeatherParameters.MidRainSunset,
            "HardRainSunset": carla.WeatherParameters.HardRainSunset,
        }

        if preset in weather_presets:
            self._world.set_weather(weather_presets[preset])
        else:
            raise ValueError(f"Unknown weather preset: {preset}")

    def _set_spectator_transform(
        self, position: list[float], rotation: list[float]
    ) -> None:
        """Set the spectator/camera position and rotation.

        Args:
            position: [x, y, z] position in meters
            rotation: [pitch, yaw, roll] in degrees
        """
        transform = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
        )
        self._spectator.set_transform(transform)

    def _create_sensors(self) -> None:
        """Create all sensors defined in settings."""
        blueprint_library = self._world.get_blueprint_library()

        for sensor_name, sensor_cfg in self.settings.get("sensors", {}).items():
            sensor_type = sensor_cfg.get("type")

            if sensor_type == "event":
                # Event cameras use a color sensor + software event simulation
                self._create_event_sensor(sensor_name, sensor_cfg, blueprint_library)
            elif sensor_type == "color":
                self._create_rgb_sensor(sensor_name, sensor_cfg, blueprint_library)
            elif sensor_type == "depth":
                self._create_depth_sensor(sensor_name, sensor_cfg, blueprint_library)
            else:
                logger.warning(f"Unknown sensor type: {sensor_type}")

    def _create_rgb_sensor(
        self,
        name: str,
        cfg: dict[str, Any],
        bp_lib: Any,  # carla.BlueprintLibrary
    ) -> None:
        """Create an RGB camera sensor."""
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))

        # Set sensor capture rate via sensor_tick
        # sensor_tick = time between captures in seconds
        # e.g., sensor_tick=0.05 means 20 Hz capture rate
        # If not set, sensor captures on every world.tick()
        bp.set_attribute("sensor_tick", "0.05")

        # Optional: motion blur, lens effects
        bp.set_attribute("motion_blur_intensity", "0.0")
        bp.set_attribute("enable_postprocess_effects", "true")

        # Position and rotation relative to spectator
        pos = cfg["position"]
        rot = cfg["rotation"]
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        # Spawn sensor attached to spectator
        sensor = self._world.spawn_actor(bp, transform, attach_to=self._spectator)

        # Set up data queue
        self._sensor_queues[name] = queue.Queue()
        sensor.listen(self._sensor_queues[name].put)

        self._sensors[name] = sensor
        self._latest_data[name] = None

        logger.info(
            f"Created RGB sensor: {name} ({cfg.get('width')}x{cfg.get('height')})"
        )

    def _create_depth_sensor(
        self,
        name: str,
        cfg: dict[str, Any],
        bp_lib: Any,  # carla.BlueprintLibrary
    ) -> None:
        """Create a depth camera sensor."""
        bp = bp_lib.find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))

        # Set sensor capture rate via sensor_tick
        # sensor_tick = time between captures in seconds
        # e.g., sensor_tick=0.05 means 20 Hz capture rate
        # If not set, sensor captures on every world.tick()
        bp.set_attribute("sensor_tick", "0.05")

        # Position and rotation relative to spectator
        pos = cfg["position"]
        rot = cfg["rotation"]
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        # Spawn sensor attached to spectator
        sensor = self._world.spawn_actor(bp, transform, attach_to=self._spectator)

        # Set up data queue
        self._sensor_queues[name] = queue.Queue()
        sensor.listen(self._sensor_queues[name].put)

        self._sensors[name] = sensor
        self._latest_data[name] = None

        logger.info(
            f"Created Depth sensor: {name} ({cfg.get('width')}x{cfg.get('height')})"
        )

    def _create_event_sensor(
        self,
        name: str,
        cfg: dict[str, Any],
        bp_lib: Any,  # carla.BlueprintLibrary
    ) -> None:
        """Create an event camera using RGB sensor + software event simulation.

        Note: CARLA has a native DVS sensor, but we use the software simulator
        for consistency with Habitat and better control over parameters.
        """
        # Create underlying RGB sensor for event simulation
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))

        # Set sensor capture rate via sensor_tick
        # Event cameras typically need high frame rates for accurate event generation
        # Default to matching world tick rate (no sensor_tick = capture every tick)
        bp.set_attribute("sensor_tick", "0.001")
        # If neither specified, sensor captures on every world.tick()

        # Disable motion blur for clean event generation
        bp.set_attribute("motion_blur_intensity", "0.0")
        bp.set_attribute("enable_postprocess_effects", "false")

        # Position and rotation relative to spectator
        pos = cfg["position"]
        rot = cfg["rotation"]
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        # Spawn sensor attached to spectator
        sensor = self._world.spawn_actor(bp, transform, attach_to=self._spectator)

        # Set up data queue
        self._sensor_queues[name] = queue.Queue()
        sensor.listen(self._sensor_queues[name].put)

        self._sensors[name] = sensor
        self._latest_data[name] = None

        # Create software event simulator
        backend = cfg.get("backend", "auto")
        contrast_threshold_pos = cfg.get("contrast_threshold_pos", 0.35)
        contrast_threshold_neg = cfg.get("contrast_threshold_neg", 0.35)

        self._event_simulators[name] = create_event_simulator(
            backend=backend,
            width=cfg["width"],
            height=cfg["height"],
            start_time=0,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
        )

        logger.info(f"Created Event sensor: {name} ({cfg['width']}x{cfg['height']})")

    def _tick_world(self) -> None:
        """Advance the simulation by one tick (synchronous mode only).

        In CARLA synchronous mode, world.tick() does three things:
        1. Advances simulation time by `fixed_delta_seconds`
        2. Runs physics for all actors (vehicles, pedestrians, etc.)
        3. Triggers sensor capture - but ONLY for sensors whose `sensor_tick`
           interval has elapsed since their last capture

        Sensors are configured with `sensor_tick` to control their capture rate.
        For example, sensor_tick=0.05 means the sensor captures at 20 Hz,
        regardless of how fast world.tick() is called. This avoids wasting
        GPU resources on frames that aren't needed.

        The spectator (camera viewpoint) has no physics - it's just a transform.
        Environmental physics (other vehicles, pedestrians) still runs.
        """
        if self._synchronous_mode:
            self._world.tick()

    def has_new_data(self, uuid: str) -> bool:
        """Check if a sensor has new data available.

        Useful for polling sensors that capture at lower rates than world tick.

        Args:
            uuid: Sensor identifier.

        Returns:
            True if new data is available in the queue.
        """
        return not self._sensor_queues[uuid].empty()

    def _get_sensor_data(self, uuid: str, timeout: float = 0.0) -> Any:
        """Get the latest data from a sensor.

        Args:
            uuid: Sensor identifier.
            timeout: Maximum time to wait for data. Default 0 = non-blocking.
                     Use timeout > 0 if you expect data but it hasn't arrived yet.

        Returns:
            Sensor data (CARLA Image object), or None if no data available.

        Note:
            Sensors only produce data at their configured `sensor_tick` rate.
            If called between sensor captures, returns the last captured data.
        """
        # Drain queue to get the most recent frame (in case multiple accumulated)
        latest = None
        try:
            while True:
                latest = self._sensor_queues[uuid].get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._latest_data[uuid] = latest
            return latest

        # If queue was empty, try waiting if timeout > 0
        if timeout > 0:
            try:
                data = self._sensor_queues[uuid].get(timeout=timeout)
                self._latest_data[uuid] = data
                return data
            except queue.Empty:
                pass

        # Return cached data if available
        return self._latest_data.get(uuid)

    def update_agent_state(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """Update the agent's pose (spectator in CARLA terms).

        Args:
            position: 3D position [x, y, z] in Habitat world coordinates.
            quaternion: Rotation quaternion [w, x, y, z] or numpy quaternion.

        Note:
            Coordinate systems:
            - Habitat (right-handed, Y-up): camera looks along -Z, +X right, +Y up
            - CARLA/UE (left-handed, Z-up): +X forward, +Y right, +Z up

            Position mapping:
                CARLA X = -Habitat Z (forward)
                CARLA Y =  Habitat X (right)
                CARLA Z =  Habitat Y (up)

            Rotation mapping (axis correspondence):
                Habitat X (right)    -> CARLA Y (right)    : Habitat roll  -> CARLA pitch
                Habitat Y (up)       -> CARLA Z (up)       : Habitat pitch -> CARLA yaw
                Habitat Z (backward) -> CARLA -X (forward) : Habitat yaw   -> -CARLA roll

            Left-handed convention: CARLA yaw is clockwise positive (opposite of right-handed).
        """
        # Extract quaternion components
        if hasattr(quaternion, "w"):
            # numpy quaternion format (w, x, y, z)
            w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
        else:
            # Array format [w, x, y, z]
            w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

        # Convert quaternion to Euler angles in Habitat frame
        # These are rotations around Habitat's X, Y, Z axes respectively

        # Rotation around Habitat X (right axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        habitat_roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Rotation around Habitat Y (up axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            habitat_pitch = np.copysign(np.pi / 2, sinp)
        else:
            habitat_pitch = np.arcsin(sinp)

        # Rotation around Habitat Z (backward axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        habitat_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Map Habitat Euler angles to CARLA Euler angles based on axis correspondence:
        # - Habitat X (right) -> CARLA Y (right): habitat_roll -> carla_pitch
        # - Habitat Y (up) -> CARLA Z (up): habitat_pitch -> carla_yaw
        # - Habitat Z (backward) -> CARLA -X: habitat_yaw -> -carla_roll
        #
        # Additionally, CARLA is left-handed so yaw direction is inverted
        carla_pitch_deg = np.degrees(habitat_roll)
        carla_yaw_deg = -np.degrees(habitat_pitch)  # Negate for left-handed convention
        carla_roll_deg = -np.degrees(habitat_yaw)  # Negate because Z -> -X

        # Position mapping: Habitat -> CARLA
        carla_x = -position[2]  # Habitat Z (backward) -> CARLA X (forward), negate
        carla_y = position[0]  # Habitat X (right) -> CARLA Y (right)
        carla_z = position[1]  # Habitat Y (up) -> CARLA Z (up)

        transform = carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=carla_z),
            carla.Rotation(
                pitch=carla_pitch_deg, yaw=carla_yaw_deg, roll=carla_roll_deg
            ),
        )
        self._spectator.set_transform(transform)

        # Note: We don't tick here anymore. The simulator should call tick()
        # explicitly after update_agent_state() to trigger sensor capture.
        # This matches the Habitat pattern where state update is separate from rendering.
        self._tick_world()

    def render_color(self, uuid: str) -> torch.Tensor:
        """Render RGB image from a color sensor.

        Args:
            uuid: Sensor unique identifier.

        Returns:
            RGB image tensor of shape (H, W, 3).
        """
        data = self._get_sensor_data(uuid)
        if data is None:
            cfg = self.get_sensor_config(uuid)
            h, w = cfg.get("height", 480), cfg.get("width", 640)
            return torch.zeros((h, w, 3), dtype=torch.uint8)

        # Convert CARLA image to numpy array
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))  # BGRA
        array = array[:, :, :3]  # Remove alpha
        array = array[:, :, ::-1]  # BGR to RGB

        return torch.from_numpy(array.copy())

    def render_depth(self, uuid: str) -> torch.Tensor:
        """Render depth image from a depth sensor.

        Args:
            uuid: Sensor unique identifier.

        Returns:
            Depth image tensor of shape (H, W) in meters.
        """
        data = self._get_sensor_data(uuid)
        if data is None:
            cfg = self.get_sensor_config(uuid)
            h, w = cfg.get("height", 480), cfg.get("width", 640)
            return torch.zeros((h, w), dtype=torch.float32)

        # Convert CARLA depth image to meters
        # CARLA encodes depth in RGB: depth = (R + G*256 + B*256*256) / (256*256*256 - 1) * 1000
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))

        # Decode depth from RGB encoding
        r = array[:, :, 2].astype(np.float32)
        g = array[:, :, 1].astype(np.float32)
        b = array[:, :, 0].astype(np.float32)

        depth = (r + g * 256 + b * 256 * 256) / (256 * 256 * 256 - 1)
        depth = depth * 1000.0  # Convert to meters (CARLA max depth is 1000m)

        return torch.from_numpy(depth)

    def render_events(
        self, uuid: str, time: int, to_numpy: bool = False
    ) -> tuple[Any, ...] | None:
        """Render events from the event camera.

        Args:
            uuid: Sensor unique identifier.
            time: Current timestamp in microseconds.
            to_numpy: Whether to convert events to numpy arrays.

        Returns:
            Tuple of (x, y, t, p) event arrays, or None if no events.
        """
        # Get RGB image from the underlying color sensor
        data = self._get_sensor_data(uuid)
        if data is None:
            return None

        # Convert CARLA image to RGB tensor
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # BGR to RGB
        rgb_tensor = torch.from_numpy(array.copy()).float() / 255.0

        # Convert to intensity (event simulator expects torch tensor)
        intensity_image = color2intensity(rgb_tensor)

        # Generate events using the software simulator
        events = self._event_simulators[uuid].image_callback(intensity_image, time)

        if to_numpy and events is not None and isinstance(events[0], torch.Tensor):
            events = [
                events[0].cpu().numpy().astype(np.uint16),
                events[1].cpu().numpy().astype(np.uint16),
                events[2].cpu().numpy().astype(np.uint64),
                events[3].cpu().numpy().astype(np.uint8),
            ]

        return events

    def close(self) -> None:
        """Clean up: destroy sensors and restore world settings."""
        # Destroy all sensors
        for name, sensor in self._sensors.items():
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
                    logger.debug(f"Destroyed sensor: {name}")
            except Exception as e:
                logger.warning(f"Failed to destroy sensor '{name}': {e}")

        self._sensors.clear()
        self._sensor_queues.clear()

        # Restore asynchronous mode
        try:
            if hasattr(self, "_world") and self._world is not None:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
                logger.info("CARLA wrapper closed, world settings restored.")
        except Exception as e:
            logger.warning(f"Failed to restore world settings: {e}")


class CarlaNativeEventWrapper(CarlaWrapper):
    """CARLA wrapper using native DVS (event camera) sensor.

    This variant uses CARLA's built-in DVS sensor instead of software
    event simulation. May offer better performance for simple use cases
    but with less control over event generation parameters.

    Note: Contrast thresholds and other parameters are set via CARLA's
    DVS blueprint attributes, which differ from our software simulator.
    """

    def _create_event_sensor(
        self,
        name: str,
        cfg: dict[str, Any],
        bp_lib: Any,  # carla.BlueprintLibrary
    ) -> None:
        """Create a native CARLA DVS sensor."""
        bp = bp_lib.find("sensor.camera.dvs")
        bp.set_attribute("image_size_x", str(cfg.get("width", 640)))
        bp.set_attribute("image_size_y", str(cfg.get("height", 480)))
        bp.set_attribute("fov", str(cfg.get("fov", 90)))

        # DVS-specific parameters
        bp.set_attribute(
            "positive_threshold", str(cfg.get("contrast_threshold_pos", 0.3))
        )
        bp.set_attribute(
            "negative_threshold", str(cfg.get("contrast_threshold_neg", 0.3))
        )
        bp.set_attribute("sigma_positive_threshold", "0.0")
        bp.set_attribute("sigma_negative_threshold", "0.0")
        bp.set_attribute("refractory_period_ns", "0")
        bp.set_attribute("use_log", "true")
        bp.set_attribute("log_eps", "0.001")

        pos = cfg.get("position", [0.0, 0.0, 0.0])
        rot = cfg.get("rotation", [0.0, 0.0, 0.0])
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        sensor = self._world.spawn_actor(bp, transform, attach_to=self._spectator)

        self._sensor_queues[name] = queue.Queue()
        sensor.listen(self._sensor_queues[name].put)

        self._sensors[name] = sensor
        self._latest_data[name] = None

        logger.info(
            f"Created Native DVS sensor: {name} ({cfg.get('width')}x{cfg.get('height')})"
        )

    def render_events(
        self, uuid: str, time: int, to_numpy: bool = False
    ) -> tuple[Any, ...] | None:
        """Get events from native CARLA DVS sensor.

        Args:
            uuid: Sensor unique identifier.
            time: Current timestamp (ignored, CARLA provides timestamps).
            to_numpy: Whether to return numpy arrays.

        Returns:
            Tuple of (x, y, t, p) event arrays, or None if no events.
        """
        data = self._get_sensor_data(uuid)
        if data is None or len(data) == 0:
            return None

        # CARLA DVSEventArray provides x, y, t, pol
        x = data.to_array_x()
        y = data.to_array_y()
        t = data.to_array_t()  # Already in microseconds
        p = data.to_array_pol()  # True/False -> 1/0

        if to_numpy:
            return (
                np.array(x, dtype=np.uint16),
                np.array(y, dtype=np.uint16),
                np.array(t, dtype=np.uint64),
                np.array(p, dtype=np.uint8),
            )
        else:
            return (
                torch.tensor(x, dtype=torch.int16),
                torch.tensor(y, dtype=torch.int16),
                torch.tensor(t, dtype=torch.int64),
                torch.tensor(p, dtype=torch.int8),
            )
