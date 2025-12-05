import cv2
import copy
import time
import yaml
import pprint
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from neurosim.habitat_wrapper import HabitatWrapper
from neurosim.utils import init_h5, append_data_to_h5, get_pose_on_navmesh

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

# Controller
from rotorpy.controllers.quadrotor_control import SE3Control

# Sensors
from rotorpy.sensors.imu import Imu

# Trajectories
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.polynomial_traj import Polynomial


# Visualization
try:
    import rerun as rr
except ImportError:
    rr = None


class Simulator:
    def __init__(
        self,
        settings: Path,
        world_rate: int = 1000,
        control_rate: int = 100,
        sim_time: int = 20,
        enable_profiling: bool = False,
    ):
        """
        settings: Path to the settings file for the habitat scene and simulator.
        if None, loads default settings.
        """
        # Visual Simulator instance
        self._hwrapper = HabitatWrapper(settings, enable_profiling=enable_profiling)

        # Dynamics Simulator instance
        self._quadsim = Multirotor(quad_params, aero=False, integration_method="euler")
        quad_state = self._hwrapper.agent.get_state()
        self._quadsim.initial_state["x"] = quad_state.position
        self._quadsim.initial_state["q"] = np.array(
            [
                quad_state.rotation.x,
                quad_state.rotation.y,
                quad_state.rotation.z,
                quad_state.rotation.w,
            ]
        )

        # Load IMU sensor
        self.has_imu_sensor = self._hwrapper.settings["imu_sensor"]
        if self.has_imu_sensor:
            self.imu_sampling_rate = self._hwrapper.settings.get(
                "imu_sampling_rate", 100
            )
            self.imu_sensor_steps = world_rate / self.imu_sampling_rate  # sim steps
            self.imu = Imu(
                p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=self.imu_sampling_rate
            )

        # Load trajectory settings
        if settings is not None:
            with open(settings, "r") as file:
                trajectory = yaml.safe_load(file)["trajectory"]
        else:
            trajectory = {"type": "constant_speed", "dist": 5, "v_avg": 1, "axis": 0}

        #! Use Rotorpy controller for simulation for now
        self._controller = SE3Control(quad_params)
        if trajectory["type"] == "constant_speed":
            self._trajectory = ConstantSpeed(
                init_pos=quad_state.position,
                dist=trajectory["dist"],
                v_avg=trajectory["v_avg"],
                axis=trajectory["axis"],
                repeat=True,
            )
        elif trajectory["type"] == "polynomial":
            self._trajectory = Polynomial(
                points=np.array(trajectory["points"]),
                v_avg=trajectory["v_avg"],
            )
        else:
            raise ValueError(
                "Invalid trajectory type. Use 'constant_speed' or 'polynomial'."
            )

        self.time = 0  # time in seconds
        self.simsteps = 0  # number of simulation steps
        self.world_rate = world_rate  # Hz
        self.control_rate = control_rate  # Hz
        self.has_color_sensor = self._hwrapper.settings["color_sensor"]
        self.color_sensor_steps = world_rate / self._hwrapper.settings.get(
            "color_sensor_rate", 25
        )  # sim steps
        self.has_depth_sensor = self._hwrapper.settings["depth_sensor"]
        self.depth_sensor_steps = world_rate / self._hwrapper.settings.get(
            "depth_sensor_rate", 10
        )  # sim steps
        self.t_step = 1 / world_rate  # seconds
        self.t_final = sim_time  # seconds
        self.width = self._hwrapper.settings["width"]
        self.height = self._hwrapper.settings["height"]
        self.state = copy.deepcopy(self._quadsim.initial_state)

    def simulate_step(self, control):
        """
        Simulate one step of the environment.
        """
        self.time += self.t_step
        self.simsteps += 1
        self.state = self._quadsim.step(self.state, control, self.t_step)

        position = np.array(
            [self.state["x"][0], self.state["x"][2], -self.state["x"][1]]
        )
        rotation = np.quaternion(
            self.state["q"][3],
            self.state["q"][0],
            self.state["q"][2],
            -self.state["q"][1],
        )
        self._hwrapper.update_agent_pose(0, position, rotation)

    def simulate_traj(
        self, save_h5: str = None, save_png: str = None, display: bool = False
    ):
        """
        Simulate the trajectory.

        Args:
            save_h5: Path to save the HDF5 file with events and intensity images.
            save_png: Path to save the PNG files with intensity and event images.
            display: If True, display the intensity and event images during simulation.
        """
        flat = self._trajectory.update(self.time)
        control = self._controller.update(self.time, self.state, flat)

        color_img, depth_img, imu_data = None, None, None

        #! Essential initialization and checks for saving and displaying data ####
        if save_h5:
            Path(save_h5).parent.mkdir(parents=True, exist_ok=True)
            h5f = init_h5(save_h5, self.height, self.width)
        if save_png or display:
            self.event_img = np.zeros(
                (self.height, self.width, 3), dtype=np.uint8
            )  # Init event im
        if save_png:
            save_png = Path(save_png)
            for subdir in ["events", "color", "depth"]:
                (save_png / subdir).mkdir(parents=True, exist_ok=True)
        if display:
            rr.init("neurosim_viz", spawn=True)
            rr.set_time("stable_time", duration=self.time)
        #! End of essential initialization and checks ############################

        latencies = []
        while self.time < self.t_final:
            for _ in range(int(self.world_rate / self.control_rate)):
                start_time = time.perf_counter()

                self.simulate_step(control)
                events = self._hwrapper.render_events(
                    self.time * 1e6, to_numpy=True
                )  # in us
                if self.has_imu_sensor and self.simsteps % self.imu_sensor_steps == 0:
                    statedot = self._quadsim.statedot(self.state, control, 0)
                    imu_data = self.imu.measurement(
                        self.state, statedot, with_noise=False
                    )
                if (
                    self.has_color_sensor
                    and self.simsteps % self.color_sensor_steps == 0
                ):
                    color_img = self._hwrapper.render_color_sensor().cpu().numpy()
                if (
                    self.has_depth_sensor
                    and self.simsteps % self.depth_sensor_steps == 0
                ):
                    depth_img = self._hwrapper.render_depth_sensor().cpu().numpy()

                latencies.append(time.perf_counter() - start_time)

                if save_h5:
                    self.save_h5_sim_data(h5f, events, color_img, depth_img, imu_data)
                if save_png or display:
                    self.save_or_display_sim_data(
                        events, color_img, depth_img, imu_data, save_png, display
                    )

            flat = self._trajectory.update(
                self.time
            )  # returns the traj point at the current time
            control = self._controller.update(self.time, self.state, flat)
        # End of simulation loop
        print("Simulation completed. \n")

        stats = {
            "Average latency": np.mean(latencies),
            "Max latency": np.max(latencies),
            "Median latency": np.median(latencies),
            "Frames": self.simsteps,
            "FPS": (self.simsteps - 10) / sum(latencies[10:]),
            "Total time": sum(latencies),
        }
        print("Simulation Latency Stats: ")
        pprint.pprint(stats)

        print(
            f"Event Simulator Backend [{self._hwrapper.settings['event_camera_backend']}] Benchmark: "
        )
        if self._hwrapper._enable_profiling:
            pprint.pprint(self._hwrapper._benchmark.to_dict())
        else:
            print("Profiling not enabled.")

        if save_h5:
            h5f.close()

    def save_h5_sim_data(self, h5f, events, color_img, depth_img, imu_data):
        """
        Save simulation data to HDF5 file and PNG files if specified.

        Args:
            h5f: HDF5 file object to save data.
            events: List of events (x, y, t, p) to save.
            color_img: Color image to save.
            depth_img: Depth image to save.
            imu_data: IMU data to save.
        """
        data = {"time": self.time, "state": self.state}
        if events is not None and len(events) > 0:
            data["events"] = events
        if imu_data is not None and self.simsteps % self.imu_sensor_steps == 0:
            data["imu"] = imu_data
        if color_img is not None and self.simsteps % self.color_sensor_steps == 0:
            data["color"] = color_img
        if depth_img is not None and self.simsteps % self.depth_sensor_steps == 0:
            data["depth"] = depth_img
        append_data_to_h5(h5f, **data)

    def save_or_display_sim_data(
        self, events, color_img, depth_img, imu_data, save_png, display
    ):
        """
        Save or display simulation data.

        Args:
            events: Array of events (x, y, t, p) to save or display.
            color_img: Color image to save or display.
            depth_img: Depth image to save or display.
            imu_data: IMU data to save or display.
            save_png: Path to save PNG files.
            display: If True, display the images.
        """
        if events is not None and len(events) > 0:
            self.event_img[events[1], events[0], events[3] * 2] = 255

        if save_png:
            if self.simsteps % self.color_sensor_steps == 0:
                plt.imsave(save_png / "events" / f"{self.simsteps}.png", self.event_img)
                if color_img is not None:
                    plt.imsave(save_png / "color" / f"{self.simsteps}.png", color_img)
            if depth_img is not None and self.simsteps % self.depth_sensor_steps == 0:
                plt.imsave(
                    save_png / "depth" / f"{self.simsteps}.png", depth_img, cmap="jet"
                )

        if display:
            rr.set_time("stable_time", duration=self.time)
            if self.simsteps % self.color_sensor_steps == 0:
                rr.log("events", rr.Image(self.event_img))

                if self._hwrapper._navmesh is not None:
                    # Create navmesh with current pose plotted on it
                    navmesh_with_pose = self._hwrapper._navmesh.copy()
                    px, py = get_pose_on_navmesh(
                        self._hwrapper._scene_bounds,
                        self.state["x"],
                        self._hwrapper.settings.get("navmesh_meters_per_pixel", 0.1),
                    )
                    # Draw a circle marker on the navmesh
                    cv2.circle(
                        navmesh_with_pose,
                        (px, py),
                        radius=2,
                        color=(255, 0, 0),
                        thickness=-1,
                    )
                    rr.log("navmesh_with_pose", rr.Image(navmesh_with_pose))

                rr.log(
                    "navigation/pose",
                    rr.Transform3D(
                        translation=self.state["x"],
                        rotation=rr.Quaternion(xyzw=self.state["q"]),
                        axis_length=5.0,
                        relation=rr.TransformRelation.ParentFromChild,
                    ),
                )
                rr.log(
                    "navigation/trajectory",
                    rr.Points3D(positions=self.state["x"][None, :]),
                )
                if color_img is not None:
                    rr.log("color", rr.Image(color_img))
                if self.has_imu_sensor:
                    rr.log("imu/acc", rr.Scalars(imu_data["accel"]))
                    rr.log("imu/gyro", rr.Scalars(imu_data["gyro"]))
            if depth_img is not None and self.simsteps % self.depth_sensor_steps == 0:
                rr.log("depth", rr.DepthImage(depth_img))

        if self.simsteps % self.color_sensor_steps == 0:
            self.event_img.fill(0)  # Reset event image after displaying or saving.
