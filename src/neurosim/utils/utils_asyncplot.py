import time
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from multiprocessing import Process, Queue, Event

warnings.filterwarnings("ignore")


@dataclass
class PlotCommand:
    """Command structure for plot operations"""

    action: str  # 'add', 'update', 'remove', 'clear'
    plot_id: str
    data: Optional[Dict[str, Any]] = None
    plot_type: str = "line"  # 'line', 'scatter', 'bar', 'hist'
    subplot_pos: Optional[Tuple[int, int]] = None  # (row, col) for specific positioning


class AsyncVisualizer:
    """
    Wrapper class to run DynamicVisualizer in a separate process
    """

    def __init__(self, figsize=(12, 8), max_plots=6):
        self.figsize = figsize
        self.max_plots = max_plots
        self.command_queue = Queue()
        self.stop_event = Event()
        self.process = None

    @staticmethod
    def _run_visualizer_process(command_queue, stop_event, figsize, max_plots):
        """Static method to run visualizer in separate process"""
        # Import matplotlib in the process and enable interactive mode
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Create a simplified visualizer for the process
        class ProcessDynamicVisualizer:
            def __init__(self):
                self.figsize = figsize
                self.max_plots = max_plots
                self.plots = {}
                self.fig = None
                self.axes = {}
                self.grid_shape = (2, 3)

            def _calculate_grid_shape(self, num_plots):
                if num_plots <= 1:
                    return (1, 1)
                elif num_plots <= 2:
                    return (1, 2)
                elif num_plots <= 4:
                    return (2, 2)
                elif num_plots <= 6:
                    return (2, 3)
                elif num_plots <= 9:
                    return (3, 3)
                else:
                    return (4, 4)

            def _setup_figure(self):
                self.fig, _ = plt.subplots(figsize=self.figsize)
                self.fig.suptitle("Dynamic Visualizer (Process)", fontsize=16)
                plt.tight_layout()
                plt.show(block=False)

            def _reorganize_subplots(self):
                if not self.plots:
                    return

                new_grid = self._calculate_grid_shape(len(self.plots))

                self.grid_shape = new_grid
                self.fig.clear()
                self.axes = {}

                for i, (plot_id, plot_data) in enumerate(self.plots.items()):
                    ax = self.fig.add_subplot(self.grid_shape[0], self.grid_shape[1], i + 1)
                    self.axes[plot_id] = ax
                    self._create_plot(plot_id, plot_data["data"], plot_data["plot_type"], ax)

                plt.tight_layout()
                plt.draw()

            def _create_plot(self, plot_id, data, plot_type, ax):
                ax.set_title(f"Plot: {plot_id}")

                if plot_type == "navigation":
                    x = data.get("x", [])
                    y = data.get("y", [])
                    limits = data.get("limits", [[-10, 10], [-20, 20]])
                    ax.scatter(x, y, c="r", s=10)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_xlim(limits[0][0], limits[0][1])
                    ax.set_ylim(limits[1][0], limits[1][1])

                elif plot_type == "imu":
                    accel = data.get("acceleration", [])
                    maxlen = data.get("maxlen", 500)
                    if len(accel) > 0:
                        ax.clear()
                        colors = ["r", "g", "b"]
                        labels = ["X", "Y", "Z"]
                        for idx in range(accel.shape[1]):
                            ax.plot(
                                accel[:, idx], c=colors[idx], label=f"{labels[idx]} Acceleration"
                            )
                        ax.set_xlim(0, maxlen)
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Acceleration (m/s²)")
                        ax.set_title("IMU Acceleration")

                elif plot_type == "line":
                    x = data.get("x", range(len(data.get("y", []))))
                    y = data.get("y", [])
                    ax.clear()  # Clear previous plot
                    ax.plot(x, y, label=plot_id)
                    ax.set_xlabel(data.get("xlabel", "X"))
                    ax.set_ylabel(data.get("ylabel", "Y"))

                elif plot_type == "scatter":
                    x = data.get("x", [])
                    y = data.get("y", [])
                    c = data.get("c", "blue")
                    ax.clear()  # Clear previous plot
                    ax.scatter(x, y, c=c, label=plot_id)
                    ax.set_xlabel(data.get("xlabel", "X"))
                    ax.set_ylabel(data.get("ylabel", "Y"))

                elif plot_type == "bar":
                    x = data.get("x", range(len(data.get("y", []))))
                    y = data.get("y", [])
                    ax.clear()  # Clear previous plot
                    ax.bar(x, y, label=plot_id)
                    ax.set_xlabel(data.get("xlabel", "X"))
                    ax.set_ylabel(data.get("ylabel", "Y"))

                elif plot_type == "hist":
                    values = data.get("values", [])
                    bins = data.get("bins", 30)
                    ax.clear()  # Clear previous plot
                    ax.hist(values, bins=bins, alpha=0.7, label=plot_id)
                    ax.set_xlabel(data.get("xlabel", "Value"))
                    ax.set_ylabel(data.get("ylabel", "Frequency"))

                elif plot_type == "image":
                    img = data.get("image", np.zeros((10, 10, 3)))
                    ax.clear()  # Clear previous plot
                    ax.imshow(img)
                    ax.set_title(data.get("title", "Image Plot"))
                    ax.axis("off")

                elif plot_type == "depth":
                    img = data.get("image", np.zeros((10, 10)))
                    ax.clear()  # Clear previous plot
                    ax.imshow(img, cmap="jet")
                    ax.set_title(data.get("title", "Depth Plot"))
                    ax.axis("off")

                elif plot_type == "events":
                    events = data["events"]
                    if "event_img" not in self.plots[plot_id]:
                        self.plots[plot_id]["event_img"] = np.zeros(
                            (data["height"], data["width"], 3), dtype=np.uint8
                        )
                    if len(events[0]) > 0:
                        self.plots[plot_id]["event_img"][events[1], events[0], events[2] * 2] = 255
                    if data.get("plot", False):
                        print(f"Plotting events for {plot_id}")
                        ax.clear()  # Clear previous plot
                        ax.imshow(self.plots[plot_id]["event_img"].copy())
                        ax.set_title(data.get("title", "Event Plot"))
                        ax.axis("off")
                        self.plots[plot_id]["event_img"].fill(0)  # Reset for next frame

                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.draw()

            def _update_plot(self, plot_id, data, plot_type):
                if plot_id not in self.axes:
                    return
                ax = self.axes[plot_id]
                self._create_plot(plot_id, data, plot_type, ax)

            def _process_commands(self):
                while not command_queue.empty():
                    try:
                        cmd = command_queue.get_nowait()

                        if cmd.action == "add":
                            if len(self.plots) < self.max_plots:
                                self.plots[cmd.plot_id] = {
                                    "data": cmd.data,
                                    "plot_type": cmd.plot_type,
                                }
                                self._reorganize_subplots()

                        elif cmd.action == "update":
                            if cmd.plot_id in self.plots:
                                self.plots[cmd.plot_id]["data"] = cmd.data
                                self._update_plot(cmd.plot_id, cmd.data, cmd.plot_type)
                            else:
                                if len(self.plots) < self.max_plots:
                                    self.plots[cmd.plot_id] = {
                                        "data": cmd.data,
                                        "plot_type": cmd.plot_type,
                                    }
                                    self._reorganize_subplots()

                        elif cmd.action == "remove":
                            if cmd.plot_id in self.plots:
                                del self.plots[cmd.plot_id]
                                if cmd.plot_id in self.axes:
                                    del self.axes[cmd.plot_id]
                                self._reorganize_subplots()

                        elif cmd.action == "clear":
                            self.plots.clear()
                            self.axes.clear()
                            self.fig.clear()

                    except:
                        break

            def _animation_func(self, frame):
                if stop_event.is_set():
                    return []
                self._process_commands()
                return []

            def run(self):
                self._setup_figure()
                ani = animation.FuncAnimation(
                    self.fig, self._animation_func, interval=50, blit=False, cache_frame_data=True
                )
                try:
                    # Keep the process alive with non-blocking updates
                    while not stop_event.is_set():
                        plt.pause(0.1)
                except KeyboardInterrupt:
                    pass
                finally:
                    plt.close("all")

        # Run the process visualizer
        viz = ProcessDynamicVisualizer()
        viz.run()

    def start(self):
        """Start the visualizer process"""
        if self.process is None or not self.process.is_alive():
            self.stop_event.clear()
            self.process = Process(
                target=self._run_visualizer_process,
                args=(self.command_queue, self.stop_event, self.figsize, self.max_plots),
            )
            self.process.start()
            time.sleep(2)  # Give process time to initialize
            if not self.process.is_alive():
                raise RuntimeError("Failed to start the visualizer process.")

    def stop(self):
        """Stop the visualizer process"""
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

    def add_plot(self, plot_id, data, plot_type="line"):
        """Add a new plot"""
        cmd = PlotCommand("add", plot_id, data, plot_type)
        self.command_queue.put(cmd)

    def update_plot(self, plot_id, data, plot_type="line"):
        """Update an existing plot"""
        cmd = PlotCommand("update", plot_id, data, plot_type)
        self.command_queue.put(cmd)

    def remove_plot(self, plot_id):
        """Remove a plot"""
        cmd = PlotCommand("remove", plot_id)
        self.command_queue.put(cmd)

    def clear_all(self):
        """Clear all plots"""
        cmd = PlotCommand("clear", "")
        self.command_queue.put(cmd)


def example_process_usage():
    """Example of using the process visualizer"""
    print("Starting process visualizer...")
    viz = AsyncVisualizer()
    viz.start()

    # Add some plots
    print("Adding plots...")
    viz.add_plot(
        "cos_wave",
        {
            "x": np.linspace(0, 4 * np.pi, 100),
            "y": np.cos(np.linspace(0, 4 * np.pi, 100)),
            "xlabel": "Time",
            "ylabel": "Amplitude",
        },
        "line",
    )

    time.sleep(1)
    viz.add_plot(
        "bar_chart",
        {
            "x": ["A", "B", "C", "D", "E"],
            "y": [23, 45, 56, 78, 32],
            "xlabel": "Category",
            "ylabel": "Value",
        },
        "bar",
    )

    # Update plots
    print("Updating plots...")
    for i in range(1000):
        time.sleep(0.5)
        viz.update_plot(
            "bar_chart",
            {
                "x": ["A", "B", "C", "D", "E"],
                "y": np.random.randint(10, 100, 5).tolist(),
                "xlabel": "Category",
                "ylabel": "Value",
            },
            "bar",
        )
        viz.update_plot(
            "cos_wave",
            {
                "x": np.linspace(0, 4 * np.pi, 100),
                "y": np.cos(np.linspace(0, 4 * np.pi, 100) + i * 0.1),
                "xlabel": "Time",
                "ylabel": "Amplitude",
            },
            "line",
        )

    print("Stopping visualizer...")
    viz.stop()


if __name__ == "__main__":
    example_process_usage()
    print("Demo completed!")
