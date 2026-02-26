import numpy as np
from types import SimpleNamespace
from numba import njit, prange
import torch

from neurosim.core.event_sim.types import Events

EVENT_TYPE = np.dtype(
    [("timestamp", "f8"), ("x", "u2"), ("y", "u2"), ("polarity", "b")], align=True
)

TOL = 0.5
MINIMUM_CONTRAST_THRESHOLD = 0.01

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_ns": 1000,
        "max_events_per_frame": 200000,
    }
)


@njit(parallel=True, fastmath=True, cache=True)
def esim(
    x_end,
    current_image,
    previous_image,
    delta_time,
    crossings,
    last_time,
    output_events,
    spikes,
    refractory_period_ns,
    max_events_per_frame,
    n_pix_row,
):
    count = 0
    max_spikes = int(delta_time / (refractory_period_ns * 1e-3))
    for x in prange(x_end):
        itdt = np.log(current_image[x])
        it = np.log(previous_image[x])
        deltaL = itdt - it

        if np.abs(deltaL) < TOL:
            continue

        pol = np.sign(deltaL)

        cross_update = pol * TOL
        crossings[x] = np.log(crossings[x]) + cross_update

        lb = crossings[x] - it
        ub = crossings[x] - itdt

        pos_check = lb > 0 and (pol == 1) and ub < 0
        neg_check = lb < 0 and (pol == -1) and ub > 0

        spike_nums = (itdt - crossings[x]) / TOL
        cross_check = pos_check + neg_check
        spike_nums = np.abs(int(spike_nums * cross_check))

        crossings[x] = itdt - cross_update
        if spike_nums > 0:
            spikes[x] = pol

        spike_nums = max_spikes if spike_nums > max_spikes else spike_nums

        current_time = last_time
        for i in range(spike_nums):
            output_events[count].x = x % n_pix_row
            output_events[count].y = x // n_pix_row
            output_events[count].timestamp = np.round(current_time * 1e-6, 6)
            output_events[count].polarity = 1 if pol > 0 else -1

            count += 1
            current_time += (delta_time) / spike_nums

            if count == max_events_per_frame:
                return count

    return count


class EventSimulator:
    def __init__(self, W, H, first_image=None, first_time=0, config=CONFIG):
        self.H = H
        self.W = W
        self.config = config
        self.last_image = None
        self.npix = H * W  # Must be set before init() is called
        self.last_time = int(first_time)

        if first_image is not None:
            self.init(first_image)

    def init(self, first_image, first_time=None):
        print("Initialized event camera simulator with sensor size:", first_image.shape)

        self.resolution = first_image.shape  # The resolution of the image

        # We ignore the 2D nature of the problem as it is not relevant here
        # It makes multi-core processing more straightforward
        first_image = first_image.reshape(-1)

        if isinstance(first_image, torch.Tensor):
            first_image = first_image.cpu().numpy()

        # Allocations
        self.last_image = first_image.copy()
        self.current_image = first_image.copy()

        if first_time is not None:
            self.last_time = first_time

        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.event_count = 0
        self.spikes = np.zeros((self.npix))

    def reset(self, first_image=None):
        """Reset simulator state."""
        self.last_image = None
        if first_image is not None:
            self.init(first_image)

    def __call__(self, image, timestamp_us):
        """Process a new image and generate events.

        Args:
            image: Raw grayscale image (H, W), positive values
            timestamp_us: Timestamp in microseconds

        Returns:
            Events(x, y, t, p) named tuple or None if no events
        """
        if self.last_image is None:
            self.init(image, timestamp_us)
            return None

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        assert timestamp_us > 0
        assert image.shape == self.resolution
        image = image.reshape(-1)  # Free operation

        np.copyto(self.current_image, image)

        delta_time = timestamp_us - self.last_time

        config = self.config
        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.spikes = np.zeros((self.npix))

        self.crossings = self.last_image.copy()
        self.event_count = esim(
            self.H * self.W,
            self.current_image,
            self.last_image,
            delta_time,
            self.crossings,
            self.last_time,
            self.output_events,
            self.spikes,
            config.refractory_period_ns,
            config.max_events_per_frame,
            self.W,
        )

        np.copyto(self.last_image, self.current_image)
        self.last_time = timestamp_us

        if self.event_count == 0:
            return None

        result = self.output_events[: self.event_count]
        result.sort(order=["timestamp"], axis=0)
        return Events(
            x=result["x"],
            y=result["y"],
            t=result["timestamp"],
            p=result["polarity"],
        )
