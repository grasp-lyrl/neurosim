import torch
import _cu_evsim_ext


def evsim_cuda(
    new_image: torch.Tensor,
    new_time: int,
    intensity_state_ub: torch.Tensor,
    intensity_state_lb: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    MINIMUM_CONTRAST_THRESHOLD_NEG: float = 0.35,
    MINIMUM_CONTRAST_THRESHOLD_POS: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated event simulation.

    Args:
        new_image: Input image tensor (H, W) on CUDA
        new_time: Timestamp in microseconds
        intensity_state_ub: Upper bound intensity state (H, W) on CUDA
        intensity_state_lb: Lower bound intensity state (H, W) on CUDA
        event_x_buf: Buffer for x coordinates of events (max size)
        event_y_buf: Buffer for y coordinates of events (max size)
        event_t_buf: Buffer for timestamps of events (max size)
        event_p_buf: Buffer for polarities of events (max size)
        MINIMUM_CONTRAST_THRESHOLD_NEG: Negative contrast threshold
        MINIMUM_CONTRAST_THRESHOLD_POS: Positive contrast threshold

    Returns:
        Tuple of (x, y, t, p) event tensors
    """
    return _cu_evsim_ext.evsim(
        new_image,
        new_time,
        intensity_state_ub,
        intensity_state_lb,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        MINIMUM_CONTRAST_THRESHOLD_NEG,
        MINIMUM_CONTRAST_THRESHOLD_POS,
    )


class EventSimulatorCUDA:

    # Maximum number of events to simulate per call
    MAX_EVENTS_SIM = 480 * 640

    # Contrast threshold for event generation (can be modified)
    MINIMUM_CONTRAST_THRESHOLD_NEG = 0.35
    MINIMUM_CONTRAST_THRESHOLD_POS = 0.35

    def __init__(
        self,
        W,
        H,
        start_time,
        first_image=None,
        device="cuda",
        contrast_threshold_neg=0.35,
        contrast_threshold_pos=0.35,
    ):
        self.H = H
        self.W = W
        self.device = device
        self.last_time = int(start_time)  # in us
        self.intensity_state_ub = None
        self.intensity_state_lb = None

        # Allow custom thresholds
        self.MINIMUM_CONTRAST_THRESHOLD_NEG = contrast_threshold_neg
        self.MINIMUM_CONTRAST_THRESHOLD_POS = contrast_threshold_pos

        # Initialize buffers immediately to avoid repeated allocations
        self._init_event_buffers()

        if first_image is not None:
            self.init(first_image)

    def _init_event_buffers(self):
        """Initialize persistent event buffers with proper memory alignment."""
        print(
            f"[evsim-cuda] Initializing event buffers for {self.MAX_EVENTS_SIM} events on {self.device}"
        )

        # Pre-allocate event buffers with proper memory alignment for coalesced access
        self.event_x_buf = torch.empty(
            self.MAX_EVENTS_SIM,
            dtype=torch.uint16,
            device=self.device,
            memory_format=torch.contiguous_format,
        ).contiguous()

        self.event_y_buf = torch.empty(
            self.MAX_EVENTS_SIM,
            dtype=torch.uint16,
            device=self.device,
            memory_format=torch.contiguous_format,
        ).contiguous()

        self.event_t_buf = torch.empty(
            self.MAX_EVENTS_SIM,
            dtype=torch.uint64,
            device=self.device,
            memory_format=torch.contiguous_format,
        ).contiguous()

        self.event_p_buf = torch.empty(
            self.MAX_EVENTS_SIM,
            dtype=torch.uint8,
            device=self.device,
            memory_format=torch.contiguous_format,
        ).contiguous()

    def init(self, current_image):
        """Initialize the event simulator with the first image."""
        print(
            "[evsim-cuda] Initialized event camera sim with sensor size: ",
            current_image.shape,
        )
        if not current_image.is_cuda:
            current_image = current_image.to(self.device)

        log_image = torch.log(current_image)
        self.intensity_state_ub = (log_image + self.MINIMUM_CONTRAST_THRESHOLD_POS).contiguous()
        self.intensity_state_lb = (log_image - self.MINIMUM_CONTRAST_THRESHOLD_NEG).contiguous()

    def set_contrast_thresholds(self, neg_threshold, pos_threshold):
        """Update contrast thresholds."""
        self.MINIMUM_CONTRAST_THRESHOLD_NEG = neg_threshold
        self.MINIMUM_CONTRAST_THRESHOLD_POS = pos_threshold
        print(f"[evsim-cuda] Updated thresholds: neg={neg_threshold}, pos={pos_threshold}")

    def image_callback(self, new_image, new_time):
        """
        Process new image and generate events.

        Args:
            new_image: The new image to process (H, W)
            new_time: Timestamp in microseconds

        Returns:
            Tuple of (x, y, t, p) event tensors or None if no events
        """
        if self.intensity_state_lb is None or self.intensity_state_ub is None:
            self.init(new_image)
            return None

        if not new_image.is_cuda:
            new_image = new_image.to(self.device)
        new_image = new_image.contiguous()

        try:
            events = evsim_cuda(
                new_image,
                int(new_time),  # in us int/np.uint64
                self.intensity_state_ub,
                self.intensity_state_lb,
                self.event_x_buf,
                self.event_y_buf,
                self.event_t_buf,
                self.event_p_buf,
                self.MINIMUM_CONTRAST_THRESHOLD_NEG,
                self.MINIMUM_CONTRAST_THRESHOLD_POS,
            )
            self.last_time = new_time
            return events

        except Exception as e:
            print(f"Error in CUDA event simulation: {e}")
            return None

    def reset(self, first_image=None):
        """Reset the simulator state."""
        self.intensity_state_ub = None
        self.intensity_state_lb = None
        if first_image is not None:
            self.init(first_image)

    def get_state(self):
        """Get current intensity states for debugging."""
        return {
            "intensity_state_ub": self.intensity_state_ub,
            "intensity_state_lb": self.intensity_state_lb,
            "last_time": self.last_time,
            "contrast_threshold_neg": self.MINIMUM_CONTRAST_THRESHOLD_NEG,
            "contrast_threshold_pos": self.MINIMUM_CONTRAST_THRESHOLD_POS,
        }

    def get_performance_info(self):
        """Get performance-related information."""
        return {
            "max_events": self.MAX_EVENTS_SIM,
            "device": self.device,
            "contrast_threshold_neg": self.MINIMUM_CONTRAST_THRESHOLD_NEG,
            "contrast_threshold_pos": self.MINIMUM_CONTRAST_THRESHOLD_POS,
            "buffer_memory_usage_mb": (
                self.event_x_buf.element_size() * self.event_x_buf.numel()
                + self.event_y_buf.element_size() * self.event_y_buf.numel()
                + self.event_t_buf.element_size() * self.event_t_buf.numel()
                + self.event_p_buf.element_size() * self.event_p_buf.numel()
            )
            / (1024 * 1024),
        }
