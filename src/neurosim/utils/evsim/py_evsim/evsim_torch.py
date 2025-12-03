import torch

#! ignoring REFRACTORY_PERIOD_NS for now, since our sim runs at 1ms and
#! we want to accumulate events at atleast 1ms. So doesn't make sense to
#! waste compute on fine grained events
REFRACTORY_PERIOD_NS = 1000

# Contrast threshold for event generation
MINIMUM_CONTRAST_THRESHOLD_NEG = 0.35
MINIMUM_CONTRAST_THRESHOLD_POS = 0.35


#! Implement caching of compiled code using
#! save_cache_artifacts and load_cache_artifacts
# @torch.compile(mode="reduce-overhead")
def esim(new_image, new_time, intensity_state_ub, intensity_state_lb):
    current_image = torch.log(new_image)
    mask_pos = current_image > intensity_state_ub
    mask_neg = current_image < intensity_state_lb
    mask = mask_pos | mask_neg
    events_yx = torch.nonzero(mask, as_tuple=True)
    events_yx = (events_yx[0].to(torch.int32), events_yx[1].to(torch.int32))  # (y, x)

    intensity_state_ub[events_yx] = current_image[events_yx] + MINIMUM_CONTRAST_THRESHOLD_POS
    intensity_state_lb[events_yx] = current_image[events_yx] - MINIMUM_CONTRAST_THRESHOLD_NEG
    intensity_state_ub[~mask] = torch.minimum(
        intensity_state_ub[~mask],
        current_image[~mask] + MINIMUM_CONTRAST_THRESHOLD_POS,
    )
    intensity_state_lb[~mask] = torch.maximum(
        intensity_state_lb[~mask],
        current_image[~mask] - MINIMUM_CONTRAST_THRESHOLD_NEG,
    )

    events_t = torch.full_like(events_yx[0], new_time, dtype=torch.uint64, device="cuda")
    events_p = torch.zeros_like(events_yx[0], dtype=torch.uint8, device="cuda")
    events_p[mask_pos[events_yx]] = 1

    return events_yx[1], events_yx[0], events_t, events_p  # x, y, t, p


class EventSimulator:
    def __init__(self, W, H, start_time, first_image=None):
        self.H = H
        self.W = W
        if first_image is not None:
            self.init(first_image)
        self.last_time = int(start_time)  # in us
        self.intensity_state_ub = None
        self.intensity_state_lb = None

    def init(self, current_image):
        print(
            "[evsim-torch] Initialized event camera sim with sensor size: ",
            current_image.shape,
        )
        self.intensity_state_ub = current_image + MINIMUM_CONTRAST_THRESHOLD_POS
        self.intensity_state_lb = current_image - MINIMUM_CONTRAST_THRESHOLD_NEG

    def image_callback(self, new_image, new_time):
        """
        new_image: The new image to process (H, W)
        """
        if self.intensity_state_lb is None:
            self.init(torch.log(new_image))
            return None

        events = esim(
            new_image,
            new_time,
            self.intensity_state_ub,
            self.intensity_state_lb,
        )
        self.last_time = new_time
        return events
