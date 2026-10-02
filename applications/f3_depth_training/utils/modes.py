"""What the network predicts: relative disparity, or metric depth in metres.

The loss, the target's units and whether evaluation first aligns the prediction differ
between the two, and only here.
"""

from torch import Tensor

from .losses import ScaleAndShiftInvariantLoss, SiLogLoss
from .metrics import METRICS, align_least_squares, depth_metrics

ALIGNED = ("abs_rel", "d1", "rmse", "abs_rel_near", "d1_near")


class RelativeDepth:
    """Disparity up to scale and shift: SSI loss, metrics after an affine fit to the truth."""

    metric_names = METRICS

    def __init__(self, min_disparity: float, max_disparity: float, loss_fn):
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.min_depth, self.max_depth = 1.0 / max_disparity, 1.0 / min_disparity
        self.loss_fn = loss_fn

    def target(self, depth: Tensor) -> tuple[Tensor, Tensor]:
        """Depth in metres, 0 where invalid -> disparity and the pixels the loss may see."""
        disparity = 1.0 / depth.clamp(
            0.5 / self.max_disparity, 1.0 / self.min_disparity
        )
        return disparity, (disparity < self.max_disparity) & (
            disparity > self.min_disparity
        )

    def to_metres(self, x: Tensor) -> Tensor:
        """Disparity -> metres, up to the gauge this mode never fixes."""
        return 1.0 / x.clamp(min=self.min_disparity)

    def metrics(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
        aligned = align_least_squares(pred, target, mask).clamp(min=self.min_disparity)
        return depth_metrics(
            1.0 / aligned, 1.0 / target.clamp(min=self.min_disparity), mask
        )


class MetricDepth:
    """Depth in metres as the camera sees it. SiLog loss, metrics as predicted.

    The network gets no intrinsics, so while the FOV is randomized the metric depth of a
    given input is ambiguous by the focal-length ratio and the head can only learn the
    average; narrowing the hfov range is what bounds that.
    """

    metric_names = (*METRICS, *(f"{k}_aligned" for k in ALIGNED))

    def __init__(
        self, min_depth: float, max_depth: float, head_max_depth: float, loss_fn
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.head_max_depth = head_max_depth
        self.loss_fn = loss_fn

    def target(self, depth: Tensor) -> tuple[Tensor, Tensor]:
        """Depth in metres, 0 where invalid -> the target and the pixels the loss may see."""
        return depth, (depth > self.min_depth) & (depth < self.max_depth)

    def to_metres(self, x: Tensor) -> Tensor:
        """Already metres."""
        return x

    def metrics(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
        """In metres as predicted, plus `*_aligned`: the same output under the relative protocol."""
        depth = pred.clamp(min=self.min_depth)
        truth = target.clamp(min=self.min_depth)
        aligned = align_least_squares(1.0 / depth, 1.0 / truth, mask).clamp(
            min=1.0 / self.max_depth
        )
        relative = depth_metrics(1.0 / aligned, truth, mask)
        return depth_metrics(depth, truth, mask) | {
            f"{k}_aligned": relative[k] for k in ALIGNED
        }


def build_mode(conf: dict, metric: bool) -> RelativeDepth | MetricDepth:
    """The mode a training config asks for; its loss has to agree with `metric`."""
    loss = conf["loss"]
    if metric:
        assert loss in ("silog", "siloggrad"), (
            f"--metric trains with silog or siloggrad, not {loss}"
        )
        alpha = conf["alpha"] if loss == "siloggrad" else 0.0
        loss_fn = SiLogLoss(conf["lambd"], alpha, conf["scales"])
        return MetricDepth(
            conf["min_depth"], conf["max_depth"], conf["head_max_depth"], loss_fn
        )
    assert loss == "ssimae", f"{loss} needs --metric"
    loss_fn = ScaleAndShiftInvariantLoss(conf["alpha"], conf["scales"])
    return RelativeDepth(conf["min_disparity"], conf["max_disparity"], loss_fn)
