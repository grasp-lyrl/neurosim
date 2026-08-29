"""The pre-tanh mean penalty, which both v23 and v24 died for lack of."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applications" / "rl"))

pytest.importorskip("wandb")

from train_sb3 import PreTanhPenalty  # noqa: E402


def _action_net(out_features: int = 3) -> torch.nn.Linear:
    torch.manual_seed(11)
    return torch.nn.Linear(4, out_features)


def test_penalty_adds_the_l2_gradient_of_the_pre_tanh_mean():
    """The added gradient must equal d/dmu of coef * mean over the batch.

    squash_output bounds the action but leaves the mean behind it free, so
    nothing opposes outward drift. Measured on v24, mean |mu| held at
    0.001-0.002 through 131k steps and then reached 0.567 (max 3.661) by
    229k as success collapsed to 0.0%.
    """
    coef = 0.25
    net = _action_net()
    features = torch.randn(8, 4)

    net.zero_grad()
    plain = net(features)
    plain.sum().backward()
    baseline = net.weight.grad.clone()

    penalty = PreTanhPenalty(net, coef)
    try:
        net.zero_grad()
        mu = net(features)
        mu.sum().backward()
        with_penalty = net.weight.grad.clone()
    finally:
        penalty.remove()

    # d/dmu of coef * sum(mu^2)/N is 2 * coef * mu / N, and the chain rule
    # carries it to the weights through the same linear map.
    net.zero_grad()
    mu = net(features)
    expected_extra = torch.autograd.grad(
        coef * mu.pow(2).sum() / mu.shape[0], net.weight
    )[0]

    assert torch.allclose(with_penalty - baseline, expected_extra, atol=1e-6)


def test_penalty_pulls_the_mean_inward_and_baseline_does_not():
    """Optimising the penalty alone must shrink the mean toward zero."""
    net = _action_net()
    with torch.no_grad():
        net.weight.mul_(0.0)
        net.bias.fill_(4.0)  # deep in the tanh rail, where the gradient dies
    features = torch.randn(16, 4)

    penalty = PreTanhPenalty(net, 0.5)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    try:
        with torch.no_grad():
            start = float(net(features).abs().mean())
        for _ in range(20):
            optimizer.zero_grad()
            # No task loss at all: the penalty is the only gradient source.
            net(features).sum().mul_(0.0).backward()
            optimizer.step()
        with torch.no_grad():
            end = float(net(features).abs().mean())
    finally:
        penalty.remove()

    assert end < start
    assert end < 4.0


def test_zero_coef_only_observes():
    """At coef 0 the hook must log the drift without touching the gradient."""
    net = _action_net()
    features = torch.randn(8, 4)

    net.zero_grad()
    net(features).sum().backward()
    baseline = net.weight.grad.clone()

    penalty = PreTanhPenalty(net, 0.0)
    try:
        net.zero_grad()
        net(features).sum().backward()
        assert torch.allclose(net.weight.grad, baseline, atol=1e-8)
        # ...but the diagnostic is still recorded, so a run that chooses not
        # to regularise can still see the drift coming.
        assert penalty.abs_mean > 0.0
    finally:
        penalty.remove()


def test_statistics_track_the_rail_crossing():
    """frac_gt1 and max_abs are what flagged v24's collapse."""
    net = _action_net(out_features=4)
    penalty = PreTanhPenalty(net, 0.0)
    try:
        with torch.no_grad():
            net.weight.mul_(0.0)
            net.bias.copy_(torch.tensor([0.1, 0.2, 3.0, -5.0]))
            net(torch.randn(6, 4))
        assert penalty.abs_mean == pytest.approx((0.1 + 0.2 + 3.0 + 5.0) / 4, abs=1e-5)
        assert penalty.frac_gt1 == pytest.approx(0.5)
        assert penalty.pop_max() == pytest.approx(5.0, abs=1e-5)
        # pop_max resets, so each rollout reports its own peak.
        with torch.no_grad():
            net(torch.randn(6, 4))
        assert penalty.pop_max() == pytest.approx(5.0, abs=1e-5)
    finally:
        penalty.remove()


def test_removal_detaches_the_hook():
    net = _action_net()
    features = torch.randn(8, 4)
    penalty = PreTanhPenalty(net, 0.5)
    penalty.remove()

    net.zero_grad()
    net(features).sum().backward()
    with_hook_removed = net.weight.grad.clone()

    net.zero_grad()
    net(features).sum().backward()
    assert torch.allclose(net.weight.grad, with_hook_removed, atol=1e-8)


def test_build_policy_config_squashes_every_actor_variant():
    """The BC clone reads this flag to decide whether to apply tanh.

    When squash_output was added to fix the action-box escape, CloneNet was
    not updated and stayed unbounded, so bc_warm_start copied action_net
    into a policy that squashed it a second time -- an expert command of 1.0
    arriving as 0.76, worst in exactly the full-authority dodges. The clone
    now mirrors the flag, which only works if the flag is actually set here.
    """
    from train_sb3 import build_policy_config  # noqa: PLC0415

    for kwargs in (
        dict(obs_mode="combined", log_std_init=-0.5, privileged=True),
        dict(obs_mode="combined", log_std_init=-0.5),
        dict(obs_mode="state", log_std_init=-0.5),
        dict(obs_mode="events", log_std_init=-0.5),
    ):
        _, policy_kwargs = build_policy_config(**kwargs)
        assert policy_kwargs["squash_output"] is True, kwargs


def test_squash_target_limit_keeps_saturated_experts_reachable():
    """Saturated +-1 expert labels must be reachable at a finite mean.

    tanh cannot reach 1, so regressing onto a saturated label drives the
    pre-tanh mean outward without bound -- the same drift measured on v24
    (mean |mu| 0.002 -> 0.567 as success fell to 0.0%).
    """
    from train_velocity_dodge_bc import SQUASH_TARGET_LIMIT  # noqa: PLC0415

    assert 0.0 < SQUASH_TARGET_LIMIT < 1.0
    pre_tanh = float(torch.atanh(torch.tensor(SQUASH_TARGET_LIMIT)))
    assert torch.isfinite(torch.tensor(pre_tanh))
    # Comfortably inside the region v24 collapsed from (max |mu| 3.66).
    assert pre_tanh < 3.0


def test_discretize_lateral_labels_one_action_once():
    """A single (3,) action must yield one label, not three.

    Testing ndim > 1 mistook a single action's three components for three
    samples, so the collate step got ragged shapes and the run died with
    "all input arrays must have the same shape".
    """
    import numpy as np  # noqa: PLC0415

    from train_velocity_dodge_bc import discretize_lateral  # noqa: PLC0415

    single = discretize_lateral(np.array([0.0, -0.9, 0.0], dtype=np.float32))
    assert np.shape(single) == ()
    assert int(single) == 0

    batch = discretize_lateral(
        np.array([[0, -1.0, 0], [0, 0.0, 0], [0, 0.9, 0]], dtype=np.float32)
    )
    assert batch.tolist() == [0, 1, 2]

    # Only the lateral component decides the class; the zeroed axes are
    # ignored rather than voting.
    assert int(discretize_lateral(np.array([9.0, 0.0, -9.0], dtype=np.float32))) == 1


def test_discretize_lateral_deadband_is_symmetric():
    import numpy as np  # noqa: PLC0415

    from train_velocity_dodge_bc import (  # noqa: PLC0415
        DISCRETE_DEADBAND,
        discretize_lateral,
    )

    eps = 1e-4
    inside = np.array(
        [[0, DISCRETE_DEADBAND - eps, 0], [0, -(DISCRETE_DEADBAND - eps), 0]],
        dtype=np.float32,
    )
    assert discretize_lateral(inside).tolist() == [1, 1]
    outside = np.array(
        [[0, DISCRETE_DEADBAND + eps, 0], [0, -(DISCRETE_DEADBAND + eps), 0]],
        dtype=np.float32,
    )
    assert discretize_lateral(outside).tolist() == [2, 0]
