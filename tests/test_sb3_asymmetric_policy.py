import gymnasium as gym
import numpy as np
import torch

from neurosim.rl.sb3_features import (
    AsymmetricActorCriticPolicy,
    CombinedEventStateExtractor,
    PRIVILEGED_KEY,
    SpatialSoftmax,
)


def _policy() -> AsymmetricActorCriticPolicy:
    torch.manual_seed(7)
    observation_space = gym.spaces.Dict(
        {
            "events": gym.spaces.Box(0.0, 1.0, (2, 16, 16), np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            PRIVILEGED_KEY: gym.spaces.Box(
                -np.inf, np.inf, (3,), np.float32
            ),
        }
    )
    return AsymmetricActorCriticPolicy(
        observation_space,
        gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=CombinedEventStateExtractor,
        normalize_images=False,
    )


def _observation(privileged: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "events": torch.rand(1, 2, 16, 16),
        "state": torch.tensor([[0.1, -0.2, 0.3, -0.4]]),
        PRIVILEGED_KEY: privileged.reshape(1, 3),
    }


def test_actor_distribution_is_invariant_to_privileged_channel() -> None:
    policy = _policy()
    observation = _observation(torch.zeros(3))
    changed = dict(observation)
    changed[PRIVILEGED_KEY] = torch.tensor([[100.0, -50.0, 25.0]])

    first = policy.get_distribution(observation).distribution.mean
    second = policy.get_distribution(changed).distribution.mean

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_critic_features_receive_privileged_channel() -> None:
    policy = _policy()
    assert policy.share_features_extractor is False
    assert policy.vf_features_extractor.use_privileged is True

    observation = _observation(torch.zeros(3))
    changed = dict(observation)
    changed[PRIVILEGED_KEY] = torch.full((1, 3), 100.0)
    base = policy.vf_features_extractor(observation)
    modified = policy.vf_features_extractor(changed)

    assert not torch.equal(base, modified)


def test_spatial_softmax_can_preserve_presence_strength() -> None:
    layer = SpatialSoftmax(include_presence=True)
    feature_map = torch.zeros(1, 2, 4, 4)
    feature_map[0, 0, 1, 2] = 3.0

    features = layer(feature_map)

    assert features.shape == (1, 8)
    torch.testing.assert_close(features[0, 4:6], torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(features[0, 6:8], torch.tensor([3.0 / 16.0, 0.0]))
