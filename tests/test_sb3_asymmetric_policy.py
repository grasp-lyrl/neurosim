import gymnasium as gym
import numpy as np
import torch
from types import SimpleNamespace

from neurosim.rl.sb3_features import (
    AsymmetricActorCriticPolicy,
    AsymmetricGruActorCriticPolicy,
    CombinedEventStateExtractor,
    PRIVILEGED_KEY,
    PrivilegedFusionEventStateExtractor,
    SpatialSoftmax,
)
from applications.rl.bc_warm_start import load_gru_clone_into_policy
from applications.rl.train_sb3 import ActorFreezeWarmupCallback


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


def _gru_policy() -> AsymmetricGruActorCriticPolicy:
    observation_space = gym.spaces.Dict(
        {
            "events": gym.spaces.Box(0.0, 1.0, (8, 60, 80), np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
            PRIVILEGED_KEY: gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
        }
    )
    return AsymmetricGruActorCriticPolicy(
        observation_space,
        gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        lr_schedule=lambda _: 5e-5,
        features_extractor_class=PrivilegedFusionEventStateExtractor,
        features_extractor_kwargs={"features_dim": 192, "event_backbone": "small"},
        normalize_images=False,
        lstm_hidden_size=128,
        n_lstm_layers=1,
        net_arch={"pi": [64], "vf": [64, 64]},
        use_sde=True,
        squash_output=True,
    )


def test_gru_actor_uses_strict_state_mask_but_critic_receives_privilege() -> None:
    policy = _gru_policy()
    obs = {
        "events": torch.rand(1, 8, 60, 80),
        "state": torch.rand(1, 18),
        PRIVILEGED_KEY: torch.zeros(1, 9),
    }
    changed = {key: value.clone() for key, value in obs.items()}
    changed["state"] += 100.0
    changed["state"][:, 6] = obs["state"][:, 6]
    changed[PRIVILEGED_KEY] += 100.0
    state = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
    starts = torch.ones(1)

    first, next_state = policy.get_distribution(obs, state, starts)
    second, _ = policy.get_distribution(changed, state, starts)
    torch.testing.assert_close(
        first.distribution.mean, second.distribution.mean, rtol=0.0, atol=0.0
    )
    assert isinstance(policy.lstm_actor, torch.nn.GRU)
    assert next_state[0].shape == (1, 1, 128)
    torch.testing.assert_close(next_state[1], torch.zeros_like(next_state[1]))
    assert not torch.equal(
        policy.vf_features_extractor(obs),
        policy.vf_features_extractor(changed),
    )


def test_gru_warm_start_transfers_only_the_actor_twin() -> None:
    policy = _gru_policy()
    target = policy.state_dict()
    clone = {}
    reverse = {
        "pi_features_extractor.": "features_extractor.",
        "lstm_actor.": "gru.",
        "mlp_extractor.policy_net.": "policy_net.",
        "action_net.": "action_net.",
    }
    expected = {}
    for policy_prefix, clone_prefix in reverse.items():
        for key, tensor in target.items():
            if not key.startswith(policy_prefix) or "privileged_fusion" in key:
                continue
            clone_key = clone_prefix + key[len(policy_prefix) :]
            value = torch.randn_like(tensor)
            clone[clone_key] = value
            expected[key] = value
    clone["side_head.weight"] = torch.randn(3, 128)

    report = load_gru_clone_into_policy(policy, clone)

    assert len(report["updated"]) == len(expected)
    assert report["skipped"] == ["side_head.weight"]
    transferred = policy.state_dict()
    for key, value in expected.items():
        torch.testing.assert_close(transferred[key], value)


def test_actor_freeze_includes_recurrent_module() -> None:
    policy = _gru_policy()
    callback = ActorFreezeWarmupCallback(4)
    callback.model = SimpleNamespace(policy=policy)

    callback._set_actor_requires_grad(False)

    actor_modules = (
        policy.pi_features_extractor,
        policy.lstm_actor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
    )
    assert all(
        not parameter.requires_grad
        for module in actor_modules
        for parameter in module.parameters()
    )
    assert not policy.log_std.requires_grad
    assert any(
        parameter.requires_grad
        for parameter in policy.vf_features_extractor.parameters()
    )
