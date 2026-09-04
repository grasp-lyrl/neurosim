"""SB3 feature extractors and policies for event-based Neurosim RL observations."""

import gymnasium as gym
import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import (
    RecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

# Key carrying the critic-only observation; the actor never reads it.
PRIVILEGED_KEY = "privileged"


class SpatialSoftmax(nn.Module):
    """Reduce a feature map to the expected (x, y) location per channel.

    Global average pooling answers "is a feature present"; for dodging, the
    decisive fact is *where* the obstacle is, since its bearing sets the
    sign of the correction. Pooling to a scalar forces that to be encoded
    in which channels fire rather than where they fire -- a much harder
    thing for a randomly-initialised CNN to discover, especially before it
    has ever produced a successful dodge.

    This layer instead makes bearing a linear readout: it takes a softmax
    over each channel's spatial map and returns the expected image
    coordinates under that distribution, giving ``2 * channels`` outputs in
    normalised [-1, 1] image coordinates.
    """

    def __init__(self, temperature: float = 1.0, include_presence: bool = False):
        super().__init__()
        self.include_presence = bool(include_presence)
        # Learnable so the network can sharpen toward argmax ("the obstacle
        # is exactly here") or soften toward the centroid of diffuse motion.
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        softmax = torch.softmax(
            (x / self.log_temperature.exp()).reshape(n, c, h * w), dim=-1
        ).reshape(n, c, h, w)

        device, dtype = x.device, x.dtype
        pos_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        pos_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)

        expected_x = (softmax.sum(dim=2) * pos_x).sum(dim=-1)
        expected_y = (softmax.sum(dim=3) * pos_y).sum(dim=-1)
        parts = [expected_x, expected_y]
        if self.include_presence:
            # Location alone is ambiguous when a feature is absent: diffuse
            # background activation also has an expected coordinate. Retain
            # raw peak and mean strength so downstream layers can decide
            # whether a localized obstacle-like feature is present at all.
            parts.extend([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))])
        return torch.cat(parts, dim=1)


class _EventBackbone(nn.Module):
    """CNN for event frames, preserving spatial location.

    ``backbone`` selects the conv trunk:
      "small"  -- the original 3-layer CNN (~26k conv params).
      "large"  -- 4 layers, roughly double the channel width at every stage
                  (~250k params, ~10x). A cheap, fast-to-train probe for
                  whether capacity is what caps the auxiliary occupancy R2 at
                  ~0.10-0.15 -- measured with both random- and oracle-driven
                  pretraining data, so more/better DATA was already ruled out
                  as the limiting factor (see pretrain_encoder.py /
                  pretrain_encoder_oracle.py).
      "efficientnet_b0" -- torchvision's EfficientNet-B0 conv trunk (~4.0M
                  params), ImageNet-pretrained, classifier stripped -- what
                  the reference (arXiv 2603.07578) uses. Its native head is
                  global-average-pool + classifier, which throws away WHERE
                  a feature fires; that is exactly what SpatialSoftmax exists
                  to avoid (see its docstring), so this keeps the pretrained
                  trunk but routes its final feature map through
                  SpatialSoftmax instead of EfficientNet's own pooling head.
                  The input stem cannot use pretrained weights (it expects 3
                  RGB channels; ours is polarity x history_frames) and is
                  freshly initialised; every other layer loads ImageNet
                  weights.
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 64,
        include_presence: bool = False,
        high_resolution: bool = False,
        backbone: str = "small",
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "efficientnet_b0":
            import torchvision

            trunk = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            ).features
            stem = trunk[0][0]
            trunk[0][0] = nn.Conv2d(
                in_channels,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=stem.bias is not None,
            )
            self.cnn = trunk
            # EfficientNet-B0's final stage is 1280 channels; SpatialSoftmax
            # works on any channel count, so `channels` (out_dim below) must
            # track what the trunk actually outputs, not the caller's value.
            channels = 1280
        elif backbone == "large":
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    128,
                    channels,
                    kernel_size=3,
                    stride=1 if high_resolution else 2,
                    padding=1,
                ),
                nn.ReLU(),
            )
        elif backbone == "small":
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    32,
                    channels,
                    kernel_size=3,
                    stride=1 if high_resolution else 2,
                    padding=1,
                ),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"unknown backbone: {backbone!r}")
        self.spatial_softmax = SpatialSoftmax(include_presence=include_presence)
        self.out_dim = (4 if include_presence else 2) * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.cnn(x))


class EventCnnExtractor(BaseFeaturesExtractor):
    """Feature extractor for ``obs_mode=events``."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        event_presence_features: bool = False,
        event_high_resolution: bool = False,
        event_backbone: str = "small",
    ):
        super().__init__(observation_space, features_dim)

        in_channels = int(observation_space.shape[0])
        self.backbone = _EventBackbone(
            in_channels=in_channels,
            include_presence=event_presence_features,
            high_resolution=event_high_resolution,
            backbone=event_backbone,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(observations.float()))


class CombinedEventStateExtractor(BaseFeaturesExtractor):
    """Feature extractor for dict observations.

    Consumes whichever of ``events`` / ``state`` / ``privileged`` the space
    provides. ``use_privileged`` controls whether the privileged channel is
    read at all, which is how one extractor class serves both the actor
    (privileged off) and the critic (privileged on).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 192,
        use_privileged: bool = True,
        event_presence_features: bool = False,
        event_high_resolution: bool = False,
        event_backbone: str = "small",
    ):
        super().__init__(observation_space, features_dim)
        spaces = observation_space.spaces
        self.use_privileged = bool(use_privileged) and PRIVILEGED_KEY in spaces

        output_dim = 0
        self.event_backbone = None
        if "events" in spaces:
            self.event_backbone = _EventBackbone(
                in_channels=int(spaces["events"].shape[0]),
                include_presence=event_presence_features,
                high_resolution=event_high_resolution,
                backbone=event_backbone,
            )
            self.event_head = nn.Sequential(
                nn.Linear(self.event_backbone.out_dim, 128), nn.ReLU()
            )
            output_dim += 128

        self.state_head = None
        if "state" in spaces:
            self.state_head = nn.Sequential(
                nn.Linear(int(spaces["state"].shape[0]), 64), nn.ReLU()
            )
            output_dim += 64

        self.privileged_head = None
        if self.use_privileged:
            self.privileged_head = nn.Sequential(
                nn.Linear(int(spaces[PRIVILEGED_KEY].shape[0]), 32), nn.ReLU()
            )
            output_dim += 32

        if output_dim == 0:
            raise ValueError("observation space provides no usable keys")
        self._features_dim = output_dim

    def forward_events(self, observations: dict[str, torch.Tensor]):
        """Event-branch features only, or ``None`` when there is no camera.

        Exposed so an auxiliary task can be supervised on the event pathway
        alone. Attaching it to the concatenated features would let the
        network satisfy the auxiliary loss from proprioception, which is the
        exact shortcut the auxiliary task exists to close off.
        """
        if self.event_backbone is None:
            return None
        return self.event_head(self.event_backbone(observations["events"].float()))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.event_backbone is not None:
            parts.append(self.forward_events(observations))
        if self.state_head is not None:
            parts.append(self.state_head(observations["state"]))
        if self.privileged_head is not None:
            parts.append(self.privileged_head(observations[PRIVILEGED_KEY]))
        return torch.cat(parts, dim=1)


class PrivilegedFusionEventStateExtractor(CombinedEventStateExtractor):
    """Keep actor feature width BC-compatible while enriching the critic.

    The event BC/GRU actor was trained on 128 event features concatenated
    with 64 state features.  Appending a privileged branch would change the
    recurrent input width and make an exact warm start impossible.  Instead,
    this extractor adds a learned privileged projection to the 64-D state
    branch.  The asymmetric actor receives a zeroed privileged tensor and is
    therefore byte-for-byte equivalent to ``CombinedEventStateExtractor``;
    the critic receives the real projection without changing feature width.
    """

    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        kwargs["use_privileged"] = False
        super().__init__(observation_space, **kwargs)
        privileged = observation_space.spaces.get(PRIVILEGED_KEY)
        self.privileged_fusion = (
            nn.Linear(int(privileged.shape[0]), 64, bias=False)
            if privileged is not None
            else None
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.event_backbone is not None:
            parts.append(self.forward_events(observations))
        if self.state_head is not None:
            state = self.state_head(observations["state"])
            if self.privileged_fusion is not None:
                state = state + self.privileged_fusion(
                    observations[PRIVILEGED_KEY]
                )
            parts.append(state)
        return torch.cat(parts, dim=1)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy where only the value function sees ``privileged``.

    SB3 hands the same observation to both feature extractors and builds a
    single ``MlpExtractor`` from one feature width, so the actor and critic
    cannot simply have different-width encoders. The split is enforced on
    the input instead: the actor's observation has ``privileged`` replaced
    by zeros before its extractor runs. A constant input carries no
    information, so the actor is genuinely blind to it while both towers
    keep identical shapes -- which also keeps checkpoints loadable by the
    stock SB3 machinery.
    """

    def __init__(self, *args, **kwargs):
        # Separate towers, so the actor's encoder is not shaped by
        # gradients flowing through the critic's privileged input.
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _blind(obs: PyTorchObs) -> PyTorchObs:
        if not isinstance(obs, dict) or PRIVILEGED_KEY not in obs:
            return obs
        blinded = dict(obs)
        blinded[PRIVILEGED_KEY] = torch.zeros_like(obs[PRIVILEGED_KEY])
        return blinded

    def extract_features(self, obs, features_extractor=None):
        if self.share_features_extractor or features_extractor is not None:
            return super().extract_features(obs, features_extractor)
        pi_features = super(ActorCriticPolicy, self).extract_features(
            self._blind(obs), self.pi_features_extractor
        )
        vf_features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return pi_features, vf_features

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Blind the actor here too.

        SB3's implementation calls ``BasePolicy.extract_features`` directly
        rather than going through :meth:`extract_features`, so without this
        override the actor would receive the privileged channel on every
        ``predict()`` -- i.e. at rollout and evaluation time, silently.
        """
        return super().get_distribution(self._blind(obs))


class AsymmetricRecurrentActorCriticPolicy(RecurrentMultiInputActorCriticPolicy):
    """Recurrent twin of :class:`AsymmetricActorCriticPolicy`.

    A single event time surface shows *where* edges fired, not how fast an
    obstacle is closing, so distance and time-to-impact are not recoverable
    from one frame -- which is what the LSTM is for. The asymmetry is
    unchanged and enforced the same way: the actor's ``privileged`` channel
    is replaced by zeros before its extractor runs, so it is genuinely blind
    to it while both towers keep identical shapes.

    ``RecurrentMultiInputActorCriticPolicy`` inherits from
    ``ActorCriticPolicy``, so the blinding overrides transplant unchanged;
    only the base class differs.
    """

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _blind(obs: PyTorchObs) -> PyTorchObs:
        if not isinstance(obs, dict) or PRIVILEGED_KEY not in obs:
            return obs
        blinded = dict(obs)
        blinded[PRIVILEGED_KEY] = torch.zeros_like(obs[PRIVILEGED_KEY])
        return blinded

    def extract_features(self, obs, features_extractor=None):
        if self.share_features_extractor or features_extractor is not None:
            return super().extract_features(obs, features_extractor)
        pi_features = super(ActorCriticPolicy, self).extract_features(
            self._blind(obs), self.pi_features_extractor
        )
        vf_features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return pi_features, vf_features

    def get_distribution(self, obs, lstm_states, episode_starts):
        """Blind the actor here too.

        SB3 calls ``BasePolicy.extract_features`` directly here rather than
        going through :meth:`extract_features`, so without this override the
        actor would receive the privileged channel at rollout and evaluation
        time, silently.
        """
        return super().get_distribution(
            self._blind(obs), lstm_states, episode_starts
        )


class AsymmetricGruActorCriticPolicy(RecurrentMultiInputActorCriticPolicy):
    """Recurrent PPO policy with a BC-compatible GRU actor.

    sb3-contrib's rollout buffer represents recurrent state as an LSTM
    ``(hidden, cell)`` pair.  The GRU uses the hidden member and returns a
    zero-valued placeholder for the unused cell member, retaining full
    compatibility with RecurrentPPO while allowing exact GRU weight transfer.
    The critic remains a separately initialized LSTM and sees privileged
    observations.  The actor sees only events and yaw error, matching the
    strict behavior-cloning observation contract.
    """

    def __init__(
        self,
        *args,
        actor_events_only_state: bool = True,
        actor_blank_previous_action: bool = True,
        **kwargs,
    ):
        self.actor_events_only_state = bool(actor_events_only_state)
        self.actor_blank_previous_action = bool(actor_blank_previous_action)
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

        n_layers = int(self.lstm_hidden_state_shape[0])
        self.lstm_actor = nn.GRU(
            self.features_dim,
            self.lstm_output_dim,
            num_layers=n_layers,
            **self.lstm_kwargs,
        ).to(self.device)
        # The parent constructed its optimizer before the LSTM was replaced.
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=learning_rate, **self.optimizer_kwargs
        )

    def _blind(self, obs: PyTorchObs) -> PyTorchObs:
        if not isinstance(obs, dict):
            return obs
        blinded = dict(obs)
        if PRIVILEGED_KEY in blinded:
            blinded[PRIVILEGED_KEY] = torch.zeros_like(blinded[PRIVILEGED_KEY])
        if "state" in blinded:
            state = blinded["state"].clone()
            if self.actor_events_only_state:
                yaw = state[..., 6:7].clone()
                state.zero_()
                state[..., 6:7] = yaw
            elif self.actor_blank_previous_action:
                action_dim = int(self.action_space.shape[0])
                state[..., -action_dim:] = 0.0
            blinded["state"] = state
        return blinded

    def extract_features(self, obs, features_extractor=None):
        if self.share_features_extractor or features_extractor is not None:
            return super().extract_features(obs, features_extractor)
        pi_features = super(ActorCriticPolicy, self).extract_features(
            self._blind(obs), self.pi_features_extractor
        )
        vf_features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return pi_features, vf_features

    def get_distribution(self, obs, lstm_states, episode_starts):
        return super().get_distribution(
            self._blind(obs), lstm_states, episode_starts
        )

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, recurrent):
        if not isinstance(recurrent, nn.GRU):
            return RecurrentMultiInputActorCriticPolicy._process_sequence(
                features, lstm_states, episode_starts, recurrent
            )

        hidden_state = lstm_states[0]
        n_seq = hidden_state.shape[1]
        sequence = features.reshape(
            (n_seq, -1, recurrent.input_size)
        ).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if torch.all(starts == 0.0):
            output, hidden_state = recurrent(sequence, hidden_state)
        else:
            outputs = []
            for step_features, episode_start in zip(sequence, starts, strict=True):
                output, hidden_state = recurrent(
                    step_features.unsqueeze(0),
                    (1.0 - episode_start).view(1, n_seq, 1) * hidden_state,
                )
                outputs.append(output)
            output = torch.cat(outputs)
        output = torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)
        return output, (hidden_state, torch.zeros_like(hidden_state))
