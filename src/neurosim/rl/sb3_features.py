"""SB3 feature extractors for event-based Neurosim RL observations."""

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _EventBackbone(nn.Module):
    """Lightweight CNN for 2-channel event frames."""

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class EventCnnExtractor(BaseFeaturesExtractor):
    """Feature extractor for ``obs_mode=events``."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        in_channels = int(observation_space.shape[0])
        self.backbone = _EventBackbone(in_channels=in_channels)
        self.head = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(observations.float()))


class CombinedEventStateExtractor(BaseFeaturesExtractor):
    """Feature extractor for dict observations {events, state}."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim)

        events_space = observation_space.spaces["events"]
        state_space = observation_space.spaces["state"]

        event_channels = int(events_space.shape[0])
        state_dim = int(state_space.shape[0])

        self.event_backbone = _EventBackbone(in_channels=event_channels)
        self.event_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.state_head = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())

        self._features_dim = 128 + 64

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        event_features = self.event_head(
            self.event_backbone(observations["events"].float())
        )
        state_features = self.state_head(observations["state"])
        return torch.cat([event_features, state_features], dim=1)
