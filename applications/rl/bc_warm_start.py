"""Load behavior-cloned weights into an SB3 PPO policy.

The clone is built to mirror ``AsymmetricActorCriticPolicy``'s actor
exactly -- same extractor (privileged head present but blinded), same
64/64 Tanh trunk, same unbounded action head -- so every cloned tensor has
a counterpart here and the warm start transfers the learned action
mapping, not just the CNN.

The critic is deliberately left at its initialisation: it was never
trained by BC, and a value head that confidently predicts nonsense is
worse than one that starts uncertain.
"""

from __future__ import annotations

import torch


CLONE_TO_POLICY = {
    "features_extractor.": ("pi_features_extractor.", "vf_features_extractor."),
    "policy_net.": ("mlp_extractor.policy_net.",),
    "action_net.": ("action_net.",),
}


def load_clone_into_policy(policy, clone_state_dict: dict) -> dict:
    """Copy clone tensors onto ``policy`` in place; report what moved."""
    target = policy.state_dict()
    updated, skipped = [], []

    for key, tensor in clone_state_dict.items():
        destinations = []
        for prefix, targets in CLONE_TO_POLICY.items():
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                destinations = [t + suffix for t in targets]
                break
        if not destinations:
            skipped.append(key)
            continue
        for dest in destinations:
            if dest not in target:
                skipped.append(f"{key} -> {dest} (absent)")
            elif target[dest].shape != tensor.shape:
                skipped.append(
                    f"{key} -> {dest} ({tuple(tensor.shape)} vs "
                    f"{tuple(target[dest].shape)})"
                )
            else:
                target[dest] = tensor.clone()
                updated.append(dest)

    policy.load_state_dict(target)
    return {"updated": updated, "skipped": skipped}


def load_gru_clone_into_policy(policy, clone_state_dict: dict) -> dict:
    """Transfer the strict GRU clone into its RecurrentPPO actor twin."""
    mappings = {
        "features_extractor.": ("pi_features_extractor.",),
        "gru.": ("lstm_actor.",),
        "policy_net.": ("mlp_extractor.policy_net.",),
        "action_net.": ("action_net.",),
    }
    target = policy.state_dict()
    updated, skipped = [], []
    for key, tensor in clone_state_dict.items():
        destinations = []
        for prefix, targets in mappings.items():
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                destinations = [target_prefix + suffix for target_prefix in targets]
                break
        if not destinations:
            skipped.append(key)
            continue
        for destination in destinations:
            if destination not in target:
                skipped.append(f"{key} -> {destination} (absent)")
            elif target[destination].shape != tensor.shape:
                skipped.append(
                    f"{key} -> {destination} ({tuple(tensor.shape)} vs "
                    f"{tuple(target[destination].shape)})"
                )
            else:
                target[destination] = tensor.clone()
                updated.append(destination)
    policy.load_state_dict(target)
    return {"updated": updated, "skipped": skipped}
