import numpy as np
import torch

from neurosim.rl.representations.event import EventRepresentationManager


def test_event_history_stacks_once_per_step():
    manager = EventRepresentationManager(
        model="event_frame",
        raw_height=4,
        raw_width=5,
        history_frames=3,
        event_device="cpu",
    )
    manager.reset_episode()
    initial = manager.observation()
    assert initial.shape == (6, 4, 5)
    assert not np.any(initial)

    manager.begin_step()
    events = type(
        "Events",
        (),
        {
            "x": torch.tensor([1]),
            "y": torch.tensor([2]),
            "t": torch.tensor([1000], dtype=torch.uint64),
            "p": torch.tensor([1]),
        },
    )()
    manager.accumulate(events)
    first = manager.observation()
    repeated = manager.observation()
    np.testing.assert_array_equal(first, repeated)
    assert first[-1, 2, 1] == 1.0
    assert not np.any(first[:4])

    manager.begin_step()
    second = manager.observation()
    assert second[-3, 2, 1] == 1.0
    assert not np.any(second[-2:])
