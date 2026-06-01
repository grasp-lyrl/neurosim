"""
Online Dataloader Module — **ARCHIVED, do not import.**

.. deprecated::
    This package targets a removed Cortex API (``neurosim.cortex.utils.ZMQNODE``
    no longer exists) and pairs depth/events by *arrival order* rather than
    timestamp. Kept only as reference while the replacement is built in
    ``neurosim.online_data`` (see ``scaling-online-depth-training-plan.md``).
    Salvage value: preallocated batch buffers + event normalization in
    ``dataloader.py``.

Original design (for reference): two processes —
1. Publisher (simulator side): runs sims, publishes data via ZMQ.
2. Subscriber (shared storage): receives via ZMQ, buffers for the DataLoader.
3. DataLoader (training side): builds PyTorch batches.

The eager re-exports below are intentionally **removed** so that
``import neurosim.sims.online_dataloader_old`` does not blow up on the missing
``neurosim.cortex`` dependency. Import the specific archived module directly if
you need to read it.
"""

__all__: list[str] = []
