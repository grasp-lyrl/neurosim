"""
Online Dataloader Module.

This module provides infrastructure for generating training data from
simulations in real-time and loading it into PyTorch batches.

The system is designed to run as two separate processes:
1. Publisher (simulator side): Runs simulations and publishes data via ZMQ
2. Subscriber (shared storage): Receives data via ZMQ and buffers it
                                for access by the DataLoader
3. DataLoader (training side): Receives data and creates PyTorch batches
"""

from .datapublisher import DataPublisher
from .datasubscriber import DataSubscriber
from .dataloader import OnlineDataLoader

__all__ = [
    # Publisher (simulator side)
    "DataPublisher",
    # Subscriber (shared storage)
    "DataSubscriber",
    # Dataloader (training side)
    "OnlineDataLoader",
]
