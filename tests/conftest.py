"""Pytest hooks for neurosim.

Run tests from the **neurosim** conda env::

    conda activate neurosim
    cd /path/to/neurosim && PYTHONNOUSERSITE=1 python -m pytest tests
"""

import os
import warnings


def pytest_configure(config) -> None:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env is not None and env != "" and env != "neurosim":
        warnings.warn(
            f"CONDA_DEFAULT_ENV is {env!r}; Habitat RL tests expect conda env 'neurosim'.",
            stacklevel=1,
        )
