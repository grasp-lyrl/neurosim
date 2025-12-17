"""Utilities for consistent logging across neurosim."""


def format_dict(kwargs: dict) -> str:
    """Format kwargs for logging, showing names instead of memory addresses."""
    items = []
    for k, v in kwargs.items():
        if callable(v):
            if hasattr(v, "__name__"):
                value_str = f"<function: {v.__name__}>"
            elif hasattr(v, "__func__"):
                value_str = f"<method: {v.__func__.__name__}>"
            else:
                value_str = "<callable>"
        elif hasattr(v, "__class__") and not isinstance(
            v, (str, int, float, bool, list, dict, tuple)
        ):
            value_str = f"<{v.__class__.__name__}>"
        else:
            value_str = repr(v)
        items.append(f"{k}={value_str}")
    return "\n" + "\n".join(items)
