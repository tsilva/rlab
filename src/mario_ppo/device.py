from __future__ import annotations


def resolve_sb3_device(device: str) -> str:
    """Resolve SB3's `auto` device with Apple MPS support."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
