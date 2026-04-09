"""
Reproducibility utilities.

Sets random seeds across all libraries (Python, NumPy, PyTorch)
so that experiments can be reproduced from a fresh run.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value. Default 42.

    Note:
        We do NOT set torch.backends.cudnn.deterministic = True here
        because it significantly slows down training. Enable it manually
        if you need bit-exact reproducibility for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(42)
    print(f"Python random: {random.random():.6f}")
    print(f"NumPy random:  {np.random.random():.6f}")
    print(f"Torch random:  {torch.rand(1).item():.6f}")
    print("Seeds set. Re-run to verify identical values.")
