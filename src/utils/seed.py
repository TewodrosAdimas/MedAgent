"""
Reproducibility and Random Seed Utilities for MedAgent.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch CPU/CUDA backends.

    Args:
        seed: Random seed integer (default: 42).
        deterministic_cuda: Whether to enforce deterministic CUDA convolution algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
