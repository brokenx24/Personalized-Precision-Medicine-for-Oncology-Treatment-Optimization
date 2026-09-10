"""
Global determinism and seed management utility across PyTorch, NumPy, and Python standard library.
"""

import os
import random
import numpy as np

def set_global_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

GLOBAL_SEED = 42
