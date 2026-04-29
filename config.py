import numpy as np
import torch
import random
import os

def seed_everything(seed=42):
    """Sets seeds for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Global configuration
SEED = 42
TARGET_INTERACTIONS = 100_000
NEGATIVE_RATIO = 4