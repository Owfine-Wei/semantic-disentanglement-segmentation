"""
Comprehensive seed locking for Hybrid MMSegmentation/PyTorch projects.

"""

import os
import random
import numpy as np
import torch
from mmengine.runner import set_random_seed

def setup_seed(seed=42, deterministic=True, is_enabled = True):

    if is_enabled:
        # 1. MMEngine high-level seed setting
        # This handles DataLoader workers, distributed rank offsets, 
        # and basic library seeds (random, numpy, torch).
        set_random_seed(seed, deterministic=deterministic)

        # 2. Redundant low-level locking (Ensures coverage for custom PyTorch code)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 3. Environment variable for Python Hashing
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 4. CuDNN Determinism
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Optional: Force PyTorch to use deterministic algorithms 
            # (Warning: Some ops might throw errors if no deterministic impl exists)
            # torch.use_deterministic_algorithms(True)
            
        print(f"Random seed fixed: {seed} (Deterministic: {deterministic})")
    else:
        print("No seed fixed")

if __name__ == "__main__":
    setup_seed(42)