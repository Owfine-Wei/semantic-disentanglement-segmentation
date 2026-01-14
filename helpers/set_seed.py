import os
import random
import numpy as np

def setup_seed(seed=42, deterministic=True, is_enabled=True,
               set_cublas_workspace: bool = True,
               cublas_workspace_config: str = ':4096:8'):
    """
    Optimized seed locking function for semantic segmentation fine-tuning.
    Ensures consistent data augmentation and shuffling while avoiding 
    RuntimeErrors from non-deterministic loss functions.
    """

    if not is_enabled:
        print("No seed fixed")
        return

    # 1. Set CuBLAS environment variable (Crucial for reproducibility on CUDA >= 10.2)
    if deterministic and set_cublas_workspace:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config
        print(f"Set CUBLAS_WORKSPACE_CONFIG={cublas_workspace_config}")

    import torch
    from mmengine.runner import set_random_seed

    # 2. MMEngine high-level seed setting 
    # Handles DataLoader workers, distributed rank offsets, and basic library seeds.
    # We set deterministic=False here to manually control it with more precision below.
    set_random_seed(seed, deterministic=False)

    # 3. Redundant low-level locking for core libraries
    # This ensures consistency for data augmentation and batch shuffling.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 4. Deterministic behavior control
    if deterministic:
        # Enable deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

if __name__ == "__main__":
    setup_seed(42)