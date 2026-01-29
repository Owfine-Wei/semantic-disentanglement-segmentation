import os
import random
import numpy as np
import torch # 移动到顶部，因为现在是核心依赖

def setup_seed(seed=42, deterministic=True, is_enabled=True,
               set_cublas_workspace: bool = True,
               cublas_workspace_config: str = ':4096:8'):
    """
    针对语义分割微调优化的种子锁定函数（脱离 MMEngine 版）。
    """

    if not is_enabled:
        print("Seed locking is disabled.")
        return

    # 1. 设置 CuBLAS 环境变量 (对 CUDA >= 10.2 的复现至关重要)
    if deterministic and set_cublas_workspace:
        # 必须在第一次调用 CUDA 算子之前设置
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config
        print(f"Set CUBLAS_WORKSPACE_CONFIG={cublas_workspace_config}")

    # 2. 基础 Python 与 NumPy 种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. PyTorch CPU 与 GPU 种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多显卡

if __name__ == "__main__":
    setup_seed(42)