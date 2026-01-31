import os
import random
import numpy as np
import torch # 移动到顶部，因为现在是核心依赖

def setup_seed(seed=42, deterministic=True):
    if not deterministic:
        return

    # 1. 基础 Python 与 NumPy 种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. PyTorch 种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # 3. 核心：CUDA 与 CuDNN 确定性设置
    if deterministic:
        # 必选：禁用 cuDNN 的非确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 必选：设置 CUBLAS 环境变量
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Seed {seed} locked with full determinism.")

if __name__ == "__main__":
    setup_seed(42)