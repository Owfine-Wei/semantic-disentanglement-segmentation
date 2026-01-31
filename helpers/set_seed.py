# helpers/set_seed.py
import os
import random
import numpy as np

def setup_seed(seed=42, deterministic=True, rank=0):
    """
    seed: 基础种子
    deterministic: 是否开启严格确定性模式
    rank: 当前进程的编号（用于 DDP 训练，防止所有卡的数据完全一致）
    """
    # 1. 计算当前进程的独立种子
    process_seed = seed + rank

    # 2. 基础 Python 与 NumPy 种子
    random.seed(process_seed)
    np.random.seed(process_seed)
    os.environ['PYTHONHASHSEED'] = str(process_seed)

    # 3. 延迟导入 torch 以确保环境变量生效
    import torch
    
    # 4. PyTorch 种子设置
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    if deterministic:
        # 5. cuDNN 确定性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 6. 强制使用确定性算法（如果遇到无法确定的算子会报错，帮助排查）
        # 某些版本如果遇到不可避免的随机性会抛出 RuntimeError
        # torch.use_deterministic_algorithms(True)

    print(f"[Rank {rank}] Seed {process_seed} locked (Deterministic: {deterministic})")

if __name__ == "__main__":
    setup_seed(42)