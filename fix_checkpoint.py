import torch
import os

# ========== 这里填入你的模型路径 ==========
# 请将下面的路径修改为你那个跑分异常的模型文件路径
FILE_PATH = "/root/autodl-tmp/models/_1_11_2026_BL+CSG+CI+back__A1.0B0.0_.pth" 
# ========================================

def fix_checkpoint(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在！请检查路径。")
        return

    print(f"正在加载模型: {file_path} ...")
    state_dict = torch.load(file_path, map_location='cpu')
    
    new_state_dict = {}
    fixed_count = 0
    
    keys = list(state_dict.keys())
    for k in keys:
        v = state_dict[k]
        new_k = k
        
        # 1. 剥离 DDP 的 module. 前缀
        if new_k.startswith('module.'):
            new_k = new_k[7:]
            
        # 2. 剥离 AuxModelWrapper 的 model. 前缀
        # 这是导致本来就有权重的模型在加载时匹配失败的主因
        if new_k.startswith('model.'):
            new_k = new_k[6:]
            fixed_count += 1
            
        new_state_dict[new_k] = v

    # 备份原文件
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(file_path, backup_path)
        print(f"原文件已备份为: {backup_path}")
    
    # 保存修复后的文件
    torch.save(new_state_dict, file_path)
    print(f"修复完成！共修正了 {fixed_count} 个参数 Key。")
    print(f"文件已覆写保存至: {file_path}")
    print("现在你可以重新运行 val.py 进行评估了。")

if __name__ == "__main__":
    if "你的模型路径" in FILE_PATH:
        print("请先修改脚本中的 FILE_PATH 变量为你实际的模型路径！")
    else:
        fix_checkpoint(FILE_PATH)
