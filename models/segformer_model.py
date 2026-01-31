import torch
import os
from .registry import register_models
from transformers import SegformerForSemanticSegmentation, SegformerConfig

@register_models("segformer")
def get_model(num_classes, checkpoint=None):
    # 本地配置和基础权重的目录（必须包含 config.json）
    local_base_path = "../segformer-b3-weights/"
    
    # 1. 无论如何，先从本地加载配置并初始化模型结构
    # 这样可以保证在断网环境下也能建立模型
    print(f"Initializing model architecture from: {local_base_path}")
    
    # 如果你要加载的是一个孤立的 .bin 或 .pth 文件
    if checkpoint and (checkpoint.endswith('.bin') or checkpoint.endswith('.pth') or checkpoint.endswith('.pt')):
        print(f"Loading specific weight file: {checkpoint}")
        
        # 先用本地 config 实例化模型
        model = SegformerForSemanticSegmentation.from_pretrained(
            local_base_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
        
        # 手动加载权重文件
        state_dict = torch.load(checkpoint, map_location='cpu')
        
        # 核心：处理 HuggingFace 官方 bin 文件的 key 差异
        # 官方 bin 文件通常直接就是 state_dict，但也可能包裹在 "model" 键下
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # 载入权重
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from {checkpoint}")
        print(f"Note: {msg}")

    # 如果 checkpoint 是个文件夹路径（标准的 transformers 格式）
    elif checkpoint and os.path.isdir(checkpoint):
        model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
        
    # 如果没有 checkpoint，直接用本地目录里的基础权重
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            local_base_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )

    # 优化移动到显存
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device, memory_format=torch.channels_last)
    return model