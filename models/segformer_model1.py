"""
SegFormer model initialization helper using MMSegmentation configs.

Provides `get_model(num_classes, checkpoint, device)` which builds a
SegFormer (MiT-B3) model from an mmseg config, updates the decode head
class count, and optionally loads weights from a checkpoint.
"""

import torch
from .registry import register_models
from transformers import SegformerForSemanticSegmentation

@register_models("segformer")
def get_model(num_classes, checkpoint=None):
    """
    加载 SegFormer-B3 模型。
    :param num_classes: 分类数量（例如 Cityscapes 为 19）
    :param checkpoint: 本地 .pth 文件路径 或 Hugging Face 模型 ID
    """
    # 官方 B3 基础模型 ID
    model_id = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

    if checkpoint and (checkpoint.endswith('.pth') or checkpoint.endswith('.bin')):
        # 情况 A: 从本地特定的权重文件（state_dict）载入
        print(f"Loading model from local checkpoint: {checkpoint}")
        
        # 先创建一个结构对应的模型
        # ignore_mismatched_sizes=True 是核心，防止 num_classes 不一致时报错
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        
        # 载入权重字典
        state_dict = torch.load(checkpoint, map_location='cpu')
        
        # 兼容性处理：如果 state_dict 包含 'model' 键（常见于某些训练框架）
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        model.load_state_dict(state_dict)
    
    elif checkpoint:
        # 情况 B: checkpoint 是一个 Hugging Face 的 ID 或者包含配置文件的文件夹
        print(f"Loading model from Hugging Face / Folder: {checkpoint}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    
    else:
        # 情况 C: 直接加载官方预训练权重
        print(f"Loading official pretrained model: {model_id}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    # === Torch 2.x 优化：转换为内存连续格式 ===
    # 这会配合你之后的图像 memory_format=torch.channels_last 达到最高推理速度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device, memory_format=torch.channels_last)

    return model
