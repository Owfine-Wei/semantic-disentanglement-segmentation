"""
Model initialization helper for FCN-ResNet50-D8 using MMSegmentation.

Provides `get_model(num_classes, checkpoint, device)` which builds an FCN
model from an mmseg config, updates class counts, and optionally loads a
checkpoint onto the requested device.
"""

import os
import torch
from mmseg.apis import init_model
from mmengine.config import Config

def get_model(num_classes=19, checkpoint=None):

    """
    Initialize a standard FCN-Res50-D8 model based on mmseg configs.
    
    Args:
        num_classes (int): Number of output classes.
        checkpoint (str, optional): Path to the pretrained weights (.pth).
        device (str): Device to run the model on.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Path to the official config file within your source code
    # Adjust this path if your mmsegmentation folder is located elsewhere
    mmseg_root = '/root/autodl-tmp/mmsegmentation/'
    config_path = os.path.join(mmseg_root, 'configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. "
                                f"Please check your mmsegmentation source path.")

    # 2. Load and update the configuration
    cfg = Config.fromfile(config_path)

    # Update number of classes to match your specific dataset
    cfg.model.decode_head.num_classes = num_classes
    
    # Update auxiliary head if it exists (standard FCN-D8 usually has one)
    if hasattr(cfg.model, 'auxiliary_head') and cfg.model.auxiliary_head is not None:
        cfg.model.auxiliary_head.num_classes = num_classes

    cfg.model.auxiliary_head = None

    # 3. Build the model and load weights
    # If checkpoint is provided, init_model will automatically call load_checkpoint
    # MODIFIED: Load checkpoint manually to handle plain state_dict and DDP prefix
    model = init_model(cfg, checkpoint=None, device=device)

    if checkpoint is not None and os.path.exists(checkpoint):
        # print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, map_location=device)

        # Handle if checkpoint is a dictionary with 'state_dict' or just the state_dict
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        # print("Checkpoint loaded successfully.")
    
    return model

if __name__ == '__main__':
    # Simple test for initialization
    try:
        model = get_model(num_classes=19)
        print("Successfully initialized FCN-Res50-D8!")
        print(f"Device: {next(model.parameters()).device}")
        
        # Verify D8 settings (Dilations should be 1, 1, 2, 4)
        dilations = [m.dilation for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
        print(f"Model successfully loaded with dilated convolutions.")
    except Exception as e:
        print(f"Initialization failed: {e}")