"""
SegFormer model initialization helper using MMSegmentation configs.

Provides `get_model(num_classes, checkpoint, device)` which builds a
SegFormer (MiT-B3) model from an mmseg config, updates the decode head
class count, and optionally loads weights from a checkpoint.
"""

import os
import torch
from .registry import register_models
from mmseg.apis import init_model
from mmengine.config import Config

@register_models("segformer")
def get_model(num_classes, checkpoint=None):
    """
    Initialize SegFormer (MiT-B3) model based on MMSegmentation configs.
    
    Args:
        num_classes (int): Number of output classes (e.g., 19 for Cityscapes, 150 for ADE20K).
        checkpoint (str, optional): Path to the pretrained weights (.pth).
        device (str): Device to run the model on.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Path to the config file
    # Updated to point to the SegFormer-B3 configuration
    mmseg_root = '/root/autodl-tmp/mmsegmentation/'
    config_path = os.path.join(mmseg_root, 'configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. "
                                f"Please check your MMSegmentation source path.")

    # 2. Load and update the configuration
    cfg = Config.fromfile(config_path)

    # SegFormer has a streamlined architecture, usually only containing a decode_head
    # Update the number of classes for the Decode Head
    cfg.model.decode_head.num_classes = num_classes
    
    # Check for auxiliary head for compatibility (Standard SegFormer doesn't have one)
    if hasattr(cfg.model, 'auxiliary_head') and cfg.model.auxiliary_head is not None:
        cfg.model.auxiliary_head.num_classes = num_classes

    # 3. Build the model
    # Initialize with checkpoint=None to handle weight loading manually later
    model = init_model(cfg, checkpoint=None, device=device)

    # 4. Manually load weights and handle DDP prefix ('module.')
    if checkpoint is not None and os.path.exists(checkpoint):
        # print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, map_location=device)

        # Extract the state_dict
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data

        # Load weights into the model
        # strict=False allows skipping mismatched layers (e.g., different class numbers in the head)
        load_info = model.load_state_dict(state_dict, strict=False)
        # print(f"Checkpoint loaded successfully. Load info: {load_info}")
    
    return model

if __name__ == '__main__':
    # Simple test for initialization
    try:
        # Example: Cityscapes with 19 classes
        num_classes = 19
        model = get_model(num_classes=num_classes)
        
        print("--- SegFormer-B3 Initialized Successfully ---")
        print(f"Device: {next(model.parameters()).device}")
        
        # Verify model structure (SegFormer backbone uses MixVisionTransformer layers)
        if hasattr(model.backbone, 'layers'):
            print("Model Backbone (Mix Transformer) structure verified.")
            
        # Basic inference test with dummy input
        dummy_input = torch.randn(1, 3, 512, 512).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy_input)
            # MMSegmentation models return a list of tensors in basic inference
            print(f"Inference test successful. Output shape: {output[0].shape}") 

    except Exception as e:
        print(f"Initialization failed: {e}")