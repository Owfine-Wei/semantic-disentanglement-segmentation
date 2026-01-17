from helpers.set_seed import setup_seed
setup_seed(42, deterministic=True)  # CUBLAS_WORKSPACE_CONFIG=':4096:8'

# test

import torch
import torch.distributed as dist 
import os 
import gc
import train
import models
import itertools
import helpers.config as config
from helpers.set_seed import setup_seed


if __name__ == '__main__':
    
    # ====================== Modified by User ======================

    # model
    model_type = 'FCN'

    # Train the Model From Scratch ?
    from_scratch = False
    # If not from scratch (.pth path)
    model_checkpoint_path = '/root/autodl-tmp/models/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth'

    # Basic Configs
    num_epochs = 15

    # Search space
    search_space = {
        'lr_backbone': [1e-6],  
        'lr_classifier': [1e-5], 
        'batch_size': [4],  # effective_batch_size = batch_size * num_gpus
    }

    # Grid search config
    grid_search_configs = list(itertools.product(
        search_space['lr_backbone'],
        search_space['lr_classifier'],
        search_space['batch_size']
    ))

    # ==============================================================

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Main loop
    for (i, (lr_backbone, lr_classifier,batch_size)) in enumerate(grid_search_configs, start=1) :

        # Create model
        # Disabling pretrained weights to avoid network issues properly
        model = models.get_model(num_classes=config.NUM_CLASSES, checkpoint = model_checkpoint_path,model_type=model_type).to(device) # modify to match your model

        train.train(model,device,num_epochs,batch_size,lr_backbone,lr_classifier,from_scratch,model_checkpoint_path)

        del model 
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 