import yaml
import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--yaml', default='', help='.yaml file for training', required=True)
arg = parser.parse_args()

def load_train_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)
        train_config = edict(train_config)
    return train_config

train_config = load_train_config(arg.yaml)

from helpers.set_seed import setup_seed
setup_seed(train_config.seed, deterministic=True)  # CUBLAS_WORKSPACE_CONFIG=':4096:8'

import os
import gc

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from configs import get_config
from helpers.Logger import Logger
from helpers.set_seed import setup_seed
from datasets.dataset_impl import load_data
from helpers.Warmup_scheduler import WarmupScheduler
from helpers.integrated_loss import compute_integrated_loss

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['SMP_SKIP_CHECKPOINT_CHECK'] = '1'


# config
config = get_config(train_config.dataset.name)

# Log
logger = Logger(date=train_config.logging.date, info=train_config.logging.info, log_root = train_config.logging.root)

# 1. 更加健壮的 rank 获取方式
# torchrun 会自动设置这些环境变量
local_rank = int(os.environ.get("LOCAL_RANK", 0)) # 默认 0 方便单卡兼容
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed = world_size > 1

if is_distributed:
    # 2. 必须先初始化进程组，再进行 cuda 操作是更稳健的做法
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # 3. 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # 打印一下，方便调试（仅在主进程打印）
    if dist.get_rank() == 0:
        print(f"Distributed mode enabled. World size: {world_size}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running in non-distributed mode.")


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs): # 去掉了 scaler 参数

    model.train()
    # ... (BN Frozen 逻辑保持不变) ...
    
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Set epoch for DistributedSampler
    if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    if not is_distributed or dist.get_rank() == 0:
        logger(f"Epoch [{epoch+1}/{num_epochs}]\n")
    
    for images, labels, mask, origin_images, origin_labels  in tqdm(train_loader):

        optimizer.zero_grad()

        # 保持 Channels Last 优化
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long)
        
        if train_config.dataset.mode == 'csg':
            origin_images = origin_images.to(device, memory_format=torch.channels_last) # 记得这里也加上
            origin_labels = origin_labels.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
        
        # === 核心修改：强制使用 BF16 ===
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            if train_config.dataset.mode == 'csg':
                combined_images = torch.cat([images, origin_images], dim=0)
                combined_main_out = model(combined_images)
                outputs_img, outputs_origin = torch.split(combined_main_out, images.size(0), dim=0)
            else:
                outputs_img = model(images)
                outputs_origin = None
            
            if outputs_img.shape[-2:] != labels.shape[-2:]:
                outputs_img = F.interpolate(outputs_img, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            integrated_loss = compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, train_config.dataset.mode, train_config.loss.alpha, train_config.loss.beta)

        # === 核心修改：移除 Scaler，直接 Backward ===
        integrated_loss.backward()
        
        # 梯度裁剪（可选，BF16下通常不裁剪也很稳，保留也没事）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()
        scheduler.step() # 这里的 scheduler 逻辑简化了，因为没有 scaler skip 的问题了

        running_loss += integrated_loss.item()
    
    return running_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        # === 新增：验证集也要 BF16 ===
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in tqdm(val_loader):
                images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.long)
                
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                loss = criterion(outputs, labels.squeeze(1))
                val_loss += loss.item()
    
    return val_loss / num_batches

def train(model, device, num_epochs, batch_size, lr_backbone, lr_classifier, from_scratch = True, model_checkpoint_path = None):

    if not is_distributed or dist.get_rank() == 0:
        logger(f"lr_backbone:{lr_backbone}, lr_classifier:{lr_classifier}, epochs:{num_epochs}, alpha:{train_config.loss.alpha}, beta:{train_config.loss.beta}\n")

    # Load dataset
    train_iter = load_data(config, mode=train_config.dataset.mode, split='train', csg_mode=train_config.dataset.csg_mode, batch_size=batch_size, num_workers=12, distributed=is_distributed)
    val_iter = load_data(config, mode='origin', split='val', batch_size=batch_size, num_workers=4, distributed=False)
    
    # Create model
    model = model.to(device, memory_format=torch.channels_last)

    # ===【新增】Torch 2.x 核心编译优化 ===
    # mode='reduce-overhead' 适合这种训练循环较小的场景
    # 如果报错，可以尝试 mode='default' 或者直接去掉 mode 参数
    try:
        model = torch.compile(model, mode='reduce-overhead')
        if not is_distributed or dist.get_rank() == 0:
            logger("Torch.compile enabled!\n")
    except Exception as e:
        print(f"Compilation failed: {e}")
    # ===================================


    if not is_distributed or dist.get_rank() == 0:
        logger(f"Model moved to {device}\n")

    # Convert to SyncBN if distributed
    if is_distributed and not train_config.train.bn_frozen:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if dist.get_rank() == 0:
            logger("Converted to SyncBatchNorm\n")

    # DDP wrapper
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)

    # Parameter grouping
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():

        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    if not is_distributed or dist.get_rank() == 0:
        logger(f"Parameter grouping: {len(backbone_params)} backbone, {len(classifier_params)} classifier\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # OPTIMIZER with different learning rates (SGD or AdamW)
    if train_config.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': classifier_params, 'lr': lr_classifier}
        ], momentum=train_config.optimizer.momentum, weight_decay=train_config.optimizer.weight_decay,fused=True)
    elif train_config.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': classifier_params, 'lr': lr_classifier}
        ], beta=train_config.optimizer.beta, weight_decay=train_config.optimizer.weight_decay,fused=True)

    # Learning rate scheduler (Polynomial with lr_end)
    total_iters = int( num_epochs * len(train_iter) )
    min_lr_ratio = 1.0 / 50.0 
    power = 0.9

    def lr_lambda(step):
        coeff = (1 - step / total_iters) ** power
        return coeff * (1 - min_lr_ratio) + min_lr_ratio

    base_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        lr_lambda, lr_lambda
    ])

    # Create scheduler using warmup_iters only.
    scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_iters=train_config.warmup.iters, warmup_factor=train_config.warmup.factor, is_enabled = train_config.warmup.enabled)
    
    # Save model path
    model_path = f"../models/{train_config.logging.date}{train_config.logging.info}_A{train_config.loss.alpha}B{train_config.loss.beta}_.pth"

    # Training loop
    origin_train_losses = []
    best_train_loss = float('inf')

    try:
        for epoch in range(num_epochs):

            # Train one epoch
            train_loss = train_epoch(model, train_iter, criterion, optimizer, scheduler, device, epoch, num_epochs)
            origin_train_losses.append(train_loss)

            gc.collect()
            torch.cuda.empty_cache()

            # Validation
            if (epoch+1) % 5 == 0:
                origin_val_loss = validate_epoch(model, val_iter, criterion, device)
            else:
                origin_val_loss = 0.00

            # Save the best model in the last 5 epochs
            if epoch > (num_epochs - 5) and train_loss < best_train_loss :
                best_train_loss = train_loss
                if not is_distributed or dist.get_rank() == 0:
                    model_to_save = model
                    if hasattr(model_to_save, 'module'): # Unwrap DDP
                        model_to_save = model_to_save.module
                        
                    torch.save(model_to_save.state_dict(), model_path)

            if not is_distributed or dist.get_rank() == 0:
                logger(f"Epoch [{epoch+1}/{num_epochs}] Summary:\n")
                logger(f"Train Loss: {train_loss:.4f} | Origin Val Loss: {origin_val_loss:.4f}\n")
                logger(f"Current LR: backbone={optimizer.param_groups[0]['lr']:.2e}, "
                      f"classifier={optimizer.param_groups[1]['lr']:.2e}\n")
                logger("-" * 60 + "\n")
                
    except KeyboardInterrupt:
        if not is_distributed or dist.get_rank() == 0:
            logger("Training interrupted by user!")
        
    except Exception as e:
        if not is_distributed or dist.get_rank() == 0:
            logger(f"Training stopped due to error: {e}")
        import traceback
        traceback.print_exc()
    
    if not is_distributed or dist.get_rank() == 0:
        logger("Training completed!\n")

    return 

def main():

    # Create model
    # Disabling pretrained weights to avoid network issues properly
    get_model_function = models.get_model(train_config.model.name)
    model = get_model_function(num_classes=config.NUM_CLASSES, checkpoint = train_config.model.checkpoint_path).to(device) # modify to match your model

    train(model,device,train_config.train.num_epochs,train_config.train.batch_size,train_config.train.lr_backbone,train_config.train.lr_classifier,train_config.model.from_scratch,train_config.model.checkpoint_path)

    del model 
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    # ===【新增】===
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__" :
    main()