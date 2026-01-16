import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

import helpers.config as config
from data_sds_cityscapes import load_data
from helpers.Logger import Logger
from helpers.integrated_loss import compute_integrated_loss
from helpers.Aux_loss import AuxModelWrapper
from helpers.Warmup_scheduler import WarmupScheduler

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['SMP_SKIP_CHECKPOINT_CHECK'] = '1'

# ========== Modified by User ==========

mode = 'csg' # origin / foreground / background / csg_only / csg+origin / nda
csg_mode = 'both'  # foreground / background / both

alpha = 0.0
beta  = 1.0

momentum = 0.9
weight_decay = 0.0001

bn_frozen = False

# Log
date = "_1_15_2026"
info = "_BL+CSG_both_"
log_root = "/root/autodl-tmp/log/"

# Auxiliary
aux_is_enabled = False
aux_weight = 0.2

# Learning rate warmup (steps)
warmup_is_enabled = True
warmup_iters = 1000
warmup_factor = 0.01

# ======================================

# Log
logger = Logger(date=date, info=info, log_root = log_root)

local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_distributed = local_rank != -1

# Set device
if is_distributed:
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not is_distributed or (dist.is_initialized() and dist.get_rank() == 0):
    logger(f"Training on device: {device}")


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, num_epochs):

    model.train()

    if bn_frozen:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                m.eval()
    
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Set epoch for DistributedSampler
    if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    if not is_distributed or dist.get_rank() == 0:
        logger(f"Epoch [{epoch+1}/{num_epochs}]\n")
    
    for images, labels, mask, origin_images, origin_labels  in train_loader:

        # Forward pass
        optimizer.zero_grad()

        # Move data to device
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        if mode == 'csg':
            origin_images = origin_images.to(device)
            origin_labels = origin_labels.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
        
        with torch.cuda.amp.autocast():
            if mode == 'csg':
                # Concatenate images to run in a single forward pass
                # This avoids inplace operation errors and improves efficiency
                combined_images = torch.cat([images, origin_images], dim=0)
                combined_main_out, combined_aux_out = model(combined_images)
                outputs_img, outputs_origin = torch.split(combined_main_out, images.size(0), dim=0)

                if combined_aux_out is not None:
                    aux_img, aux_origin = torch.split(combined_aux_out, images.size(0), dim=0)
            else:
                outputs_img, aux_img = model(images)
                outputs_origin, aux_origin = None, None
            
            # Resize outputs to match labels if necessary
            if outputs_img.shape[-2:] != labels.shape[-2:]:
                outputs_img = F.interpolate(outputs_img, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            if aux_is_enabled:
                if aux_img.shape[-2:] != labels.shape[-2:]:
                        aux_img = F.interpolate(aux_img, size=labels.shape[-2:], mode='bilinear', align_corners=False)


            main_integrated_loss = compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, mode, alpha, beta)
            if aux_is_enabled:
                aux_integrated_loss = compute_integrated_loss(aux_img, labels, mask, aux_origin, origin_labels, criterion, mode, alpha, beta)

                integrated_loss = main_integrated_loss + aux_weight * aux_integrated_loss
            else:
                integrated_loss = main_integrated_loss

        # Backward pass
        scaler.scale(integrated_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # Gradient clipping

        # Step the optimizer and scheduler
        # We only step the scheduler if the scaler actually performed an optimizer step (didn't skip due to Inf/NaN)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_after >= scale_before:
            scheduler.step()

        running_loss += integrated_loss.item()
    
    return running_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels, mask, origin_images, origin_labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs, _ = model(images)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, labels.squeeze(1))
            val_loss += loss.item()
    
    return val_loss / num_batches

def train(model, device, num_epochs, batch_size, lr_backbone, lr_classifier, from_scratch = True, model_checkpoint_path = None):

    if not is_distributed or dist.get_rank() == 0:
        logger(f"lr_backbone:{lr_backbone}, lr_classifier:{lr_classifier}, epochs:{num_epochs}, alpha:{alpha}, beta:{beta}\n")

    # Load dataset
    train_iter = load_data(root=config.DATA_DIR, mode=mode, split='train', csg_mode=csg_mode, batch_size=batch_size, num_workers=12, distributed=is_distributed)
    val_iter = load_data(root=config.DATA_DIR, mode='origin', split='val', batch_size=batch_size, num_workers=4, distributed=False)
    
    # Create model
    model = model.to(device)

    if not is_distributed or dist.get_rank() == 0:
        logger(f"Model moved to {device}\n")

    # Convert to SyncBN if distributed
    if is_distributed and not bn_frozen:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if dist.get_rank() == 0:
            logger("Converted to SyncBatchNorm\n")

    # Auxiliary Loss
    model = AuxModelWrapper(model, aux_is_enabled)

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
    
    # SGD OPTIMIZER with different learning rates
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': classifier_params, 'lr': lr_classifier}
    ], momentum=momentum, weight_decay=weight_decay)


    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

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
    scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_iters=warmup_iters, warmup_factor=warmup_factor, is_enabled = warmup_is_enabled)
    
    # Save model path
    model_path = f"/root/autodl-tmp/models/{date}{info}_A{alpha}B{beta}_.pth"

    # Training loop
    origin_val_losses = []
    best_val_loss = float('inf')

    try:
        for epoch in range(num_epochs):

            # Train one epoch
            train_loss = train_epoch(model, train_iter, criterion, optimizer, scheduler, scaler, device, epoch, num_epochs)
            
            gc.collect()
            torch.cuda.empty_cache()

            # Validation
            origin_val_loss = validate_epoch(model, val_iter, criterion, device)
            origin_val_losses.append(origin_val_loss)

            # Save the best model in the last 5 epochs
            if epoch > (num_epochs - 5) and origin_val_loss < best_val_loss :
                best_val_loss = origin_val_loss
                if not is_distributed or dist.get_rank() == 0:
                    model_to_save = model
                    if hasattr(model_to_save, 'module'): # Unwrap DDP
                        model_to_save = model_to_save.module
                    if hasattr(model_to_save, 'model'):  # Unwrap AuxModelWrapper
                        model_to_save = model_to_save.model
                        
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