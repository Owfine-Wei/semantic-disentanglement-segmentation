# ========================================================
# 1. 第一优先级：设置环境变量 (必须在 import torch 之前)
# ========================================================
import os
# 如果是多卡训练或使用了特殊的卷积算子，必须设置这个
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

# ========================================================
# 2. 第二优先级：基础库导入
# ========================================================
import yaml
import argparse
from easydict import EasyDict as edict
import gc

# ========================================================
# 3. 第三优先级：配置加载与种子初始化
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--yaml', default='', help='.yaml file for training', required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank') # 增加 DDP 支持
    return parser.parse_args()

args = parse_args()

def load_train_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return edict(config)

train_config = load_train_config(args.yaml)

# 导入并立即运行种子设置
from helpers.set_seed import setup_seed
# 如果是单卡，rank 默认为 0；如果是 DDP，传入 args.local_rank
setup_seed(train_config.seed, deterministic=True, rank=getattr(args, 'local_rank', 0))

# ========================================================
# 4. 第四优先级：Torch 及其他依赖导入 (此时种子已锁死)
# ========================================================
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
from helpers.integrated_loss import compute_integrated_loss

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    
    # 冻结 BN (Freeze BN)
    if train_config.train.bn_frozen:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Set epoch for DistributedSampler
    if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    if not is_distributed or dist.get_rank() == 0:
        logger(f"Epoch [{epoch+1}/{num_epochs}]\n")
    
    for images, labels, mask, origin_images, origin_labels in tqdm(train_loader):

        optimizer.zero_grad()

        # 保持 Channels Last 优化
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long)
        
        if train_config.dataset.mode == 'csg':
            origin_images = origin_images.to(device, memory_format=torch.channels_last) # 记得这里也加上
            origin_labels = origin_labels.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.float32)
        
        # === 核心修改：强制使用 BF16 ===
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            if train_config.dataset.mode == 'csg':
                logits_img, features_img = model(images, return_features=True, return_dict=False) 
                logits_origin_img, features_origin_img = model(origin_images, return_features=True, return_dict=False)
            else:
                logits_img = model(images, return_features=False, return_dict=False)
                features_img = None
                logits_origin_img = None
                features_origin_img = None

            integrated_loss = compute_integrated_loss(logits_img, labels, mask, logits_origin_img, origin_labels, features_img, features_origin_img, criterion, train_config.dataset.mode, train_config.loss.alpha, train_config.loss.beta)

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
        # 使用 autocast 开启自动混合精度
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in tqdm(val_loader, desc="Validating"):
                # 1. 搬运数据
                images = batch[0].to(device)
                labels = batch[1].to(device, dtype=torch.long)
                
                # 2. 前向传播
                outputs = model(images)
                
                # 3. 提取 Logits 并转为 float32 以确保 Loss 计算稳定
                # 注意：将 logits 转回 float32 是解决很多奇怪报错的关键
                logits = outputs.float() 

                # 4. 尺寸对齐 (SegFormer 输出是 1/4)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                # 5. 计算 Loss
                # 使用 squeeze(1) 时要小心 batch 为 1 的情况，建议明确维度
                # 如果 labels 已经是 [N, H, W]，则不需要 squeeze
                target = labels.squeeze(1) if labels.dim() == 4 else labels
                
                loss = criterion(logits, target)
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

    raw_model = model.module if hasattr(model, 'module') else model
    backbone_prefix = raw_model.base_model_prefix
    
    for name, param in model.named_parameters():

        if backbone_prefix in name:
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

    # 计算总迭代步数
    total_iters = int(num_epochs * len(train_iter))
    # 预热步数：建议设置为总步数的 5% 或 1500 次迭代
    warmup_iters = 1500 
    power = 0.9
    min_lr = 1e-7 # 建议设为极小值，Poly 衰减的终点通常趋向于 0

    def lr_lambda(current_step):
        # 1. Linear Warmup 阶段
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        
        # 2. Polynomial Decay 阶段
        # 计算从 warmup 结束到训练结束的进度
        progress = (current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
        progress = min(progress, 1.0) # 确保不会超过 1
        
        coeff = (1.0 - progress) ** power
        
        # 返回的是相对于初始 LR 的倍数
        return coeff

    # 注意：LambdaLR 会将返回的 lambda 值与对应参数组的 base_lr 相乘
    # 确保你的 optimizer 有两个参数组 (Backbone 和 Head)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[lr_lambda, lr_lambda]
    )

    # Save model path
    model_path = f"../models/{train_config.logging.date}{train_config.logging.info}_A{train_config.loss.alpha}B{train_config.loss.beta}_.bin"

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