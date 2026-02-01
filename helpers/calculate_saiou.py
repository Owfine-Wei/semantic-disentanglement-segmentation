"""
SA-IoU helpers.

Provides utilities to compute IoU for foreground/background class groups
and the weighted SA-IoU (stand-alone IoU) combining them.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def cal_fore_back_iou(model, val_loader, train_ids, device):
    """
    计算给定类别集合的 Dataset-level 平均 IoU (对齐学术标准)。
    
    逻辑：累加所有样本的交集和并集像素总和，最后统一求商。
    """
    model.eval()
    
    # K 是要计算的类别数量
    num_target_classes = len(train_ids)
    # 将 train_ids 转为 tensor (1, K, 1)，用于并行广播比较
    target_ids_tensor = torch.tensor(train_ids, device=device, dtype=torch.long).view(1, -1, 1)
    
    # 初始化全局累加器：存储每个类在整个数据集上的交集和并集总数
    total_inter_accumulator = torch.zeros(num_target_classes, device=device)
    total_union_accumulator = torch.zeros(num_target_classes, device=device)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # 兼容性解包，假设 masks 在第三个位置
            images, labels, masks = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # 尺寸对齐
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            preds = torch.argmax(outputs, dim=1)

            # --- 向量化处理 ---
            # 展平像素维度 (B, H, W) -> (B, N)
            flat_labels = labels.view(labels.size(0), -1)
            flat_preds = preds.view(preds.size(0), -1)
            # mask=0 为有效，1 为忽略；unsqueeze 方便广播
            valid_mask = (masks.view(masks.size(0), -1) == 0).unsqueeze(1) # (B, 1, N)

            # --- 广播机制计算交集与并集 ---
            # label_in_train: (B, K, N) - 第b张图中，第n个像素是否属于第k个类且有效
            label_in_train = (flat_labels.unsqueeze(1) == target_ids_tensor) & valid_mask
            pred_in_train = (flat_preds.unsqueeze(1) == target_ids_tensor) & valid_mask

            # 计算当前 batch 的交集和并集像素总数
            # 对 Batch(0) 和 Pixel(2) 维度求和，得到 (K,)
            batch_inter = (label_in_train & pred_in_train).float().sum(dim=(0, 2))
            batch_union = (label_in_train | pred_in_train).float().sum(dim=(0, 2))

            # --- 累加到全局累加器 ---
            total_inter_accumulator += batch_inter
            total_union_accumulator += batch_union

    # --- 计算最终结果 ---
    # 计算每个类别的全局 IoU: (K,)
    # 使用 1e-8 防止全集中无该类时产生的除零错误
    avg_ious = total_inter_accumulator / (total_union_accumulator + 1e-8)
    
    # 转为字典输出
    final_iou_dict = {
        str(tid): avg_ious[i].item() 
        for i, tid in enumerate(train_ids)
    }

    # 计算所有选中类别的平均 mIoU
    final_avg_iou = avg_ious.mean().item() if num_target_classes > 0 else 0.0

    return final_iou_dict, final_avg_iou

def cal_sa_iou(model, fore_loader, back_loader, fore_ids, back_ids, device):
    """
    Compute foreground IoU, background IoU and SA-IoU.

    SA-IoU is the class-group-weighted average of foreground and background
    IoUs, weighted by the number of classes in each group.
    """

    fiou_dict, fore_iou = cal_fore_back_iou(model, back_loader, fore_ids, device)
    biou_dict, back_iou = cal_fore_back_iou(model, fore_loader, back_ids, device)

    fore_classes = len(fore_ids)
    back_classes = len(back_ids)
    data_classes = fore_classes + back_classes

    sa_iou = (fore_classes / data_classes) * fore_iou + (back_classes / data_classes) * back_iou

    return fiou_dict, fore_iou, biou_dict, back_iou, sa_iou

    



