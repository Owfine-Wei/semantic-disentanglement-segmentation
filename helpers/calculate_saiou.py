"""
SA-IoU helpers.

Provides utilities to compute IoU for foreground/background class groups
and the weighted SA-IoU (stand-alone IoU) combining them.
"""

import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F

def cal_fore_back_iou(model, val_loader, train_ids, device):
    """
    Compute average IoU over images for a set of class ids.
    
    Args:
        model: The segmentation model.
        val_loader: DataLoader returning (images, labels, masks, ...).
        train_ids: List of integer class IDs to calculate IoU for.
        device: Torch device (cpu or cuda).
        
    Returns:
        iou_dict (dict): A dictionary {class_id: mean_iou}.
        mean_iou (float): The average of the mean_ious across all requested classes.
    """
    
    model.eval()
    
    # 1. 在循环外部初始化累加器，用于记录所有图片的总 IoU
    # key为str(tid)是为了方便后续处理，也可以直接用int
    total_iou_dict = {str(tid): 0.0 for tid in train_ids}
    
    total_images = 0 # 记录总图片数量

    with torch.no_grad():
        for images, labels, masks, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            # 假设 mask 中 0 为有效区域，1 为无效区域
            masks = masks.to(device, dtype=torch.long)

            batch_size = images.size(0)
            
            # 模型推理
            outputs = model(images)

            # 对齐尺寸：如果输出尺寸和标签不一致，进行插值
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            # 获取预测结果 (B, H, W)
            preds = torch.argmax(outputs, dim=1)

            # 生成有效掩码 (B, H, W)，根据逻辑描述：只有 masks=0 的地方才有效
            # 这里的 valid_area 对应 user 逻辑中的 (1 - masks)
            valid_area = (masks == 0)

            for tid in train_ids:
                # 2. 找到 Labels 中为 tid 且在有效区域的像素
                # eq(tid) 等价于 == tid
                tid_in_labels = labels.eq(tid) & valid_area
                
                # 3. 找到 Preds 中为 tid 且在有效区域的像素
                tid_in_preds = preds.eq(tid) & valid_area

                # 4. 计算交集和并集
                intersection = tid_in_labels & tid_in_preds
                union = tid_in_labels | tid_in_preds

                # 5. 计算当前 Batch 中每张图片的 IoU
                # sum(dim=(1, 2)) 将 (B, H, W) 压缩为 (B, )，即算出每张图的像素数
                # float() 转换是为了进行除法运算
                inter_sum = intersection.float().sum(dim=(1, 2))
                union_sum = union.float().sum(dim=(1, 2))
                
                # 计算 IoU: Intersection / Union
                # 添加 1e-6 是为了防止 Union 为 0 时产生除零错误 (NaN)
                ious = inter_sum / (union_sum + 1e-6)

                # 6. 将当前 Batch 的 IoU 求和并累加到字典中
                total_iou_dict[str(tid)] += ious.sum().item()
            
            # 更新处理过的图片总数
            total_images += batch_size

    # 7. 计算最终结果
    # 此时 total_iou_dict 中存储的是所有图片 IoU 的总和，需要除以图片总数得到平均值
    final_iou_dict = {}
    average_iou_sum = 0.0
    
    for tid in train_ids:
        tid_str = str(tid)
        # 计算该类别的 Mean IoU
        mIoU = total_iou_dict[tid_str] / total_images
        final_iou_dict[tid_str] = mIoU
        average_iou_sum += mIoU

    # 计算所有选中类别的平均 mIoU
    final_avg_iou = average_iou_sum / len(train_ids) if len(train_ids) > 0 else 0.0

    return final_iou_dict, final_avg_iou

def cal_sa_iou(model, fore_loader, back_loader, fore_ids, back_ids, device):
    """
    Compute foreground IoU, background IoU and SA-IoU.

    SA-IoU is the class-group-weighted average of foreground and background
    IoUs, weighted by the number of classes in each group.
    """

    fiou_dict, fore_iou = cal_fore_back_iou(model, fore_loader, fore_ids, device)
    biou_dict, back_iou = cal_fore_back_iou(model, back_loader, back_ids, device)

    fore_classes = len(fore_ids)
    back_classes = len(back_ids)
    data_classes = fore_classes + back_classes

    sa_iou = (fore_classes / data_classes) * fore_iou + (back_classes / data_classes) * back_iou

    return fiou_dict, fore_iou, biou_dict, back_iou, sa_iou

    



