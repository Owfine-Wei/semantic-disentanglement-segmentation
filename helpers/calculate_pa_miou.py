import torch
import torch.nn.functional as F
from tqdm import tqdm

def calculate_metrics(model, val_loader, device, num_classes, train_id_dict=None):
    """
    使用 Dataset-level 逻辑计算指标（与 torchmetrics/学术论文对齐）。
    
    逻辑：先累加所有图片的 Intersection 和 Union，最后统一求商。
    """
    model.eval()

    # 初始化全局累加器：形状为 (num_classes,)
    # 用于存储整个验证集中每个类别的交集和并集像素总数
    total_inter_accumulator = torch.zeros(num_classes, device=device)
    total_union_accumulator = torch.zeros(num_classes, device=device)
    
    # 像素准确率累加器
    total_correct_pixels = 0
    total_valid_pixels = 0

    ignore_index = 255
    # 类别 ID 张量，用于广播比较 (1, K, 1)
    target_ids_tensor = torch.tensor(range(num_classes), device=device).view(1, -1, 1)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.long)
            
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            preds = torch.argmax(outputs, dim=1)

            # --- 1. 像素准确率 (Pixel Accuracy) 计算 ---
            valid_mask = (labels != ignore_index)
            total_correct_pixels += ((preds == labels) & valid_mask).sum().item()
            total_valid_pixels += valid_mask.sum().item()

            # --- 2. 向量化计算交集与并集 ---
            # 展平维度：(B, H, W) -> (B, N)
            flat_labels = labels.view(labels.size(0), -1)
            flat_preds = preds.view(preds.size(0), -1)
            flat_valid = valid_mask.view(valid_mask.size(0), -1).unsqueeze(1) # (B, 1, N)

            # 广播比较：(B, 1, N) == (1, K, 1) -> (B, K, N)
            # 找到属于每个类别的像素位置
            label_in_train = (flat_labels.unsqueeze(1) == target_ids_tensor) & flat_valid
            pred_in_train = (flat_preds.unsqueeze(1) == target_ids_tensor) & flat_valid

            # 计算当前 Batch 的交集和并集：在 Batch 和 Pixel 维度同时求和
            # 结果形状为 (K,)，即每个类别的总像素数
            batch_inter = (label_in_train & pred_in_train).float().sum(dim=(0, 2))
            batch_union = (label_in_train | pred_in_train).float().sum(dim=(0, 2))

            # --- 3. 累加到全局 ---
            total_inter_accumulator += batch_inter
            total_union_accumulator += batch_union

    # --- 4. 计算最终指标 ---
    # 计算每个类别的 IoU (Dataset-level)
    # 使用 epsilon 避免分母为 0
    class_ious = total_inter_accumulator / (total_union_accumulator + 1e-8)
    
    # 转换为 Python 列表
    class_ious_list = class_ious.cpu().tolist()

    # 构建语义字典
    if train_id_dict:
        id_to_name = {v: k for k, v in train_id_dict.items()}
        miou_dict = {id_to_name.get(i, f"ID_{i}"): iou for i, iou in enumerate(class_ious_list)}
    else:
        miou_dict = {str(i): iou for i, iou in enumerate(class_ious_list)}

    # 计算平均 mIoU (所有类别的平均)
    mean_iou = class_ious.mean().item()

    # 计算全局像素准确率
    pixel_accuracy = total_correct_pixels / (total_valid_pixels + 1e-8)

    return miou_dict, mean_iou, pixel_accuracy