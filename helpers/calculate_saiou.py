"""
SA-IoU helpers.

Provides utilities to compute IoU for foreground/background class groups
and the weighted SA-IoU (stand-alone IoU) combining them.
"""

import torch
import torch.nn.functional as F


def cal_fore_back_iou(model, val_loader, train_ids, device):
    """
    Compute average IoU over images for a set of class ids.

    The function treats `train_ids` as the target class group (foreground or
    background). For each image it computes IoU between the union of pixels
    belonging to `train_ids` in labels and predictions.
    """

    model.eval()

    total_iou = 0.0
    num = 0

    with torch.no_grad():
        for images, labels, _, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            batch_size = images.size(0)

            outputs = model(images)

            # Resize outputs to match labels size if necessary
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            preds = torch.argmax(outputs, dim=1)

            # Build boolean masks marking pixels that belong to `train_ids`
            label_in_train = torch.zeros_like(labels, dtype=torch.bool)
            pred_in_train = torch.zeros_like(preds, dtype=torch.bool)
            for tid in train_ids:
                label_in_train |= (labels == tid)
                pred_in_train |= (preds == tid)

            # Intersection: pixels where prediction equals label and label is in group
            intersection = (preds == labels) & label_in_train

            # Union: pixels where either label or prediction is in the group
            union = label_in_train | pred_in_train

            # accumulate per-image IoU
            for i in range(batch_size):
                total_iou += (intersection[i].sum().item() / union[i].sum().item())
                num += 1

    return total_iou / num


def cal_sa_iou(model, fore_loader, back_loader, fore_ids, back_ids, device):
    """
    Compute foreground IoU, background IoU and SA-IoU.

    SA-IoU is the class-group-weighted average of foreground and background
    IoUs, weighted by the number of classes in each group.
    """

    fore_iou = cal_fore_back_iou(model, fore_loader, fore_ids, device)
    back_iou = cal_fore_back_iou(model, back_loader, back_ids, device)

    fore_classes = len(fore_ids)
    back_classes = len(back_ids)
    data_classes = fore_classes + back_classes

    sa_iou = (fore_classes / data_classes) * fore_iou + (back_classes / data_classes) * back_iou

    return fore_iou, back_iou, sa_iou

    



