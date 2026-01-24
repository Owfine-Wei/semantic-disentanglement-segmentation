"""
Compute mean IoU and pixel accuracy on the validation set.

This helper uses torchmetrics' MulticlassJaccardIndex (mIoU) and
MulticlassAccuracy (pixel accuracy).
"""

import torch
import torch.nn.functional as F

from torchmetrics.classification import (MulticlassJaccardIndex, MulticlassAccuracy)


def calculate_metrics(model, val_loader, device, num_classes):
    
    """
    Evaluate `model` on `val_loader` and return (mIoU, pixel_accuracy).

    Args:
        model: segmentation model; expected to return logits or (logits, ...).
        val_loader: validation DataLoader yielding (images, labels, ...).
        device: torch device to run metrics on.
        num_classes: number of segmentation classes (without 'ignore').

    Returns:
        (mean_iou, pixel_accuracy) as torch tensors from torchmetrics.
    """

    model.eval()

    # Mean IoU (Jaccard) metric, ignoring void index 255
    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=255,
        average='macro'  # macro = mean IoU (mIoU)
    )

    # Pixel accuracy metric
    pa_metric = MulticlassAccuracy(
        num_classes=num_classes,
        ignore_index=255,
        average='micro'  # micro = overall pixel accuracy
    )

    # Disable distributed sync if attribute exists (avoids extra overhead)
    if hasattr(miou_metric, 'sync_on_compute'):
        miou_metric.sync_on_compute = False
    if hasattr(pa_metric, 'sync_on_compute'):
        pa_metric.sync_on_compute = False

    miou_metric = miou_metric.to(device)
    pa_metric = pa_metric.to(device)

    with torch.no_grad():
        for images, labels, _, _, _ in val_loader:
            # move tensors to target device
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            # model may return logits or (logits, aux)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # resize logits to label size when necessary
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False
                )

            # predicted class per pixel
            predictions = torch.argmax(outputs, dim=1)

            # accumulate metric state
            miou_metric.update(predictions, labels)
            pa_metric.update(predictions, labels)

    # compute final metrics
    mean_iou = miou_metric.compute()
    pixel_accuracy = pa_metric.compute()

    return mean_iou, pixel_accuracy