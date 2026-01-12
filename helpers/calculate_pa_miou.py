import torch
import torch.nn.functional as F

import helpers.config as config
from torchmetrics.classification import (MulticlassJaccardIndex,MulticlassAccuracy)

def calculate_metrics(model, val_loader, device, num_classes=config.NUM_CLASSES):
    
    model.eval()
    
    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=255,
        average='macro'       # macro = mean IoU (mIoU)
    )

    pa_metric = MulticlassAccuracy(
        num_classes=num_classes,
        ignore_index=255,
        average='micro'       # micro = Pixel Accuracy
    )
    
    # Try to disable sync if possible through attributes (works in many versions)
    if hasattr(miou_metric, 'sync_on_compute'):
        miou_metric.sync_on_compute = False
    if hasattr(pa_metric, 'sync_on_compute'):
        pa_metric.sync_on_compute = False
    
    miou_metric = miou_metric.to(device)
    pa_metric = pa_metric.to(device)

    with torch.no_grad():
        for images, labels, _, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images)
            
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Resize outputs to match labels size if necessary
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            predictions = torch.argmax(outputs, dim=1)
            
            miou_metric.update(predictions, labels)
            pa_metric.update(predictions,labels)
    
    mean_iou = miou_metric.compute()
    pixel_accuracy = pa_metric.compute()
    
    return mean_iou, pixel_accuracy