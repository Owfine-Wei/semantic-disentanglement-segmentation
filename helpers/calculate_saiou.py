import torch
import torch.nn.functional as F

def cal_fore_back_iou(model, val_loader, train_ids, device):

    model.eval()
    
    total_iou = 0
    num = 0

    with torch.no_grad():

        for images, labels, _, _, _ in val_loader:

            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            batch_size = images.size(0)

            outputs = model(images)
            
            # Resize outputs to match labels size if necessary
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1)
            
            # Create masks for train_ids
            label_in_train = torch.zeros_like(labels, dtype=torch.bool)
            pred_in_train = torch.zeros_like(preds, dtype=torch.bool)
            
            for tid in train_ids:
                label_in_train |= (labels == tid)
                pred_in_train |= (preds == tid)
            
            # Intersection: label == pred AND label in train_ids
            # (If label == pred and label in train_ids, then pred is also in train_ids)
            intersection = (preds == labels) & label_in_train
            
            # Union: label in train_ids OR pred in train_ids
            union = label_in_train | pred_in_train
            
            # IoU for each image
            for i in range(batch_size):
                total_iou += (intersection[i].sum().item() / union[i].sum().item())
                num += 1

                # if (num % 100 == 0) :
                #     print(f"Processed {num} images ...\n")
        
    return total_iou / num


def cal_sa_iou(model, fore_loader, back_loader, fore_ids, back_ids, device):

    # print(f"Calculate Foreground Val IoU\n")
    fore_iou = cal_fore_back_iou(model,fore_loader,fore_ids,device)

    # print(f"Calculate Background Val IoU\n") 
    back_iou = cal_fore_back_iou(model,back_loader,back_ids,device)
    
    fore_classes = len(fore_ids)
    back_classes = len(back_ids)
    data_classes = fore_classes + back_classes

    sa_iou = (fore_classes/data_classes) * fore_iou + (back_classes/data_classes) * back_iou

    return fore_iou, back_iou, sa_iou

    



