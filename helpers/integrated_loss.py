import torch.nn.functional as F
import torch

def compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, mode, alpha, beta):
        
        # Calculate loss
        loss_img = criterion(outputs_img, labels.squeeze(1))
        if mode=='csg+origin':

            # Resize outputs_origin to match origin_labels if necessary
            if outputs_origin.shape[-2:] != origin_labels.shape[-2:]:
                outputs_origin = F.interpolate(outputs_origin, size=origin_labels.shape[-2:], mode='bilinear', align_corners=False)

            loss_origin = criterion(outputs_origin, origin_labels.squeeze(1))
            
            # Resize mask to match outputs_img if necessary (for consistency loss)
            if mask.shape[-2:] != outputs_img.shape[-2:]:
                    mask = F.interpolate(mask.float().unsqueeze(1), size=outputs_img.shape[-2:], mode='nearest').squeeze(1).long()

            mask = mask.unsqueeze(1)

            outputs_origin_frozen = outputs_origin.detach()

            diff = (outputs_img - outputs_origin_frozen) * (1.0 - mask)

            valid_pixels = torch.sum(1.0 - mask) * outputs_img.size(1) 

            # MSE Loss
            consist_loss = torch.sum(diff**2)  / (valid_pixels + 1e-6)

            # # CE Loss
            # consist_loss = compute_ce_loss(outputs_img, outputs_origin_frozen, mask)

            integrated_loss = loss_img + alpha * consist_loss + beta * loss_origin
            
        else:
            integrated_loss = loss_img

        return integrated_loss


def compute_ce_loss(student_logits, teacher_logits, mask, T=1):
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)

    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none') # (B, C, H, W)

    loss_pixel = kl_div.sum(dim=1)  # (B, H, W)
    valid_mask = (1 - mask).squeeze(1) # (B, H, W)

    loss_c = (loss_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return loss_c * (T ** 2)
