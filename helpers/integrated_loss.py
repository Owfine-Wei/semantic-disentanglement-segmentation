"""
Integrated loss utilities.

Provides a small helper to compute the combined loss used in training:
- classification loss on the (possibly class-erased) image (`loss_img`),
- optional consistency loss between current outputs and original-image outputs,
- optional origin-image classification loss. 
The combined loss is returnedas a single tensor.
"""

import torch.nn.functional as F
import torch


# def compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, mode, alpha, beta):
#     """Compute integrated loss used for CSG training.

#     Args:
#         outputs_img: logits from the class-erased image branch.
#         labels: corresponding label tensor for the class-erased image.
#         mask: binary mask marking erased/ignored pixels (0.0/1.0).
#         outputs_origin: logits from the original image branch.
#         origin_labels: labels for the original image.
#         criterion: segmentation loss function (CrossEntropyLoss).
#         mode: when 'csg+origin', include consistency and origin losses.
#         alpha: weight for the consistency loss.
#         beta: weight for the origin classification loss.

#     Returns:
#         integrated_loss: scalar tensor combining the selected loss terms.
#     """

#     # classification loss on the processed (csg) image
#     loss_img = criterion(outputs_img, labels.squeeze(1))

#     if mode == 'csg+origin':
#         # ensure outputs_origin matches origin_labels spatial size
#         if outputs_origin.shape[-2:] != origin_labels.shape[-2:]:
#             outputs_origin = F.interpolate(
#                 outputs_origin, size=origin_labels.shape[-2:], mode='bilinear', align_corners=False
#             )

#         # classification loss on the original image
#         loss_origin = criterion(outputs_origin, origin_labels.squeeze(1))

#         # resize mask to outputs_img size for consistency computation
#         if mask.shape[-2:] != outputs_img.shape[-2:]:
#             mask = F.interpolate(mask.float().unsqueeze(1), size=outputs_img.shape[-2:], mode='nearest').squeeze(1).long()

#         # add channel dim for broadcasting
#         mask = mask.unsqueeze(1) # B 1 H W

#         # freeze the gradient propagation from origin outputs
#         outputs_origin_frozen = outputs_origin.detach() # B num_classes H W

#         # difference only on non-masked (valid) pixels
#         diff = (outputs_img - outputs_origin_frozen) * (1.0 - mask) # B num_classes H W
#         diff_square = torch.sum(diff**2, dim=(1,2,3), keepdim = True) # B 1 1 1

#         # number of valid pixels times number of channels (classes)
#         valid_pixels = torch.sum((1.0 - mask), dim=(1,2,3), keepdim = True)  
#         # B 1 1 1

#         # mean squared error per sample over valid pixels (small epsilon to avoid div0)
#         consist_loss = diff_square / (valid_pixels + 1e-6)

#         # mean consist_loss in one batch
#         consist_loss = consist_loss.mean()

#         integrated_loss = loss_img + alpha * consist_loss + beta * loss_origin
#     else:
#         integrated_loss = loss_img

#     return integrated_loss

def compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, mode, alpha, beta):
    """Compute integrated loss used for CSG training with CE-based consistency."""

    # classification loss on the processed (csg) image
    loss_img = criterion(outputs_img, labels.squeeze(1))

    if mode == 'csg+origin':
        # ensure outputs_origin matches origin_labels spatial size
        if outputs_origin.shape[-2:] != origin_labels.shape[-2:]:
            outputs_origin = F.interpolate(
                outputs_origin, size=origin_labels.shape[-2:], mode='bilinear', align_corners=False
            )

        # classification loss on the original image
        loss_origin = criterion(outputs_origin, origin_labels.squeeze(1))

        # resize mask to outputs_img size for consistency computation
        if mask.shape[-2:] != outputs_img.shape[-2:]:
            mask = F.interpolate(mask.float().unsqueeze(1), size=outputs_img.shape[-2:], mode='nearest').squeeze(1).long()

        # add channel dim for broadcasting
        mask = mask.unsqueeze(1) # B 1 H W

        # -------------------------------------------------------
        # 改动部分：将 MSE 更改为 Cross-Entropy 形式的一致性损失
        # -------------------------------------------------------
        
        # 1. 冻结原图分支梯度，并将其转化为概率分布 (Target)
        # 使用 Softmax 将 Logits 转化为分布
        outputs_origin_soft = F.softmax(outputs_origin.detach(), dim=1) 

        # 2. 对擦除后的分支计算 Log-Softmax (Prediction)
        outputs_img_log_soft = F.log_softmax(outputs_img, dim=1)

        # 3. 计算逐像素的交叉熵: - \sum(p * log(q))
        # 这里的 ce_per_pixel 形状为 (B, H, W)
        ce_per_pixel = -(outputs_origin_soft * outputs_img_log_soft).sum(dim=1, keepdim=True)

        # 4. 仅在非遮挡区域 (valid pixels) 应用损失
        # mask 为 1 表示擦除/忽略，1.0 - mask 表示保留的区域
        valid_mask = (1.0 - mask)
        consist_loss_map = ce_per_pixel * valid_mask

        # 5. 计算有效像素的平均损失
        diff_square = torch.sum(consist_loss_map, dim=(1, 2, 3), keepdim=True) # 沿用原变量名以减少结构改动
        valid_pixels = torch.sum(valid_mask, dim=(1, 2, 3), keepdim=True)

        consist_loss = diff_square / (valid_pixels + 1e-6)
        consist_loss = consist_loss.mean()
        # -------------------------------------------------------

        integrated_loss = loss_img + alpha * consist_loss + beta * loss_origin
    else:
        integrated_loss = loss_img

    return integrated_loss