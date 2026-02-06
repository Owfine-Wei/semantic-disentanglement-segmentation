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


def compute_integrated_loss(logits_img, labels, masks, logits_origin_img, origin_labels, features_img, features_origin_img, criterion, mode, alpha, beta):
    """
    Compute integrated loss used for CSG training.

    Args:
        logits_img: logits from the class-erased image branch.
        labels: corresponding label tensor for the class-erased image.
        masks: binary masks marking erased/ignored pixels (0.0/1.0).
        logits_origin_img: logits from the original image branch.
        origin_labels: labels for the original image.
        criterion: segmentation loss function (CrossEntropyLoss).
        mode: when 'csg', include consistency and origin losses.
        alpha: weight for the consistency loss.
        beta: weight for the origin classification loss.

    Returns:
        integrated_loss: scalar tensor combining the selected loss terms.
    """

    if logits_img.shape[-2:] != labels.shape[-2:]:
        logits_img = F.interpolate(logits_img, size=labels.shape[-2:], mode='bilinear', align_corners=False)

    # classification loss on the processed (csg) image
    loss_img = criterion(logits_img, labels.squeeze(1))

    if mode == 'csg' :
        # ensure logits_origin_img matches origin_labels spatial size
        if logits_origin_img.shape[-2:] != origin_labels.shape[-2:]:
            logits_origin_img = F.interpolate(
                logits_origin_img, size=origin_labels.shape[-2:], mode='bilinear', align_corners=False
            )

        # classification loss on the original image
        loss_origin = criterion(logits_origin_img, origin_labels.squeeze(1))

        # resize masks to logits_img size for consistency computation
        if masks.shape[-2:] != features_origin_img.shape[-2:]:
            masks = F.interpolate(masks.float().unsqueeze(1), size=features_origin_img.shape[-2:], mode='nearest').squeeze(1).float()

        # add channel dim for broadcasting
        masks = masks.unsqueeze(1) # B 1 H W

        # freeze the gradient propagation from origin outputs
        features_origin_img_frozen = features_origin_img.detach() # B num_classes H W

        # difference only on non-masksed (valid) pixels
        diff = (features_img - features_origin_img_frozen) * (1.0 - masks) # B num_classes H W
        diff_square = torch.sum(diff**2, dim=(1,2,3), keepdim = True) # B 1 1 1

        num_channels = features_img.shape[1]
        spatial_valid_pixels = torch.sum(1.0 - masks) # 空间点数
        total_valid_elements = spatial_valid_pixels * num_channels + 1e-6 # 总元素数

        # mean squared error per sample over valid pixels (small epsilon to avoid div0)
        consist_loss = diff_square / (total_valid_elements)

        # mean consist_loss in one batch
        consist_loss = consist_loss.mean()

        integrated_loss = loss_img + alpha * consist_loss + beta * loss_origin

        print(f'loss_img{loss_img:.6f}, loss_consist{consist_loss:.6f}, loss_origin{loss_origin:.6f}')

    else:
        integrated_loss = loss_img

    return integrated_loss
