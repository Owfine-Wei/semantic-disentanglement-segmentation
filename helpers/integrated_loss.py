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


def compute_integrated_loss(outputs_img, labels, mask, outputs_origin, origin_labels, criterion, mode, alpha, beta):
    """
    Compute integrated loss used for CSG training.

    Args:
        outputs_img: logits from the class-erased image branch.
        labels: corresponding label tensor for the class-erased image.
        mask: binary mask marking erased/ignored pixels (0.0/1.0).
        outputs_origin: logits from the original image branch.
        origin_labels: labels for the original image.
        criterion: segmentation loss function (CrossEntropyLoss).
        mode: when 'csg+origin', include consistency and origin losses.
        alpha: weight for the consistency loss.
        beta: weight for the origin classification loss.

    Returns:
        integrated_loss: scalar tensor combining the selected loss terms.
    """

    # classification loss on the processed (csg) image
    loss_img = criterion(outputs_img, labels.squeeze(1))

    if mode == 'csg' :
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

        # freeze the gradient propagation from origin outputs
        outputs_origin_frozen = outputs_origin.detach() # B num_classes H W

        # difference only on non-masked (valid) pixels
        diff = (outputs_img - outputs_origin_frozen) * (1.0 - mask) # B num_classes H W
        diff_square = torch.sum(diff**2, dim=(1,2,3), keepdim = True) # B 1 1 1

        # number of valid pixels times number of channels (classes)
        valid_pixels = torch.sum((1.0 - mask), dim=(1,2,3), keepdim = True)  
        # B 1 1 1

        # mean squared error per sample over valid pixels (small epsilon to avoid div0)
        consist_loss = diff_square / (valid_pixels + 1e-6)

        # mean consist_loss in one batch
        consist_loss = consist_loss.mean()

        integrated_loss = loss_img + alpha * consist_loss + beta * loss_origin
    else:
        integrated_loss = loss_img

    return integrated_loss
