"""
Generate samples with selected classes erased.

Given an image and its label map, randomly erase a small number of
classes (replace pixels with the image mean and label with 255) from the
foreground, background, or both according to `mode`.
"""

import random
import torch

def classes_erased_samples_generator(config, origin_image, origin_label, mode):
    """Return image/label/mask with selected classes erased.

    Args:
        origin_image: tensor image with shape (C,H,W).
        origin_label: tensor label map with shape (H,W) or (1,H,W).
        mode: one of 'foreground', 'background', or 'both' selecting which
              class group to sample erased classes from.

    Returns:
        csg_img: image tensor where erased-class pixels are set to channel mean, with shape (C,H,W).
        csg_label: label tensor where erased-class pixels are set to 255 (ignore), with shape (H,W).
        csg_mask: float mask (0/1) marking erased pixels (including 255), with shape (H,W).
    """

    # compute per-channel mean to fill erased regions
    mean_rgb = origin_image.mean(dim=(1, 2), keepdim=True)

    # classes present in the current label map
    present_classes = torch.unique(origin_label)

    # choose valid classes depending on mode and exclude ignore index 255
    if mode == 'foreground':
        valid_classes = [tid for tid in present_classes if tid in config.FOREGROUND_TRAINIDS and tid != 255]
    elif mode == 'background':
        valid_classes = [tid for tid in present_classes if tid in config.BACKGROUND_TRAINIDS and tid != 255]
    elif mode == 'both':
        valid_classes = [tid for tid in present_classes if tid in config.TRAINIDS and tid != 255]
    else:
        valid_classes = []

    # sample up to `num_erased_class` classes to erase
    if len(valid_classes) > 0:
        count = min(len(valid_classes), config.num_erased_class)
        erased_classes = random.sample(valid_classes, count)
    else:
        erased_classes = []

    erased_classes = torch.tensor(erased_classes).long()

    # boolean mask where label equals any erased class
    csg_mask = torch.isin(origin_label, erased_classes)

    # image: replace erased pixels with per-channel mean
    csg_img = origin_image.clone()
    csg_img = torch.where(csg_mask, mean_rgb, origin_image)

    # label: mark erased pixels as 255 (ignore index)
    csg_label = origin_label.clone()
    csg_label = torch.where(csg_mask, 255, origin_label)

    # final mask includes erased classes and existing 255 pixels
    csg_mask = torch.isin(origin_label, torch.cat((erased_classes, torch.tensor([255]))))
    csg_mask = csg_mask.float()

    return csg_img, csg_label, csg_mask

