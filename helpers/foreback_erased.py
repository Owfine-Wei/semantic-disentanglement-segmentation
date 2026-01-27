import torch

def foreback_erased(config, origin_image, origin_label ,mode):

    # compute per-channel mean to fill erased regions
    mean_rgb = origin_image.mean(dim=(1, 2), keepdim=True)

    # choose valid classes depending on mode and exclude ignore index 255
    if mode == 'foreground':
        valid_classes = config.FOREGROUND_TRAINIDS
    elif mode == 'background':
        valid_classes = config.BACKGROUND_TRAINIDS

    erased_classes = torch.tensor(valid_classes).long()

    # boolean mask where label equals any erased class
    erased_mask = torch.isin(origin_label, erased_classes)

    # image: replace erased pixels with per-channel mean
    erased_img = origin_image.clone()
    erased_img = torch.where(erased_mask, mean_rgb, origin_image)

    # label: mark erased pixels as 255 (ignore index)
    erased_label = origin_label.clone()
    erased_label = torch.where(erased_mask, 255, origin_label)

    # final mask includes erased classes and existing 255 pixels
    erased_mask = torch.isin(origin_label, torch.cat((erased_classes, torch.tensor([255]))))
    erased_mask = erased_mask.float()

    return erased_img, erased_label, erased_mask