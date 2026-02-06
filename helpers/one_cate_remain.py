import torch

def one_cate_remain(config, origin_image, origin_label, mode):

    # compute per-channel mean to fill erased regions
    mean_rgb = torch.tensor(config.RGB_MEAN).view(3,1,1)

    # choose valid classes depending on mode and exclude ignore index 255
    if mode == 'flat':
        valid_classes = list(set(config.TRAINIDS)-set(config.FLAT_TRAINIDS))
    elif mode == 'construction':
        valid_classes = list(set(config.TRAINIDS)-set(config.CONSTRUCTION_TRAINIDS))
    elif mode == 'object':
        valid_classes = list(set(config.TRAINIDS)-set(config.OBJECT_TRAINIDS))
    elif mode == 'nature':
        valid_classes = list(set(config.TRAINIDS)-set(config.NATURE_TRAINIDS))
    elif mode == 'sky':
        valid_classes = list(set(config.TRAINIDS)-set(config.SKY_TRAINIDS))
    elif mode == 'human':
        valid_classes = list(set(config.TRAINIDS)-set(config.HUMAN_TRAINIDS))
    elif mode == 'vehicle':
        valid_classes = list(set(config.TRAINIDS)-set(config.VEHICLE_TRAINIDS))

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