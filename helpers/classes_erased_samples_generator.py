import cv2
import random
import torch

import helpers.config as config

def classes_erased_samples_generator(origin_image, origin_label, mode):

    # 2. Caculate the RGB mean
    mean_rgb = origin_image.mean(dim=(1,2), keepdim=True)

    # 3. Create Mask
    
    # Get classes present in the current image
    present_classes = torch.unique(origin_label)

    # Randomly select classes to erase from present classes
    if mode == 'foreground' :
        valid_classes = [tid for tid in present_classes if tid in config.FOREGROUND_TRAINIDS and tid != 255]
    elif mode == 'background' :
        valid_classes = [tid for tid in present_classes if tid in config.BACKGROUND_TRAINIDS and tid != 255]
    elif mode == 'both' :
        valid_classes = [tid for tid in present_classes if tid in config.TRAINIDS and tid != 255]

    if len(valid_classes) > 0:
        count = min(len(valid_classes), config.num_erased_class)
        erased_classes = random.sample(valid_classes, count)
    else:
        erased_classes = []

    erased_classes = torch.tensor(erased_classes).long()

    # Create mask where label equals any of the erased_classes
    csg_mask = torch.isin(origin_label, erased_classes)

    # Image
    csg_img = origin_img.copy()
    csg_img = torch.where(csg_mask, mean_rgb, origin_image)
    
    # Label
    csg_label = origin_label.copy()
    csg_label = torch.where(csg_mask, mean_rgb, origin_label)
    
    # mask
    csg_mask = torch.isin( origin_label, erased_classes + [255] )
    csg_mask = csg_mask.float()

    # return Tensor
    return csg_img, csg_label, csg_mask

