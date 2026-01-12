import cv2
import random
import numpy as np

import helpers.config as config

def classes_erased_samples_generator(img_path, label_path, mode):

    # 1. Read image
    img = cv2.imread(img_path) # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # 2. Caculate the RGB mean
    mean_rgb = np.mean(img, axis=(0, 1))

    # 3. Create Mask
    
    # Get classes present in the current image
    present_classes = np.unique(label)

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

    # Create mask where label equals any of the erased_classes
    csg_mask = np.isin(label, erased_classes)

    # Image
    csg_img = img.copy()
    csg_img[csg_mask] = mean_rgb
    
    # Label
    csg_label = label.copy()
    csg_label[csg_mask] = 255
    
    # mask
    csg_mask = np.isin( label, erased_classes + [255] )
    csg_mask = csg_mask.astype(float) * 255.0

    # return Numpy.Array
    return csg_img, csg_label, csg_mask

