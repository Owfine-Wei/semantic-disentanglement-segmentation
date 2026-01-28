
"""
Project configuration: dataset paths, preprocessing values,
class definitions, and visualization settings used across training and
evaluation of the semantic segmentation project.
"""

from .registry import register_configs

@register_configs("bdd10k")
class BDD10kConfig:

    # ==============================================================================
    # DIRECTORIES
    # ==============================================================================

    # BDD10k Data Root
    DATA_DIR = '../data/BDD10k/'
    TRAIN_IMG_DIR = '../data/BDD10k/bdd100k_images_10k/10k/train'
    TRAIN_LABEL_DIR = '../data/BDD10k/bdd100k_seg_maps/labels/train'
    VAL_IMG_DIR = '../data/BDD10k/bdd100k_images_10k/10k/val'
    VAL_LABEL_DIR = '../data/BDD10k/bdd100k_seg_maps/labels/val'
    
    IMG_SUFFIX = '.jpg'
    LABEL_SUFFIX = '_train_id.png'
    DATA_SPLIT = ['train', 'val']

    # Foreground/Background Visualization Results
    FORE_VISUAL_DIR = ''
    BACK_VISUAL_DIR = ''

    # BDD10k Visualization Results
    VAL_IMGS_DIR = ''
    VISUAL_IMGS_DIR = ''

    # MODEL (Where your model saved)
    MODEL_ROOT = '../models/'

    # Dictionary
    IMG_DIR = {DATA_SPLIT[0]:TRAIN_IMG_DIR,
               DATA_SPLIT[1]: VAL_IMG_DIR}
    
    LABEL_DIR = {DATA_SPLIT[0]:TRAIN_LABEL_DIR,
               DATA_SPLIT[1]: VAL_LABEL_DIR}
    

    # ==============================================================================
    # DATA PROCESSING
    # ==============================================================================

    CROP_SIZE = (720, 1280)

    RGB_MEAN = [123.675/255.0, 116.28/255.0, 103.53/255.0]

    RGB_STD = [58.395/255.0, 57.12/255.0, 57.375/255.0]

    # ==============================================================================
    # BDD10k CLASSES
    # ==============================================================================

    NUM_CLASSES = 19

    # Train IDs
    BACKGROUND_TRAINIDS = [
        0,   # road
        1,   # sidewalk
        2,   # building
        3,   # wall
        8,   # vegetation
        9,   # terrain
        10   # sky
    ]

    FOREGROUND_TRAINIDS = [
        4,   # fence
        5,   # pole
        6,   # traffic light
        7,   # traffic sign
        11,  # person
        12,  # rider
        13,  # car
        14,  # truck
        15,  # bus
        16,  # train
        17,  # motorcycle
        18   # bicycle
    ]

    TRAINIDS = [
        0,    # road
        1,    # sidewalk
        2,    # building
        3,    # wall
        4,    # fence
        5,    # pole
        6,    # traffic light
        7,    # traffic sign
        8,    # vegetation
        9,    # terrain
        10,   # sky
        11,   # person
        12,   # rider
        13,   # car
        14,   # truck
        15,   # bus
        16,   # train
        17,   # motorcycle
        18,   # bicycle
        255,  # ignore
    ]

    TRAIN_ID_DICT = {
        "road": 0,
        "sidewalk": 1,
        "building": 2,
        "wall": 3,
        "fence": 4,
        "pole": 5,
        "traffic light": 6,
        "traffic sign": 7,
        "vegetation": 8,
        "terrain": 9,
        "sky": 10,
        "person": 11,
        "rider": 12,
        "car": 13,
        "truck": 14,
        "bus": 15,
        "train": 16,
        "motorcycle": 17,
        "bicycle": 18,
        "ignore": 255
    }

    IMG_SIZE = (720, 1280)


    # ==============================================================================
    # ERASED CLASSES SAMPLES (CES)
    # ==============================================================================

    num_erased_class = 1


    # ==============================================================================
    # CLASS COLORS
    # ==============================================================================

    CLASS_COLORS = {
        0:   (128, 64, 128),   # road
        1:   (244, 35, 232),   # sidewalk
        2:   (70, 70, 70),     # building
        3:   (102, 102, 156),  # wall
        4:   (190, 153, 153),  # fence
        5:   (153, 153, 153),  # pole
        6:   (250, 170, 30),   # traffic light
        7:   (220, 220, 0),    # traffic sign
        8:   (107, 142, 35),   # vegetation
        9:   (152, 251, 152),  # terrain
        10:  (70, 130, 180),   # sky
        11:  (220, 20, 60),    # person
        12:  (255, 0, 0),      # rider
        13:  (0, 0, 142),      # car
        14:  (0, 0, 70),       # truck
        15:  (0, 60, 100),     # bus
        16:  (0, 80, 100),     # train
        17:  (0, 0, 230),      # motorcycle
        18:  (119, 11, 32),    # bicycle
        255: (255, 255, 255)   # ignore/void
    }


