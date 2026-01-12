# ==============================================================================
# CONFIG: Paths, data stats, model, classes, and training
# ==============================================================================

# ==============================================================================
# DIRECTORIES
# ==============================================================================

# Cityscapes Data Root
DATA_DIR = '/root/autodl-tmp/data/Cityscapes/'

# Foreground/Background Data Root
FOREBACK_DATA_DIR = '/root/autodl-tmp/data/ForeBackground/'

# Foreground/Background Images
FORE_IMGS_DIR = '/root/autodl-tmp/data/ForeBackground/leftImg8bit_fg/'
BACK_IMGS_DIR = '/root/autodl-tmp/data/ForeBackground/leftImg8bit_bg/'

# Foreground/Background Labels
FORE_LABELS_DIR = '/root/autodl-tmp/data/ForeBackground/gtFine_fg/'
BACK_LABELS_DIR = '/root/autodl-tmp/data/ForeBackground/gtFine_bg/'

# Foreground/Background Masks
FORE_MASK_DIR = '/root/autodl-tmp/data/ForeBackground/mask_fg/'
BACK_MASK_DIR = '/root/autodl-tmp/data/ForeBackground/mask_bg/'

# Foreground/Background Visualization Results
FORE_VISUAL_DIR = '/root/autodl-tmp/data/Result/Fore_Test_Result/'
BACK_VISUAL_DIR = '/root/autodl-tmp/data/Result/Back_Test_Result/'

# Cityscapes Visualization Results
VAL_IMGS_DIR = '/root/autodl-tmp/data/Cityscapes/leftImg8bit/val/'
VISUAL_IMGS_DIR = '/root/autodl-tmp/data/Cityscapes/leftImgs8bit/val/'

# MODEL (Where your model saved)
MODEL_ROOT = '/root/autodl-tmp/models/'


# Data Dictionary
DIRS = {
    'origin': {
        'imgs': DATA_DIR + 'leftImg8bit/',
        'labels': DATA_DIR + 'gtFine/',
        'mask': ''
    },
    'foreground': {
        'imgs': FORE_IMGS_DIR,
        'labels': FORE_LABELS_DIR,
        'mask': FORE_MASK_DIR
    },
    'background': {
        'imgs': BACK_IMGS_DIR,
        'labels': BACK_LABELS_DIR,
        'mask': BACK_MASK_DIR
    },
    'csg': {
        'imgs': DATA_DIR + 'leftImg8bit/',
        'labels': DATA_DIR + 'gtFine/'
    }
}

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

CROP_SIZE = {
    'h': 512,
    'w': 1024
}

RGB_MEAN = [123.675/255.0, 116.28/255.0, 103.53/255.0]

RGB_STD = [58.395/255.0, 57.12/255.0, 57.375/255.0]

# ==============================================================================
# CITYSCAPES CLASSES
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
    255   # ignore
]


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


