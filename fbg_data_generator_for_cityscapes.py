"""
Create foreground-only and background-only datasets from Cityscapes.

This script extracts images and labels containing only foreground or only
background by replacing the opposite region with the image mean and setting
its label to the ignore index. It also writes binary masks used by the
dataset creation.
"""

import argparse
import os
import cv2
import numpy as np
from configs import get_config
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create Foreground/Background Datasets')
parser.add_argument('--dataset_name', default='', help='the origin dataset name', required=True)
arg = parser.parse_args()

config = get_config(arg.dataset_name)

def create_foreback_data():
    """
    Walk Cityscapes splits and generate fore/back images, labels and masks.

    The function expects Cityscapes layout under `config.DATA_DIR` and writes
    outputs to directories defined in `config`, so users need to check on `config.py` before using this script.
    """

    # Make directories
    directories = [
        config.BACK_IMGS_DIR,
        config.FORE_IMGS_DIR,
        config.BACK_LABELS_DIR,
        config.FORE_LABELS_DIR,
        config.BACK_MASK_DIR,
        config.FORE_MASK_DIR
    ]
    for directory in directories:
        if directory: 
            os.makedirs(directory, exist_ok=True)
            

    print(f"Data Root: {config.DATA_DIR}\n")
    print(f"Background Output (Images only containing Background): {config.BACK_IMGS_DIR}\n")
    print(f"Foreground Output (Images only containing Foreground): {config.FORE_IMGS_DIR}\n")
    print(f"Background Label Output: {config.BACK_LABELS_DIR}\n")
    print(f"Foreground Label Output: {config.FORE_LABELS_DIR}\n")
    print(f"Background Mask Output: {config.BACK_MASK_DIR}\n")
    print(f"Foreground Mask Output: {config.FORE_MASK_DIR}\n")

    data_split_list = ['train', 'val', 'test']

    for data_split in data_split_list:

        leftImg8bit_dir = os.path.join(config.DATA_DIR, 'leftImg8bit', data_split)
        gtFine_dir = os.path.join(config.DATA_DIR, 'gtFine', data_split)

        if not os.path.exists(leftImg8bit_dir) or not os.path.exists(gtFine_dir):
            print(f"Error: Input directories not found.\n{leftImg8bit_dir}\n{gtFine_dir}")
            return

        # iterate over images in the split
        for root, dirs, files in os.walk(leftImg8bit_dir):
            for file_name in tqdm(files):
                if not file_name.endswith('_leftImg8bit.png'):
                    break

                city = os.path.basename(root)

                # construct image and label paths
                img_path = os.path.join(root, file_name)
                prefix = file_name.replace('_leftImg8bit.png', '_')
                gt_file_name = prefix + 'gtFine_labelTrainIds.png'
                gt_path = os.path.join(gtFine_dir, city, gt_file_name)

                if not os.path.exists(gt_path):
                    print(f"Warning: Label file not found for {file_name}, skipping.")
                    break

                # read image (BGR) and label (grayscale train ids)
                img = cv2.imread(img_path)
                label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                if img is None or label is None:
                    print(f"Error reading files: {file_name}")
                    continue

                # print(f"Processing {file_name}...")

                # per-image BGR mean used to fill erased regions
                mean_bgr = np.mean(img, axis=(0, 1))

                # prepare output directories per city
                bg_out_city_dir = os.path.join(config.FORE_IMGS_DIR, data_split, city)
                fg_out_city_dir = os.path.join(config.BACK_IMGS_DIR, data_split, city)
                bg_label_out_city_dir = os.path.join(config.FORE_LABELS_DIR, data_split, city)
                fg_label_out_city_dir = os.path.join(config.BACK_LABELS_DIR, data_split, city)
                bg_mask_out_city_dir = os.path.join(config.FORE_MASK_DIR, data_split, city)
                fg_mask_out_city_dir = os.path.join(config.BACK_MASK_DIR, data_split, city)

                os.makedirs(bg_out_city_dir, exist_ok=True)
                os.makedirs(fg_out_city_dir, exist_ok=True)
                os.makedirs(bg_label_out_city_dir, exist_ok=True)
                os.makedirs(fg_label_out_city_dir, exist_ok=True)
                os.makedirs(bg_mask_out_city_dir, exist_ok=True)
                os.makedirs(fg_mask_out_city_dir, exist_ok=True)

                # --- Foreground Set ---
                # mask where pixels belong to background classes
                is_bg = np.isin(label, config.BACKGROUND_TRAINIDS)
                img_fg_only = img.copy()
                img_fg_only[is_bg] = mean_bgr

                label_fg_only = label.copy()
                label_fg_only[is_bg] = 255  # mark ignored pixels

                # save foreground-only image and label
                fg_img_name = file_name.replace('.png', '_bg.png')
                fg_label_name = gt_file_name.replace('.png', '_bg.png')
                cv2.imwrite(os.path.join(fg_out_city_dir, fg_img_name), img_fg_only)
                cv2.imwrite(os.path.join(fg_label_out_city_dir, fg_label_name), label_fg_only)

                # save background mask used when creating foreground set
                fg_mask_name = file_name.replace('.png', '_bg_mask.png')
                is_bg = is_bg | (label_fg_only == 255)
                cv2.imwrite(os.path.join(fg_mask_out_city_dir, fg_mask_name), is_bg.astype(np.uint8) * 255)

                # --- Background Set ---
                # mask where pixels belong to foreground classes
                is_fg = np.isin(label, config.FOREGROUND_TRAINIDS)
                img_bg_only = img.copy()
                img_bg_only[is_fg] = mean_bgr

                label_bg_only = label.copy()
                label_bg_only[is_fg] = 255

                # save background-only image and label
                bg_img_name = file_name.replace('.png', '_fg.png')
                bg_label_name = gt_file_name.replace('.png', '_fg.png')
                cv2.imwrite(os.path.join(bg_out_city_dir, bg_img_name), img_bg_only)
                cv2.imwrite(os.path.join(bg_label_out_city_dir, bg_label_name), label_bg_only)

                # save foreground mask used when creating background set
                bg_mask_name = file_name.replace('.png', '_fg_mask.png')
                is_fg = is_fg | (label_bg_only == 255)
                cv2.imwrite(os.path.join(bg_mask_out_city_dir, bg_mask_name), is_fg.astype(np.uint8) * 255)

    print("All processing done!")


if __name__ == '__main__':
    create_foreback_data()