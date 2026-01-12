import os
import cv2
import numpy as np

import helpers.config as config


def create_foreback_data():
    
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

        # Go through all files under leftImg8bit/val 
        for root, dirs, files in os.walk(leftImg8bit_dir):
            for file_name in files:
                if not file_name.endswith('_leftImg8bit.png'):
                    break
                
                # City
                city = os.path.basename(root)
                
                # 1. Image path
                img_path = os.path.join(root, file_name)
                
                # Eg: frankfurt_000000_000294_leftImg8bit.png -> frankfurt_000000_000294_gtFine_labelTrainIds.png
                prefix = file_name.replace('_leftImg8bit.png', '_')
                gt_file_name = prefix + 'gtFine_labelTrainIds.png'
                gt_path = os.path.join(gtFine_dir, city, gt_file_name)
                
                if not os.path.exists(gt_path):
                    print(f"Warning: Label file not found for {file_name}, skipping.")
                    break
                    
                # 2. Read image
                img = cv2.imread(img_path) # BGR
                label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or label is None:
                    print(f"Error reading files: {file_name}")
                    continue

                print(f"Processing {file_name}...")

                # 3. Caculate the BGR mean
                mean_bgr = np.mean(img, axis=(0, 1))
                
                # Output dir
                bg_out_city_dir = os.path.join(config.BACK_IMGS_DIR, data_split, city)
                fg_out_city_dir = os.path.join(config.FORE_IMGS_DIR, data_split, city)
                bg_label_out_city_dir = os.path.join(config.BACK_LABELS_DIR, data_split, city)
                fg_label_out_city_dir = os.path.join(config.FORE_LABELS_DIR, data_split, city)
                bg_mask_out_city_dir = os.path.join(config.BACK_MASK_DIR, data_split, city)
                fg_mask_out_city_dir = os.path.join(config.FORE_MASK_DIR, data_split, city)
                
                os.makedirs(bg_out_city_dir, exist_ok=True)
                os.makedirs(fg_out_city_dir, exist_ok=True)
                os.makedirs(bg_label_out_city_dir, exist_ok=True)
                os.makedirs(fg_label_out_city_dir, exist_ok=True)
                os.makedirs(bg_mask_out_city_dir, exist_ok=True)
                os.makedirs(fg_mask_out_city_dir, exist_ok=True)

                # 4. Create Mask
                
                # --- Foreground Set ---
                # Image
                is_bg = np.isin(label, config.BACKGROUND_TRAINIDS)
                img_fg_only = img.copy()
                img_fg_only[is_bg] = mean_bgr
                
                # Label
                label_fg_only = label.copy()
                label_fg_only[is_bg] = 255  # ignore index
                
                # save in Foreground
                # Add suffix _fg
                fg_img_name = file_name.replace('.png', '_fg.png')
                fg_label_name = gt_file_name.replace('.png', '_fg.png')
                cv2.imwrite(os.path.join(fg_out_city_dir, fg_img_name), img_fg_only)
                cv2.imwrite(os.path.join(fg_label_out_city_dir, fg_label_name), label_fg_only)

                # Save Background Mask (used to create Foreground Set)
                # This mask shows where the background is (True/255)
                bg_mask_name = file_name.replace('.png', '_bg_mask.png')
                cv2.imwrite(os.path.join(bg_mask_out_city_dir, bg_mask_name), is_bg.astype(np.uint8) * 255)
                
                # --- Background Set  ---
                # Image
                is_fg = np.isin(label, config.FOREGROUND_TRAINIDS)
                img_bg_only = img.copy()
                img_bg_only[is_fg] = mean_bgr
                
                # Label
                label_bg_only = label.copy()
                label_bg_only[is_fg] = 255
                
                # save in Background
                # Add suffix _bg
                bg_img_name = file_name.replace('.png', '_bg.png')
                bg_label_name = gt_file_name.replace('.png', '_bg.png')
                cv2.imwrite(os.path.join(bg_out_city_dir, bg_img_name), img_bg_only)
                cv2.imwrite(os.path.join(bg_label_out_city_dir, bg_label_name), label_bg_only)

                # Save Foreground Mask (used to create Background Set)
                # This mask shows where the foreground is (True/255)
                fg_mask_name = file_name.replace('.png', '_fg_mask.png')
                cv2.imwrite(os.path.join(fg_mask_out_city_dir, fg_mask_name), is_fg.astype(np.uint8) * 255)

    print("All processing done!")

if __name__ == '__main__':
    create_foreback_data()
