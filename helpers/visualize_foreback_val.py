import os
import cv2
import torch
import random
import matplotlib.pyplot as plt

from fcn_model import get_model
import helpers.config as config
from helpers.visualize_val import (preprocess_image, decode_segmap, overlap_images)

# ======== Modified by User ========
model_path = ''
# ==================================

# Main
def visualize_foreback_test(model, device, img_root, mask_root, output_dir, mean, std, task_name):
    print(f"\n--- Processing {task_name} ---")
    print(f"Root: {img_root}")
    print(f"Output: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Collect Forebackground Val Images
    test_images = []
    test_masks = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                test_images.append(os.path.join(root, file))
    for root, dirs, files in os.walk(mask_root):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                test_masks.append(os.path.join(root, file))
    
    if not test_images:
        print(f"No images found in {img_root}")
        return

    # Choose Images Randomly
    num_samples = 20
    total_images = len(test_images)
    num_to_select = min(num_samples, total_images)
    
    # Generate random indices
    random_indices = random.sample(range(total_images), num_to_select)
    selected_images = [test_images[i] for i in random_indices]
    selected_masks = [test_masks[i] for i in random_indices]
    
    print(f"Processing {len(selected_images)} images...")

    with torch.no_grad():

        for i, img_path in enumerate(selected_images):
            mask_path = selected_masks[i]

            print(f"Processing: {img_path}")
            
            img_tensor, original_img = preprocess_image(img_path, mean, std)
            
            # Read mask image and convert to tensor
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask_tensor: 1 where 255, 0 where 0
            mask_tensor = torch.from_numpy((mask_img == 255)).long().to(device)
            
            img_tensor = img_tensor.to(device)
            
            output, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0)

            # Apply mask: 255 where mask is 0, keep pred where mask is 1
            pred_mask = pred.clone()
            pred_mask[mask_tensor == 0] = 255
            pred_mask = pred_mask.cpu().numpy()

            # Make Predictions(Colors)
            seg_color = decode_segmap(pred_mask, nc=256) 
            
            # Make Overlap Images
            overlap = overlap_images(original_img, seg_color)
            
            filename = os.path.basename(img_path)

            # Save overlap separately
            cv2.imwrite(os.path.join(output_dir, f"overlay_{filename}"), cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))
            
            # Make Comparison Chart
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(seg_color)
            axes[1].set_title("Segmentation")
            axes[1].axis('off')
            
            axes[2].imshow(overlap)
            axes[2].set_title("Overlap")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{filename}"))
            plt.close()
            
    print(f"Done with {task_name}!")


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")

    model = get_model(num_classes=19, checkpoint = model_path) 
    
    model.to(device)
    model.eval()

    # Process Foreground
    visualize_foreback_test(model, device, config.DIRS['foreground']['val'], config.DIRS['foreground']['mask'], config.FORE_VISUAL_DIR, config.RGB_MEAN, config.RGB_STD, "Foreground")

    # Process Background
    visualize_foreback_test(model, device, config.DIRS['background']['val'], config.DIRS['background']['mask'], config.BACK_VISUAL_DIR, config.RGB_MEAN, config.RGB_STD, "Background")

if __name__ == '__main__':
    main()
