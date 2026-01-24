"""
Visualize foreground/background validation sets with model predictions.

This helper samples a small number of images from foreground/background
validation directories, runs the model, applies the corresponding mask, and
saves comparison overlays and charts into `output_dir`.
"""

import os
import cv2
import torch
import random
import matplotlib.pyplot as plt

import models 
from datasets import get_config
from helpers.visualize_val import (preprocess_image, decode_segmap, overlay_images)

# ======== Modified by User ========

dataset_name = 'cityscapes'

model_name = 'fcn'

model_path = ''

# ==================================

config = get_config(dataset_name)

def visualize_foreback_test(model, device, img_root, mask_root, output_dir, mean, std, task_name):
    """
    Sample images, run model, and save overlays and comparison charts.

    Args:
        model: segmentation model (in eval mode).
        device: torch device.
        img_root: directory with images to visualize.
        mask_root: directory with binary masks (255 where keep predictions).
        output_dir: where to save outputs.
        mean, std: normalization used by `preprocess_image`.
        task_name: human-readable name for logging.
    """

    print(f"\n--- Processing {task_name} ---")
    print(f"Root: {img_root}")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Collect images and masks
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

    # randomly sample up to `num_samples` images
    num_samples = 20
    total_images = len(test_images)
    num_to_select = min(num_samples, total_images)
    random_indices = random.sample(range(total_images), num_to_select)
    selected_images = [test_images[i] for i in random_indices]
    selected_masks = [test_masks[i] for i in random_indices]

    print(f"Processing {len(selected_images)} images...")

    with torch.no_grad():
        for i, img_path in enumerate(selected_images):
            mask_path = selected_masks[i]

            print(f"Processing: {img_path}")

            # preprocess image and keep original RGB for visualization
            img_tensor, original_img = preprocess_image(img_path, mean, std)

            # load mask and convert to tensor (1 where valid, 0 elsewhere)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_tensor = torch.from_numpy((mask_img == 255)).long().to(device)

            img_tensor = img_tensor.to(device)

            output, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0)

            # apply mask: set prediction to 255 (ignore) where mask is 0
            pred_mask = pred.clone()
            pred_mask[mask_tensor == 0] = 255
            pred_mask = pred_mask.cpu().numpy()

            # convert predicted ids to color map
            seg_color = decode_segmap(pred_mask, nc=256)

            # create overlay image
            overlay = overlay_images(original_img, seg_color)

            filename = os.path.basename(img_path)

            # make dir for one group of images
            group_dir = os.path.join(output_dir, filename)
            os.makedirs(group_dir, exist_ok=True)

            # save origin and seg_color and overlay (convert RGB->BGR for cv2)
            cv2.imwrite(os.path.join(group_dir, f"origin_{filename}"), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(group_dir, f"segcolor_{filename}"), cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(group_dir, f"overlay_{filename}"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # save a comparison chart: original | segmentation | overlay
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(seg_color)
            axes[1].set_title("Segmentation")
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title("overlay")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"comparison_{filename}"))
            plt.close()

    print(f"Done with {task_name}!")


def main():
    """Load model and visualize foreground and background validation sets."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")

    get_model_function = models.get_model(model_name)
    model = get_model_function(num_classes=19, checkpoint=model_path)
    model.to(device)
    model.eval()

    # visualize foreground and background samples
    visualize_foreback_test(
        model,
        device,
        config.DIRS['foreground']['val'],
        config.DIRS['foreground']['mask'],
        config.FORE_VISUAL_DIR,
        config.RGB_MEAN,
        config.RGB_STD,
        "Foreground",
    )

    visualize_foreback_test(
        model,
        device,
        config.DIRS['background']['val'],
        config.DIRS['background']['mask'],
        config.BACK_VISUAL_DIR,
        config.RGB_MEAN,
        config.RGB_STD,
        "Background",
    )


if __name__ == '__main__':
    main()
