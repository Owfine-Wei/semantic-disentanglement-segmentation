"""
Visualization helpers for validation: decode, overlay and save results.

Provides small utilities to preprocess images, convert label ids to colors,
overlay predictions on the original image, and save sample visualizations.
"""

import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import models
from datasets import get_config

# ======== Modified by User ========

dataset_name = 'cityscapes'

model_type = 'FCN'

model_path = ''

# ==================================

config = get_config(dataset_name)

# To_Tensor and Normalize the Image
def preprocess_image(image_path, mean, std):
    """Read image, apply ToTensor+Normalize and return (1,C,H,W), original RGB."""

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img_tensor = transform(img)
    return img_tensor.unsqueeze(0), original_img


# Convert numeric predictions (ids) into color map
def decode_segmap(image, nc=21):
    """Map integer label ids to RGB colors using `config.CLASS_COLORS`."""

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l

        if l in config.CLASS_COLORS:
            r[idx] = config.CLASS_COLORS[l][0]
            g[idx] = config.CLASS_COLORS[l][1]
            b[idx] = config.CLASS_COLORS[l][2]
        else:
            r[idx] = 0
            g[idx] = 0
            b[idx] = 0

    rgb = np.stack([r, g, b], axis=2)
    return rgb


# overlay the original image with predictions(colors)
def overlay_images(original_img, seg_map, alpha=0.5):
    """Blend `seg_map` onto `original_img` with opacity `alpha`."""

    original_np = np.array(original_img)

    if original_np.shape[:2] != seg_map.shape[:2]:
        seg_map = cv2.resize(seg_map, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = (original_np * (1 - alpha) + seg_map * alpha).astype(np.uint8)
    return overlay


def visualize_val(model, device, img_root, output_dir, mean, std):
    """Sample images from `img_root`, run model and save visual results."""

    os.makedirs(output_dir, exist_ok=True)

    # Collect Val Images
    test_images = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                test_images.append(os.path.join(root, file))

    if not test_images:
        print(f"No images found in {img_root}")
        return

    # Choose Images Randomly (up to `num_samples`)
    num_samples = 20
    selected_images = random.sample(test_images, min(num_samples, len(test_images)))

    print(f"Processing {len(selected_images)} images...")

    with torch.no_grad():
        for i, img_path in enumerate(selected_images):
            print(f"Processing: {img_path}")

            img_tensor, original_img = preprocess_image(img_path, mean, std)
            img_tensor = img_tensor.to(device)

            output, _ = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            seg_color = decode_segmap(pred, nc=256)

            overlay = overlay_images(original_img, seg_color)

            filename = os.path.basename(img_path)

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

    print(f"Done! Results saved to {output_dir}")


def main():
    """Load model and run `visualize_val` over `config.VAL_IMGS_DIR`."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")

    model = models.get_model(num_classes=19, checkpoint=model_path, model_type = model_type)

    model.to(device)
    model.eval()

    # Test
    visualize_val(model, device, config.VAL_IMGS_DIR, config.VISUAL_IMGS_DIR, config.RGB_MEAN, config.RGB_STD)


if __name__ == '__main__':
    main()
