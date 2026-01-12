import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data_sds_cityscapes import SDS_CityScapes
import helpers.config as config


# ------------------ Editable globals (change and run immediately) ------------------
# dataset mode: origin|foreground|background|csg+origin|csg_only|nda
MODE = 'csg+origin'
# split: train|val|test
SPLIT = 'train'
# csg mode (if using csg modes), e.g. None or 'some_mode'
CSG_MODE = 'background'
# random seed
SEED = 77
# number of samples to save
NUM_SAMPLES = 5
# output directory
OUTDIR = '/root/autodl-tmp/outputs/test_data_samples'
# ----------------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def colorize_label(label_np, color_map):
    h, w = label_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_map.items():
        mask = label_np == k
        if mask.any():
            color_img[mask] = v
    return color_img


def unnormalize_image(img_tensor):
    # img_tensor: C,H,W, normalized with config.RGB_MEAN/std after scaling to [0,1]
    mean = torch.tensor(config.RGB_MEAN).view(3, 1, 1)
    std = torch.tensor(config.RGB_STD).view(3, 1, 1)
    img = img_tensor.clone().cpu() * std + mean
    img = img.numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_samples(root, mode, split, csg_mode, seed, num_samples, outdir):
    dataset = SDS_CityScapes(root, mode, split, csg_mode)
    length = len(dataset)
    if length == 0:
        raise RuntimeError(f"Dataset is empty: mode={mode}, split={split}, root={root}")

    rng = random.Random(seed)
    indices = rng.sample(range(length), k=min(num_samples, length))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, idx in enumerate(indices, start=1):
        sample = dataset[idx]

        # sample layout: (image, label, mask, origin_image, origin_label)
        # prefer origin image/label if present
        img_tensor = sample[0]
        label_tensor = sample[1]

        # img_tensor might already be normalized; unnormalize for visualization
        try:
            img_np = unnormalize_image(img_tensor)
        except Exception:
            # fallback: handle if tensor is 0-1 already without normalize
            img_np = img_tensor.clone().cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        label_np = label_tensor.clone().cpu().numpy().astype(np.int32)

        color_label = colorize_label(label_np, config.CLASS_COLORS)

        # overlay
        alpha = 0.6
        overlay = (img_np.astype(np.float32) * (1 - alpha) + color_label.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Save (convert RGB->BGR for cv2)
        fname_base = f"sample_{i:02d}_idx_{idx}"
        rgb_path = outdir / (fname_base + "_rgb.png")
        label_path = outdir / (fname_base + "_label_color.png")
        overlay_path = outdir / (fname_base + "_overlay.png")

        cv2.imwrite(str(rgb_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(label_path), cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        saved.append((str(rgb_path), str(label_path), str(overlay_path)))

    return saved


def main():
    # Use globals defined above so user can edit file and re-run
    set_seed(SEED, deterministic=True)

    root = config.DATA_DIR
    saved = save_samples(root, MODE, SPLIT, CSG_MODE, SEED, NUM_SAMPLES, OUTDIR)

    # for s in saved:
        # print("Saved:")
        # print("  RGB:", s[0])
        # print("  Label:", s[1])
        # print("  Overlay:", s[2])


if __name__ == "__main__":
    main()
