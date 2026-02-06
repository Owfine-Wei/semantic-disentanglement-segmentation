import os
import numpy as np
from PIL import Image
import torch
from configs import get_config
from datasets.dataset_impl import load_data
<<<<<<< HEAD

# Configurable params
dataset_name = 'cityscapes'
modes = ['origin', 'foreground', 'background', 'csg']  # 'nda' ignored
csg_modes = ['foreground', 'background', 'both']
split = 'train'
samples_per_variant = 5
=======
from tqdm import *

# Configurable params
dataset_name = 'cityscapes'
modes = ['origin', 'csg', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'] 
split = 'train'
samples_per_variant = 25
>>>>>>> master
out_root = 'visualizations'  # will create visualizations/<dataset>/...

# ImageNet mean/std for un-normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def tensor_to_pil(img_t):
    """
    Accepts a torch.Tensor in shape (C,H,W) or (H,W,C) or (B,C,H,W).
    Assumes RGB normalized by ImageNet mean/std if 3 channels.
    Returns PIL.Image.
    """
    if img_t is None:
        return None
    t = img_t.detach().cpu() if isinstance(img_t, torch.Tensor) else torch.tensor(img_t)
    # if batch dim
    if t.dim() == 4:
        t = t[0]
    # if H,W,C -> convert
    if t.dim() == 3 and t.shape[2] in (1,3) and t.shape[0] not in (1,3):
        t = t.permute(2,0,1)
    # now expect (C,H,W) or (H,W)
    if t.dim() == 3 and t.shape[0] == 3:
        t = t * IMAGENET_STD + IMAGENET_MEAN
        t = t.clamp(0,1)
        arr = (t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        return Image.fromarray(arr)
    if t.dim() == 3 and t.shape[0] == 1:
        arr = (t[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    if t.dim() == 2:
        arr = (t.numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    # fallback
    arr = (t.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def save_array_as_png(arr, path):
    if arr is None:
        return
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    else:
        a = np.array(arr)
        
    # if float mask in [0,1], scale
    if a.dtype == np.float32 or a.dtype == np.float64:
        a = (a * 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    
    Image.fromarray(a).save(path)

def process_loader_and_save(loader, out_dir, max_samples):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for batch in loader:
        # loader may yield different tuples; normalize to 5-length tuple
        if isinstance(batch, (list,tuple)):
            parts = list(batch) + [None]*5
            images, labels, masks, origin_images, origin_labels = parts[:5]
        else:
            # unexpected, skip
            continue

        # handle batched tensors
        batch_size = 1
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_size = images.shape[0]

        for i in range(batch_size):
            if saved >= max_samples:
                break
            # --- 修复后的 idx 函数 ---
            def idx(x):
                if x is None:
                    return None
                # 只要是 Tensor 且第一维大小等于 batch_size，就尝试解包
                if isinstance(x, torch.Tensor) and x.shape[0] == batch_size:
                    return x[i]
                return x
            # -----------------------
            img = idx(images)
            lbl = idx(labels)
            msk = idx(masks)
            oimg = idx(origin_images)
            olbl = idx(origin_labels)

            name = f'sample_{saved:02d}'
            if img is not None:
                pil = tensor_to_pil(img)
                pil.save(os.path.join(out_dir, name + '_image.png'))
            if oimg is not None:
                pil_o = tensor_to_pil(oimg)
                pil_o.save(os.path.join(out_dir, name + '_origin_image.png'))
            if lbl is not None:
                save_array_as_png(lbl, os.path.join(out_dir, name + '_label.png'))
            if olbl is not None:
                save_array_as_png(olbl, os.path.join(out_dir, name + '_origin_label.png'))
            if msk is not None:
                save_array_as_png(msk, os.path.join(out_dir, name + '_mask.png'))

            saved += 1
        if saved >= max_samples:
            break

def main():
    config = get_config(dataset_name)
<<<<<<< HEAD
    for mode in modes:
        if mode == 'csg':
            for csg_mode in csg_modes:
                loader = load_data(config, mode, split, csg_mode)
                out_dir = os.path.join(out_root, dataset_name, mode, csg_mode)
                process_loader_and_save(loader, out_dir, samples_per_variant)
        else:
            # for non-csg, csg_mode arg can be ignored or set to 'both' in loader
            loader = load_data(config, mode, split, 'both')
            out_dir = os.path.join(out_root, dataset_name, mode)
            process_loader_and_save(loader, out_dir, samples_per_variant)
=======
    for mode in tqdm(modes):
        loader = load_data(config, mode, split)
        out_dir = os.path.join(out_root, dataset_name, mode)
        process_loader_and_save(loader, out_dir, samples_per_variant)
>>>>>>> master

if __name__ == '__main__':
    main()