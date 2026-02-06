import os
import numpy as np
from PIL import Image
import torch
from configs import get_config
from datasets.dataset_impl import load_data
from tqdm import *

# Configurable params
dataset_name = 'cityscapes'
modes = ['origin', 'csg', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'] 
split = 'train'
samples_per_variant = 25
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

def process_loader_and_save(loader, out_dir, max_samples, mode):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    
    # 用于统计类别数
    total_class_counts = 0
    processed_images_for_stats = 0
    
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            parts = list(batch) + [None]*5
            images, labels, masks, origin_images, origin_labels = parts[:5]
        else:
            continue

        batch_size = 1
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_size = images.shape[0]

        for i in range(batch_size):
            if saved >= max_samples:
                break
                
            def idx(x):
                if x is None: return None
                if isinstance(x, torch.Tensor) and x.shape[0] == batch_size:
                    return x[i]
                return x

            img = idx(images)
            lbl = idx(labels)
            # ... (获取其他 idx 变量) ...

            # --- 新增：统计类别的逻辑 ---
            if mode == 'origin' and lbl is not None:
                # 获取唯一类别 ID
                unique_classes = torch.unique(lbl)
                # 过滤掉 Cityscapes 常用的忽略标签 255 (如果存在)
                unique_classes = unique_classes[unique_classes != 255]
                
                num_classes = len(unique_classes)
                total_class_counts += num_classes
                processed_images_for_stats += 1
            # --------------------------

            # 保存图像逻辑 (保持不变)
            name = f'sample_{saved:02d}'
            if img is not None:
                pil = tensor_to_pil(img)
                pil.save(os.path.join(out_dir, name + '_image.png'))
            # ... (此处省略原有的其他 save 逻辑) ...

            saved += 1
        
        if saved >= max_samples:
            break

    # 计算均值并返回
    avg_classes = 0
    if mode == 'origin' and processed_images_for_stats > 0:
        avg_classes = total_class_counts / processed_images_for_stats
    
    return avg_classes

def main():
    config = get_config(dataset_name)
    origin_avg = 0
    
    for mode in tqdm(modes):
        loader = load_data(config, mode, split)
        out_dir = os.path.join(out_root, dataset_name, mode)
        
        # 传入 mode 并接收返回值
        avg = process_loader_and_save(loader, out_dir, samples_per_variant, mode)
        
        if mode == 'origin':
            origin_avg = avg

    print("-" * 30)
    print(f"Dataset: {dataset_name} | Split: {split}")
    print(f"Average unique classes per image (origin): {origin_avg:.2f}")
    print("-" * 30)

if __name__ == '__main__':
    main()