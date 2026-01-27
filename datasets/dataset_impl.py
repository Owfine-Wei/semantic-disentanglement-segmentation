import os
import torch
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from helpers.classes_erased_samples_generator import classes_erased_samples_generator
from helpers.foreback_erased import foreback_erased

import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class Origin_Dataset(Dataset):
    """
    Robust Dataset class that aligns images and labels based on config suffixes.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.crop_size = config.CROP_SIZE
        # 你的 config 可能没有定义 mean/std，这里假设你有，或者用 ImageNet 默认值
        self.normalize = transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)

        # 1. 确定根路径 (使用 os.path.join 更安全)
        self.img_root = config.IMG_DIR[split]
        self.label_root = config.LABEL_DIR[split]

        if not os.path.exists(self.img_root) or not os.path.exists(self.label_root):
            raise FileNotFoundError(f"Directory not found: {self.img_root} or {self.label_root}")

        self.files = [] # 存储 (image_path, label_path) 元组

        # 2. 从 Config 获取后缀 (这是解耦的关键)
        # 如果 config 没有定义，给默认值 (兼容 Cityscapes)
        img_suffix = getattr(config, 'IMG_SUFFIX', '_leftImg8bit.png')
        label_suffix = getattr(config, 'LABEL_SUFFIX', '_gtFine_labelTrainIds.png')

        print(f"[{split}] Scanning images in {self.img_root}...")
        
        # 3. 遍历图片目录，动态寻找对应的标签
        for root, _, filenames in os.walk(self.img_root):
            for filename in filenames:
                if filename.endswith(img_suffix):
                    # 获取图片绝对路径
                    img_path = os.path.join(root, filename)
                    
                    # --- 核心逻辑：推导标签路径 ---
                    
                    # 1. 计算相对路径 (例如: 'frankfurt/file.png' 或 'file.png')
                    rel_path = os.path.relpath(root, self.img_root)
                    
                    # 2. 替换文件名后缀 (从 img_suffix -> label_suffix)
                    label_filename = filename.replace(img_suffix, label_suffix)
                    
                    # 3. 拼接标签完整路径 (保持相同的目录结构)
                    label_path = os.path.join(self.label_root, rel_path, label_filename)

                    # 4. 强校验：标签文件必须存在
                    if os.path.exists(label_path):
                        self.files.append((img_path, label_path))
                    else:
                        print(f"Warning: Label not found for {filename}, skipping.")

        print(f"Found {len(self.files)} paired samples for {split}.")
        
        if len(self.files) == 0:
            raise RuntimeError(f"No valid pairs found! Check your IMG_SUFFIX '{img_suffix}' or paths.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 从元组中解包路径，保证绝对一一对应
        img_path, label_path = self.files[idx]

        # 读取图片
        image_pil = Image.open(img_path).convert('RGB')
        label_pil = Image.open(label_path)

        image = np.array(image_pil)
        label = np.array(label_pil)

        # 转 Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        # 数据增强 (Train Split)
        if self.split == 'train':
            # 获取随机裁剪参数
            # 注意：RandomCrop 需要输入的形状是 (C, H, W) 或 (H, W)
            # image 是 (3, H, W)，label 是 (H, W)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
            
            image = TF.crop(image, i, j, h, w)
            # label 需要unsqueeze再crop吗？TF.crop支持 2D 或 3D tensor
            # 为了保险起见，保持你的写法，或者直接传 label (TF.crop 支持 (H,W) tensor)
            label = TF.crop(label, i, j, h, w) 
            
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

        image = self.normalize(image)
        label = label.long()

        return image, label, None, None, None

class FOREBACK_Dataset(Origin_Dataset):
    """
    Robust Dataset for foreground-only or background-only images.
    Ensures strict alignment between Image, Label, and Mask.
    """

    def __init__(self, config, split, mode):
        super().__init__(config, split)
        self.mode = mode  # 'foreground' or 'background'

    def __getitem__(self, idx):
        # 解包三元组，保证绝对一一对应
        img_path, label_path = self.files[idx]

        # 读取文件
        image_pil = Image.open(img_path).convert('RGB')
        label_pil = Image.open(label_path)
        
        image = np.array(image_pil)
        label = np.array(label_pil)

        # 转 Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        # 数据增强 (同步变换)
        if self.split == 'train':
            # 获取随机裁剪参数
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
            
            image = TF.crop(image, i, j, h, w)
            # Label 和 Mask 可能需要 unsqueeze 再 squeeze 才能正确处理 (如果是2D tensor)
            # 或者直接传入，TF.crop 现在的版本支持 2D Tensor
            label = TF.crop(label, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

        # 前景背景擦除
        image, label, mask = foreback_erased(self.config, image, label, self.mode)

        image = self.normalize(image)
        label = label.long()
        mask = mask.long()

        # 返回 image, label, mask, 其他, 其他
        return image, label, mask, None, None
        

class CSG_Dataset(Origin_Dataset):
    """
    Robust Dataset for class-erased (CSG) samples.
    Aligns original images and labels, then generates CSG samples on-the-fly.
    """

    def __init__(self, config, split, csg_mode):
        super().__init__(config, split)
        self.csg_mode = csg_mode

    def __getitem__(self, idx):
        # 解包路径
        img_path, label_path = self.files[idx]

        # 读取原始数据
        origin_image_pil = Image.open(img_path).convert('RGB')
        origin_label_pil = Image.open(label_path)

        origin_image_np = np.array(origin_image_pil)
        origin_label_np = np.array(origin_label_pil)

        # To tensors
        origin_image = torch.from_numpy(origin_image_np).permute(2, 0, 1).float() / 255.0
        origin_label = torch.from_numpy(origin_label_np).long()

        # 4. 同步裁剪 (在生成 CSG 样本之前进行裁剪，提高效率)
        i, j, h, w = transforms.RandomCrop.get_params(origin_image, output_size=self.crop_size)
        origin_image = TF.crop(origin_image, i, j, h, w)
        # 注意：这里直接传 label 即可，TF.crop 支持 (H, W)
        origin_label = TF.crop(origin_label, i, j, h, w)

        # 5. 生成类擦除样本 (基于裁剪后的区域生成)
        # 注意：这个函数必须支持处理 Tensor
        image, label, mask = classes_erased_samples_generator(
            self.config, origin_image, origin_label, self.csg_mode
        )

        # 6. 同步水平翻转 (覆盖所有 5 个返回项)
        if self.split == 'train' and random.random() > 0.5:
            origin_image = TF.hflip(origin_image)
            origin_label = TF.hflip(origin_label)
            image = TF.hflip(image)
            label = TF.hflip(label)
            mask = TF.hflip(mask)

        # 归一化和格式转换
        origin_image_normalized = self.normalize(origin_image)
        image_normalized = self.normalize(image)
        mask = mask.long()

        # 返回项顺序: image, label, mask, origin_image, origin_label
        return image_normalized, label, mask, origin_image_normalized, origin_label


class NDA_Dataset(Dataset):
    """Concatenate Origin, Foreground and Background datasets in one view."""

    def __init__(self, config, split):
        self.origin_dataset = Origin_Dataset(config, split)
        self.foreground_dataset = FOREBACK_Dataset(config, 'foreground', split)
        self.background_dataset = FOREBACK_Dataset(config, 'background', split)
        self.len_origin = len(self.origin_dataset)
        self.len_fore = len(self.foreground_dataset)
        self.len_back = len(self.background_dataset)

    def __len__(self):
        return self.len_origin + self.len_fore + self.len_back

    def __getitem__(self, idx):
        if idx < self.len_origin:
            return self.origin_dataset[idx]
        elif idx < (self.len_origin + self.len_fore):
            return self.foreground_dataset[idx - self.len_origin]
        else:
            return self.background_dataset[idx - self.len_origin - self.len_fore]

class SDS_Dataset(Dataset):
    """Factory wrapper returning one of the dataset views by `mode`."""

    def __init__(self, config, mode, split, csg_mode=None):
        self.mode = mode
        if mode == 'origin':
            self.dataset = Origin_Dataset(config, split)
        elif mode == 'foreground':
            self.dataset = FOREBACK_Dataset(config, 'foreground', split)
        elif mode == 'background':
            self.dataset = FOREBACK_Dataset(config, 'background', split)
        elif mode == 'csg':
            self.dataset = CSG_Dataset(config, csg_mode, split )
        elif mode == 'nda':
            self.dataset = NDA_Dataset(config, split)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    """
    Custom collate that handles None fields in dataset items.

    The dataset items are tuples like (img, label, mask, origin_img, origin_label)
    where some entries may be None. This function applies default_collate to
    each column if the first element is not None, otherwise returns None.
    """

    transposed = list(zip(*batch))
    return [
        default_collate(samples) if samples[0] is not None else None
        for samples in transposed
    ]

def load_data(config, mode, split, csg_mode=None, batch_size=1, num_workers=4, distributed=False):
    """
    Create DataLoader for selected dataset `mode`.

    Supports distributed sampling when `distributed=True`.
    """

    dataset = SDS_Dataset(config, mode, split, csg_mode)
    sampler = DistributedSampler(dataset) if distributed else None
    shuffle = True if not distributed else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True if split == 'train' else False,
    )
    return dataloader

def check_data_shapes(config, mode, split, csg_mode=None):
    """Small helper to instantiate loader and print one batch for sanity check."""

    dataloader = load_data(config, mode, split, csg_mode)
    for data in dataloader:
        print(f"Sample Batch Received. Mode: {mode}")
        break

if __name__ == "__main__" :
    pass