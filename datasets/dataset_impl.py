import os
import torch
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate
from torch.utils.data.distributed import DistributedSampler
<<<<<<< HEAD

from helpers.classes_erased_samples_generator import classes_erased_samples_generator
from helpers.foreback_erased import foreback_erased
=======
from torchvision.transforms import InterpolationMode

from helpers.classes_erased_samples_generator import classes_erased_samples_generator
from helpers.one_cate_remain import one_cate_remain
>>>>>>> master

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
        img_suffix = config.IMG_SUFFIX
        label_suffix = config.LABEL_SUFFIX

        # print(f"[{split}] Scanning images in {self.img_root}...")
        
        # 3. 遍历图片目录，动态寻找对应的标签
        for root, _, filenames in os.walk(self.img_root):
            for filename in filenames:

                # 找到当前 filename 匹配的具体 suffix
                matched_suffix = (
                    next((s for s in img_suffix if filename.endswith(s)), None)
                    if isinstance(img_suffix, tuple)
                    else img_suffix if filename.endswith(img_suffix)
                    else None
                )

                if matched_suffix is None:
                    continue

                # 图片路径
                img_path = os.path.join(root, filename)

                # 相对路径
                rel_path = os.path.relpath(root, self.img_root)

                # 推导 label 文件名
                label_filename = filename.replace(matched_suffix, label_suffix)

                # 拼 label 路径
                label_path = os.path.join(self.label_root, rel_path, label_filename)

                # 校验
                if os.path.exists(label_path):
                    self.files.append((img_path, label_path))
                else:
                    print(f"Warning: Label not found for {filename}, skipping.")

        # print(f"Found {len(self.files)} paired samples for {split}.")
        
        if len(self.files) == 0:
            raise RuntimeError(f"No valid pairs found! Check your IMG_SUFFIX '{img_suffix}' or paths.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label_path = self.files[idx]

        # 读取图片（先不转 numpy，PIL 的 .size 效率很高）
        image_pil = Image.open(img_path).convert('RGB')
        
        if image_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]): 
            # 方案 A: 递归取下一张（简单暴力）
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)
            
        label_pil = Image.open(label_path)
        if label_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]):
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)

        # 校验通过后再进行转换
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

<<<<<<< HEAD
=======
        elif self.split == 'val' and self.config.SCALE_SIZE is not None:

            image = TF.resize(image, self.config.SCALE_SIZE, 
                              interpolation=TF.InterpolationMode.BILINEAR)
            
            label = TF.resize(label.unsqueeze(0), self.config.SCALE_SIZE, 
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            

>>>>>>> master
        image = self.normalize(image)
        label = label.long()

        return image, label, None, None, None

<<<<<<< HEAD
class FOREBACK_Dataset(Origin_Dataset):
=======
class OneCateRemain_Dataset(Origin_Dataset): # only for validation
>>>>>>> master
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

        # 读取图片（先不转 numpy，PIL 的 .size 效率很高）
        image_pil = Image.open(img_path).convert('RGB')
        
        if image_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]): 
            # 方案 A: 递归取下一张（简单暴力）
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)
            
        label_pil = Image.open(label_path)
        if label_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]):
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)

        # 校验通过后再进行转换
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
<<<<<<< HEAD
        image, label, mask = foreback_erased(self.config, image, label, self.mode)
=======
        image, label, mask = one_cate_remain(self.config, image, label, self.mode)
>>>>>>> master

        image = self.normalize(image)
        label = label.long()
        mask = mask.float()

        # 返回 image, label, mask, 其他, 其他
        return image, label, mask, None, None
        

class CSG_Dataset(Origin_Dataset):
    """
    Robust Dataset for class-erased (CSG) samples.
<<<<<<< HEAD
    Aligns original images and labels, then generates CSG samples on-the-fly.
    """

    def __init__(self, config, split, csg_mode):
        super().__init__(config, split)
        self.csg_mode = csg_mode
=======
    Refactored to handle Resize and Pad in PIL mode to avoid Tensor dimension errors.
    """

    def __init__(self, config, split):
        super().__init__(config, split)
        self.ratio_range = (0.5, 2.0)
        # 预先准备好填充值 (PIL 需要 tuple, Tensor 需要 scalar)
        self.mean_padding_pil = (123.68, 116.28, 103.53)
        self.ignore_index = 255
>>>>>>> master

    def __getitem__(self, idx):
        # 解包路径
        img_path, label_path = self.files[idx]

<<<<<<< HEAD
        # 读取图片（先不转 numpy，PIL 的 .size 效率很高）
        origin_image_pil = Image.open(img_path).convert('RGB')
        
        if origin_image_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]): 
            # 方案 A: 递归取下一张（简单暴力）
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)
            
        origin_label_pil = Image.open(label_path)
        if origin_label_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]):
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)

        # 校验通过后再进行转换
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
=======
        # 1. 读取为 PIL
        origin_image_pil = Image.open(img_path).convert('RGB')
        origin_label_pil = Image.open(label_path)

        # 校验尺寸，不对则递归读取下一个
        if origin_image_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]): 
            return self.__getitem__((idx + 1) % len(self.files))
        if origin_label_pil.size != (self.config.IMG_SIZE[-1], self.config.IMG_SIZE[-2]):
            return self.__getitem__((idx + 1) % len(self.files))

        # =========================================================
        # 2. 随机缩放 (在 PIL 阶段进行，避免 Tensor 维度报错)
        # =========================================================
        scale = random.uniform(self.ratio_range[0], self.ratio_range[1])
        
        # PIL.size 返回 (W, H)
        w_cur, h_cur = origin_image_pil.size
        new_h, new_w = int(h_cur * scale), int(w_cur * scale)
        
        # PIL Resize
        origin_image_pil = origin_image_pil.resize((new_w, new_h), Image.BILINEAR)
        # 标签必须用 NEAREST
        origin_label_pil = origin_label_pil.resize((new_w, new_h), Image.NEAREST)

        # =========================================================
        # 3. 填充 (Pad) - 也在 PIL 阶段做，支持 Tuple 填充
        # =========================================================
        crop_h, crop_w = self.crop_size
        pad_h = max(0, crop_h - new_h)
        pad_w = max(0, crop_w - new_w)
        
        if pad_h > 0 or pad_w > 0:
            # TF.pad 对 PIL Image 支持 tuple fill
            origin_image_pil = TF.pad(origin_image_pil, (0, 0, pad_w, pad_h), fill=self.mean_padding_pil)
            origin_label_pil = TF.pad(origin_label_pil, (0, 0, pad_w, pad_h), fill=self.ignore_index)

        # =========================================================
        # 4. 转 Tensor (统一转换点)
        # =========================================================
        # Image: (H, W, 3) -> (3, H, W), float 0-1
        origin_image = TF.to_tensor(origin_image_pil) 
        
        # Label: (H, W) -> int64
        # 这一步非常关键：np.array之后直接转 tensor，保持 2D 形状 (H, W)
        origin_label = torch.from_numpy(np.array(origin_label_pil)).long()
        
        # =========================================================
        # 5. 随机裁剪 (Random Crop)
        # =========================================================
        # TF.crop 同时支持 (C, H, W) 和 (H, W) 的 tensor
        i, j, h, w = transforms.RandomCrop.get_params(origin_image, output_size=self.crop_size)
        
        origin_image = TF.crop(origin_image, i, j, h, w)
        origin_label = TF.crop(origin_label, i, j, h, w)

        # =========================================================
        # 6. 生成类擦除样本 (CSG)
        # =========================================================
        # 注意：CSG 生成器通常期望 label 是 (H, W) 或 (1, H, W)。
        # 如果生成器内部需要 3D label，请在传入时 unsqueeze，但在返回时一定要保持 label 是 2D
        image, label, mask = classes_erased_samples_generator(self.config, origin_image, origin_label)

        # 确保 label 是 2D (防止 generator 返回了 3D)
        if label.dim() == 3 and label.shape[0] == 1:
            label = label.squeeze(0)

        # =========================================================
        # 7. 同步水平翻转
        # =========================================================
        if random.random() > 0.5:
>>>>>>> master
            origin_image = TF.hflip(origin_image)
            origin_label = TF.hflip(origin_label)
            image = TF.hflip(image)
            label = TF.hflip(label)
            mask = TF.hflip(mask)

<<<<<<< HEAD
        # 归一化和格式转换
        origin_image_normalized = self.normalize(origin_image)
        image_normalized = self.normalize(image)
        mask = mask.float()

        # 返回项顺序: image, label, mask, origin_image, origin_label
        return image_normalized, label, mask, origin_image_normalized, origin_label


class NDA_Dataset(Dataset):
    """Concatenate Origin, Foreground and Background datasets in one view."""

    def __init__(self, config, split):
        self.origin_dataset = Origin_Dataset(config, split)
        self.foreground_dataset = FOREBACK_Dataset(config, split, 'foreground')
        self.background_dataset = FOREBACK_Dataset(config, split, 'background')
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
            self.dataset = FOREBACK_Dataset(config, split, 'foreground')
        elif mode == 'background':
            self.dataset = FOREBACK_Dataset(config, split, 'background')
        elif mode == 'csg':
            self.dataset = CSG_Dataset(config, split, csg_mode )
        elif mode == 'nda':
            self.dataset = NDA_Dataset(config, split)
=======
        # =========================================================
        # 8. 归一化和格式收尾
        # =========================================================
        # 假设 self.normalize 只是 Normalize((mean), (std))，不包含 ToTensor
        origin_image_normalized = self.normalize(origin_image)
        image_normalized = self.normalize(image)
        
        # 处理 Mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()
        
        # 确保 Mask 也是 2D (或者 3D 单通道，取决于你的 Loss 函数需求)
        # 通常 mask 用于计算 loss，保持 (H, W) 最安全，如果需要 (1, H, W) 可以在 loss 前加
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # === 最终检查 ===
        # origin_label 和 label 必须是 (H, W)
        # origin_image 和 image 必须是 (3, H, W)
        
        return image_normalized, label, mask, origin_image_normalized, origin_label


class SDS_Dataset(Dataset):
    """Factory wrapper returning one of the dataset views by `mode`."""

    def __init__(self, config, mode, split):
        self.mode = mode
        if mode == 'origin':
            self.dataset = Origin_Dataset(config, split)
        elif mode == 'flat':
            self.dataset = OneCateRemain_Dataset(config, split, 'flat')
        elif mode == 'construction':
            self.dataset = OneCateRemain_Dataset(config, split, 'construction')
        elif mode == 'object':
            self.dataset = OneCateRemain_Dataset(config, split, 'object')
        elif mode == 'nature':
            self.dataset = OneCateRemain_Dataset(config, split, 'nature')
        elif mode == 'sky':
            self.dataset = OneCateRemain_Dataset(config, split, 'sky')
        elif mode == 'human':
            self.dataset = OneCateRemain_Dataset(config, split, 'human')
        elif mode == 'vehicle':
            self.dataset = OneCateRemain_Dataset(config, split, 'vehicle')
        elif mode == 'csg':
            self.dataset = CSG_Dataset(config, split)
>>>>>>> master
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    # 彻底过滤掉 batch 里的 None 样本（如果 Dataset 出了错直接返回 None 的话）
    batch = [s for s in batch if s is not None]
    
    if not batch:
        return [None] * 5  # 返回 5 个占位符解构

    # 针对你设计的 5 个字段进行处理
    transposed = list(zip(*batch))
    output = []
    
    for samples in transposed:
        # 2.x 的 default_collate 依然不能处理包含 None 的 samples 列表
        # 所以这里的 all 检查在 2.x 中依然是必须的
        if all(x is not None for x in samples):
            output.append(default_collate(list(samples)))
        else:
            output.append(None) # 保持你的占位符设计
            
    return output

<<<<<<< HEAD
def load_data(config, mode, split, csg_mode=None, batch_size=1, num_workers=4, distributed=False):
=======
def load_data(config, mode, split, batch_size=1, num_workers=4, distributed=False):
>>>>>>> master
    """
    Create DataLoader for selected dataset `mode`.

    Supports distributed sampling when `distributed=True`.
    """

<<<<<<< HEAD
    dataset = SDS_Dataset(config, mode, split, csg_mode)
=======
    dataset = SDS_Dataset(config, mode, split)
>>>>>>> master
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

<<<<<<< HEAD
def check_data_shapes(config, mode, split, csg_mode=None):
    """Small helper to instantiate loader and print one batch for sanity check."""

    dataloader = load_data(config, mode, split, csg_mode)
=======
def check_data_shapes(config, mode, split):
    """Small helper to instantiate loader and print one batch for sanity check."""

    dataloader = load_data(config, mode, split)
>>>>>>> master
    for data in dataloader:
        print(f"Sample Batch Received. Mode: {mode}")
        break

if __name__ == "__main__" :
    pass