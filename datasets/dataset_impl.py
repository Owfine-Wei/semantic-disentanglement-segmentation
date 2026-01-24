import os
import cv2
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from helpers.classes_erased_samples_generator import classes_erased_samples_generator

class Origin_Dataset(Dataset):
    """
    Dataset for original Cityscapes images and trainIds labels.

    Returns tuples of (image_tensor, label_tensor, None, None, None) so it
    can be used interchangeably with other dataset wrappers in this file.
    """

    def __init__(self, config, split):
        # Basic Configs
        self.config = config   # config file
        self.split = split # train / val / test

        # Img / Label Directory
        self.img_dir = config.DIRS['origin']['imgs'] + self.split
        self.label_dir = config.DIRS['origin']['labels'] + self.split

        # Store crop size and normalization for transforms
        self.crop_size = config.CROP_SIZE
        self.normalize = transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)

        self.images = []
        self.labels = []

        if not os.path.exists(self.img_dir) or not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir} or {self.label_dir}")

        for root_path, _, files in os.walk(self.img_dir):
            for file in files:
                self.images.append(os.path.join(root_path, file))
        self.images.sort()

        for root_path, _, files in os.walk(self.label_dir):
            for file in files:
                self.labels.append(os.path.join(root_path, file))
        self.labels.sort()

        if len(self.images) == 0:
            print(f"Warning: No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # to tensor CxHxW, normalized to [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        if self.split == 'train':
            # synchronized random crop and horizontal flip for image+label
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label.unsqueeze(0), i, j, h, w).squeeze(0)
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

        image = self.normalize(image)
        return image, label, None, None, None
        
class FOREBACK_Dataset(Dataset):
    """
    Dataset for foreground-only or background-only images.

    The `mode` argument selects which split ('foreground'|'background') to
    load. Returns same item format as other datasets.
    """

    def __init__(self, config, mode, split):
        self.config = config
        self.split = split
        self.mode = mode   # foreground / background

        self.img_dir = config.DIRS[mode]['imgs'] + self.split
        self.label_dir = config.DIRS[mode]['labels'] + self.split

        self.crop_size = config.CROP_SIZE
        self.normalize = transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)

        self.images = []
        self.labels = []

        if not os.path.exists(self.img_dir) or not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir} or {self.label_dir}")

        for root_path, _, files in os.walk(self.img_dir):
            for file in files:
                self.images.append(os.path.join(root_path, file))
        self.images.sort()

        for root_path, _, files in os.walk(self.label_dir):
            for file in files:
                self.labels.append(os.path.join(root_path, file))
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        if self.split == 'train':
            # sync random crop + flip
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label.unsqueeze(0), i, j, h, w).squeeze(0)
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

        image = self.normalize(image)
        return image, label, None, None, None
        

class CSG_Dataset(Dataset):
    """
    Dataset generating class-erased (CSG) samples, optionally with origin.

    Items are (image, label, mask, origin_image, origin_label).

    """

    def __init__(self, config, csg_mode, split ):
        self.config = config
        self.split = split
        self.csg_mode = csg_mode

        self.img_dir = config.DIRS['csg']['imgs'] + self.split
        self.label_dir = config.DIRS['csg']['labels'] + self.split

        self.crop_size = config.CROP_SIZE
        self.normalize = transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)

        self.images = []
        self.labels = []

        for root_path, _, files in os.walk(self.img_dir):
            for file in files:
                self.images.append(os.path.join(root_path, file))
        self.images.sort()

        for root_path, _, files in os.walk(self.label_dir):
            for file in files:
                self.labels.append(os.path.join(root_path, file))
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        origin_image_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
        origin_image_np = cv2.cvtColor(origin_image_np, cv2.COLOR_BGR2RGB)
        origin_label_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # to tensors
        origin_image = torch.from_numpy(origin_image_np).permute(2, 0, 1).float() / 255.0
        origin_label = torch.from_numpy(origin_label_np).long()

        # synchronized crop for origin and derived samples
        i, j, h, w = transforms.RandomCrop.get_params(origin_image, output_size=self.crop_size)
        origin_image = TF.crop(origin_image, i, j, h, w)
        origin_label = TF.crop(origin_label.unsqueeze(0), i, j, h, w).squeeze(0)

        # generate class-erased sample (image,label,mask)
        image, label, mask = classes_erased_samples_generator(self.config, origin_image, origin_label, self.csg_mode)

        # synchronized horizontal flip
        if random.random() > 0.5:
            origin_image = TF.hflip(origin_image)
            origin_label = TF.hflip(origin_label)
            image = TF.hflip(image)
            label = TF.hflip(label)
            mask = TF.hflip(mask)

        origin_image = self.normalize(origin_image)
        image = self.normalize(image)
        mask = mask.long()

        return image, label, mask, origin_image, origin_label


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