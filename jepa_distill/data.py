"""Dataset for Stage 2: one shared 224-px view per image (teacher & student
see the SAME pixels - see docs/STAGE2_DISTILL_DESIGN.md)."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageListDataset(Dataset):
    """Reads an image list (names or relative paths, one per line)."""

    def __init__(self, list_file, images_dir, train=True, size=224, limit=0):
        self.images_dir = Path(images_dir)
        names = [Path(l.strip()).name for l in open(list_file) if l.strip()]
        if limit:
            names = names[:limit]
        self.names = names
        self.tf = build_transform(train, size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.images_dir / self.names[i]).convert("RGB")
        return self.tf(img), i
