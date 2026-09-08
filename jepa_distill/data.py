"""Dataset for Stage 2: one shared 224-px view per image (teacher & student
see the SAME pixels - see docs/STAGE2_DISTILL_DESIGN.md)."""
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REPO = Path(__file__).resolve().parents[1]


def _resolve(p):
    """Resolve relative config paths against the repo root, not whatever
    directory the terminal happens to be in - training is launched via
    tmux/nohup where cwd is easy to lose track of."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p


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


def _build_name_index(images_dir):
    """BDD100K on this mirror is sharded into subfolders (observed:
    trainA/trainB/testA/testB nested under both images/100k/train/ and
    images/100k/test/ - a quirk of this Kaggle mirror's packaging, not a
    layout we can assume is flat). Walk once, map every filename found
    anywhere under images_dir to its full path."""
    index = {}
    for root, _dirs, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                index.setdefault(fn, os.path.join(root, fn))  # first copy found wins
    return index


class ImageListDataset(Dataset):
    """Reads an image list (names or relative paths, one per line).
    Filters out names with no file anywhere under images_dir - loudly, since
    a large missing fraction means the download itself is incomplete (FLAG 4,
    docs/REPO_SURVEY.md), not something to silently train around."""

    def __init__(self, list_file, images_dir, train=True, size=224, limit=0):
        images_dir = _resolve(images_dir)
        index = _build_name_index(images_dir)
        raw_names = [Path(l.strip()).name for l in open(_resolve(list_file)) if l.strip()]
        self.paths = [index[n] for n in raw_names if n in index]
        missing = len(raw_names) - len(self.paths)
        if missing:
            frac = missing / len(raw_names)
            print(f"WARNING: {missing}/{len(raw_names)} images in {list_file} "
                  f"({frac:.2%}) not found anywhere under {images_dir} - excluded from this run.")
            if frac > 0.01:
                raise SystemExit(
                    f"FLAG: {frac:.1%} of listed images are missing - looks like an "
                    f"incomplete download, not a few stray files. Investigate (see "
                    f"docs/REPO_SURVEY.md FLAG 4) before training on a silently-shrunk dataset.")
        if limit:
            self.paths = self.paths[:limit]
        self.names = [os.path.basename(p) for p in self.paths]  # kept for the day/night probe
        self.tf = build_transform(train, size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), i