"""Dataset for Stage 2: one shared 224-px view per image (teacher & student
see the SAME pixels - see docs/STAGE2_DISTILL_DESIGN.md)."""
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


class ImageListDataset(Dataset):
    """Reads an image list (names or relative paths, one per line).
    Filters out names with no file on disk - loudly, since a large missing
    fraction means the download itself is incomplete (FLAG 4,
    docs/REPO_SURVEY.md), not something to silently train around."""

    def __init__(self, list_file, images_dir, train=True, size=224, limit=0):
        self.images_dir = _resolve(images_dir)
        raw_names = [Path(l.strip()).name for l in open(_resolve(list_file)) if l.strip()]
        names = [n for n in raw_names if (self.images_dir / n).exists()]
        missing = len(raw_names) - len(names)
        if missing:
            frac = missing / len(raw_names)
            print(f"WARNING: {missing}/{len(raw_names)} images in {list_file} "
                  f"({frac:.2%}) not found under {self.images_dir} - excluded from this run.")
            if frac > 0.01:
                raise SystemExit(
                    f"FLAG: {frac:.1%} of listed images are missing - looks like an "
                    f"incomplete download, not a few stray files. Investigate (see "
                    f"docs/REPO_SURVEY.md FLAG 4) before training on a silently-shrunk dataset.")
        if limit:
            names = names[:limit]
        self.names = names
        self.tf = build_transform(train, size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.images_dir / self.names[i]).convert("RGB")
        return self.tf(img), i
