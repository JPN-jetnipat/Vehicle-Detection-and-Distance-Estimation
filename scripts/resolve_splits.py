"""Expand bare-filename split lists into absolute image paths ultralytics can load.

dataset/yolo/splits/{train_100,val,test}.txt (copied from the handoff repo)
list bare image filenames, one per line, with no directory component.
Ultralytics' txt-list loader only rewrites lines that start with "./", and
even then does a literal os.sep-based "/images/" substring match to find the
matching label file - bare filenames without that structure resolve to
nothing (see the plan discussion / smoke-test failure this fixes).

This writes a sibling "<name>_resolved.txt" next to each source list, with
each bare filename expanded to its absolute path under
dataset/yolo/images/train/. Resolved files are machine-specific (absolute
paths) and gitignored - regenerate them via main() on whichever machine runs
training, before every training run (cheap: a few tens of thousands of
lines).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "dataset" / "yolo" / "splits"
TRAIN_IMAGES_DIR = ROOT / "dataset" / "yolo" / "images" / "train"

SOURCE_LISTS = ["train_100", "train_50", "train_25", "train_10", "val"]


def resolve_split(source_name: str) -> Path:
    src = SPLITS_DIR / f"{source_name}.txt"
    dst = SPLITS_DIR / f"{source_name}_resolved.txt"
    names = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolved = [str(TRAIN_IMAGES_DIR / name) for name in names]
    dst.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return dst


def main() -> None:
    for name in SOURCE_LISTS:
        if not (SPLITS_DIR / f"{name}.txt").exists():
            continue
        dst = resolve_split(name)
        print(f"{name}.txt -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
