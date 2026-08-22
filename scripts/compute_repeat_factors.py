"""Repeat-factor sampling math for class-imbalance oversampling (RFS / IRFS).

RFS: Gupta et al. 2019 (LVIS), image-level repeat factor.
  f_c  = fraction of training images containing >=1 instance of class c
  r_c  = max(1, sqrt(t / f_c))
  r_i  = max over classes c present in image i of r_c

IRFS: Yaman et al. 2023 (arXiv:2305.08069), instance-aware repeat factor,
geometric-mean variant (Eq. 3 in the paper) - the one the paper reports as
its main result:
  f_(i,c) = same image-level fraction as RFS's f_c
  f_(b,c) = fraction of all bounding boxes in the training set belonging to c
  r_c     = max(1, sqrt(t / sqrt(f_(i,c) * f_(b,c))))
  r_i     computed the same way as RFS (max over classes present in the image)

Pure functions only - no filesystem I/O beyond load_class_counts(), so the
math can be reused/tested independently of the repo's file layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClassCounts:
    """Per-image class presence/instance counts for one training split."""

    image_names: list[str]
    # image_name -> {class_id: instance_count}; only classes present in that image
    per_image_classes: dict[str, dict[int, int]]
    class_names: dict[int, str]


def load_class_counts(labels_dir: Path, split_list: Path, class_names: dict[int, str]) -> ClassCounts:
    """Read-only: tallies class counts per image from existing label files.

    Never writes to labels_dir. Images with a missing or empty label file are
    treated as background images (no instances of any class).
    """
    image_names = [line.strip() for line in split_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_image_classes: dict[str, dict[int, int]] = {}
    for image_name in image_names:
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        counts: dict[int, int] = {}
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counts[class_id] = counts.get(class_id, 0) + 1
        per_image_classes[image_name] = counts
    return ClassCounts(image_names=image_names, per_image_classes=per_image_classes, class_names=class_names)


def _image_frequencies(counts: ClassCounts) -> dict[int, float]:
    """f_c: fraction of images containing >=1 instance of class c."""
    n_images = len(counts.image_names)
    image_hits = {c: 0 for c in counts.class_names}
    for classes in counts.per_image_classes.values():
        for c in classes:
            image_hits[c] += 1
    return {c: image_hits[c] / n_images for c in counts.class_names}


def _instance_frequencies(counts: ClassCounts) -> dict[int, float]:
    """f_(b,c): fraction of all bounding boxes belonging to class c."""
    instance_totals = {c: 0 for c in counts.class_names}
    for classes in counts.per_image_classes.values():
        for c, n in classes.items():
            instance_totals[c] += n
    total_instances = sum(instance_totals.values())
    if total_instances == 0:
        return {c: 0.0 for c in counts.class_names}
    return {c: instance_totals[c] / total_instances for c in counts.class_names}


def _resolve_target_class_ids(counts: ClassCounts, target_classes: list[str] | None) -> set[int] | None:
    if target_classes is None:
        return None
    name_to_id = {name: cid for cid, name in counts.class_names.items()}
    unknown = set(target_classes) - set(name_to_id)
    if unknown:
        raise ValueError(f"Unknown target class name(s): {sorted(unknown)}")
    return {name_to_id[name] for name in target_classes}


def compute_rfs_factors(
    counts: ClassCounts, t: float, target_classes: list[str] | None = None
) -> dict[int, dict[str, float]]:
    """Per-class RFS stats. Returns {class_id: {"f_c": ..., "r_c": ...}}.

    f_c is always computed for every class. When target_classes is given,
    r_c is forced to 1.0 for any class not in that list, regardless of its
    measured f_c - i.e. that class is never oversampled.
    """
    target_ids = _resolve_target_class_ids(counts, target_classes)
    f_c = _image_frequencies(counts)
    result = {}
    for c in counts.class_names:
        if target_ids is not None and c not in target_ids:
            r_c = 1.0
        else:
            r_c = max(1.0, (t / f_c[c]) ** 0.5) if f_c[c] > 0 else 1.0
        result[c] = {"f_c": f_c[c], "r_c": r_c}
    return result


def compute_irfs_factors(
    counts: ClassCounts, t: float, target_classes: list[str] | None = None
) -> dict[int, dict[str, float]]:
    """Per-class IRFS stats. Returns {class_id: {"f_i_c", "f_b_c", "r_c"}}.

    Frequencies are always computed for every class. Same target_classes
    forcing behavior as compute_rfs_factors.
    """
    target_ids = _resolve_target_class_ids(counts, target_classes)
    f_i_c = _image_frequencies(counts)
    f_b_c = _instance_frequencies(counts)
    result = {}
    for c in counts.class_names:
        blended = (f_i_c[c] * f_b_c[c]) ** 0.5
        if target_ids is not None and c not in target_ids:
            r_c = 1.0
        else:
            r_c = max(1.0, (t / blended) ** 0.5) if blended > 0 else 1.0
        result[c] = {"f_i_c": f_i_c[c], "f_b_c": f_b_c[c], "r_c": r_c}
    return result


def compute_image_repeat_factors(counts: ClassCounts, class_factors: dict[int, dict[str, float]]) -> dict[str, float]:
    """r_i = max over classes c present in image i of r_c.

    Background images (no labeled instances) get r_i = 1.0.
    """
    image_r: dict[str, float] = {}
    for image_name, classes in counts.per_image_classes.items():
        image_r[image_name] = max((class_factors[c]["r_c"] for c in classes), default=1.0)
    return image_r


def stochastic_round(value: float, rng: random.Random) -> int:
    """floor(value) + 1 with probability = fractional part of value.

    Matches Detectron2's RepeatFactorTrainingSampler rounding behavior.
    """
    floor = int(value)
    frac = value - floor
    return floor + (1 if rng.random() < frac else 0)


def compute_repeat_counts(image_r: dict[str, float], seed: int = 42) -> dict[str, int]:
    """Deterministic per-image integer repeat counts via stochastic rounding.

    Images are processed in sorted-name order so the result is reproducible
    regardless of upstream dict iteration order, given the same seed.
    """
    rng = random.Random(seed)
    return {name: stochastic_round(image_r[name], rng) for name in sorted(image_r)}
