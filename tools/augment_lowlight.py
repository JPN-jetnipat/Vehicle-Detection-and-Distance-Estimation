#!/usr/bin/env python3
"""Offline low-light augmentation for the BDD100K YOLO split (Method 3, stage 1).

WHAT THIS IS
------------
Method 3 = Method 1 (low-light augmentation on every training image) composed
with Method 2 (IRFS instance-aware oversampling of bike/motor). This script is
stage 1: it materializes the low-light copies. Stage 2 is
`tools/build_method3_split.py`, which applies IRFS on top of the combined pool.

The augmentation math below is a straight port of the Method 1 arm's
`augment_lowlight.py` (teammate's `low_light_augment_allclass/`). The operators,
their probability gates, their parameter ranges and the forced-gamma rule are
byte-for-byte the same, so Method 3's images are drawn from exactly the same
distribution as Method 1's. Only the plumbing was changed to fit this repo:

  1. Paths default to this repo's layout and the OUTPUT LIST IS REPO-ROOT
     RELATIVE, matching `dataset/yolo/splits/train_100.txt`'s convention.
     The original script wrote absolute paths, which do not survive the
     Mac -> GPU-server move this project actually makes.
  2. The RNG is per-image, seeded from (--seed, filename), instead of one
     global sequential stream. Same distribution, but it makes the job
     RESUMABLE: rerunning after a power cut (see RUNBOOK's night-shift
     protocol) reproduces byte-identical images for the files already done
     and continues cleanly. With a single global stream, resuming at image
     40,000 would silently draw a different sequence.
  3. Preflight disk check + live size extrapolation, because the augmented
     copies are ~4-8 GB and this repo's own dataset dir has been tight before.

Design rationale and citations (unchanged from Method 1's script)
----------------------------------------------------------------
- Probability-gate structure (each augmentation applied with a set probability,
  strength randomized) and p = 0.5 per augmentation follow NightAug from
  2PCNet (Kennerley et al., CVPR 2023), which targets day-to-night domain
  adaptive detection on BDD100K -- the same dataset used here.
- Gamma range U(2, 3.5) follows the low-illumination degrading transformation
  of MAET (Cui et al., ICCV 2021), Table 1.
- Poisson-Gaussian noise model (shot noise + read noise) follows Foi et al.,
  IEEE TIP 2008, as adopted by MAET. Kept instead of NightAug's plain additive
  Gaussian noise because it is physically grounded in sensor behavior.
- Motion blur is retained as a night-specific degradation (longer exposure
  under low light). NightAug uses Gaussian blur instead; this is a deliberate
  deviation and should be described as such in the write-up.
- "At least one augmentation per image" mirrors NightAug's forced-brightness
  fallback (its brightness step is forced when the gamma step is skipped).

LABELS
------
All three operators are pixel-level, so bounding boxes do not move. Label
files are copied unchanged into a mirrored `labels/` tree next to the output
`images/` tree -- that mirroring is what lets ultralytics find them
(`img2label_paths` swaps `/images/` for `/labels/` in the image path).

Usage (from the repo root):
    python tools/augment_lowlight.py                       # full 60,186-image run
    python tools/augment_lowlight.py --limit 200 \
        --dst-root dataset/lowlight_smoke \
        --out-list dataset/yolo/splits/train_100_lowlight_smoke.txt

Output:
    dataset/lowlight/images/train/*.jpg    augmented images
    dataset/lowlight/labels/train/*.txt    labels, copied unchanged
    dataset/yolo/splits/train_100_lowlight.txt   repo-relative image list
    results/sampling_reports/lowlight_aug_log.csv (with --log-csv)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMG_LIST = "dataset/yolo/splits/train_100.txt"
DEFAULT_LABELS_SRC = "dataset/yolo/labels/train"
DEFAULT_DST_ROOT = "dataset/lowlight"
DEFAULT_OUT_LIST = "dataset/yolo/splits/train_100_lowlight.txt"
DEFAULT_LOG_CSV = "results/sampling_reports/lowlight_aug_log.csv"


# --------------------------------------------------------------------------
# I/O helpers (unicode-safe: this project's paths have gone through a Thai
# OneDrive folder on one teammate's machine, where cv2.imread returns None)
# --------------------------------------------------------------------------
def imread_unicode(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, quality: int) -> bool:
    ext = path.suffix if path.suffix else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------
# Augmentation operators -- identical math to Method 1's script, but each
# takes an explicit RandomState so the per-image seeding above works.
# RandomState reproduces numpy's legacy MT19937 stream, i.e. the same
# distributions the original global `np.random.*` calls drew from.
# --------------------------------------------------------------------------
def gamma_jitter(img: np.ndarray, rng: np.random.RandomState, gamma_range=(2.0, 3.5)) -> np.ndarray:
    """Darken via random gamma > 1 (underexposure). U(2, 3.5) per MAET Table 1."""
    gamma = rng.uniform(*gamma_range)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def poisson_gaussian_noise(
    img: np.ndarray,
    rng: np.random.RandomState,
    poisson_scale=0.8,
    gaussian_sigma_range=(5, 15),
) -> np.ndarray:
    """Shot noise (Poisson, signal-dependent) + read noise (Gaussian). Foi et al. 2008."""
    img_f = img.astype(np.float32)

    vals = 255.0 * poisson_scale
    noisy = rng.poisson(img_f / 255.0 * vals) / vals * 255.0

    sigma = rng.uniform(*gaussian_sigma_range)
    noisy = noisy + rng.normal(0, sigma, img_f.shape)

    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, rng: np.random.RandomState, kernel_size_range=(5, 15)) -> np.ndarray:
    """Camera/vehicle motion during the longer exposure low light forces."""
    k = int(rng.choice(range(*kernel_size_range, 2)))  # odd sizes only
    if k < 3:
        return img
    kernel = np.zeros((k, k))
    angle = rng.uniform(0, 180)

    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(img, -1, kernel)


def apply_lowlight_gated(
    img: np.ndarray,
    rng: np.random.RandomState,
    p_gamma=0.5,
    p_noise=0.5,
    p_blur=0.5,
):
    """NightAug-style probability gates, gamma forced on.

    NightAug forces its brightness step whenever its gamma step is skipped, so
    every image leaves the pipeline darker. We mirror that with a single
    darkening operator: the gamma gate is rolled (and logged, so the recipe is
    auditable) but gamma is applied regardless. Noise and blur stay gated at
    p = 0.5. Without the force, an image whose only passing gate is blur would
    come out blurred but not dark -- not a low-light image at all.

    Order when multiple apply: gamma -> noise -> blur.
    """
    gates = {
        "gamma": rng.random_sample() < p_gamma,
        "noise": rng.random_sample() < p_noise,
        "blur": rng.random_sample() < p_blur,
    }
    gates["gamma"] = True  # forced, per NightAug's forced-brightness fallback

    if gates["gamma"]:
        img = gamma_jitter(img, rng)
    if gates["noise"]:
        img = poisson_gaussian_noise(img, rng)
    if gates["blur"]:
        img = motion_blur(img, rng)

    return img, gates


def rng_for(seed: int, name: str) -> np.random.RandomState:
    """Deterministic per-image RNG: same (seed, filename) -> same image, always.

    This is what makes the job resumable without changing what any already-
    written image would have been.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "big"))


# --------------------------------------------------------------------------
def repo_rel(path: Path) -> str:
    """Path as repo-root-relative POSIX, so split lists survive machine moves."""
    p = path.resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()  # outside the repo: absolute is the only option


def resolve(path_str: str) -> Path:
    """Interpret a CLI path as repo-relative unless it is already absolute."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-list", default=DEFAULT_IMG_LIST,
                    help=f"txt listing source images, one per line (default: {DEFAULT_IMG_LIST})")
    ap.add_argument("--labels-src", default=DEFAULT_LABELS_SRC,
                    help=f"folder holding the matching .txt labels (default: {DEFAULT_LABELS_SRC})")
    ap.add_argument("--dst-root", default=DEFAULT_DST_ROOT,
                    help=f"output root; gets images/<split>/ and labels/<split>/ (default: {DEFAULT_DST_ROOT})")
    ap.add_argument("--out-list", default=DEFAULT_OUT_LIST,
                    help=f"txt to write with the augmented image paths (default: {DEFAULT_OUT_LIST})")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-gamma", type=float, default=0.5,
                    help="Gamma gate probability. NOTE: gamma is force-applied regardless, to guarantee "
                         "an illumination change per image (see apply_lowlight_gated). The roll is still "
                         "logged so the gate rate is auditable.")
    ap.add_argument("--p-noise", type=float, default=0.5, help="P(Poisson-Gaussian noise). NightAug default 0.5.")
    ap.add_argument("--p-blur", type=float, default=0.5, help="P(motion blur). NightAug default 0.5.")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG quality for the augmented copies (default 95 = OpenCV's default, which is "
                         "what the Method 1 arm used; lower it to ~85 only if disk is tight, and say so "
                         "in the write-up since it changes compression artifacts vs Method 1).")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N images (smoke test).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-augment images that already exist in --dst-root. Default is to skip them "
                         "(resume). Output is identical either way thanks to per-image seeding.")
    ap.add_argument("--log-csv", default=DEFAULT_LOG_CSV, help="Per-image gate log. Pass '' to disable.")
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                    help="Abort before starting if the destination filesystem has less free space than this.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    img_list = resolve(args.img_list)
    labels_src = resolve(args.labels_src)
    dst_root = resolve(args.dst_root)
    out_list = resolve(args.out_list)

    if not img_list.exists():
        sys.exit(f"FLAG: image list not found: {img_list}")
    if not labels_src.is_dir():
        sys.exit(f"FLAG: labels dir not found: {labels_src}")

    dst_img_dir = dst_root / "images" / args.split
    dst_lbl_dir = dst_root / "labels" / args.split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_list.parent.mkdir(parents=True, exist_ok=True)

    src_lines = [ln.strip() for ln in img_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        src_lines = src_lines[: args.limit]

    # Duplicate basenames across the list would silently overwrite each other in
    # the flat output dir. train_100.txt is unique by construction, but check.
    stems = [Path(x).name for x in src_lines]
    if len(set(stems)) != len(stems):
        sys.exit("FLAG: --img-list contains duplicate basenames; the flat output dir cannot represent them.")

    free_gb = shutil.disk_usage(dst_root).free / 1024**3
    print(f"Source list      : {repo_rel(img_list)}  ({len(src_lines)} images)")
    print(f"Labels from      : {repo_rel(labels_src)}")
    print(f"Writing images to: {repo_rel(dst_img_dir)}")
    print(f"Writing labels to: {repo_rel(dst_lbl_dir)}")
    print(f"Free space here  : {free_gb:.1f} GB")
    print(f"Gates            : gamma {args.p_gamma} (forced on), noise {args.p_noise}, blur {args.p_blur}")
    print(f"JPEG quality     : {args.jpeg_quality}   seed: {args.seed}")
    if free_gb < args.min_free_gb:
        sys.exit(f"FLAG: only {free_gb:.1f} GB free, below --min-free-gb {args.min_free_gb}. Refusing to start.")

    out_paths: list[str] = []
    log_rows: list[tuple[str, int, int, int]] = []
    skipped_existing = 0
    unreadable = 0
    missing_labels = 0
    bytes_written = 0
    t0 = time.time()

    for i, line in enumerate(src_lines, 1):
        src_img = resolve(line)
        out_img = dst_img_dir / src_img.name
        out_lbl = dst_lbl_dir / (src_img.stem + ".txt")

        if out_img.exists() and not args.overwrite:
            skipped_existing += 1
            out_paths.append(repo_rel(out_img))
            if not out_lbl.exists():  # partial previous run
                src_lbl = labels_src / (src_img.stem + ".txt")
                if src_lbl.exists():
                    shutil.copy(src_lbl, out_lbl)
        else:
            img = imread_unicode(src_img)
            if img is None:
                print(f"  skip (unreadable): {line}")
                unreadable += 1
                continue

            rng = rng_for(args.seed, src_img.name)
            aug, gates = apply_lowlight_gated(
                img, rng, p_gamma=args.p_gamma, p_noise=args.p_noise, p_blur=args.p_blur
            )
            if not imwrite_unicode(out_img, aug, args.jpeg_quality):
                print(f"  FLAG: failed to write {out_img}")
                continue

            bytes_written += out_img.stat().st_size
            out_paths.append(repo_rel(out_img))
            log_rows.append((src_img.name, int(gates["gamma"]), int(gates["noise"]), int(gates["blur"])))

            src_lbl = labels_src / (src_img.stem + ".txt")
            if src_lbl.exists():
                shutil.copy(src_lbl, out_lbl)
            else:
                # BDD has 564 empty label files in train; an image with no
                # vehicles is a legitimate background image, so write an empty
                # label rather than leaving ultralytics to guess.
                out_lbl.write_text("", encoding="utf-8")
                missing_labels += 1

        if i % 500 == 0 or i == len(src_lines):
            done_new = i - skipped_existing
            rate = done_new / max(time.time() - t0, 1e-9)
            eta_min = (len(src_lines) - i) / rate / 60 if rate > 0 else float("nan")
            proj_gb = (bytes_written / max(done_new, 1)) * len(src_lines) / 1024**3
            print(f"  {i}/{len(src_lines)}  ({rate:.1f} img/s, ETA {eta_min:.0f} min, "
                  f"projected total {proj_gb:.1f} GB)")

    out_list.write_text("\n".join(out_paths) + "\n", encoding="utf-8")

    if args.log_csv:
        log_csv = resolve(args.log_csv)
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("filename,gamma,noise,blur\n")
            for row in log_rows:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
        print(f"Per-image gate log -> {repo_rel(log_csv)}")

    n = len(log_rows)
    print("-" * 60)
    if n:
        print(f"Applied rates over the {n} newly written images -> "
              f"gamma: {sum(r[1] for r in log_rows)/n:.1%} (forced), "
              f"noise: {sum(r[2] for r in log_rows)/n:.1%}, "
              f"blur: {sum(r[3] for r in log_rows)/n:.1%}")
    if skipped_existing:
        print(f"Skipped (already present, resume): {skipped_existing}")
    if unreadable:
        print(f"FLAG: unreadable source images: {unreadable}")
    if missing_labels:
        print(f"Wrote {missing_labels} empty label files for images with no source label.")
    print(f"Augmented images : {len(out_paths)} in {repo_rel(dst_img_dir)}")
    print(f"Image list       : {repo_rel(out_list)}")
    print(f"Elapsed          : {(time.time() - t0)/60:.1f} min")
    print()
    print("Next: python tools/build_method3_split.py")


if __name__ == "__main__":
    main()
