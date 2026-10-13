"""
Offline low-light augmentation for YOLO-format datasets (Method 1).

Pipeline: gamma jitter + Poisson-Gaussian noise + motion blur, each applied
under a probability gate with randomized strength.

Design rationale and citations
------------------------------
- Probability-gate structure (each augmentation applied with a set probability,
  strength randomized) and p = 0.5 per augmentation follow NightAug from
  2PCNet (Kennerley et al., CVPR 2023), which targets day-to-night domain
  adaptive detection on BDD100K -- the same dataset used here.
- Gamma range U(2, 3.5) follows the low-illumination degrading transformation
  of MAET (Cui et al., ICCV 2021), Table 1.
- Poisson-Gaussian noise model (shot noise + read noise) follows Foi et al.,
  IEEE TIP 2008, as adopted by MAET. This is kept instead of NightAug's plain
  additive Gaussian noise because it is physically grounded in sensor behavior.
- Motion blur is retained as a night-specific degradation (longer exposure
  under low light). NightAug uses Gaussian blur instead; this is a deliberate
  deviation and should be described as such in the write-up.
- "At least one augmentation per image" mirrors NightAug's forced-brightness
  fallback (its brightness step is forced when the gamma step is skipped).

Dataset layout note
-------------------
This project's dataset is NOT organized as a plain images/ folder -- training
uses a .txt file listing absolute image paths, one per line (e.g.
train_100_images.txt from prepare_train_yaml.py). This script reads that list,
augments each image, writes augmented copies to a SEPARATE output folder
(never touches the original bdd100k_split_handoff data), copies each image's
matching label file unchanged (pixel-level augmentations don't move bboxes),
and writes a NEW .txt list pointing at the augmented images -- so it plugs
into data.yaml the same way the original list does.

Usage:
    python augment_lowlight.py \
        --img_list "C:\\...\\splits_local\\train_100_images.txt" \
        --labels_src "C:\\...\\bdd100k_split_handoff\\dataset\\yolo\\labels\\train" \
        --dst_root "C:\\...\\baby\\bdd100k_lowlight_aug" \
        --out_list "C:\\...\\splits_local\\train_100_images_lowlight.txt"

Output:
    <dst_root>/images/train/*.jpg   (augmented images)
    <dst_root>/labels/train/*.txt   (copied labels, unchanged)
    <out_list>                      (.txt list of augmented image paths)
"""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


def imread_unicode(path: Path):
    """
    Read an image from a path that may contain non-ASCII characters.
    cv2.imread uses the ANSI code page on Windows and fails on Thai/Unicode
    paths, so the bytes are read by Python and decoded by OpenCV instead.
    Returns None if the file cannot be read or decoded.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """
    Write an image to a path that may contain non-ASCII characters.
    Mirrors imread_unicode: OpenCV encodes to a buffer, Python writes the file.
    """
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def gamma_jitter(img: np.ndarray, gamma_range=(2.0, 3.5)) -> np.ndarray:
    """
    Darken image via random gamma > 1 (simulates underexposure).
    Range U(2, 3.5) follows MAET (Cui et al., ICCV 2021), Table 1.
    """
    gamma = np.random.uniform(*gamma_range)
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def poisson_gaussian_noise(
    img: np.ndarray, poisson_scale=0.8, gaussian_sigma_range=(5, 15)
) -> np.ndarray:
    """
    Sensor noise at low light: shot noise (Poisson, signal-dependent) plus
    read noise (Gaussian, signal-independent).
    Model follows Foi et al., IEEE TIP 2008, as adopted by MAET.
    """
    img_f = img.astype(np.float32)

    vals = 255.0 * poisson_scale
    noisy = np.random.poisson(img_f / 255.0 * vals) / vals * 255.0

    sigma = np.random.uniform(*gaussian_sigma_range)
    noisy = noisy + np.random.normal(0, sigma, img_f.shape)

    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, kernel_size_range=(5, 15)) -> np.ndarray:
    """
    Motion blur from camera/vehicle movement during the longer exposure times
    required in low light. Deviates from NightAug, which uses Gaussian blur.
    """
    k = int(np.random.choice(range(*kernel_size_range, 2)))  # odd sizes only
    if k < 3:
        return img
    kernel = np.zeros((k, k))
    angle = np.random.uniform(0, 180)

    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(img, -1, kernel)


def apply_lowlight_gated(img: np.ndarray, p_gamma=0.5, p_noise=0.5, p_blur=0.5):
    """
    Probability-gate pipeline following NightAug (2PCNet, CVPR 2023): each
    augmentation is gated independently with randomized strength.

    Illumination guarantee: NightAug forces its brightness step whenever its
    gamma step is skipped, so every image leaves the pipeline darker. We mirror
    that with a single darkening operator: if the gamma gate rolls False, gamma
    is applied anyway. Noise and blur remain probability-gated at p = 0.5.
    Without this, an image whose only passing gate is blur would come out
    blurred but not dark -- not a low-light image at all.

    Order when multiple are applied: gamma -> noise -> blur.
    Returns (augmented_img, gates_dict) so the caller can log what was applied.
    """
    gates = {
        "gamma": np.random.random() < p_gamma,
        "noise": np.random.random() < p_noise,
        "blur": np.random.random() < p_blur,
    }
    # Forced illumination change, per NightAug's forced-brightness fallback.
    gates["gamma"] = True

    if gates["gamma"]:
        img = gamma_jitter(img)
    if gates["noise"]:
        img = poisson_gaussian_noise(img)
    if gates["blur"]:
        img = motion_blur(img)

    return img, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_list", required=True, help="Path to .txt file listing source image paths (e.g. train_100_images.txt)")
    ap.add_argument("--labels_src", required=True, help="Folder containing matching .txt labels (e.g. .../labels/train)")
    ap.add_argument("--dst_root", required=True, help="Output root for augmented images+labels (separate from original)")
    ap.add_argument("--out_list", required=True, help="Path to write the new .txt list of augmented image paths")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_gamma", type=float, default=0.5, help="Gamma gate probability. Note: gamma is force-applied regardless, to guarantee an illumination change per image (see apply_lowlight_gated)")
    ap.add_argument("--p_noise", type=float, default=0.5, help="Probability of Poisson-Gaussian noise (NightAug default: 0.5)")
    ap.add_argument("--p_blur", type=float, default=0.5, help="Probability of motion blur (NightAug default: 0.5)")
    ap.add_argument("--log_csv", default=None, help="Optional CSV log of which augmentations were applied per image")
    args = ap.parse_args()

    np.random.seed(args.seed)

    labels_src = Path(args.labels_src)
    # Resolve to absolute so the generated .txt list works regardless of the
    # working directory YOLO is launched from.
    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_img_dir = dst_root / "images" / args.split
    dst_lbl_dir = dst_root / "labels" / args.split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(args.img_list, "r", encoding="utf-8") as f:
        img_paths = [line.strip() for line in f if line.strip()]

    print(f"Read {len(img_paths)} image paths from {args.img_list}")
    print(f"Probabilities -> gamma: {args.p_gamma}, noise: {args.p_noise}, blur: {args.p_blur}")

    out_paths = []
    log_rows = []
    for i, img_path_str in enumerate(img_paths, 1):
        img_path = Path(img_path_str)
        img = imread_unicode(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        aug, gates = apply_lowlight_gated(
            img, p_gamma=args.p_gamma, p_noise=args.p_noise, p_blur=args.p_blur
        )
        out_img_path = dst_img_dir / img_path.name
        if not imwrite_unicode(out_img_path, aug):
            print(f"Failed to write: {out_img_path}")
            continue
        out_paths.append(str(out_img_path))
        log_rows.append((img_path.name, gates["gamma"], gates["noise"], gates["blur"]))

        lbl_path = labels_src / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)
        else:
            print(f"Warning: no label found for {img_path.name}")

        if i % 500 == 0:
            print(f"Processed {i}/{len(img_paths)}")

    with open(args.out_list, "w", encoding="utf-8") as f:
        f.write("\n".join(out_paths) + "\n")

    if args.log_csv:
        with open(args.log_csv, "w", encoding="utf-8") as f:
            f.write("filename,gamma,noise,blur\n")
            for row in log_rows:
                f.write(f"{row[0]},{int(row[1])},{int(row[2])},{int(row[3])}\n")
        print(f"Per-image augmentation log written to {args.log_csv}")

    n = len(log_rows)
    if n:
        print(
            f"Applied rates -> gamma: {sum(r[1] for r in log_rows)/n:.1%}, "
            f"noise: {sum(r[2] for r in log_rows)/n:.1%}, "
            f"blur: {sum(r[3] for r in log_rows)/n:.1%} "
            f"(every image got at least one)"
        )

    print(f"Done. {len(out_paths)} augmented images written to {dst_img_dir}")
    print(f"Labels copied to {dst_lbl_dir}")
    print(f"New image list written to {args.out_list}")


if __name__ == "__main__":
    main()