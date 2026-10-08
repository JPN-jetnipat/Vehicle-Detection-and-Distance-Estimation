"""Preview adaptive_enhance at a few fixed gamma values on one image.

Only the gamma stage changes - CLAHE and the bilateral denoise run exactly as
in preprocessing_filters.adaptive_enhance. Used to pick a gentler gamma for
night frames where oncoming headlights bloom under the auto-chosen gamma
(which runs up to ENHANCE_GAMMA_MAX on very dark frames).

Usage:
    python scripts/gamma_sweep.py dataset/yolo/images/val/b1cd1e94-26dd524f.jpg
    python scripts/gamma_sweep.py <image> --gammas 1.2 1.5 1.8 2.2
    python scripts/gamma_sweep.py <image> --gammas 0.9 0.8 0.7 0.6 --gamma-only

gamma > 1 brightens (out = in ** (1/gamma)), gamma < 1 darkens. --gamma-only
skips CLAHE and the bilateral denoise, to see the gamma stage on its own.

Writes to results/gamma_sweep/<image stem>/<run>/ (run = pipeline + gammas): the original, one file per gamma,
the current adaptive_enhance output for reference, and a labeled comparison grid.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import preprocessing_filters as pf

ROOT = Path(__file__).resolve().parent.parent


def gamma_l(l_chan: np.ndarray, gamma: float) -> np.ndarray:
    normalized = l_chan.astype(np.float64) / 255.0
    return np.clip(np.power(normalized, 1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)


def enhance_with_gamma(img: np.ndarray, gamma: float, gamma_only: bool = False) -> np.ndarray:
    if gamma_only:
        return pf._apply_on_l_channel(img, lambda l: gamma_l(l, gamma))
    clahe = cv2.createCLAHE(clipLimit=pf.CLAHE_CLIP_LIMIT, tileGridSize=pf.CLAHE_TILE_GRID)
    img = pf._apply_on_l_channel(img, lambda l: gamma_l(l, gamma))
    img = pf._apply_on_l_channel(img, clahe.apply)
    return cv2.bilateralFilter(img, pf.BILATERAL_D, pf.BILATERAL_SIGMA_COLOR, pf.BILATERAL_SIGMA_SPACE)


def auto_gamma(img: np.ndarray) -> float:
    """The gamma adaptive_enhance would pick for this image (same formula as _gamma_correct_l)."""
    mean_l = max(float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].mean()), 1.0)
    if mean_l >= pf.ENHANCE_TARGET_MEAN:
        return 1.0
    gamma = np.log(mean_l / 255.0) / np.log(pf.ENHANCE_TARGET_MEAN / 255.0)
    return float(np.clip(gamma, pf.ENHANCE_GAMMA_MIN, pf.ENHANCE_GAMMA_MAX))


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 15 + 20, 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.2, 1.5, 1.8, 2.2])
    parser.add_argument("--gamma-only", action="store_true", help="Skip CLAHE and bilateral denoise.")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    run = ("gamma_only" if args.gamma_only else "full") + "_" + "_".join(f"{g:.1f}" for g in args.gammas)
    out_dir = ROOT / "results" / "gamma_sweep" / args.image.stem / run
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [("original", img)]
    for g in args.gammas:
        panels.append((f"gamma_{g:.1f}", enhance_with_gamma(img, g, args.gamma_only)))
    g_auto = auto_gamma(img)
    panels.append((f"current_adaptive_gamma_{g_auto:.2f}", pf.adaptive_enhance(img)))

    for i, (name, panel) in enumerate(panels):
        cv2.imwrite(str(out_dir / f"{i}_{name}.jpg"), panel)

    # 2 columns, padded with black if the panel count is odd
    tiles = [label(p, n) for n, p in panels]
    if len(tiles) % 2:
        tiles.append(np.zeros_like(img))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(out_dir / "comparison.jpg"), np.vstack(rows))
    print(f"Saved {len(panels)} images + comparison.jpg to {out_dir}")


if __name__ == "__main__":
    main()
