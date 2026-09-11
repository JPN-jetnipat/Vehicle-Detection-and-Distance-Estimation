"""Photometric image filters for conditional dataset preprocessing.

Every filter takes a BGR uint8 image (cv2.imread's default) and returns a
BGR uint8 image of the same shape. None of these touch geometry, so label
files (bounding boxes) never need to be adjusted when an image passes
through one of these.

Add a new filter by writing a new `img -> img` function here and wiring it
into scripts/preprocess_router.py's ROUTING_TABLE - no other file needs to
change.
"""

import cv2
import numpy as np

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

MEDIAN_KSIZE = 5

# Dark Channel Prior defaults (He, Sun & Tang, 2011)
DCP_PATCH_SIZE = 15
DCP_OMEGA = 0.95
DCP_T0 = 0.1
DCP_GUIDED_RADIUS = 40
DCP_GUIDED_EPS = 1e-3


def identity(img: np.ndarray) -> np.ndarray:
    """Raw copy - no photometric change."""
    return img


def derain(img: np.ndarray) -> np.ndarray:
    """Median blur to knock down rain streaks."""
    return cv2.medianBlur(img, MEDIAN_KSIZE)


def _apply_on_l_channel(img: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = fn(l_chan)
    lab = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def histeq_lab(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization on the LAB L channel."""
    return _apply_on_l_channel(img, cv2.equalizeHist)


def clahe_bilateral(img: np.ndarray) -> np.ndarray:
    """CLAHE on the LAB L channel, followed by a bilateral filter.

    The bilateral pass suppresses the noise CLAHE amplifies while keeping
    edges (needed for small objects like motor/bike) intact.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img = _apply_on_l_channel(img, clahe.apply)
    return cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)


def _dark_channel(img: np.ndarray, patch_size: int) -> np.ndarray:
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def _atmospheric_light(img: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    flat_dark = dark_channel.ravel()
    flat_img = img.reshape(-1, 3)
    num_pixels = max(int(flat_dark.size * 0.001), 1)
    top_indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    brightest = flat_img[top_indices]
    return np.max(brightest, axis=0).astype(np.float64)


def dcp_dehaze(img: np.ndarray) -> np.ndarray:
    """Dark Channel Prior dehazing, transmission map refined with a guided filter."""
    img_f = img.astype(np.float64)
    dark = _dark_channel(img_f, DCP_PATCH_SIZE)
    atmosphere = np.clip(_atmospheric_light(img_f, dark), 1.0, 255.0)

    normalized = img_f / atmosphere
    transmission_raw = 1.0 - DCP_OMEGA * _dark_channel(normalized, DCP_PATCH_SIZE)

    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = cv2.ximgproc.guidedFilter(
        guide=guide,
        src=transmission_raw.astype(np.float32),
        radius=DCP_GUIDED_RADIUS,
        eps=DCP_GUIDED_EPS,
    )
    transmission = np.clip(transmission, DCP_T0, 1.0)[:, :, None]

    result = (img_f - atmosphere) / transmission + atmosphere
    return np.clip(result, 0, 255).astype(np.uint8)
