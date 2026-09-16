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

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 40

MEDIAN_KSIZE = 5
DERAIN_GUIDED_RADIUS = 9   # base/detail split point: bigger than a rain streak, smaller than real structure
DERAIN_GUIDED_EPS = 1e-2

# Gamma target: raise mean L (0-255, LAB) toward this before CLAHE.
# CLAHE alone only redistributes *local* contrast - on a genuinely dark frame
# (mean L well under this) it leaves the frame looking almost as dark as the
# input because there's no local contrast to redistribute in the first place.
ENHANCE_TARGET_MEAN = 110.0
ENHANCE_GAMMA_MIN = 1.0   # never darken (gamma < 1 would darken already-bright frames)
ENHANCE_GAMMA_MAX = 3.5   # cap so near-black frames don't get pushed into flat gray

# Dark Channel Prior defaults (He, Sun & Tang, 2011)
DCP_PATCH_SIZE = 15
DCP_OMEGA = 0.85          # softer than the textbook 0.95: less over-darkening on
                          # frames that are mostly sky/road with only mild haze
DCP_T0 = 0.2              # higher transmission floor than the textbook 0.1: the
                          # haze equation divides by transmission, so a low floor
                          # amplifies ordinary JPEG block noise in flat sky/haze
                          # regions into visible blockiness ("ภาพแตก")
DCP_GUIDED_RADIUS = 40
DCP_GUIDED_EPS = 1e-3
DCP_ATM_BLUR_KSIZE = 41   # smooths out small point light sources before atmosphere estimation
DCP_ATM_MAX = 220         # cap below full saturation (255) so glare pixels can't stand in for sky/airlight


def identity(img: np.ndarray) -> np.ndarray:
    """Raw copy - no photometric change."""
    return img


def derain(img: np.ndarray) -> np.ndarray:
    """Remove rain-streak-like high-frequency noise without blurring the whole frame.

    A single global median blur (the previous implementation) can't tell a
    rain streak from a building edge or license-plate text - it softens
    everything uniformly, which is why the output looked "blurry/broken" but
    the streaks themselves barely moved (median blur only erases noise
    *thinner* than its kernel; on frames with no visible streaks it just
    destroys detail for nothing).

    Instead: split the image into a base layer (edge-preserving guided
    filter - keeps real structure sharp) and a detail layer (base minus
    original - carries thin streak-like noise plus fine texture). Median-blur
    only the detail layer to knock out the streak-like component, then add it
    back onto the untouched base. Real edges live mostly in the base layer
    and pass through unblurred; only the noise-like residual gets filtered.
    """
    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = cv2.ximgproc.guidedFilter(
        guide=guide, src=img, radius=DERAIN_GUIDED_RADIUS, eps=DERAIN_GUIDED_EPS
    )
    detail = img.astype(np.int16) - base.astype(np.int16)
    detail_u8 = np.clip(detail + 128, 0, 255).astype(np.uint8)
    detail_denoised = cv2.medianBlur(detail_u8, MEDIAN_KSIZE).astype(np.int16) - 128
    result = base.astype(np.int16) + detail_denoised
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_on_l_channel(img: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = fn(l_chan)
    lab = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _gamma_correct_l(l_chan: np.ndarray) -> np.ndarray:
    """Power-law brighten the L channel so its mean approaches ENHANCE_TARGET_MEAN.

    This is what actually fixes "light is still low": CLAHE only reshuffles
    *local* contrast and cannot raise the overall brightness of a frame that
    is dark everywhere (a real night frame has little local contrast to
    redistribute in the first place). Gamma correction lifts every pixel
    (dark pixels more than bright ones) before CLAHE sharpens local detail.

    Self-limiting: a frame already at or above the target is returned
    untouched, so this is safe to run on well-exposed daytime frames too.
    """
    mean_l = float(l_chan.mean())
    if mean_l < 1.0:
        mean_l = 1.0
    if mean_l >= ENHANCE_TARGET_MEAN:
        return l_chan
    gamma = np.log(mean_l / 255.0) / np.log(ENHANCE_TARGET_MEAN / 255.0)
    gamma = float(np.clip(gamma, ENHANCE_GAMMA_MIN, ENHANCE_GAMMA_MAX))
    normalized = l_chan.astype(np.float64) / 255.0
    brightened = np.power(normalized, 1.0 / gamma) * 255.0
    return np.clip(brightened, 0, 255).astype(np.uint8)


def adaptive_enhance(img: np.ndarray) -> np.ndarray:
    """Gamma-brighten, then CLAHE, then a gentle bilateral denoise.

    Order matters: gamma first raises overall exposure so CLAHE has real
    local contrast to work with; CLAHE second sharpens local detail (needed
    for small objects like motor/bike); bilateral last cleans up the noise
    CLAHE amplifies. sigmaColor/sigmaSpace are kept low (40) so the filter
    doesn't blur across the strong edges around streetlights and headlights,
    which produces a smeared halo look.

    Named "adaptive" rather than "low_light" because the gamma stage scales
    itself to the input and no-ops on an already-bright frame - so the same
    function serves genuinely dark night frames and merely flat overcast
    daytime ones, doing only as much as each needs.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img = _apply_on_l_channel(img, _gamma_correct_l)
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
    # Average the candidate pixels (not a per-channel max): taking the max of
    # each channel independently stitches together mismatched channels from
    # different candidate pixels (e.g. a red sign's R with a white lamp's B),
    # synthesizing a color-biased atmosphere that causes colored halos around
    # isolated bright light sources - exactly the artifact that shows up on
    # foggy dawn/dusk street scenes with streetlights already on.
    return np.mean(brightest, axis=0).astype(np.float64)


def dcp_dehaze(img: np.ndarray) -> np.ndarray:
    """Dark Channel Prior dehazing, transmission map refined with a guided filter.

    Atmospheric light is estimated from a blurred copy of the image so that
    small, isolated overexposed light sources (streetlights, headlights,
    signage - a few pixels wide) don't get picked as "sky/airlight" just
    because they're locally the brightest, least-hazy-looking patch. Without
    the blur, a scene with no real hazy-sky region (e.g. a foggy dusk street
    with the lights already on) hands the estimator a pool of pure
    (255, 255, 255) glare pixels, and the resulting near-white atmosphere
    crushes every darker pixel toward black when the haze equation is
    inverted.
    """
    img_f = img.astype(np.float64)
    blurred = cv2.blur(img, (DCP_ATM_BLUR_KSIZE, DCP_ATM_BLUR_KSIZE)).astype(np.float64)
    atm_dark = _dark_channel(blurred, DCP_PATCH_SIZE)
    atmosphere = np.clip(_atmospheric_light(blurred, atm_dark), 1.0, DCP_ATM_MAX)

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
