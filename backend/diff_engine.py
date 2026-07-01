"""
Difference detection engine for CAD drawing comparison.

This module implements the core diff algorithm that identifies and localizes
changes between two preprocessed CAD images. It combines two complementary
approaches:

1. **SSIM (Structural Similarity Index)** — Computes a per-pixel similarity
   map that captures perceptual differences in structure, luminance, and
   contrast. The difference map (1 - SSIM) highlights regions where the
   two images diverge structurally.

2. **Absolute pixel difference** — A simpler, direct subtraction that catches
   any pixel-level change, including subtle shifts that SSIM might underweight.

The two signals are combined via logical OR on their thresholded binary masks,
ensuring we catch both structural and pixel-level changes.

Critical CAD-specific step — **morphological dilation**:
    CAD drawings consist of thin lines (often 1-3 pixels wide). When a line
    shifts by even a few pixels, the diff produces two parallel thin streaks
    (one for the old position, one for the new). Without dilation, contour
    detection fragments each streak into dozens of tiny disconnected regions,
    creating noise. Dilation with a small kernel (5x5) merges these nearby
    thin-line fragments into coherent change regions that correspond to
    actual engineering modifications rather than pixel-level noise.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from typing import List, Dict, Tuple


# Minimum contour area in pixels to be considered a real change.
# Anything smaller is treated as noise (scanner artifacts, anti-aliasing residue).
MIN_CONTOUR_AREA = 50

# Dilation kernel size. 5x5 is large enough to bridge thin-line fragments
# without merging genuinely separate change regions.
DILATION_KERNEL_SIZE = 5


def compute_ssim_diff(
    gray_a: np.ndarray, gray_b: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute the SSIM difference map between two grayscale images.

    SSIM produces a per-pixel similarity score in [-1, 1]. We convert this
    to a difference map in [0, 255] by computing (1 - ssim_map), scaling,
    and casting to uint8. Regions with low similarity (= high difference)
    become bright in the output.

    Args:
        gray_a: Grayscale reference image.
        gray_b: Grayscale comparison image.

    Returns:
        Tuple of:
        - ssim_score: Overall SSIM score (float in [-1, 1]).
        - diff_map: Per-pixel difference map (uint8, 0=identical, 255=maximally different).
    """
    # full=True returns the per-pixel SSIM map alongside the scalar score.
    # win_size must be odd and <= image dimensions; 7 is a safe default.
    win_size = min(7, min(gray_a.shape[:2]) | 1)  # ensure odd
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    ssim_score, ssim_map = structural_similarity(
        gray_a, gray_b, full=True, win_size=win_size
    )

    # Convert similarity map to difference map: higher = more different
    diff_map = ((1.0 - ssim_map) * 255).astype(np.uint8)

    return ssim_score, diff_map


def compute_absdiff(
    bin_a: np.ndarray, bin_b: np.ndarray
) -> np.ndarray:
    """
    Compute the absolute pixel-wise difference between two binary images.

    This is a straightforward subtraction that catches any pixel that changed
    between the two versions. On binarized CAD images, this directly shows
    where lines were added, removed, or shifted.

    Args:
        bin_a: Binarized reference image.
        bin_b: Binarized comparison image.

    Returns:
        Absolute difference image (uint8).
    """
    return cv2.absdiff(bin_a, bin_b)


def create_combined_mask(
    ssim_diff: np.ndarray, abs_diff: np.ndarray
) -> np.ndarray:
    """
    Combine SSIM and absolute-difference signals into a single binary mask.

    Each signal is independently thresholded using Otsu's method (which
    adapts to the actual distribution of difference values), then the two
    masks are merged with logical OR. This ensures we capture both
    structural differences (SSIM) and raw pixel changes (absdiff).

    After thresholding, morphological dilation is applied to merge nearby
    thin-line fragments into coherent regions. This is the single most
    important step for CAD drawings — without it, a single shifted line
    produces dozens of tiny contours instead of one meaningful region.

    Args:
        ssim_diff: SSIM difference map (uint8).
        abs_diff: Absolute difference image (uint8).

    Returns:
        Binary mask (uint8, 0 or 255) with dilated change regions.
    """
    # Smooth the difference maps slightly before thresholding to reduce
    # isolated noise while preserving real structural differences.
    blurred_ssim = cv2.GaussianBlur(ssim_diff, (5, 5), 0)
    blurred_abs = cv2.GaussianBlur(abs_diff, (5, 5), 0)

    # Threshold SSIM diff — Otsu's adapts to the actual noise level
    _, ssim_mask = cv2.threshold(
        blurred_ssim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Threshold absolute diff similarly
    _, abs_mask = cv2.threshold(
        blurred_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Combine both masks — a pixel is "changed" if either method flags it
    combined = cv2.bitwise_or(ssim_mask, abs_mask)

    # Close small holes and gaps first, then dilate to unify line fragments.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=2)

    return dilated


def extract_changed_regions(
    mask: np.ndarray, image_shape: Tuple[int, int]
) -> List[Dict]:
    """
    Extract contours from the binary mask and return structured region data.

    Each contour is converted to a bounding box, and its centroid is
    classified into one of nine spatial zones (3x3 grid). Contours with
    area below MIN_CONTOUR_AREA are discarded as noise.

    Args:
        mask: Binary mask of changed regions (uint8, 0 or 255).
        image_shape: (height, width) of the original image, used for
                     spatial classification.

    Returns:
        List of dicts, each with keys: 'bbox', 'area', 'centroid', 'location'.
        Sorted by area descending (largest change first).
    """
    # findContours returns the outermost contours only (RETR_EXTERNAL)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = image_shape
    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out noise: tiny contours from anti-aliasing or scanner artifacts
        if area < MIN_CONTOUR_AREA:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Compute centroid using image moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback: center of bounding box
            cx = x + w // 2
            cy = y + h // 2

        # Classify spatial location based on centroid position in a 3x3 grid
        location = _classify_location(cx, cy, img_w, img_h)

        regions.append({
            "bbox": [x, y, w, h],
            "area": int(area),
            "centroid": (cx, cy),
            "location": location,
        })

    # Sort by area descending — largest change first
    regions.sort(key=lambda r: r["area"], reverse=True)

    return regions


def _classify_location(cx: int, cy: int, img_w: int, img_h: int) -> str:
    """
    Classify a centroid's position into one of nine spatial zones.

    The image is divided into a 3x3 grid (thirds horizontally and vertically).
    The centroid's position within this grid determines its human-readable
    location label.

    Args:
        cx: Centroid x-coordinate.
        cy: Centroid y-coordinate.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Location string like "top-left", "center", "bottom-right", etc.
    """
    # Determine horizontal third
    if cx < img_w / 3:
        col = "left"
    elif cx < 2 * img_w / 3:
        col = "center"
    else:
        col = "right"

    # Determine vertical third
    if cy < img_h / 3:
        row = "top"
    elif cy < 2 * img_h / 3:
        row = "center"
    else:
        row = "bottom"

    # Combine: "center-center" simplifies to "center"
    if row == "center" and col == "center":
        return "center"

    return f"{row}-{col}"


def detect_differences(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    bin_a: np.ndarray,
    bin_b: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, List[Dict]]:
    """
    Full difference detection pipeline.

    Orchestrates SSIM computation, absolute diff, mask creation, and
    contour extraction into a single call.

    Args:
        gray_a: Grayscale reference image.
        gray_b: Grayscale comparison image.
        bin_a: Binarized reference image.
        bin_b: Binarized comparison image.

    Returns:
        Tuple of:
        - ssim_score: Overall structural similarity score.
        - diff_map: Per-pixel SSIM difference map (for heatmap visualization).
        - mask: Binary mask of detected changes (for overlay visualization).
        - regions: List of structured change region dicts.
    """
    # Step 1: SSIM-based structural difference
    ssim_score, ssim_diff = compute_ssim_diff(gray_a, gray_b)

    # Step 2: Pixel-level absolute difference on binarized images
    abs_diff = compute_absdiff(bin_a, bin_b)

    # Step 3: Combine both signals and apply morphological cleanup
    mask = create_combined_mask(ssim_diff, abs_diff)

    # Step 4: Extract and classify change regions
    regions = extract_changed_regions(mask, gray_a.shape[:2])

    return ssim_score, ssim_diff, mask, regions
