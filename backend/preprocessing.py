"""
Image preprocessing pipeline for CAD drawing comparison.

This module handles three critical preparation steps before diff detection:
1. **Resizing** — Normalizes both images to identical dimensions so that
   pixel-wise comparison is geometrically valid.
2. **Alignment** — Uses ORB feature matching + RANSAC homography to correct
   slight translational/rotational shifts between scans of the same drawing.
3. **Binarization** — Converts grayscale images to pure black-and-white using
   Otsu's adaptive threshold, which is essential for CAD drawings where the
   signal is thin dark lines on a white background.

Design rationale (CAD-specific):
    Natural photographs have smooth gradients and dense texture. CAD drawings
    are fundamentally different: sparse, high-contrast line art on a uniform
    background. Standard SSIM on raw grayscale CAD images picks up scanner
    noise, slight gray-level variations in the background, and anti-aliasing
    artifacts as false positives. Binarization eliminates these issues by
    reducing each pixel to "line" or "background", making the subsequent
    diff detection far more reliable.
"""

import cv2
import numpy as np
from typing import Tuple


def resize_to_match(
    image_a: np.ndarray, image_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize both images to the same dimensions based on the smaller image.

    We choose the smaller image's dimensions as the target because upscaling
    introduces interpolation artifacts, while downscaling is lossy but
    preserves relative spatial relationships more faithfully.

    Args:
        image_a: First input image (BGR, any resolution).
        image_b: Second input image (BGR, any resolution).

    Returns:
        Tuple of (resized_a, resized_b), both with identical (h, w).
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    # Use the minimum dimensions to avoid upscaling artifacts
    target_h = min(h_a, h_b)
    target_w = min(w_a, w_b)

    resized_a = cv2.resize(
        image_a, (target_w, target_h), interpolation=cv2.INTER_AREA
    )
    resized_b = cv2.resize(
        image_b, (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    return resized_a, resized_b


def align_images(
    reference: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Align the target image to the reference using ORB feature matching.

    CAD drawings scanned or exported at different times may have slight
    positional shifts, rotations, or scale differences. This function
    detects matching keypoints in both images using ORB (Oriented FAST
    and Rotated BRIEF), then computes a homography matrix via RANSAC
    to warp the target into the reference's coordinate frame.

    ORB was chosen over SIFT/SURF because:
    - It is free and unencumbered by patents.
    - It is fast (binary descriptors vs. floating-point).
    - It works well on high-contrast line art (strong corners).

    If fewer than 4 matches are found (the minimum for homography),
    the function returns the target unchanged — this is a safe fallback
    because the images are likely already well-aligned.

    Args:
        reference: The reference image (BGR color).
        target: The target image to warp into alignment (BGR color).

    Returns:
        The target image warped to align with the reference, or the
        original target if alignment was not possible.
    """
    # Convert to grayscale for feature detection
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector with a generous feature count
    # More features = more robust matching, especially on sparse CAD drawings
    orb = cv2.ORB_create(nfeatures=5000)

    # Detect keypoints and compute descriptors
    kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
    kp_tgt, desc_tgt = orb.detectAndCompute(gray_tgt, None)

    # If either image has no detectable features, skip alignment
    if desc_ref is None or desc_tgt is None:
        return target

    # Use Brute-Force matcher with Hamming distance (appropriate for ORB's
    # binary descriptors) and cross-check for symmetric matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_tgt, desc_ref)

    # Sort by distance (lower = better match quality)
    matches = sorted(matches, key=lambda m: m.distance)

    # Need at least 4 point correspondences to compute a homography
    if len(matches) < 4:
        return target

    # Extract matched point coordinates
    pts_tgt = np.float32(
        [kp_tgt[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    pts_ref = np.float32(
        [kp_ref[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # Compute homography using RANSAC to reject outlier matches
    # RANSAC threshold of 5.0 pixels is generous enough for CAD drawings
    H, mask = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, 5.0)

    if H is None:
        return target

    # Warp the target image to align with the reference
    h, w = reference.shape[:2]
    aligned = cv2.warpPerspective(target, H, (w, h))

    return aligned


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR color image to single-channel grayscale.

    Args:
        image: Input BGR image.

    Returns:
        Single-channel grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(gray_image: np.ndarray) -> np.ndarray:
    """
    Binarize a grayscale image using Otsu's adaptive thresholding.

    This is critical for CAD drawing comparison. CAD images are fundamentally
    binary in nature (lines vs. background), but scanned/exported versions
    contain anti-aliasing, slight gray gradients, and scanner noise. Otsu's
    method automatically determines the optimal threshold by minimizing
    intra-class variance, effectively separating line pixels from background
    without manual tuning.

    After thresholding, we invert the result so that lines (the content we
    care about) become white (255) and background becomes black (0). This
    makes subsequent morphological operations and contour detection work
    correctly, since OpenCV's findContours looks for white objects on black.

    Args:
        gray_image: Single-channel grayscale image.

    Returns:
        Binary image where line content is white (255) and background is
        black (0).
    """
    # Otsu's method: cv2.THRESH_OTSU automatically computes the optimal
    # threshold value, so the first threshold argument (0) is ignored.
    # THRESH_BINARY_INV inverts so lines become white.
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def preprocess_pair(
    raw_a: np.ndarray, raw_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a pair of CAD drawing images.

    Orchestrates resize → align → grayscale → binarize, returning all
    intermediate representations needed by downstream modules.

    Args:
        raw_a: Raw reference image as loaded by cv2.imread (BGR).
        raw_b: Raw comparison image as loaded by cv2.imread (BGR).

    Returns:
        Tuple of six arrays:
        - color_a: Resized color reference image (for visualization).
        - color_b: Resized + aligned color comparison image (for visualization).
        - gray_a: Grayscale reference (for SSIM computation).
        - gray_b: Grayscale aligned comparison (for SSIM computation).
        - bin_a: Binarized reference (for CAD-specific diff detection).
        - bin_b: Binarized aligned comparison (for CAD-specific diff detection).
    """
    # Step 1: Resize both to matching dimensions
    color_a, color_b = resize_to_match(raw_a, raw_b)

    # Step 2: Align image B to image A's coordinate frame
    color_b = align_images(color_a, color_b)

    # Step 3: Convert to grayscale for diff algorithms
    gray_a = convert_to_grayscale(color_a)
    gray_b = convert_to_grayscale(color_b)

    # Step 4: Binarize for CAD-specific line isolation
    bin_a = binarize(gray_a)
    bin_b = binarize(gray_b)

    return color_a, color_b, gray_a, gray_b, bin_a, bin_b
