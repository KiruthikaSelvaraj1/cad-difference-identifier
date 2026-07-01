"""
Visualization module for CAD drawing difference results.

Generates four output images that together provide a comprehensive visual
understanding of the detected changes:

1. **Highlighted Regions** — Bounding boxes drawn on Image B around each
   detected change, with region index labels for cross-referencing with
   the statistics table.

2. **Heatmap Overlay** — The SSIM difference map rendered as a JET colormap
   and alpha-blended onto Image B, showing change intensity as a continuous
   gradient (blue=no change → red=maximum change).

3. **Side-by-Side** — Image A and Image B placed next to each other with a
   dividing line, enabling direct visual comparison.

4. **Overlay Blend** — Image A and Image B alpha-blended at 50% opacity,
   with the binary change mask overlaid in red. This makes it easy to see
   exactly what moved or changed.

All outputs are saved as PNG files in the outputs/ directory.
"""

import cv2
import numpy as np
import os
from typing import List, Dict


def draw_bounding_boxes(
    image: np.ndarray, regions: List[Dict], output_path: str
) -> str:
    """
    Draw labeled bounding boxes around each detected change region.

    Each box is drawn in green (high contrast against both white CAD
    backgrounds and dark lines) with a sequential label number. The label
    is placed just above the top-left corner of each box.

    Args:
        image: The comparison image (BGR color) to draw on.
        regions: List of region dicts with 'bbox' key.
        output_path: File path to save the output PNG.

    Returns:
        The output file path (same as output_path).
    """
    # Work on a copy to avoid mutating the original
    result = image.copy()

    for idx, region in enumerate(regions, start=1):
        x, y, w, h = region["bbox"]

        # Green bounding box with 2px line weight
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label background and arrow for clearer region marking
        label = f"#{idx}"
        label_x = x
        label_y = max(y - 18, 20)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            result,
            (label_x - 4, label_y - text_size[1] - 8),
            (label_x + text_size[0] + 8, label_y + 4),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            result,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Draw an arrow from the label down to the top-left corner of the box
        arrow_tip = (x + 10, y + 10)
        arrow_start = (label_x + text_size[0] // 2, label_y - 4)
        cv2.arrowedLine(
            result,
            arrow_start,
            arrow_tip,
            (0, 255, 0),
            2,
            tipLength=0.2,
        )

    cv2.imwrite(output_path, result)
    return output_path


def generate_heatmap(
    image: np.ndarray, diff_map: np.ndarray, output_path: str
) -> str:
    """
    Generate a heatmap overlay showing difference intensity.

    The SSIM difference map is converted to a JET colormap (blue=low,
    red=high) and alpha-blended onto the comparison image at 60% opacity.
    This provides an intuitive visualization of where changes are
    concentrated and how severe they are.

    Args:
        image: The comparison image (BGR color) as the base layer.
        diff_map: SSIM difference map (uint8, single channel).
        output_path: File path to save the output PNG.

    Returns:
        The output file path.
    """
    # Apply JET colormap to the difference map
    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

    # Resize heatmap to match image dimensions (should already match,
    # but this guards against edge cases)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Alpha blend: 40% original + 60% heatmap
    overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)

    cv2.imwrite(output_path, overlay)
    return output_path


def generate_side_by_side(
    image_a: np.ndarray, image_b: np.ndarray, output_path: str
) -> str:
    """
    Create a side-by-side composite of both images with a dividing line.

    The two images are placed horizontally adjacent with a 4-pixel white
    separator line between them. Labels "Image A (Reference)" and
    "Image B (Comparison)" are added at the top of each half.

    Args:
        image_a: The reference image (BGR color).
        image_b: The comparison image (BGR color).
        output_path: File path to save the output PNG.

    Returns:
        The output file path.
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    # Use the maximum height for the canvas
    max_h = max(h_a, h_b)
    separator_width = 4

    # Create white canvas
    canvas = np.ones(
        (max_h, w_a + separator_width + w_b, 3), dtype=np.uint8
    ) * 255

    # Place Image A on the left
    canvas[:h_a, :w_a] = image_a

    # Draw separator line (gray)
    canvas[:, w_a:w_a + separator_width] = (180, 180, 180)

    # Place Image B on the right
    canvas[:h_b, w_a + separator_width:w_a + separator_width + w_b] = image_b

    # Add labels
    cv2.putText(
        canvas, "Image A (Reference)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2
    )
    cv2.putText(
        canvas, "Image B (Comparison)", (w_a + separator_width + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2
    )

    cv2.imwrite(output_path, canvas)
    return output_path


def generate_overlay(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    output_path: str,
) -> str:
    """
    Generate a blended overlay with changes highlighted in red.

    Both images are alpha-blended at 50% opacity to show the combined
    drawing, then the change mask is overlaid in semi-transparent red.
    This makes it immediately obvious where content differs between
    the two versions.

    Args:
        image_a: The reference image (BGR color).
        image_b: The comparison image (BGR color).
        mask: Binary change mask (uint8, 0 or 255).
        output_path: File path to save the output PNG.

    Returns:
        The output file path.
    """
    # Blend both images at 50/50
    blended = cv2.addWeighted(image_a, 0.5, image_b, 0.5, 0)

    # Create a red highlight layer
    red_overlay = np.zeros_like(blended)
    red_overlay[:, :] = (0, 0, 255)  # BGR red

    # Resize mask to match if needed
    mask_resized = cv2.resize(mask, (blended.shape[1], blended.shape[0]))

    # Apply red highlight only where mask is non-zero
    # Use the mask as an alpha channel for the red overlay
    mask_3ch = cv2.merge([mask_resized, mask_resized, mask_resized])
    mask_float = mask_3ch.astype(np.float32) / 255.0

    # Blend: where mask=1, show 60% red + 40% original blend
    result = (
        blended.astype(np.float32) * (1 - mask_float * 0.6)
        + red_overlay.astype(np.float32) * (mask_float * 0.6)
    ).astype(np.uint8)

    cv2.imwrite(output_path, result)
    return output_path


def generate_all_visualizations(
    color_a: np.ndarray,
    color_b: np.ndarray,
    diff_map: np.ndarray,
    mask: np.ndarray,
    regions: List[Dict],
    output_dir: str,
    session_id: str,
) -> Dict[str, str]:
    """
    Generate all four visualization outputs and save them to disk.

    Args:
        color_a: Resized color reference image.
        color_b: Resized + aligned color comparison image.
        diff_map: SSIM difference map.
        mask: Binary change mask.
        regions: List of detected change regions.
        output_dir: Directory to save output files.
        session_id: Unique identifier for this comparison session,
                    used as a filename prefix to avoid collisions.

    Returns:
        Dict mapping output type to file path:
        - 'highlighted': Bounding box visualization path.
        - 'heatmap': Heatmap overlay path.
        - 'side_by_side': Side-by-side composite path.
        - 'overlay': Red-highlighted overlay path.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate each visualization
    highlighted_path = draw_bounding_boxes(
        color_b, regions,
        os.path.join(output_dir, f"{session_id}_highlighted.png"),
    )

    heatmap_path = generate_heatmap(
        color_b, diff_map,
        os.path.join(output_dir, f"{session_id}_heatmap.png"),
    )

    side_by_side_path = generate_side_by_side(
        color_a, color_b,
        os.path.join(output_dir, f"{session_id}_side_by_side.png"),
    )

    overlay_path = generate_overlay(
        color_a, color_b, mask,
        os.path.join(output_dir, f"{session_id}_overlay.png"),
    )

    return {
        "highlighted": highlighted_path,
        "heatmap": heatmap_path,
        "side_by_side": side_by_side_path,
        "overlay": overlay_path,
    }
