"""
Statistics computation module for CAD drawing difference analysis.

This module transforms raw contour data into structured, human-interpretable
metrics. It computes both aggregate statistics (total change count, percentage
of image affected) and per-region details (bounding box, area, spatial
location classification).

The output is designed to feed directly into:
1. The API response JSON (via Pydantic serialization).
2. The rule-based summarizer (which uses these metrics to generate
   natural language descriptions).
"""

from typing import List, Dict, Tuple


def compute_statistics(
    regions: List[Dict], image_shape: Tuple[int, int]
) -> Dict:
    """
    Compute aggregate and per-region statistics from detected change regions.

    Args:
        regions: List of region dicts from diff_engine.extract_changed_regions().
                 Each dict has keys: 'bbox' [x,y,w,h], 'area' (int),
                 'centroid' (x,y), 'location' (str).
        image_shape: (height, width) of the compared images, used to
                     calculate the percentage of total area changed.

    Returns:
        Dict with structure matching the Statistics Pydantic model:
        {
            "region_count": int,
            "percent_changed": float (rounded to 2 decimals),
            "total_area_changed": int,
            "regions": [
                {
                    "bbox": [x, y, w, h],
                    "area": int,
                    "location": str
                },
                ...
            ]
        }

    Note:
        - Regions are pre-sorted largest-to-smallest by area (done in
          diff_engine), so the output preserves that ordering.
        - percent_changed is computed as (total_changed_pixels / total_pixels) * 100.
        - If no regions are detected, all values are zero.
    """
    img_h, img_w = image_shape
    total_image_area = img_h * img_w

    # Sum up all region areas
    total_area_changed = sum(r["area"] for r in regions)

    # Calculate percentage of image that changed
    # Guard against division by zero (shouldn't happen with valid images)
    if total_image_area > 0:
        percent_changed = round(
            (total_area_changed / total_image_area) * 100, 2
        )
    else:
        percent_changed = 0.0

    # Build per-region detail list (strip internal fields like 'centroid')
    region_details = [
        {
            "bbox": r["bbox"],
            "area": r["area"],
            "location": r["location"],
        }
        for r in regions
    ]

    return {
        "region_count": len(regions),
        "percent_changed": percent_changed,
        "total_area_changed": total_area_changed,
        "regions": region_details,
    }
