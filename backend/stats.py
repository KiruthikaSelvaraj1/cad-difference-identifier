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


def _label_for_change_type(change_type: str | None) -> str:
    normalized = (change_type or "modification").lower().replace(" ", "_")
    mapping = {
        "addition": "additions",
        "removal": "removals",
        "modification": "modifications",
        "positional_shift": "positional_shifts",
    }
    return mapping.get(normalized, "modifications")


def compute_statistics(
    regions: List[Dict],
    image_shape: Tuple[int, int],
    text_changes: List[Dict] | None = None,
    has_ssim_signal: bool = False,
    has_absdiff_signal: bool = False,
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
    region_details = []
    change_breakdown = {
        "additions": 0,
        "removals": 0,
        "modifications": 0,
        "positional_shifts": 0,
    }
    max_region_severity = 0.0
    for r in regions:
        region_area_pct = (r["area"] / total_image_area * 100) if total_image_area else 0.0
        if region_area_pct > 5:
            severity = "critical"
        elif region_area_pct >= 1:
            severity = "moderate"
        else:
            severity = "minor"

        if severity == "critical":
            max_region_severity = max(max_region_severity, 100.0)
        elif severity == "moderate":
            max_region_severity = max(max_region_severity, 50.0)
        else:
            max_region_severity = max(max_region_severity, 10.0)

        change_type = r.get("change_type", "modification")
        change_breakdown[_label_for_change_type(change_type)] += 1

        region_details.append(
            {
                "bbox": r["bbox"],
                "area": r["area"],
                "location": r["location"],
                "severity": severity,
                "change_type": change_type,
            }
        )

    has_text_change = bool(text_changes)
    if percent_changed > 10 or has_text_change:
        change_severity = "major_revision"
    elif percent_changed >= 2:
        change_severity = "moderate_revision"
    else:
        change_severity = "minor_revision"

    signal_count = 0
    if has_ssim_signal:
        signal_count += 1
    if has_absdiff_signal:
        signal_count += 1
    if has_text_change:
        signal_count += 1

    if signal_count >= 3:
        confidence_score = 95.0
    elif signal_count == 2:
        confidence_score = 80.0
    elif signal_count == 1:
        confidence_score = 60.0
    else:
        confidence_score = 45.0

    baseline_region_count = max(3, 10)
    region_count_component = min(100.0, (len(regions) / baseline_region_count) * 100.0)
    percent_component = min(100.0, percent_changed * 1.2)
    text_component = 100.0 if has_text_change else 0.0
    severity_component = max_region_severity

    impact_score = round(
        (percent_component * 0.4)
        + (region_count_component * 0.2)
        + (text_component * 0.25)
        + (severity_component * 0.15),
        1,
    )
    impact_score = min(100.0, max(0.0, impact_score))

    if impact_score < 25.0:
        impact_label = "Low Impact"
    elif impact_score <= 60.0:
        impact_label = "Moderate Impact"
    else:
        impact_label = "High Impact"

    return {
        "region_count": len(regions),
        "percent_changed": percent_changed,
        "total_area_changed": total_area_changed,
        "regions": region_details,
        "change_severity": change_severity,
        "confidence_score": round(confidence_score, 1),
        "change_breakdown": change_breakdown,
        "impact_score": impact_score,
        "impact_label": impact_label,
    }
