"""
Rule-based natural language summary generator for CAD drawing differences.

This module generates grammatically correct, engineering-flavored summary
paragraphs from computed change statistics — with zero external API calls,
zero latency, zero cost, and full offline availability.

Design decision: Why rule-based instead of an LLM?
    While an LLM (e.g., GPT-4, Gemini) could generate more varied prose,
    it introduces multiple engineering risks for a CAD comparison tool:
    1. **Reliability** — API outages, rate limits, and network failures would
       break the summarization step. A rule-based system never fails.
    2. **Cost** — Per-token API costs accumulate quickly in high-volume
       industrial use (e.g., reviewing hundreds of drawing revisions).
    3. **Latency** — LLM calls add 1-5 seconds. Rule-based generation is
       sub-millisecond.
    4. **Privacy** — CAD drawings may contain proprietary engineering data.
       Sending descriptions to external APIs creates IP exposure risk.
    5. **Determinism** — For audit trails, having a predictable, reproducible
       summary from the same inputs is valuable.

    The rule-based approach uses sentence templates with random variation
    to avoid sounding robotic, while maintaining full control over output
    quality and factual accuracy.
"""

import random
from typing import List, Dict


def generate_summary(statistics: Dict) -> str:
    """
    Generate a 3-5 sentence natural language summary of detected changes.

    The summary follows a consistent structure:
    1. Opening: state total number of changed regions and overall result.
    2. Middle: describe the 1-3 largest regions by location and nature.
    3. Closing: state the overall percentage of the drawing affected.

    Sentence phrasing is varied using random.choice among 2-3 alternative
    templates per sentence type, so repeated runs produce natural variation
    without sacrificing clarity.

    Args:
        statistics: Dict with keys 'region_count', 'percent_changed',
                    'total_area_changed', 'regions' (list of dicts with
                    'bbox', 'area', 'location').

    Returns:
        A grammatically correct paragraph string (3-5 sentences).
    """
    region_count = statistics["region_count"]
    percent_changed = statistics["percent_changed"]
    regions = statistics["regions"]

    # === Handle the trivial case: no changes detected ===
    if region_count == 0:
        return (
            "No differences were detected between the two drawings. "
            "The images appear to be identical within the detection threshold. "
            "No modifications, additions, or removals of drawing elements were found."
        )

    sentences = []

    # === Opening sentence: count and high-level result ===
    opening_templates = [
        f"The comparison identified {_num_word(region_count)} changed region{'s' if region_count != 1 else ''} between the two drawings.",
        f"Analysis found {_num_word(region_count)} distinct change region{'s' if region_count != 1 else ''} across the two images.",
        f"The two images differ in {_num_word(region_count)} detected region{'s' if region_count != 1 else ''}.",
    ]
    sentences.append(random.choice(opening_templates))

    # === Describe the largest regions by location ===
    top_regions = regions[:3]  # Already sorted largest-first
    primary_change = top_regions[0]
    primary_location = _readable_location(primary_change["location"])
    primary_area = primary_change["area"]

    primary_bbox = _bbox_text(primary_change["bbox"])
    primary_change_type = (primary_change.get("change_type") or "modification").replace("_", " ")
    sentences.append(
        f"The largest difference is located in the {primary_location}, bounded by {primary_bbox}, "
        f"and appears to be a {primary_change_type} covering about {primary_area:,} pixels."
    )

    if len(top_regions) > 1:
        secondary_change = top_regions[1]
        secondary_location = _readable_location(secondary_change["location"])
        secondary_bbox = _bbox_text(secondary_change["bbox"])
        secondary_change_type = (secondary_change.get("change_type") or "modification").replace("_", " ")
        sentences.append(
            f"A second significant change appears in the {secondary_location}, bounded by {secondary_bbox}, "
            f"indicating another clearly localized {secondary_change_type}."
        )

    if len(top_regions) > 2:
        third_change = top_regions[2]
        third_location = _readable_location(third_change["location"])
        third_bbox = _bbox_text(third_change["bbox"])
        third_change_type = (third_change.get("change_type") or "modification").replace("_", " ")
        sentences.append(
            f"A further smaller change was detected in the {third_location}, bounded by {third_bbox}, "
            f"showing another {third_change_type} region within the comparison."
        )

    # === Mention the overall distribution of changes ===
    location_summary = _summarize_locations(regions)
    if location_summary:
        sentences.append(
            f"Detected changes are concentrated in the {location_summary}, "
            f"with clearly labeled bounding boxes marking each area of difference."
        )

    # === Closing sentence: severity and extent ===
    severity = _severity_label(percent_changed)
    sentences.append(
        f"Overall, these {severity} modifications affect approximately {percent_changed}% of the total drawing area."
    )

    return " ".join(sentences)


def generate_difference_explanation(statistics: Dict) -> str:
    """
    Create a concise, user-friendly explanation of how the images differ.

    The output highlights the number of detected regions, the main locations
    of the changes, and the overall severity in a readable format.
    """
    region_count = statistics.get("region_count", 0)
    regions = statistics.get("regions", [])
    percent_changed = statistics.get("percent_changed", 0.0)

    if region_count == 0:
        return (
            "No visible differences were detected between the two files. "
            "The drawings appear to be identical at the current analysis threshold."
        )

    top_regions = regions[:3]
    location_names = [region["location"] for region in top_regions]

    if len(location_names) == 1:
        location_text = location_names[0]
    elif len(location_names) == 2:
        location_text = f"{location_names[0]} and {location_names[1]}"
    else:
        location_text = f"{location_names[0]}, {location_names[1]}, and {location_names[2]}"

    primary_bbox = _bbox_text(top_regions[0]["bbox"]) if top_regions else "(x=0, y=0, w=0, h=0)"
    return (
        f"Detected {region_count} distinct change area(s). "
        f"The main differences are concentrated around {location_text}. "
        f"The largest region is outlined at {primary_bbox}, and the overall change level is about {percent_changed:.1f}% of the drawing area."
    )


def _readable_location(location: str) -> str:
    """
    Convert location keys like "center-right" into a more natural phrase.
    """
    return location.replace('-', ' ')


def _unified_location(location: str) -> str:
    """
    Convert location strings into consistent, readable phrases.
    """
    mapping = {
        'top-left': 'upper left',
        'top-center': 'top center',
        'top-right': 'upper right',
        'center-left': 'middle left',
        'center': 'center',
        'center-right': 'middle right',
        'bottom-left': 'lower left',
        'bottom-center': 'bottom center',
        'bottom-right': 'lower right',
    }
    return mapping.get(location, _readable_location(location))


def _summarize_locations(regions: List[Dict]) -> str:
    """
    Build a short human-readable summary of the most common locations.
    """
    location_counts = {}
    for region in regions:
        loc = _readable_location(region["location"])
        location_counts[loc] = location_counts.get(loc, 0) + 1

    # Sort locations by frequency, then return the top 2 locations
    sorted_locations = sorted(
        location_counts.items(), key=lambda item: item[1], reverse=True
    )
    if not sorted_locations:
        return ""

    top_locations = [loc for loc, _ in sorted_locations[:2]]
    if len(top_locations) == 1:
        return top_locations[0]

    return f"{top_locations[0]} and {top_locations[1]}"


def _bbox_text(bbox: List[int]) -> str:
    """
    Convert a bounding box list to a human-readable coordinate string.
    """
    x, y, w, h = bbox
    return f"(x={x}, y={y}, w={w}, h={h})"


def _severity_label(percent_changed: float) -> str:
    """
    Choose a severity descriptor based on the percentage of area changed.
    """
    if percent_changed < 2.0:
        return "minor"
    if percent_changed < 10.0:
        return "moderate"
    return "significant"


def _num_word(n: int) -> str:
    """
    Convert small integers to their English word form for more natural prose.

    Numbers above 10 are returned as digits, since word forms for large
    numbers become unwieldy ("one hundred and forty-seven").

    Args:
        n: Non-negative integer.

    Returns:
        English word (e.g., "three") for n <= 10, or digit string for n > 10.
    """
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten",
    }
    return words.get(n, str(n))
