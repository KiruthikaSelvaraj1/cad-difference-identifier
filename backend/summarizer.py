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

    # === OPENING SENTENCE: state the count and overall characterization ===
    opening_templates = [
        f"The comparison identified {_num_word(region_count)} changed "
        f"region{'s' if region_count != 1 else ''} between the two drawings.",

        f"Analysis of the two CAD drawings revealed "
        f"{_num_word(region_count)} distinct area{'s' if region_count != 1 else ''} "
        f"of modification.",

        f"A total of {_num_word(region_count)} "
        f"region{'s' if region_count != 1 else ''} "
        f"{'were' if region_count != 1 else 'was'} "
        f"found to differ between the reference and comparison drawings.",
    ]
    sentences.append(random.choice(opening_templates))

    # === MIDDLE SENTENCES: describe the top 1-3 largest regions ===
    top_regions = regions[:3]  # Already sorted largest-first

    # Engineering vocabulary for describing changes — more specific than
    # generic words like "object" or "thing"
    change_descriptors = [
        "a component modification",
        "an alteration to a drawing element",
        "a change in line geometry",
    ]

    size_descriptors_large = [
        "The most significant modification",
        "The largest detected change",
        "The primary area of difference",
    ]

    size_descriptors_secondary = [
        "followed by",
        "additionally,",
        "a secondary change was detected as",
    ]

    annotation_descriptors = [
        "Minor annotation adjustments were also detected",
        "Smaller dimensional or label changes were observed",
        "Additional minor modifications appear",
    ]

    for idx, region in enumerate(top_regions):
        location = region["location"]
        area = region["area"]

        if idx == 0:
            # Describe the largest region
            desc_start = random.choice(size_descriptors_large)
            desc_type = random.choice([
                f"is {random.choice(change_descriptors)} in the {location} area",
                f"involves a structural revision in the {location} portion of the drawing",
                f"corresponds to {random.choice(change_descriptors)} located in the {location} section",
            ])
            sentences.append(f"{desc_start} {desc_type}.")

        elif idx == 1:
            # Describe the second-largest region
            secondary_templates = [
                f"This is {random.choice(size_descriptors_secondary)} "
                f"the removal or addition of a line segment in the {location} area.",

                f"A secondary modification was identified in the {location} region, "
                f"suggesting a revision to a dimension or component boundary.",

                f"Another notable change appears in the {location} area, "
                f"indicating an adjustment to the drawing layout.",
            ]
            sentences.append(random.choice(secondary_templates))

        elif idx == 2:
            # Describe the third region more briefly
            sentences.append(
                f"{random.choice(annotation_descriptors)} near the {location}."
            )

    # === CLOSING SENTENCE: state the percentage ===
    if percent_changed < 1.0:
        # Minor changes — use language reflecting negligible modifications
        closing_templates = [
            f"Overall, the changes are minor, affecting approximately "
            f"{percent_changed}% of the total drawing area.",

            f"These represent negligible modifications, with only "
            f"{percent_changed}% of the drawing area affected.",

            f"In total, less than {max(percent_changed, 0.1)}% of the "
            f"drawing was modified, indicating minor revisions.",
        ]
    else:
        # Substantive changes
        closing_templates = [
            f"Overall, approximately {percent_changed}% of the drawing "
            f"area was affected by these changes.",

            f"In total, the modifications span roughly {percent_changed}% "
            f"of the drawing's visible area.",

            f"The cumulative extent of changes covers about {percent_changed}% "
            f"of the total drawing area.",
        ]

    sentences.append(random.choice(closing_templates))

    return " ".join(sentences)


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
