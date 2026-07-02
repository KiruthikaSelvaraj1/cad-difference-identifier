"""Generate analytics charts for change distribution by region."""

import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANGE_TYPE_COLORS = {
    "addition": "#10B981",
    "removal": "#EF4444",
    "modification": "#F59E0B",
    "positional_shift": "#6366F1",
}


def generate_analytics_chart(regions: List[Dict], output_path: str) -> str:
    """Create a horizontal bar chart showing region areas by change type."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ordered_regions = sorted(regions, key=lambda item: item.get("area", 0), reverse=True)
    labels = [f"#{index}" for index in range(1, len(ordered_regions) + 1)]
    areas = [max(int(region.get("area", 0)), 0) for region in ordered_regions]
    colors = [
        CHANGE_TYPE_COLORS.get(
            str(region.get("change_type", "modification")).lower().replace(" ", "_"),
            CHANGE_TYPE_COLORS["modification"],
        )
        for region in ordered_regions
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(
        figsize=(8.0, max(2.6, 0.48 * max(len(ordered_regions), 1) + 1.0)),
        dpi=140,
    )

    y_positions = np.arange(len(ordered_regions))
    ax.barh(y_positions, areas, color=colors, edgecolor="#E5E7EB", height=0.72)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Region area (pixels)", fontsize=10)
    ax.set_title("Change Distribution by Region", fontsize=12, pad=8)
    ax.xaxis.grid(True, alpha=0.25)
    ax.yaxis.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#E5E7EB")

    if not ordered_regions:
        ax.text(0.5, 0.5, "No significant regions detected", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return output_path
