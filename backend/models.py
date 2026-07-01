"""
Pydantic schemas for the CAD Review Studio application.

This module defines all data transfer objects (DTOs) used in the API response.
Pydantic models enforce strict type validation at serialization boundaries,
ensuring the frontend always receives well-structured, predictable JSON.
"""

from pydantic import BaseModel, Field
from typing import List


class TextChange(BaseModel):
    old_text: str = Field(..., description="Text present in the first image")
    new_text: str = Field(..., description="Text present in the second image")
    change_type: str = Field(..., description="One of modified, added, or removed")
    location: List[int] = Field(..., description="OCR bounding box [x, y, w, h]")


class RegionDetail(BaseModel):
    """
    Describes a single detected change region in the compared images.

    Attributes:
        bbox: Bounding box as [x, y, width, height] in pixel coordinates.
              (x, y) is the top-left corner of the rectangle.
        area: Total pixel area enclosed by the bounding box.
        location: Human-readable spatial classification of the region's centroid
                  (e.g., "top-left", "center", "bottom-right"). Determined by
                  dividing the image into a 3x3 grid.
    """
    bbox: List[int] = Field(
        ...,
        description="Bounding box [x, y, width, height] in pixels",
        min_length=4,
        max_length=4,
    )
    area: int = Field(..., description="Pixel area of the bounding box", ge=0)
    location: str = Field(
        ...,
        description="Spatial location classification (e.g., 'top-left', 'center')",
    )
    severity: str = Field(
        default="minor",
        description="Severity of the region: critical, moderate, or minor",
    )


class Statistics(BaseModel):
    """
    Aggregate statistics about the differences detected between two images.

    These metrics give a quantitative overview of what changed, how much
    changed, and where the changes are concentrated.

    Attributes:
        region_count: Number of distinct changed regions after noise filtering.
        percent_changed: Percentage of total image area affected by changes.
        total_area_changed: Sum of all bounding-box pixel areas.
        regions: Per-region detail list, sorted largest-to-smallest by area.
    """
    region_count: int = Field(..., description="Total changed region count", ge=0)
    percent_changed: float = Field(
        ...,
        description="Percentage of total image area changed",
        ge=0.0,
        le=100.0,
    )
    total_area_changed: int = Field(
        ...,
        description="Total pixel area covered by changes",
        ge=0,
    )
    regions: List[RegionDetail] = Field(
        default_factory=list,
        description="List of individual change regions",
    )
    change_severity: str = Field(
        default="minor_revision",
        description="Overall severity classification",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence score between 0 and 100",
    )


class CompareResponse(BaseModel):
    """
    Full API response for the POST /compare endpoint.

    Contains URLs to all generated visualization images, quantitative
    statistics, and a natural-language summary paragraph.

    Attributes:
        image_a_url: Served URL for the uploaded reference image.
        image_b_url: Served URL for the uploaded comparison image.
        diff_visualization_url: Side-by-side composite of both images.
        highlighted_regions_url: Image B with bounding boxes drawn on changes.
        heatmap_url: Heatmap overlay showing diff intensity.
        overlay_url: Blended overlay with changes highlighted in red.
        statistics: Quantitative change metrics.
        summary: Rule-based natural language summary paragraph.
    """
    image_a_url: str = Field(..., description="URL for the reference image")
    image_b_url: str = Field(..., description="URL for the comparison image")
    diff_visualization_url: str = Field(
        ..., description="URL for the side-by-side visualization"
    )
    highlighted_regions_url: str = Field(
        ..., description="URL for the bounding-box visualization"
    )
    heatmap_url: str = Field(..., description="URL for the heatmap overlay")
    overlay_url: str = Field(
        ..., description="URL for the blended overlay with changes in red"
    )
    statistics: Statistics = Field(..., description="Change statistics")
    summary: str = Field(..., description="Natural language change summary")
    difference_explanation: str = Field(
        ..., description="Plain-English explanation of how the images differ"
    )
    text_changes: List[TextChange] = Field(
        default_factory=list,
        description="OCR-based text and dimension changes",
    )
