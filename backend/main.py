"""
FastAPI application for AI-Based Image Difference Detection.

This is the main entry point for the backend server. It exposes a single
POST /compare endpoint that accepts two CAD drawing images, runs the full
diff pipeline (preprocess → detect → visualize → summarize), and returns
a structured JSON response with visualization URLs, statistics, and a
natural language summary.

The server also serves static files from the outputs/ directory so that
the frontend can display generated visualizations via URL.

Run with: uvicorn backend.main:app --reload --port 8000
"""

import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.models import CompareResponse, Statistics, RegionDetail
from backend.preprocessing import preprocess_pair
from backend.diff_engine import detect_differences
from backend.visualization import generate_all_visualizations
from backend.stats import compute_statistics
from backend.summarizer import generate_summary


# === Application Setup ===

app = FastAPI(
    title="Image Diff AI",
    description=(
        "AI-Based Image Difference Detection, Visualization, and "
        "Automated Change Summarization for CAD drawings."
    ),
    version="1.0.0",
)

# Enable CORS so the Streamlit frontend can call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where generated visualizations are saved and served
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the outputs directory as a static file server
# This allows the frontend to fetch visualization images via URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Allowed image MIME types — only JPG/JPEG and PNG
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

# Allowed file extensions as a fallback check
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _validate_upload(file: UploadFile, label: str) -> None:
    """
    Validate that an uploaded file is an acceptable image.

    Checks both the MIME content type (from the upload header) and the
    file extension. This dual check prevents both accidental uploads
    of wrong file types and deliberate MIME spoofing.

    Args:
        file: The uploaded file from the multipart request.
        label: Human-readable label ("Image A" or "Image B") for error messages.

    Raises:
        HTTPException: 400 if the file type is not JPG/JPEG/PNG.
    """
    # Check MIME type from the Content-Type header
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_CONTENT_TYPES:
        # Fallback: also check file extension
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{label}: Unsupported file type '{content_type}' "
                    f"(extension: '{ext}'). Only JPG, JPEG, and PNG files "
                    f"are accepted."
                ),
            )


def _read_image(file_bytes: bytes, label: str) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV image array.

    Validates that the image is not corrupted or unreadable by checking
    that cv2.imdecode returns a valid array.

    Args:
        file_bytes: Raw file content as bytes.
        label: Human-readable label for error messages.

    Returns:
        Decoded BGR image as a numpy array.

    Raises:
        HTTPException: 400 if the image cannot be decoded.
    """
    # Convert bytes to numpy array, then decode as a color image
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}: The uploaded file could not be read as an image. "
                f"The file may be corrupted or in an unsupported format."
            ),
        )

    return image


@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two CAD drawing images",
    description=(
        "Upload two CAD drawing images (Image A as reference, Image B as "
        "comparison). The system detects differences, generates visualizations, "
        "computes statistics, and returns a natural language summary."
    ),
)
async def compare_images(
    image_a: UploadFile = File(..., description="Reference CAD drawing (JPG/PNG)"),
    image_b: UploadFile = File(..., description="Comparison CAD drawing (JPG/PNG)"),
) -> CompareResponse:
    """
    Main comparison endpoint — orchestrates the full diff pipeline.

    Pipeline steps:
    1. Validate uploads (file type + readability).
    2. Preprocess (resize, align, grayscale, binarize).
    3. Detect differences (SSIM + absdiff + morphology + contours).
    4. Generate visualizations (bounding boxes, heatmap, side-by-side, overlay).
    5. Compute statistics (counts, percentages, per-region details).
    6. Generate natural language summary from statistics.
    7. Return structured JSON response with all URLs and data.

    Args:
        image_a: The reference/original CAD drawing upload.
        image_b: The comparison/modified CAD drawing upload.

    Returns:
        CompareResponse with visualization URLs, statistics, and summary.
    """
    # === Step 1: Validate uploads ===
    _validate_upload(image_a, "Image A")
    _validate_upload(image_b, "Image B")

    # Read file contents
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    # Decode images
    img_a = _read_image(bytes_a, "Image A")
    img_b = _read_image(bytes_b, "Image B")

    # Generate a unique session ID for output file naming
    session_id = str(uuid.uuid4())[:8]

    # Save uploaded originals to outputs for serving via URL
    orig_a_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_a.png")
    orig_b_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_b.png")
    cv2.imwrite(orig_a_path, img_a)
    cv2.imwrite(orig_b_path, img_b)

    # === Step 2: Preprocess ===
    color_a, color_b, gray_a, gray_b, bin_a, bin_b = preprocess_pair(img_a, img_b)

    # === Step 3: Detect differences ===
    ssim_score, diff_map, mask, regions = detect_differences(
        gray_a, gray_b, bin_a, bin_b
    )

    # === Step 4: Generate visualizations ===
    viz_paths = generate_all_visualizations(
        color_a, color_b, diff_map, mask, regions, OUTPUT_DIR, session_id
    )

    # === Step 5: Compute statistics ===
    stats = compute_statistics(regions, gray_a.shape[:2])

    # === Step 6: Generate summary ===
    summary = generate_summary(stats)

    # === Step 7: Build and return response ===
    # Convert file paths to URLs (relative to the static mount)
    base_url = "/outputs"

    response = CompareResponse(
        image_a_url=f"{base_url}/{session_id}_original_a.png",
        image_b_url=f"{base_url}/{session_id}_original_b.png",
        diff_visualization_url=f"{base_url}/{session_id}_side_by_side.png",
        highlighted_regions_url=f"{base_url}/{session_id}_highlighted.png",
        heatmap_url=f"{base_url}/{session_id}_heatmap.png",
        overlay_url=f"{base_url}/{session_id}_overlay.png",
        statistics=Statistics(
            region_count=stats["region_count"],
            percent_changed=stats["percent_changed"],
            total_area_changed=stats["total_area_changed"],
            regions=[
                RegionDetail(
                    bbox=r["bbox"],
                    area=r["area"],
                    location=r["location"],
                )
                for r in stats["regions"]
            ],
        ),
        summary=summary,
    )

    return response


@app.get("/", summary="Health check")
async def root():
    """
    Root endpoint — simple health check to verify the server is running.
    """
    return {
        "status": "ok",
        "service": "Image Diff AI",
        "version": "1.0.0",
        "docs": "/docs",
    }
