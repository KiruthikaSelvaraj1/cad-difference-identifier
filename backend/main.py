"""
FastAPI application for CAD Review Studio.

This is the main entry point for the backend server. It exposes a single
POST /compare endpoint that accepts two CAD drawing files (images or PDFs),
runs the full diff pipeline (preprocess -> detect -> visualize -> summarize),
and returns a structured JSON response with visualization URLs, statistics,
and a natural language summary.

The server also:
- Serves static files from the outputs/ directory for visualization images.
- Serves a built-in HTML frontend at the root URL (no Streamlit needed).

Supported input formats:
- Images: JPG, JPEG, PNG
- Documents: PDF (first page is extracted and converted to an image)

Run with: uvicorn backend.main:app --reload --port 8000
"""

import os
import uuid
import cv2
import numpy as np
import fitz  # PyMuPDF — for converting PDF pages to images
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

from backend.models import CompareResponse, Statistics, RegionDetail
from backend.preprocessing import preprocess_pair
from backend.diff_engine import detect_differences
from backend.visualization import generate_all_visualizations
from backend.stats import compute_statistics
from backend.summarizer import generate_summary, generate_difference_explanation
from backend.text_diff import detect_text_changes
from backend.report_generator import ReportGenerator
from backend.analytics_chart import generate_analytics_chart


# === Application Setup ===

app = FastAPI(
    title="CAD Review Studio",
    description=(
        "Professional CAD revision review for detecting visual, textual, "
        "and dimensional changes across drawing versions. "
        "Accepts JPG, PNG, and PDF inputs."
    ),
    version="1.0.0",
)

# Enable CORS so any frontend can call the API
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
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_GENERATOR = ReportGenerator(REPORT_DIR)

# Mount the outputs directory as a static file server
# This allows the frontend to fetch visualization images via URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Allowed MIME types — images + PDF
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/pdf",
}

# Allowed file extensions as a fallback check
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


def _validate_upload(file: UploadFile, label: str) -> None:
    """
    Validate that an uploaded file is an acceptable image or PDF.

    Checks both the MIME content type (from the upload header) and the
    file extension. This dual check prevents both accidental uploads
    of wrong file types and deliberate MIME spoofing.

    Args:
        file: The uploaded file from the multipart request.
        label: Human-readable label ("Image A" or "Image B") for error messages.

    Raises:
        HTTPException: 400 if the file type is not JPG/JPEG/PNG/PDF.
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
                    f"(extension: '{ext}'). Accepted formats: "
                    f"JPG, JPEG, PNG, and PDF."
                ),
            )


def _is_pdf(file_bytes: bytes, filename: str) -> bool:
    """
    Check if the uploaded file is a PDF.

    Uses both the file header (magic bytes) and the filename extension
    for reliable detection.

    Args:
        file_bytes: Raw file content.
        filename: Original filename from the upload.

    Returns:
        True if the file is a PDF, False otherwise.
    """
    # PDF files start with '%PDF' magic bytes
    if file_bytes[:4] == b'%PDF':
        return True
    # Fallback: check file extension
    ext = os.path.splitext(filename or "")[1].lower()
    return ext == ".pdf"


def _pdf_to_image(file_bytes: bytes, label: str) -> np.ndarray:
    """
    Convert the first page of a PDF to an OpenCV image.

    Uses PyMuPDF (fitz) to render the first page of the PDF at high
    resolution (300 DPI) and convert it to a BGR numpy array suitable
    for OpenCV processing.

    CAD drawings are frequently exported as PDFs, so this conversion
    step is essential for a practical comparison tool. Only the first
    page is used — multi-page PDFs are treated as single-page drawings.

    Args:
        file_bytes: Raw PDF file content.
        label: Human-readable label for error messages.

    Returns:
        Decoded BGR image as a numpy array.

    Raises:
        HTTPException: 400 if the PDF cannot be read or rendered.
    """
    try:
        # Open PDF from memory bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        if doc.page_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"{label}: The PDF file contains no pages.",
            )

        # Render the first page at 300 DPI for high-quality conversion
        # Default PDF resolution is 72 DPI, so zoom factor = 300/72 ≈ 4.17
        page = doc[0]
        zoom = 300 / 72  # 300 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert PyMuPDF pixmap to numpy array
        # pix.samples is raw pixel bytes in RGB format
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # Convert RGB to BGR for OpenCV compatibility
        if pix.n == 4:
            # RGBA → BGR (drop alpha, convert color order)
            image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            # RGB → BGR
            image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale → BGR
            image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        doc.close()
        return image

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}: Failed to convert PDF to image. "
                f"The file may be corrupted or password-protected. "
                f"Error: {str(e)}"
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


def _load_file(file_bytes: bytes, filename: str, label: str) -> np.ndarray:
    """
    Load an uploaded file as an OpenCV image, handling both images and PDFs.

    This is the unified entry point for file loading. It detects whether
    the file is a PDF (based on magic bytes and extension) and routes
    to the appropriate conversion function.

    Args:
        file_bytes: Raw file content.
        filename: Original filename from the upload.
        label: Human-readable label for error messages.

    Returns:
        Decoded BGR image as a numpy array.
    """
    if _is_pdf(file_bytes, filename):
        return _pdf_to_image(file_bytes, label)
    else:
        return _read_image(file_bytes, label)


@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two CAD drawings",
    description=(
        "Upload two CAD drawing files — images (JPG/PNG) or PDFs. "
        "The system detects differences, generates visualizations, "
        "computes statistics, and returns a natural language summary."
    ),
)
async def compare_images(
    image_a: UploadFile = File(..., description="Reference CAD drawing (JPG/PNG/PDF)"),
    image_b: UploadFile = File(..., description="Comparison CAD drawing (JPG/PNG/PDF)"),
) -> CompareResponse:
    """
    Main comparison endpoint — orchestrates the full diff pipeline.

    Pipeline steps:
    1. Validate uploads (file type + readability).
    2. Load files (auto-detects PDF vs image, converts PDF to image).
    3. Preprocess (resize, align, grayscale, binarize).
    4. Detect differences (SSIM + absdiff + morphology + contours).
    5. Generate visualizations (bounding boxes, heatmap, side-by-side, overlay).
    6. Compute statistics (counts, percentages, per-region details).
    7. Generate natural language summary from statistics.
    8. Return structured JSON response with all URLs and data.

    Args:
        image_a: The reference/original CAD drawing upload (JPG/PNG/PDF).
        image_b: The comparison/modified CAD drawing upload (JPG/PNG/PDF).

    Returns:
        CompareResponse with visualization URLs, statistics, and summary.
    """
    # === Step 1: Validate uploads ===
    _validate_upload(image_a, "Image A")
    _validate_upload(image_b, "Image B")

    # Read file contents
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    # === Step 2: Load files (handles both images and PDFs) ===
    img_a = _load_file(bytes_a, image_a.filename, "Image A")
    img_b = _load_file(bytes_b, image_b.filename, "Image B")

    # Generate a unique session ID for output file naming
    session_id = str(uuid.uuid4())[:8]

    # Save uploaded originals to outputs for serving via URL
    orig_a_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_a.png")
    orig_b_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_b.png")
    cv2.imwrite(orig_a_path, img_a)
    cv2.imwrite(orig_b_path, img_b)

    # === Step 3: Preprocess ===
    color_a, color_b, gray_a, gray_b, bin_a, bin_b = preprocess_pair(img_a, img_b)

    # === Step 4: Detect differences ===
    ssim_score, diff_map, mask, regions = detect_differences(
        gray_a, gray_b, bin_a, bin_b
    )
    text_changes = detect_text_changes(img_a, img_b)

    # === Step 5: Generate visualizations ===
    viz_paths = generate_all_visualizations(
        color_a, color_b, diff_map, mask, regions, OUTPUT_DIR, session_id
    )

    # === Step 6: Compute statistics ===
    stats = compute_statistics(
        regions,
        gray_a.shape[:2],
        text_changes=text_changes,
        has_ssim_signal=bool(np.count_nonzero(mask)),
        has_absdiff_signal=bool(np.count_nonzero(mask)),
    )

    # === Step 7: Generate summary ===
    summary = generate_summary(stats)
    difference_explanation = generate_difference_explanation(stats)

    analytics_chart_path = os.path.join(OUTPUT_DIR, f"{session_id}_analytics.png")
    generate_analytics_chart(
        [
            {
                **region,
                "change_type": region.get("change_type", "modification"),
            }
            for region in regions
        ],
        analytics_chart_path,
    )

    # === Step 8: Build and return response ===
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
                    severity=r.get("severity", "minor"),
                    change_type=r.get("change_type", "modification"),
                )
                for r in stats["regions"]
            ],
            change_severity=stats["change_severity"],
            confidence_score=stats["confidence_score"],
            change_breakdown=stats.get("change_breakdown", {}),
            impact_score=stats.get("impact_score", 0.0),
            impact_label=stats.get("impact_label", "Low Impact"),
        ),
        summary=summary,
        difference_explanation=difference_explanation,
        text_changes=text_changes,
        analytics_chart_url=f"{base_url}/{session_id}_analytics.png",
    )

    return response


@app.post(
    "/compare/report",
    summary="Generate a downloadable PDF report",
    description="Runs the full comparison and returns a PDF report for download.",
)
async def compare_and_download_report(
    image_a: UploadFile = File(..., description="Reference CAD drawing (JPG/PNG/PDF)"),
    image_b: UploadFile = File(..., description="Comparison CAD drawing (JPG/PNG/PDF)"),
) -> FileResponse:
    _validate_upload(image_a, "Image A")
    _validate_upload(image_b, "Image B")

    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    img_a = _load_file(bytes_a, image_a.filename, "Image A")
    img_b = _load_file(bytes_b, image_b.filename, "Image B")

    session_id = str(uuid.uuid4())[:8]
    orig_a_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_a.png")
    orig_b_path = os.path.join(OUTPUT_DIR, f"{session_id}_original_b.png")
    cv2.imwrite(orig_a_path, img_a)
    cv2.imwrite(orig_b_path, img_b)

    color_a, color_b, gray_a, gray_b, bin_a, bin_b = preprocess_pair(img_a, img_b)
    _, diff_map, mask, regions = detect_differences(gray_a, gray_b, bin_a, bin_b)
    text_changes = detect_text_changes(img_a, img_b)
    viz_paths = generate_all_visualizations(color_a, color_b, diff_map, mask, regions, OUTPUT_DIR, session_id)
    stats = compute_statistics(
        regions,
        gray_a.shape[:2],
        text_changes=text_changes,
        has_ssim_signal=bool(np.count_nonzero(mask)),
        has_absdiff_signal=bool(np.count_nonzero(mask)),
    )
    summary = generate_summary(stats)

    with open(viz_paths["highlighted"], "rb") as fh:
        highlighted_bytes = fh.read()
    with open(viz_paths["heatmap"], "rb") as fh:
        heatmap_bytes = fh.read()
    with open(orig_a_path, "rb") as fh:
        image_a_bytes = fh.read()
    with open(orig_b_path, "rb") as fh:
        image_b_bytes = fh.read()

    report_path = REPORT_GENERATOR.generate_report(
        summary=summary,
        statistics=stats,
        text_changes=text_changes,
        image_a=image_a_bytes,
        image_b=image_b_bytes,
        highlighted_image=highlighted_bytes,
        heatmap_image=heatmap_bytes,
    )

    return FileResponse(report_path, filename=os.path.basename(report_path), media_type="application/pdf")


# === HTML Frontend ===
# Serves a built-in web UI at the root URL — no Streamlit or separate server needed.
# Just open http://localhost:8000 in your browser.

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAD Review Studio — CAD Drawing Comparison</title>
    <meta name="description" content="Professional CAD revision review for visual, textual, and dimensional change detection across drawing versions">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0f1117;
            --bg-secondary: #1a1d29;
            --bg-card: #1e2130;
            --bg-card-hover: #252840;
            --border: #2d3148;
            --text-primary: #e8eaf0;
            --text-secondary: #9ca3b8;
            --text-muted: #6b7394;
            --accent: #6c63ff;
            --accent-hover: #7f78ff;
            --accent-glow: rgba(108, 99, 255, 0.25);
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --gradient-1: linear-gradient(135deg, #6c63ff 0%, #3b82f6 100%);
            --gradient-2: linear-gradient(135deg, #1e2130 0%, #252840 100%);
            --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 8px 48px rgba(0, 0, 0, 0.4);
            --radius: 12px;
            --radius-lg: 16px;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(108, 99, 255, 0.2), transparent 30%),
                linear-gradient(135deg, var(--bg-primary) 0%, #11151f 100%);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

        /* Header */
        header {
            text-align: center;
            padding: 48px 0 32px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 40px;
        }
        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        header p { color: var(--text-secondary); font-size: 1.05rem; font-weight: 300; }

        /* Upload Section */
        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }
        .upload-card {
            background: var(--bg-card);
            border: 2px dashed var(--border);
            border-radius: var(--radius-lg);
            padding: 32px 24px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .upload-card:hover, .upload-card.dragover {
            border-color: var(--accent);
            background: var(--bg-card-hover);
            box-shadow: 0 0 20px var(--accent-glow);
        }
        .upload-card.has-file { border-style: solid; border-color: var(--success); }
        .upload-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
        .upload-card .subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 16px; }
        .upload-card .formats { color: var(--text-muted); font-size: 0.75rem; margin-top: 12px; }
        .upload-card input[type="file"] {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            opacity: 0; cursor: pointer;
        }
        .upload-icon { font-size: 2.5rem; margin-bottom: 12px; }
        .file-name {
            color: var(--success); font-weight: 500; font-size: 0.9rem;
            margin-top: 8px; word-break: break-all;
        }
        .preview-img {
            max-width: 100%; max-height: 200px; margin-top: 16px;
            border-radius: 8px; border: 1px solid var(--border);
        }

        /* Compare Button */
        .btn-compare {
            display: block; width: 100%; padding: 16px 32px;
            background: var(--gradient-1); color: white;
            border: none; border-radius: var(--radius);
            font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 600;
            cursor: pointer; transition: all 0.3s ease;
            box-shadow: 0 4px 16px var(--accent-glow);
            margin-bottom: 40px;
        }
        .btn-compare:hover { transform: translateY(-2px); box-shadow: 0 6px 24px var(--accent-glow); }
        .btn-compare:disabled {
            opacity: 0.4; cursor: not-allowed;
            transform: none; box-shadow: none;
        }
        .btn-compare.loading { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }

        /* Results Section */
        #results { display: none; }
        .section-title {
            font-size: 1.4rem; font-weight: 600; margin-bottom: 20px;
            display: flex; align-items: center; gap: 10px;
        }

        /* Summary Box */
        .summary-box {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-left: 4px solid #3b82f6;
            border-radius: var(--radius);
            padding: 24px 28px;
            margin-bottom: 32px;
            font-size: 1.05rem;
            line-height: 1.8;
            color: var(--text-primary);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }

        /* Bounding Box Grid */
        .bbox-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }
        .bbox-card {
            background: linear-gradient(135deg, rgba(37, 40, 64, 0.95) 0%, rgba(24, 28, 47, 0.95) 100%);
            border: 1px solid rgba(108, 99, 255, 0.28);
            border-radius: var(--radius);
            padding: 16px 18px;
            box-shadow: var(--shadow);
        }
        .bbox-card .bbox-title {
            color: #8b9bff;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .bbox-card .bbox-coords {
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: 8px;
        }
        .bbox-card .bbox-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            flex-wrap: wrap;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 16px; margin-bottom: 28px;
        }
        .stat-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 20px; text-align: center;
        }
        .stat-card .value {
            font-size: 2rem; font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stat-card .label { color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px; }

        /* Region Table */
        .table-wrap {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); overflow: hidden; margin-bottom: 32px;
        }
        table { width: 100%; border-collapse: collapse; }
        th {
            background: var(--bg-secondary); padding: 12px 16px;
            text-align: left; font-weight: 600; font-size: 0.85rem;
            color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;
        }
        td { padding: 12px 16px; border-top: 1px solid var(--border); font-size: 0.9rem; }
        tr:hover td { background: var(--bg-card-hover); }
        .location-badge {
            background: var(--accent-glow); color: var(--accent);
            padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;
        }

        /* Visualization Grid */
        .viz-grid { display: grid; grid-template-columns: 1fr; gap: 24px; margin-bottom: 32px; }
        .viz-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .viz-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: var(--radius); overflow: hidden;
        }
        .viz-card h4 {
            padding: 14px 18px; font-size: 0.9rem; font-weight: 600;
            border-bottom: 1px solid var(--border);
        }
        .viz-card img { width: 100%; display: block; }

        /* Divider */
        .divider { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

        /* Footer */
        footer {
            text-align: center; padding: 32px 0; color: var(--text-muted);
            font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .upload-grid, .stats-grid, .viz-pair {
                grid-template-columns: 1fr;
            }
            header h1 { font-size: 1.6rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CAD Review Studio</h1>
            <p>Professional CAD revision review with clear change highlights and report-ready insights</p>
        </header>

        <!-- Upload Section -->
        <div class="upload-grid">
            <div class="upload-card" id="card-a" ondragover="handleDragOver(event, 'card-a')" ondragleave="handleDragLeave(event, 'card-a')" ondrop="handleDrop(event, 'file-a', 'card-a')">
                <div class="upload-icon">📄</div>
                <h3>Image A (Reference)</h3>
                <p class="subtitle">Upload the original / reference drawing</p>
                <div id="preview-a"></div>
                <p class="formats">JPG, JPEG, PNG, or PDF</p>
                <input type="file" id="file-a" accept=".jpg,.jpeg,.png,.pdf" onchange="handleFileSelect(this, 'card-a', 'preview-a')">
            </div>
            <div class="upload-card" id="card-b" ondragover="handleDragOver(event, 'card-b')" ondragleave="handleDragLeave(event, 'card-b')" ondrop="handleDrop(event, 'file-b', 'card-b')">
                <div class="upload-icon">📄</div>
                <h3>Image B (Comparison)</h3>
                <p class="subtitle">Upload the modified / comparison drawing</p>
                <div id="preview-b"></div>
                <p class="formats">JPG, JPEG, PNG, or PDF</p>
                <input type="file" id="file-b" accept=".jpg,.jpeg,.png,.pdf" onchange="handleFileSelect(this, 'card-b', 'preview-b')">
            </div>
        </div>

        <button class="btn-compare" id="btn-compare" disabled onclick="compareImages()">
            Compare Drawings
        </button>

        <!-- Results Section -->
        <div id="results">
            <hr class="divider">

            <div class="section-title">AI Change Summary</div>
            <div class="summary-box" id="summary-text"></div>

            <div class="section-title">Change Statistics</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="value" id="stat-regions">0</div>
                    <div class="label">Changed Regions</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="stat-percent">0%</div>
                    <div class="label">Area Changed</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="stat-pixels">0</div>
                    <div class="label">Total Pixels Changed</div>
                </div>
            </div>

            <div class="section-title">Detected Bounding Boxes</div>
            <div class="bbox-grid" id="bbox-grid"></div>

            <div id="region-table-wrap"></div>

            <hr class="divider">
            <div class="section-title">Visualizations</div>

            <div class="viz-grid">
                <div class="viz-card">
                    <h4>Side-by-Side Comparison</h4>
                    <img id="viz-sidebyside" src="" alt="Side-by-side comparison">
                </div>
                <div class="viz-pair">
                    <div class="viz-card">
                        <h4>Highlighted Change Regions</h4>
                        <img id="viz-highlighted" src="" alt="Highlighted regions">
                    </div>
                    <div class="viz-card">
                        <h4>Heatmap Overlay</h4>
                        <img id="viz-heatmap" src="" alt="Heatmap overlay">
                    </div>
                </div>
                <div class="viz-card">
                    <h4>Overlay Blend (Changes in Red)</h4>
                    <img id="viz-overlay" src="" alt="Overlay blend">
                </div>
            </div>
        </div>

        <footer>
            CAD Review Studio &mdash; Professional CAD revision review with clear change highlights and report-ready insights &mdash; Runs fully offline, zero API cost
        </footer>
    </div>

    <script>
        // === File Upload Handling ===
        function handleFileSelect(input, cardId, previewId) {
            const card = document.getElementById(cardId);
            const preview = document.getElementById(previewId);
            if (input.files && input.files[0]) {
                const file = input.files[0];
                card.classList.add('has-file');
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        preview.innerHTML = '<div class="file-name">' + file.name + '</div>' +
                            '<img class="preview-img" src="' + e.target.result + '">';
                    };
                    reader.readAsDataURL(file);
                } else {
                    preview.innerHTML = '<div class="file-name">' + file.name + ' (PDF)</div>';
                }
            }
            updateCompareButton();
        }

        function handleDragOver(e, cardId) {
            e.preventDefault();
            document.getElementById(cardId).classList.add('dragover');
        }
        function handleDragLeave(e, cardId) {
            document.getElementById(cardId).classList.remove('dragover');
        }
        function handleDrop(e, inputId, cardId) {
            e.preventDefault();
            document.getElementById(cardId).classList.remove('dragover');
            const input = document.getElementById(inputId);
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }

        function updateCompareButton() {
            const a = document.getElementById('file-a').files.length > 0;
            const b = document.getElementById('file-b').files.length > 0;
            document.getElementById('btn-compare').disabled = !(a && b);
        }

        // === Compare API Call ===
        async function compareImages() {
            const btn = document.getElementById('btn-compare');
            const fileA = document.getElementById('file-a').files[0];
            const fileB = document.getElementById('file-b').files[0];

            if (!fileA || !fileB) return;

            btn.disabled = true;
            btn.textContent = 'Analyzing differences...';
            btn.classList.add('loading');
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('image_a', fileA);
            formData.append('image_b', fileB);

            try {
                const resp = await fetch('/compare', { method: 'POST', body: formData });

                if (!resp.ok) {
                    const err = await resp.json();
                    alert('Error: ' + (err.detail || 'Unknown error'));
                    return;
                }

                const data = await resp.json();
                displayResults(data);
            } catch (e) {
                alert('Connection error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Compare Drawings';
                btn.classList.remove('loading');
            }
        }

        // === Display Results ===
        function displayResults(data) {
            const stats = data.statistics;

            // Summary
            document.getElementById('summary-text').textContent = data.summary;

            // Stats
            document.getElementById('stat-regions').textContent = stats.region_count;
            document.getElementById('stat-percent').textContent = stats.percent_changed + '%';
            document.getElementById('stat-pixels').textContent = stats.total_area_changed.toLocaleString();

            // Bounding boxes
            const bboxGrid = document.getElementById('bbox-grid');
            if (stats.regions.length > 0) {
                let html = '';
                stats.regions.forEach((r, i) => {
                    const [x, y, w, h] = r.bbox;
                    html += '<div class="bbox-card">' +
                        '<div class="bbox-title">Bounding Box ' + (i + 1) + '</div>' +
                        '<div class="bbox-coords">x: ' + x + ', y: ' + y + ', w: ' + w + ', h: ' + h + '</div>' +
                        '<div class="bbox-meta">' +
                        '<span class="location-badge">' + r.location + '</span>' +
                        '<span>Area: ' + r.area.toLocaleString() + ' px</span>' +
                        '</div>' +
                        '</div>';
                });
                bboxGrid.innerHTML = html;
            } else {
                bboxGrid.innerHTML = '';
            }

            // Region table
            const tableWrap = document.getElementById('region-table-wrap');
            if (stats.regions.length > 0) {
                let html = '<div class="table-wrap"><table><thead><tr>' +
                    '<th>#</th><th>Location</th><th>Area (px)</th><th>Bounding Box</th>' +
                    '</tr></thead><tbody>';
                stats.regions.forEach((r, i) => {
                    html += '<tr><td>' + (i + 1) + '</td>' +
                        '<td><span class="location-badge">' + r.location + '</span></td>' +
                        '<td>' + r.area.toLocaleString() + '</td>' +
                        '<td>(' + r.bbox.join(', ') + ')</td></tr>';
                });
                html += '</tbody></table></div>';
                tableWrap.innerHTML = html;
            } else {
                tableWrap.innerHTML = '';
            }

            // Visualizations
            document.getElementById('viz-sidebyside').src = data.diff_visualization_url;
            document.getElementById('viz-highlighted').src = data.highlighted_regions_url;
            document.getElementById('viz-heatmap').src = data.heatmap_url;
            document.getElementById('viz-overlay').src = data.overlay_url;

            // Show results
            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


@app.get("/health", summary="Service health check")
async def health() -> dict:
    """Return a simple status payload for health checks."""
    return {"status": "ok", "service": "cad-review-studio"}


@app.get("/", response_class=HTMLResponse, summary="Web UI")
async def root():
    """
    Serve the built-in HTML frontend.

    No Streamlit or separate frontend server needed — just open
    http://localhost:8000 in your browser.
    """
    return HTML_PAGE
