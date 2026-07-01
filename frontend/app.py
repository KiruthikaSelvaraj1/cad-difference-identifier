"""
Streamlit frontend for AI-Based Image Difference Detection.

This provides a user-friendly web interface for uploading two CAD drawing
images, sending them to the FastAPI backend for comparison, and displaying
the results including visualizations, statistics, and the AI-generated
change summary.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
from PIL import Image
import io


st.set_page_config(
    page_title="Image Diff AI — CAD Drawing Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = "http://localhost:8000"

st.markdown(
    """
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111c31;
            --panel-2: #17253f;
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #818cf8;
            --success: #34d399;
            --danger: #fb7185;
        }
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111c31 50%, #17253f 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #f8fafc !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            font-weight: 700;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
        }
        .stTextInput > div > div > input,
        .stFileUploader > div {
            background-color: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 10px;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .card {
            background: linear-gradient(135deg, rgba(23,37,63,0.95), rgba(17,28,49,0.95));
            border: 1px solid rgba(129,140,248,0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(2,6,23,0.35);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
            border: 1px solid rgba(56,189,248,0.15);
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .section-title {
            color: #8b5cf6;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            letter-spacing: 0.03em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """
    Main Streamlit application function.

    Renders the upload interface, sends images to the backend,
    and displays all comparison results.
    """
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0 1.2rem 0;">
            <h1 style="margin-bottom: 0.2rem; font-size: 2rem;">🔍 Image Diff AI</h1>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 900px; margin: 0 auto;">
                AI-Based Image Difference Detection, Visualization, and Automated Change Summarization for CAD Drawings.
                This system compares two versions of an image, highlights changed regions, and generates a human-readable summary.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style='color:#cbd5e1; font-size:0.95rem; line-height:1.8;'>
            Comparing two versions of an image manually is time-consuming and prone to human error,
            especially when the differences are subtle or distributed across multiple regions.
            Automated image comparison systems are widely used in quality inspection, surveillance,
            document verification, construction monitoring, medical imaging, and version tracking.
        </p>
        <p style='color:#cbd5e1; font-size:0.95rem; line-height:1.8;'>
            The objective of this project is to develop an intelligent application capable of identifying
            visual differences between two images, highlighting changed regions, and generating a
            human-readable summary describing the detected changes.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Expected Outputs</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul style='color:#cbd5e1; font-size:0.95rem; line-height:1.8;'>
            <li>Original Image A</li>
            <li>Original Image B</li>
            <li>Difference Visualization</li>
            <li>Highlighted Changed Regions with bounding boxes and arrows</li>
            <li>Difference Statistics</li>
            <li>AI-Generated Summary Paragraph</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Functional Requirements Covered</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul style='color:#cbd5e1; font-size:0.95rem; line-height:1.8;'>
            <li>FR-1: Image upload with validation of supported formats.</li>
            <li>FR-2: Preprocessing, resizing, alignment, and normalization.</li>
            <li>FR-3: Difference detection using structural and pixel-level comparison.</li>
            <li>FR-4: Visualization with bounding boxes, heatmap, side-by-side comparison, and overlay.</li>
            <li>FR-5: Difference statistics including count, percentage, area, and coordinates.</li>
            <li>FR-6: AI-based summary generated from detected changes.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload CAD Drawings")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>Image A (Reference)</div>", unsafe_allow_html=True)
        file_a = st.file_uploader(
            "Upload the reference/original drawing",
            type=["jpg", "jpeg", "png"],
            key="image_a",
            label_visibility="collapsed",
        )
        if file_a:
            st.image(file_a, caption="Image A — Reference", use_column_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Image B (Comparison)</div>", unsafe_allow_html=True)
        file_b = st.file_uploader(
            "Upload the modified/comparison drawing",
            type=["jpg", "jpeg", "png"],
            key="image_b",
            label_visibility="collapsed",
        )
        if file_b:
            st.image(file_b, caption="Image B — Comparison", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    compare_clicked = st.button(
        "🔬 Compare Drawings",
        type="primary",
        disabled=(file_a is None or file_b is None),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if compare_clicked and file_a and file_b:
        with st.spinner("🔄 Analyzing differences... This may take a moment."):
            try:
                # Send both files to the FastAPI backend
                response = requests.post(
                    f"{BACKEND_URL}/compare",
                    files={
                        "image_a": (file_a.name, file_a.getvalue(), file_a.type),
                        "image_b": (file_b.name, file_b.getvalue(), file_b.type),
                    },
                    timeout=120,
                )

                if response.status_code != 200:
                    st.error(
                        f"❌ Backend error ({response.status_code}): "
                        f"{response.json().get('detail', 'Unknown error')}"
                    )
                    return

                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to the backend server. "
                    "Make sure the FastAPI server is running on "
                    f"`{BACKEND_URL}`. Start it with:\n\n"
                    "```bash\n"
                    "uvicorn backend.main:app --reload --port 8000\n"
                    "```"
                )
                return
            except requests.exceptions.Timeout:
                st.error(
                    "❌ The request timed out. The images may be too large "
                    "or the server is overloaded."
                )
                return
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                return

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Comparison Results")

        st.markdown("<div class='section-title'>🤖 AI Change Summary</div>", unsafe_allow_html=True)
        st.info(result["summary"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Change Statistics</div>", unsafe_allow_html=True)

        stats = result["statistics"]

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.markdown('<div class="metric-card"><div class="metric-label">Changed Regions</div><div class="metric-value">{}</div></div>'.format(stats["region_count"]), unsafe_allow_html=True)
        with metric_cols[1]:
            st.markdown('<div class="metric-card"><div class="metric-label">Area Changed</div><div class="metric-value">{}%</div></div>'.format(stats["percent_changed"]), unsafe_allow_html=True)
        with metric_cols[2]:
            st.markdown('<div class="metric-card"><div class="metric-label">Total Pixels Changed</div><div class="metric-value">{:,}</div></div>'.format(stats["total_area_changed"]), unsafe_allow_html=True)

        if stats["regions"]:
            st.markdown("<div class='section-title' style='margin-top: 1rem;'>📍 Detected Bounding Boxes</div>", unsafe_allow_html=True)
            bbox_cols = st.columns(2)
            for idx, region in enumerate(stats["regions"], start=1):
                bbox = region["bbox"]
                col = bbox_cols[(idx - 1) % 2]
                with col:
                    col.markdown(
                        f"""
                        <div class="card" style="margin-bottom: 0.6rem;">
                            <div style="color:#38bdf8; font-weight:700; font-size:0.9rem;">Bounding Box {idx}</div>
                            <div style="margin-top:0.25rem; font-weight:600;">x: {bbox[0]}, y: {bbox[1]}, w: {bbox[2]}, h: {bbox[3]}</div>
                            <div style="margin-top:0.35rem; color:#94a3b8;">Location: {region['location']} · Area: {region['area']:,} px</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<div class='section-title' style='margin-top: 1rem;'>📋 Region Details</div>", unsafe_allow_html=True)
            table_data = []
            for idx, region in enumerate(stats["regions"], start=1):
                bbox = region["bbox"]
                table_data.append({
                    "Region #": idx,
                    "Location": region["location"],
                    "Area (px)": region["area"],
                    "Bounding Box (x, y, w, h)": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
                })
            st.table(table_data)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🖼️ Visualizations</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Side-by-Side Comparison</div>", unsafe_allow_html=True)
        _display_backend_image(result["diff_visualization_url"])

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("<div class='section-title'>Highlighted Change Regions</div>", unsafe_allow_html=True)
            _display_backend_image(result["highlighted_regions_url"])
        with viz_col2:
            st.markdown("<div class='section-title'>Heatmap Overlay</div>", unsafe_allow_html=True)
            _display_backend_image(result["heatmap_url"])

        st.markdown("<div class='section-title'>Overlay Blend (Changes in Red)</div>", unsafe_allow_html=True)
        _display_backend_image(result["overlay_url"])
        st.markdown('</div>', unsafe_allow_html=True)

    elif compare_clicked:
        st.warning("⚠️ Please upload both images before comparing.")


def _display_backend_image(image_url: str) -> None:
    """
    Fetch and display an image from the backend server.

    The backend serves visualization images as static files. This function
    fetches them via HTTP and displays them in the Streamlit UI.

    Args:
        image_url: Relative URL path to the image on the backend.
    """
    try:
        full_url = f"{BACKEND_URL}{image_url}"
        resp = requests.get(full_url, timeout=30)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content))
            st.image(img, use_column_width=True)
        else:
            st.warning(f"Could not load image: {image_url}")
    except Exception as e:
        st.warning(f"Error loading image: {str(e)}")


if __name__ == "__main__":
    main()
