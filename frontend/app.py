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


# === Page Configuration ===
st.set_page_config(
    page_title="Image Diff AI — CAD Drawing Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# === Backend API URL ===
BACKEND_URL = "http://localhost:8000"


def main():
    """
    Main Streamlit application function.

    Renders the upload interface, sends images to the backend,
    and displays all comparison results.
    """
    # === Header ===
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="color: #1E88E5; margin-bottom: 0.2rem;">
                🔍 Image Diff AI
            </h1>
            <p style="color: #666; font-size: 1.1rem;">
                AI-Based Image Difference Detection, Visualization &amp;
                Automated Change Summarization for CAD Drawings
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # === Upload Section ===
    st.subheader("📤 Upload CAD Drawings")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Image A (Reference)**")
        file_a = st.file_uploader(
            "Upload the reference/original drawing",
            type=["jpg", "jpeg", "png"],
            key="image_a",
            label_visibility="collapsed",
        )
        if file_a:
            st.image(file_a, caption="Image A — Reference", use_column_width=True)

    with col_b:
        st.markdown("**Image B (Comparison)**")
        file_b = st.file_uploader(
            "Upload the modified/comparison drawing",
            type=["jpg", "jpeg", "png"],
            key="image_b",
            label_visibility="collapsed",
        )
        if file_b:
            st.image(file_b, caption="Image B — Comparison", use_column_width=True)

    st.divider()

    # === Compare Button ===
    compare_clicked = st.button(
        "🔬 Compare Drawings",
        type="primary",
        disabled=(file_a is None or file_b is None),
    )

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

        # === Display Results ===
        st.divider()
        st.subheader("📊 Comparison Results")

        # --- AI Summary (prominent display) ---
        st.markdown("### 🤖 AI Change Summary")
        st.info(result["summary"])

        st.divider()

        # --- Statistics ---
        st.markdown("### 📈 Change Statistics")

        stats = result["statistics"]

        # Metric cards in a row
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric(
                label="Changed Regions",
                value=stats["region_count"],
            )
        with metric_cols[1]:
            st.metric(
                label="Area Changed (%)",
                value=f"{stats['percent_changed']}%",
            )
        with metric_cols[2]:
            st.metric(
                label="Total Pixels Changed",
                value=f"{stats['total_area_changed']:,}",
            )

        # Per-region detail table
        if stats["regions"]:
            st.markdown("#### Region Details")
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

        st.divider()

        # --- Visualizations ---
        st.markdown("### 🖼️ Visualizations")

        # Side-by-side comparison
        st.markdown("#### Side-by-Side Comparison")
        _display_backend_image(result["diff_visualization_url"])

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### Highlighted Change Regions")
            _display_backend_image(result["highlighted_regions_url"])

        with viz_col2:
            st.markdown("#### Heatmap Overlay")
            _display_backend_image(result["heatmap_url"])

        # Overlay blend
        st.markdown("#### Overlay Blend (Changes in Red)")
        _display_backend_image(result["overlay_url"])

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
