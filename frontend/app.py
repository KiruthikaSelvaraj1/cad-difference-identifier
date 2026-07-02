# Streamlit frontend for AI-based image comparison.
# This UI handles file upload, request submission, and presentation of results.

import io

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(
    page_title="CAD Review Studio — CAD Revision Review",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000"

st.markdown(
    """
    <style>
        :root {
            --bg: #f7f8fc;
            --panel: #ffffff;
            --panel-alt: #f9fbff;
            --panel-soft: #f4f7fb;
            --text: #172033;
            --muted: #5f6d83;
            --accent: #4f46e5;
            --accent-2: #7c3aed;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #dfe7f5;
            --soft-blue: #eef4ff;
            --soft-cream: #fcfaf5;
            --soft-mint: #f0fdf7;
        }
        * { font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif; }
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f3f6ff 45%, #f9f9f7 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #14213d !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.72rem 1.2rem;
            font-weight: 700;
            width: 100%;
            box-shadow: 0 8px 20px rgba(79, 70, 229, 0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.24);
        }
        .stTextInput > div > div > input,
        .stFileUploader > div {
            background-color: var(--panel);
            border: 1px dashed #9cb2df;
            border-radius: 14px;
            color: var(--text);
            padding: 0.65rem 0.75rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
        }
        .stFileUploader > div:hover {
            border-color: var(--accent);
            box-shadow: 0 4px 16px rgba(79,70,229,0.12);
        }
        .stAlert, .stInfo, .stSuccess, .stWarning {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 6px 18px rgba(15,23,42,0.05);
        }
        .stSuccess {
            background: #f0fdf7;
            border-left: 4px solid var(--success);
            color: #047857;
        }
        .stWarning {
            background: #fff8eb;
            border-left: 4px solid var(--warning);
            color: #b45309;
        }
        .stError {
            background: #fef2f2;
            border-left: 4px solid var(--danger);
            color: #b91c1c;
        }
        .card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(244,247,255,0.98));
            border: 1px solid rgba(223,231,245,0.9);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.04);
            margin-bottom: 1rem;
        }
        .card-alt {
            background: linear-gradient(135deg, rgba(251,252,255,0.98), rgba(240,245,255,0.98));
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(248,250,252,0.98), rgba(237,244,255,0.98));
            border: 1px solid rgba(79,70,229,0.14);
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
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .section-title {
            color: #4338ca;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            letter-spacing: 0.02em;
        }
        .pill {
            display: inline-block;
            padding: 0.36rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            box-shadow: 0 6px 16px rgba(16,185,129,0.18);
        }
        .header-shell {
            text-align: center;
            padding: 0.35rem 0 1rem 0;
        }
        .header-badge {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            font-weight: 700;
            margin-bottom: 0.6rem;
            box-shadow: 0 6px 16px rgba(79,70,229,0.16);
        }
        .header-accent {
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, #4f46e5, #7c3aed, #10b981);
            margin: 0.8rem auto 0 auto;
            width: 140px;
        }
        .upload-confirmation-card {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.75rem 0.85rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #f8fbff 0%, #f1f6ff 100%);
            border: 1px solid #dce8ff;
            box-shadow: 0 8px 20px rgba(15,23,42,0.04);
            margin-bottom: 0.7rem;
        }
        .upload-confirmation-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            font-weight: 800;
            flex-shrink: 0;
        }
        .upload-confirmation-content {
            min-width: 0;
        }
        .upload-confirmation-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #14213d;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .upload-confirmation-meta {
            color: #5b6b80;
            font-size: 0.84rem;
            margin-top: 0.12rem;
        }
        .upload-success-message {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.7rem;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .remove-button > button {
            border: 1px solid #d8e3f7;
            border-radius: 999px;
            background: white;
            color: #5b6b80;
            min-width: 2.2rem;
            height: 2.2rem;
            padding: 0;
            box-shadow: 0 4px 10px rgba(15,23,42,0.04);
        }
        .remove-button > button:hover {
            color: var(--danger);
            border-color: #f7c1c1;
        }
        .stExpander {
            border: 1px solid #e3eaf7;
            border-radius: 14px;
            background: rgba(255,255,255,0.82);
            margin-bottom: 1rem;
        }
        .impact-card {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #e3ebf7;
            box-shadow: 0 8px 20px rgba(15,23,42,0.04);
            margin-bottom: 1rem;
        }
        .impact-score-circle {
            width: 96px;
            height: 96px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 800;
            background: conic-gradient(#10B981 0% 100%, #e5e7eb 100% 100%);
            color: #0f172a;
            box-shadow: inset 0 0 0 8px white;
        }
        .impact-score-circle > span { background: white; border-radius: 50%; width: 72px; height: 72px; display: flex; align-items: center; justify-content: center; }
        .impact-copy .impact-label { font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; color: #5b6b80; }
        .impact-copy .impact-value { font-size: 1.15rem; font-weight: 700; color: #0f172a; }
        .breakdown-card {
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            background: #ffffff;
            border: 1px solid #e8eef8;
            box-shadow: 0 6px 16px rgba(15,23,42,0.04);
            min-height: 96px;
        }
        .breakdown-card .metric-name { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5b6b80; }
        .breakdown-card .metric-value { font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-top: 0.25rem; }
        .analytics-card {
            border-radius: 16px;
            padding: 1rem;
            background: #ffffff;
            border: 1px solid #e8eef8;
            box-shadow: 0 10px 24px rgba(15,23,42,0.04);
        }
        .analytics-card .section-title { margin-bottom: 0.7rem; }
        .comparison-table .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.sidebar.title("CAD Review Studio")
    st.sidebar.caption("Professional review for CAD drawing changes")
    st.sidebar.markdown("### Workflow")
    st.sidebar.markdown("1. Upload two versions of a drawing.\n2. Review highlighted differences and OCR text changes.\n3. Download a polished PDF report.")
    if st.sidebar.button("Reset", use_container_width=True):
        st.session_state.pop("image_a", None)
        st.session_state.pop("image_b", None)
        st.rerun()

    st.markdown(
        """
        <div class="header-shell">
            <div class="header-badge">Professional Review Platform</div>
            <h1 style="margin-bottom: 0.25rem; font-size: 2.05rem;">CAD Review Studio</h1>
            <p style="color: #5b6b80; font-size: 1.0rem; max-width: 900px; margin: 0 auto;">
                CAD Review Studio for precise revision review, clear change highlights, and OCR-based dimension checks.
            </p>
            <div class="header-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Requirements ▾", expanded=False):
        st.write("Upload two CAD drawing revisions, review highlighted changes, inspect OCR text updates, and download a PDF report.")

    file_a = None
    file_b = None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload CAD Drawings")

    col_a, col_b = st.columns(2)

    with col_a:
        file_a = _render_upload_slot(
            label="Image A (Reference)",
            widget_key="image_a",
            prompt="Upload the reference/original drawing",
            caption="Image A — Reference",
        )

    with col_b:
        file_b = _render_upload_slot(
            label="Image B (Comparison)",
            widget_key="image_b",
            prompt="Upload the modified/comparison drawing",
            caption="Image B — Comparison",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card card-alt">', unsafe_allow_html=True)
    compare_clicked = st.button(
        "🔬 Compare Drawings",
        type="primary",
        disabled=(file_a is None or file_b is None),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if compare_clicked and file_a and file_b:
        with st.spinner("Analyzing drawings..."):
            try:
                content_type_a = file_a.type or "application/octet-stream"
                content_type_b = file_b.type or "application/octet-stream"
                response = requests.post(
                    f"{BACKEND_URL}/compare",
                    files={
                        "image_a": (file_a.name, file_a.getvalue(), content_type_a),
                        "image_b": (file_b.name, file_b.getvalue(), content_type_b),
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

        stats_payload = result.get("statistics", {})
        severity_label = stats_payload.get("change_severity", "minor_revision")
        severity_display = {
            "major_revision": ("Major", "#ef4444"),
            "moderate_revision": ("Moderate", "#f59e0b"),
            "minor_revision": ("Minor", "#10b981"),
        }
        badge_text, badge_color = severity_display.get(severity_label, ("Minor", "#10b981"))
        st.markdown(f"<div class='pill' style='background:{badge_color};'>{badge_text} Revision</div>", unsafe_allow_html=True)

        confidence = float(stats_payload.get("confidence_score", 0.0))
        st.progress(confidence / 100)
        if confidence >= 80:
            st.markdown(f"<div style='color:#047857; font-weight:700;'>Confidence: {confidence:.1f}% · High confidence</div>", unsafe_allow_html=True)
        elif confidence >= 60:
            st.markdown(f"<div style='color:#b45309; font-weight:700;'>Confidence: {confidence:.1f}% · Medium confidence</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#b91c1c; font-weight:700;'>Confidence: {confidence:.1f}% · Lower confidence</div>", unsafe_allow_html=True)

        if st.button("📄 Download PDF Report", use_container_width=True):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/compare/report",
                    files={
                        "image_a": (file_a.name, file_a.getvalue(), file_a.type or "application/octet-stream"),
                        "image_b": (file_b.name, file_b.getvalue(), file_b.type or "application/octet-stream"),
                    },
                    timeout=180,
                )
                if response.status_code == 200:
                    st.download_button("Save PDF", response.content, file_name="comparison_report.pdf", mime="application/pdf")
                else:
                    st.error("PDF report generation failed.")
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")

        tabs = st.tabs(["Visual Comparison", "Statistics", "Text/Dimension Changes", "Summary Report"])

        with tabs[0]:
            _display_backend_image(result["diff_visualization_url"], caption="Side-by-side comparison")
            col1, col2 = st.columns(2)
            with col1:
                _display_backend_image(result["highlighted_regions_url"], caption="Highlighted change regions")
            with col2:
                _display_backend_image(result["heatmap_url"], caption="Heatmap overlay")
            _display_backend_image(result["overlay_url"], caption="Overlay blend")

        with tabs[1]:
            stats = result["statistics"]
            impact_score = float(stats.get("impact_score", 0.0))
            impact_label = stats.get("impact_label", "Low Impact")
            impact_color = "#10B981" if impact_label == "Low Impact" else "#F59E0B" if impact_label == "Moderate Impact" else "#EF4444"
            st.markdown(
                f"""
                <div class="impact-card">
                    <div class="impact-score-circle" style="background: conic-gradient({impact_color} 0% {impact_score:.0f}%, #e5e7eb {impact_score:.0f}% 100%);">
                        <span>{impact_score:.0f}</span>
                    </div>
                    <div class="impact-copy">
                        <div class="impact-label">Overall Impact</div>
                        <div class="impact-value">{impact_label}</div>
                        <div style="color:#5b6b80; margin-top:0.2rem;">Weighted from change extent, region count, OCR evidence, and severity.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            breakdown = stats.get("change_breakdown", {})
            breakdown_items = [
                ("Additions", breakdown.get("additions", 0), "#10B981"),
                ("Removals", breakdown.get("removals", 0), "#EF4444"),
                ("Modifications", breakdown.get("modifications", 0), "#F59E0B"),
                ("Positional Shifts", breakdown.get("positional_shifts", 0), "#6366F1"),
            ]
            cols = st.columns(4)
            for col, (name, value, color) in zip(cols, breakdown_items):
                with col:
                    st.markdown(
                        f"""
                        <div class="breakdown-card" style="border-top: 4px solid {color};">
                            <div class="metric-name">{name}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Changed Regions", stats["region_count"])
            with metric_cols[1]:
                st.metric("Area Changed", f"{stats['percent_changed']}%")
            with metric_cols[2]:
                st.metric("Total Pixels", f"{stats['total_area_changed']:,}")

            st.markdown("### Image A vs Image B Properties")
            try:
                image_a_bytes = file_a.getvalue() if file_a is not None else b""
                image_b_bytes = file_b.getvalue() if file_b is not None else b""
                def _image_stats(payload: bytes) -> dict:
                    if not payload:
                        return {"resolution": "n/a", "file_size_kb": 0.0, "line_density": 0.0}
                    arr = np.frombuffer(payload, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        return {"resolution": "n/a", "file_size_kb": round(len(payload) / 1024, 1), "line_density": 0.0}
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    line_density = round((np.count_nonzero(binary > 0) / binary.size) * 100, 2)
                    return {
                        "resolution": f"{img.shape[1]} × {img.shape[0]}",
                        "file_size_kb": round(len(payload) / 1024, 1),
                        "line_density": line_density,
                    }

                a_stats = _image_stats(image_a_bytes)
                b_stats = _image_stats(image_b_bytes)
                comparison_df = pd.DataFrame(
                    [
                        {"Property": "Resolution", "Image A": a_stats["resolution"], "Image B": b_stats["resolution"]},
                        {"Property": "File Size (KB)", "Image A": a_stats["file_size_kb"], "Image B": b_stats["file_size_kb"]},
                        {"Property": "Line Density (%)", "Image A": a_stats["line_density"], "Image B": b_stats["line_density"]},
                    ]
                )
                st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
                styled_df = comparison_df.style.apply(
                    lambda row: ["font-weight: 700" if row.name == 0 else "" for _ in row],
                    axis=1,
                ).set_properties(**{"background-color": "#ffffff"})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.warning(f"Unable to compute structural comparison metrics: {exc}")

            if result.get("analytics_chart_url"):
                st.markdown("### Change Distribution by Region")
                st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
                _display_backend_image(result["analytics_chart_url"], caption="Region area distribution")
                st.markdown('</div>', unsafe_allow_html=True)

            if stats["regions"]:
                region_rows = []
                for region in stats["regions"]:
                    region_rows.append({
                        "Location": region["location"],
                        "Severity": region.get("severity", "minor"),
                        "Change Type": region.get("change_type", "modification"),
                        "Area": region["area"],
                        "BBox": tuple(region["bbox"]),
                    })
                st.dataframe(pd.DataFrame(region_rows), use_container_width=True)

        with tabs[2]:
            text_changes = result.get("text_changes", [])
            if text_changes:
                df = pd.DataFrame(text_changes)
                df["Change Type"] = df["change_type"].str.title()
                df["Location"] = df["location"].apply(lambda loc: f"x={loc[0]}, y={loc[1]}, w={loc[2]}, h={loc[3]}")
                df = df[["old_text", "new_text", "Change Type", "Location"]]
                df = df.set_index(pd.Index(range(1, len(df) + 1)))

                def style_text_table(row: pd.Series) -> list[str]:
                    change_type = str(row["Change Type"]).lower()
                    if change_type == "modified":
                        return ["background-color:#fff7ed; font-weight:600"] * len(row)
                    if change_type == "added":
                        return ["background-color:#ecfdf5; font-weight:600"] * len(row)
                    return ["background-color:#fef2f2; font-weight:600"] * len(row)

                st.dataframe(df.style.apply(style_text_table, axis=1), use_container_width=True)
            else:
                st.info("No OCR text or dimension differences were detected.")

        with tabs[3]:
            st.info(result["summary"])
            st.success(result.get("difference_explanation", "No additional explanation was generated."))
        st.markdown('</div>', unsafe_allow_html=True)

    elif compare_clicked:
        st.warning("⚠️ Please upload both images before comparing.")


def _render_upload_slot(label: str, widget_key: str, prompt: str, caption: str) -> UploadedFile | None:
    st.markdown(f"<div class='section-title'>{label}</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        prompt,
        type=["jpg", "jpeg", "png", "pdf"],
        key=widget_key,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        size_kb = round(uploaded_file.size / 1024, 1) if getattr(uploaded_file, "size", None) else 0
        col_card, col_remove = st.columns([6, 1])
        with col_card:
            st.markdown(
                f"""
                <div class="upload-confirmation-card">
                    <div class="upload-confirmation-icon">✓</div>
                    <div class="upload-confirmation-content">
                        <div class="upload-confirmation-title">{uploaded_file.name}</div>
                        <div class="upload-confirmation-meta">{size_kb:.1f} KB · Ready for review</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_remove:
            if st.button("✕", key=f"remove_{widget_key}", help=f"Remove {label}", use_container_width=True):
                st.session_state.pop(widget_key, None)
                st.rerun()

        if uploaded_file.name.lower().endswith(".pdf"):
            st.markdown(
                "<div class='upload-success-message'>✅ PDF detected — the first page will be converted automatically.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.image(uploaded_file, caption=caption, use_column_width=True)

    return uploaded_file


def _display_backend_image(image_url: str, caption: str | None = None) -> None:
    try:
        full_url = f"{BACKEND_URL}{image_url}"
        resp = requests.get(full_url, timeout=30)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content))
            st.image(img, caption=caption, width=700)
        else:
            st.warning(f"Could not load image: {image_url}")
    except Exception as e:
        st.warning(f"Error loading image: {str(e)}")


if __name__ == "__main__":
    main()
