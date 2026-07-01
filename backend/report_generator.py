import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image as RLImage


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        summary: str,
        statistics: Dict,
        text_changes: List[Dict],
        image_a: bytes,
        image_b: bytes,
        highlighted_image: bytes,
        heatmap_image: bytes,
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        filename = f"comparison_report_{uuid.uuid4().hex[:8]}.pdf"
        output_path = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("CAD Review Studio — Comparison Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {timestamp}", styles["Heading4"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(summary, styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Statistics", styles["Heading2"]))
        data = [
            ["Metric", "Value"],
            ["Changed regions", statistics.get("region_count", 0)],
            ["Percent changed", f"{statistics.get('percent_changed', 0)}%"],
            ["Total area changed", statistics.get("total_area_changed", 0)],
            ["Severity", statistics.get("change_severity", "minor_revision")],
            ["Confidence", f"{statistics.get('confidence_score', 0)}%"],
        ]
        table = Table(data, colWidths=[2.2 * inch, 2.6 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Images", styles["Heading2"]))
        for label, payload in [("Image A", image_a), ("Image B", image_b), ("Highlighted Regions", highlighted_image), ("Heatmap", heatmap_image)]:
            img_buffer = BytesIO(payload)
            story.append(Paragraph(label, styles["Heading4"]))
            story.append(RLImage(img_buffer, width=4.8 * inch, height=3.2 * inch))
            story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("Text and Dimension Changes", styles["Heading2"]))
        if text_changes:
            text_rows = [["Old Value", "New Value", "Change Type", "Location"]]
            for change in text_changes:
                text_rows.append([
                    change.get("old_text", ""),
                    change.get("new_text", ""),
                    change.get("change_type", ""),
                    str(change.get("location", [])),
                ])
            text_table = Table(text_rows, repeatRows=1, colWidths=[1.25 * inch, 1.25 * inch, 1.1 * inch, 1.8 * inch])
            text_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(text_table)
        else:
            story.append(Paragraph("No OCR text changes detected.", styles["BodyText"]))

        doc.build(story)
        return output_path
