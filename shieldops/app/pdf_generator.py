"""
ShieldOps — Dispute Evidence Packet PDF Generator
=================================================
Generates professional, presentation-ready dispute defense packets
and fraud investigation reports using ReportLab.
"""

import io
import datetime as dt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


def generate_dispute_pdf(record: dict) -> io.BytesIO:
    """
    Generates a Dispute Evidence Packet PDF for a given audit log record.
    
    Expected record keys:
        - entity_type: 'chargeback' or 'return_abuse'
        - entity_id: str (e.g. 'ORD_1042', 'CUST_038')
        - risk_score: float (0.0 to 1.0)
        - risk_tier: 'high', 'medium', or 'low'
        - explanation: str (evidence summary / fraud explanation)
        - source: 'llm_generated' or 'template_fallback'
        - recommended_action: str
        - estimated_cost_impact: float or int
        - created_at: str (ISO datetime)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography & styles
    primary_color = colors.HexColor("#16213E")
    secondary_color = colors.HexColor("#0F3460")
    text_dark = colors.HexColor("#1E293B")
    text_muted = colors.HexColor("#64748B")
    box_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    tier = record.get("risk_tier", "medium").lower()
    if tier == "high":
        tier_color = colors.HexColor("#991B1B")
        tier_bg = colors.HexColor("#FEE2E2")
    elif tier == "medium":
        tier_color = colors.HexColor("#92400E")
        tier_bg = colors.HexColor("#FEF3C7")
    else:
        tier_color = colors.HexColor("#166534")
        tier_bg = colors.HexColor("#DCFCE7")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=text_muted,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=text_dark,
    )

    body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
    )

    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=text_muted,
        alignment=1,  # Center
    )

    story = []

    # 1. Header Banner Table
    header_data = [
        [
            Paragraph("<b>SHIELDOPS</b> | Risk Intelligence", ParagraphStyle("HdrL", parent=body_bold, textColor=primary_color, fontSize=11)),
            Paragraph(f"Generated: {dt.datetime.now().strftime('%d %b %Y, %H:%M UTC')}", ParagraphStyle("HdrR", parent=subtitle_style, alignment=2)),
        ]
    ]
    hdr_table = Table(header_data, colWidths=[3.5 * inch, 4.0 * inch])
    hdr_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(hdr_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=14))

    # 2. Main Title
    entity_type = record.get("entity_type", "chargeback")
    doc_title = "Dispute Evidence Packet" if entity_type == "chargeback" else "Return Abuse Investigation Packet"
    story.append(Paragraph(doc_title, title_style))
    story.append(Paragraph(
        "Official merchant risk assessment & evidence summary prepared for payment gateway dispute resolution.",
        subtitle_style,
    ))
    story.append(Spacer(1, 12))

    # 3. Case Metadata & Overview Table
    entity_id = record.get("entity_id", "N/A")
    risk_score = float(record.get("risk_score", 0.0))
    cost_impact = float(record.get("estimated_cost_impact", 0.0))
    source_label = "Google Gemini Intelligence" if record.get("source") == "llm_generated" else "Deterministic Template Fallback"

    summary_data = [
        [
            Paragraph("<b>Target Entity:</b>", body_style),
            Paragraph(f"<b>{entity_id}</b> ({'Order ID' if entity_type == 'chargeback' else 'Customer ID'})", body_style),
            Paragraph("<b>Risk Score:</b>", body_style),
            Paragraph(f"<b>{risk_score * 100:.1f}%</b>", ParagraphStyle("Score", parent=body_bold, textColor=tier_color)),
        ],
        [
            Paragraph("<b>Case Category:</b>", body_style),
            Paragraph(entity_type.replace("_", " ").title(), body_style),
            Paragraph("<b>Risk Tier:</b>", body_style),
            Paragraph(f"<b>{tier.upper()} RISK</b>", ParagraphStyle("TierBadge", parent=body_bold, textColor=tier_color)),
        ],
        [
            Paragraph("<b>Financial Impact:</b>", body_style),
            Paragraph(f"Rs. {cost_impact:,.2f}", body_style),
            Paragraph("<b>Analysis Engine:</b>", body_style),
            Paragraph(source_label, body_style),
        ],
        [
            Paragraph("<b>Case Logged At:</b>", body_style),
            Paragraph(record.get("created_at", dt.datetime.now().isoformat()), body_style),
            Paragraph("<b>Defense Status:</b>", body_style),
            Paragraph("Evidence Pre-Drafted", body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[1.4 * inch, 2.3 * inch, 1.4 * inch, 2.4 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # 4. Evidence / Explanation Callout Box
    story.append(Paragraph("1. Fraud Risk Assessment & Evidence Narrative", section_heading))
    
    explanation_text = record.get("explanation", "No detailed explanation recorded.")
    explanation_data = [
        [Paragraph(f"<b>Merchant Evidence Summary:</b><br/><br/>{explanation_text}", callout_style)]
    ]
    explanation_table = Table(explanation_data, colWidths=[7.5 * inch])
    explanation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("LINELEFT", (0, 0), (-1, -1), 4, secondary_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(explanation_table)
    story.append(Spacer(1, 14))

    # 5. Recommended Action
    story.append(Paragraph("2. Recommended Defense & Mitigation Action", section_heading))
    action_text = record.get("recommended_action", "Review transaction logs and contact customer.")
    action_data = [
        [
            Paragraph("<b>Recommended Merchant Action:</b>", body_bold),
            Paragraph(action_text, body_style),
        ]
    ]
    action_table = Table(action_data, colWidths=[2.2 * inch, 5.3 * inch])
    action_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 14))

    # 6. Audit Trail & Verification Section
    story.append(Paragraph("3. Dispute Audit Trail & Compliance Verification", section_heading))
    audit_data = [
        [
            Paragraph("<b>Audit Parameter</b>", body_bold),
            Paragraph("<b>Verification Details</b>", body_bold),
        ],
        [
            Paragraph("Model Pipeline", body_style),
            Paragraph("ShieldOps RandomForest Tabular Ensemble v1.0", body_style),
        ],
        [
            Paragraph("LLM Synthesis Layer", body_style),
            Paragraph("Google Gemini Generative Defense Protocol", body_style),
        ],
        [
            Paragraph("Audit Trail DB Record ID", body_style),
            Paragraph(f"AUD-{record.get('id', 'N/A')}-{record.get('entity_id', 'N/A')}", body_style),
        ],
        [
            Paragraph("Dispute Submission Intended For", body_style),
            Paragraph("Payment Gateway Chargeback Portal / Acquiring Bank Representative", body_style),
        ],
    ]
    audit_table = Table(audit_data, colWidths=[2.4 * inch, 5.1 * inch])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 22))

    # 7. Document Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=border_color, spaceBefore=4, spaceAfter=8))
    story.append(Paragraph(
        "<b>Confidentiality Notice:</b> This dispute evidence packet contains proprietary merchant risk scores and customer transaction history. Generated automatically by ShieldOps AI Risk Manager.",
        footer_style,
    ))

    # Build document
    doc.build(story)
    buffer.seek(0)
    return buffer
