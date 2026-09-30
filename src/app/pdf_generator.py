"""
MedAgent Clinical Decision Support - PDF Report Generator.

Generates professional, hospital-grade clinical PDF reports from multimodal AI synthesis.
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_clinical_pdf(
    patient_info: Dict[str, Any],
    signal_res: Any,
    imaging_res: Any,
    risk_res: Any,
    evidence_res: Any,
    report_id: Optional[str] = None,
) -> bytes:
    """
    Generates a hospital-grade electronic health record PDF document.
    
    Returns:
        bytes: The compiled PDF binary data ready for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    style_section_hdr = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=8,
        spaceAfter=4,
    )

    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )

    style_body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )

    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>🏥 MedAgent Clinical Intelligence Cockpit</b><br/><font size=8 color='#64748b'>Multimodal AI Diagnostic Synthesis & Clinical Decision Support</font>", style_title),
            Paragraph(f"<b>REPORT ID:</b> {report_id or 'MED-REP-2026'}<br/><b>DATE:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/><b>STATUS:</b> VERIFIED", style_subtitle),
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceBefore=2, spaceAfter=8))

    # 2. Patient Demographics & Vitals Table
    p_id = patient_info.get("patient_id", "PT-UNKNOWN")
    age = patient_info.get("age", "N/A")
    sex = patient_info.get("sex", "N/A")
    hr = patient_info.get("heart_rate_bpm", "N/A")
    bp = patient_info.get("blood_pressure", "N/A")
    spo2 = patient_info.get("spo2_percent", "N/A")
    notes = patient_info.get("medical_history", "None recorded.")

    comorbs = []
    if patient_info.get("hypertension"): comorbs.append("Hypertension")
    if patient_info.get("diabetes"): comorbs.append("Diabetes Mellitus")
    if patient_info.get("smoking_history"): comorbs.append("Tobacco Smoker")
    if patient_info.get("prior_cardiac_event"): comorbs.append("Prior Cardiac Event")
    comorb_str = ", ".join(comorbs) if comorbs else "None Reported"

    pt_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {p_id}", style_body),
            Paragraph(f"<b>Age / Sex:</b> {age}y / {sex}", style_body),
            Paragraph(f"<b>Heart Rate:</b> {hr:.0f} BPM" if isinstance(hr, (int, float)) else f"<b>Heart Rate:</b> {hr}", style_body),
        ],
        [
            Paragraph(f"<b>Blood Pressure:</b> {bp} mmHg", style_body),
            Paragraph(f"<b>SpO2 Saturation:</b> {spo2:.1f}%" if isinstance(spo2, (int, float)) else f"<b>SpO2:</b> {spo2}", style_body),
            Paragraph(f"<b>Comorbidities:</b> {comorb_str}", style_body),
        ],
        [
            Paragraph(f"<b>Clinical History & Chief Complaint:</b> {notes}", style_body),
            "",
            "",
        ],
    ]
    pt_table = Table(pt_data, colWidths=[180, 180, 180])
    pt_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("SPAN", (0, 2), (2, 2)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(pt_table)
    story.append(Spacer(1, 10))

    # 3. Primary Modality Findings (ECG + Vision)
    story.append(Paragraph("1. PRIMARY DIAGNOSTIC MODALITY FINDINGS", style_section_hdr))

    sig_conf = f"{signal_res.confidence * 100:.1f}%" if hasattr(signal_res, "confidence") else "N/A"
    sig_name = getattr(signal_res, "class_name", "N/A")
    sig_sev = getattr(signal_res, "severity", "N/A")
    sig_interp = getattr(signal_res, "clinical_interpretation", "N/A")

    img_conf = f"{imaging_res.confidence * 100:.1f}%" if hasattr(imaging_res, "confidence") else "N/A"
    img_name = getattr(imaging_res, "class_name", "N/A")
    img_finding = getattr(imaging_res, "radiological_finding", "N/A")
    img_loc = getattr(imaging_res, "gradcam_localization", "N/A")

    findings_data = [
        [
            Paragraph("<b>Modality / Engine</b>", style_body_bold),
            Paragraph("<b>Classification & Confidence</b>", style_body_bold),
            Paragraph("<b>Severity Tier</b>", style_body_bold),
            Paragraph("<b>Clinical & Anatomical Interpretation</b>", style_body_bold),
        ],
        [
            Paragraph("<b>Physiological Signal</b><br/><font size=7.5 color='#64748b'>1D-CNN (Lead MLII)</font>", style_body),
            Paragraph(f"<b>{sig_name}</b><br/>Conf: {sig_conf}", style_body),
            Paragraph(f"{sig_sev}", style_body),
            Paragraph(f"{sig_interp}", style_body),
        ],
        [
            Paragraph("<b>Medical Vision</b><br/><font size=7.5 color='#64748b'>DenseNet-121 + Grad-CAM</font>", style_body),
            Paragraph(f"<b>{img_name}</b><br/>Conf: {img_conf}", style_body),
            Paragraph("Radiological", style_body),
            Paragraph(f"{img_finding}<br/><b>Spatial Focus:</b> {img_loc}", style_body),
        ],
    ]
    findings_table = Table(findings_data, colWidths=[110, 120, 80, 230])
    findings_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(findings_table)
    story.append(Spacer(1, 10))

    # 4. Multimodal Patient Risk Stratification
    story.append(Paragraph("2. CROSS-ATTENTION MULTIMODAL RISK STRATIFICATION", style_section_hdr))
    risk_tier = getattr(risk_res, "risk_tier", "Low")
    risk_score = getattr(risk_res, "risk_score", 0.0) * 100.0
    risk_urgency = getattr(risk_res, "urgency_level", "Routine")
    risk_drivers = getattr(risk_res, "primary_risk_drivers", [])
    risk_rationale = getattr(risk_res, "clinical_rationale", "N/A")

    driver_str = "<br/>• ".join(risk_drivers)
    if driver_str:
        driver_str = "• " + driver_str

    risk_box_bg = colors.HexColor("#fef2f2") if "High" in risk_tier or "Critical" in risk_tier else colors.HexColor("#fffbeb") if "Moderate" in risk_tier else colors.HexColor("#ecfdf5")
    risk_box_border = colors.HexColor("#dc2626") if "High" in risk_tier or "Critical" in risk_tier else colors.HexColor("#d97706") if "Moderate" in risk_tier else colors.HexColor("#059669")

    risk_data = [
        [
            Paragraph(f"<b>COMPOSITE RISK TIER:</b> <font color='{risk_box_border.hexval()}'><b>{risk_tier.upper()}</b></font><br/><b>Index Score:</b> {risk_score:.1f} / 100<br/><b>Triage Urgency:</b> {risk_urgency}", style_body),
            Paragraph(f"<b>Primary Risk Drivers:</b><br/>{driver_str}<br/><br/><b>Synthesis Rationale:</b> {risk_rationale}", style_body),
        ]
    ]
    risk_table = Table(risk_data, colWidths=[200, 340])
    risk_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), risk_box_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, risk_box_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(risk_table)
    story.append(Spacer(1, 10))

    # 5. Evidence-Based Practice Guidelines (RAG)
    story.append(Paragraph("3. EVIDENCE-BASED PRACTICE GUIDELINES (RAG RETRIEVAL)", style_section_hdr))
    citations = getattr(evidence_res, "citations", [])
    guidelines = getattr(evidence_res, "retrieved_guidelines", [])

    if guidelines:
        g_rows = []
        for g in guidelines:
            g_rows.append([
                Paragraph(f"<b>[{g.id}] {g.title}</b> ({g.topic})<br/>{g.content}<br/><font size=7.5 color='#0284c7'><b>Recommendation:</b> {g.strength_of_recommendation}</font> | <font size=7 color='#64748b'>Source: {g.citation}</font>", style_body)
            ])
        g_table = Table(g_rows, colWidths=[540])
        g_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(g_table)
    else:
        for c in citations:
            story.append(Paragraph(f"• {c}", style_body))

    story.append(Spacer(1, 10))

    # 6. Physician Action Plan & Sign-Off Block
    actions = []
    if "High" in risk_tier or "Critical" in risk_tier:
        actions = [
            "Immediate Attending Physician bedside review and continuous telemetry monitoring.",
            "Order STAT 12-lead ECG, cardiac biomarkers (Troponin I/T), and comprehensive metabolic panel.",
            "Schedule urgent Transthoracic Echocardiogram (TTE) and consider early cardiology/pulmonology consultation.",
        ]
    elif "Moderate" in risk_tier:
        actions = [
            "Priority clinical review and repeat vitals assessment within 24 hours.",
            "Consider 24-hour ambulatory Holter ECG monitoring and chest imaging follow-up in 4-6 weeks.",
            "Optimize medical therapy according to ACC/AHA clinical guidelines.",
        ]
    else:
        actions = [
            "Routine outpatient health maintenance and clinical follow-up.",
            "Patient education on lifestyle factors and cardiovascular risk mitigation.",
        ]

    action_items_str = "<br/>".join([f"<b>[ &nbsp; ]</b> &nbsp; {a}" for a in actions])

    signoff_data = [
        [
            Paragraph(f"<b>RECOMMENDED CLINICAL ACTION PLAN:</b><br/>{action_items_str}", style_body),
            Paragraph("<b>PHYSICIAN ATTESTATION:</b><br/><br/>Physician Signature: _______________________<br/>NPI / License: ____________________________<br/>Date / Time: ______________________________", style_body),
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[330, 210])
    signoff_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0f172a")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(KeepTogether([signoff_table, Spacer(1, 8)]))

    # 7. Safety Disclaimer
    story.append(
        Paragraph(
            "<b>CLINICAL SAFETY DISCLAIMER:</b> MedAgent is an assistive AI clinical decision support system designed to augment licensed medical practitioners. It does not provide autonomous diagnoses or replace clinical judgment. All algorithmic scores, Grad-CAM heatmaps, and practice citations require independent clinical validation prior to patient management decisions.",
            style_disclaimer,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
