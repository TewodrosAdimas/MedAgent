"""
Clinical Report Generation Agent for MedAgent.

Synthesizes structured, transparent, and explainable clinical decision-support summaries
combining physiological signals, medical imaging, Grad-CAM localizations, and RAG guidelines.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.evidence_agent import EvidenceResult
from src.agents.imaging_agent import ImagingAnalysisResult
from src.agents.risk_agent import RiskAssessmentResult
from src.agents.signal_agent import SignalAnalysisResult


@dataclass
class ClinicalReport:
    report_id: str
    patient_id: str
    created_at: str
    markdown_content: str
    structured_data: Dict[str, Any]


class ReportAgent:
    """
    Agent responsible for generating structured, clinician-facing medical reports.
    """

    def generate_report(
        self,
        patient_info: Dict[str, Any],
        signal_res: SignalAnalysisResult,
        imaging_res: ImagingAnalysisResult,
        risk_res: RiskAssessmentResult,
        evidence_res: EvidenceResult,
    ) -> ClinicalReport:
        """
        Synthesizes the complete multi-agent clinical summary report.
        """
        patient_id = patient_info.get("patient_id", "ANON-001")
        age = patient_info.get("age", "N/A")
        sex = patient_info.get("sex", "N/A")
        hr = patient_info.get("heart_rate_bpm", "N/A")
        bp = patient_info.get("blood_pressure", "N/A")
        spo2 = patient_info.get("spo2_percent", "N/A")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = f"MED-REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Action plan formulation
        action_items = []
        if risk_res.risk_tier_id == 2:  # High / Critical
            action_items.append("Urgent Attending Physician evaluation and continuous telemetry monitoring.")
            if signal_res.predicted_class in [2, 3]:
                action_items.append("Order 12-lead diagnostic ECG, serum electrolytes (K+, Mg2+), and cardiac troponin.")
            if imaging_res.predicted_class in [1, 2, 4]:
                action_items.append("Stat transthoracic echocardiogram (TTE) and evaluate BNP/NT-proBNP.")
        elif risk_res.risk_tier_id == 1:  # Moderate
            action_items.append("Priority outpatient clinical review and repeat vitals assessment.")
            action_items.append("Consider ambulatory 24h Holter monitoring and follow-up imaging in 4-6 weeks.")
        else:  # Low
            action_items.append("Routine healthcare monitoring. No immediate emergent interventions indicated.")

        # Build Markdown
        md_lines = [
            "# MEDAGENT CLINICAL DECISION SUPPORT REPORT",
            "**Notice:** AI-Generated Diagnostic Decision Support for Licensed Clinical Review Only.",
            "",
            "---",
            f"**Report ID:** `{report_id}` | **Patient ID:** `{patient_id}` | **Date:** `{timestamp}`",
            f"**Demographics:** Age {age} | Sex: {sex} | **Vitals:** HR {hr} BPM | BP {bp} | SpO2 {spo2}%",
            "---",
            "",
            "## 1. PRIMARY AI FINDINGS",
            f"### A. Physiological Signal Intelligence (ECG)",
            f"- **Classification:** **{signal_res.class_name}** (Confidence: `{signal_res.confidence*100:.1f}%`)",
            f"- **Severity Tier:** `{signal_res.severity}`",
            f"- **Interpretation:** {signal_res.clinical_interpretation}",
            "",
            f"### B. Medical Imaging Intelligence (Chest Radiograph)",
            f"- **Classification:** **{imaging_res.class_name}** (Confidence: `{imaging_res.confidence*100:.1f}%`)",
            f"- **Severity Tier:** `{imaging_res.severity}`",
            f"- **Radiological Finding:** {imaging_res.radiological_finding}",
            f"- **Grad-CAM Localization:** {imaging_res.gradcam_localization}",
            "",
            "---",
            "## 2. MULTIMODAL RISK STRATIFICATION",
            f"- **Overall Risk Level:** **{risk_res.risk_tier.upper()}** (Composite Score: `{risk_res.risk_score:.2f}` / 1.00)",
            f"- **Clinical Urgency:** `{risk_res.urgency_level}`",
            f"- **Primary Contributing Drivers:**",
        ]

        for d in risk_res.primary_risk_drivers:
            md_lines.append(f"  - {d}")

        md_lines.extend([
            f"- **Assessment Rationale:** {risk_res.clinical_rationale}",
            "",
            "---",
            "## 3. EVIDENCE-BASED PRACTICE GUIDELINES (RAG)",
        ])

        for c in evidence_res.citations:
            md_lines.append(f"- **Guideline:** {c}")

        md_lines.extend([
            "",
            "---",
            "## 4. RECOMMENDED CLINICAL ACTION PLAN",
        ])

        for item in action_items:
            md_lines.append(f"- [ ] {item}")

        md_lines.extend([
            "",
            "---",
            "> **CLINICAL SAFETY DISCLAIMER:** MedAgent is an explainable clinical decision-support AI platform designed to assist healthcare professionals. It does not replace medical judgment. Final diagnosis, care pathway, and treatment decisions remain the sole responsibility of the attending physician.",
        ])

        markdown_content = "\n".join(md_lines)

        structured_data = {
            "report_id": report_id,
            "patient_id": patient_id,
            "timestamp": timestamp,
            "patient_info": patient_info,
            "signal_analysis": {
                "class_name": signal_res.class_name,
                "confidence": signal_res.confidence,
                "interpretation": signal_res.clinical_interpretation,
                "abnormalities": signal_res.abnormal_features,
            },
            "imaging_analysis": {
                "class_name": imaging_res.class_name,
                "confidence": imaging_res.confidence,
                "finding": imaging_res.radiological_finding,
                "gradcam_localization": imaging_res.gradcam_localization,
            },
            "risk_assessment": {
                "risk_tier": risk_res.risk_tier,
                "risk_score": risk_res.risk_score,
                "urgency_level": risk_res.urgency_level,
                "drivers": risk_res.primary_risk_drivers,
            },
            "evidence_citations": evidence_res.citations,
            "recommended_actions": action_items,
        }

        return ClinicalReport(
            report_id=report_id,
            patient_id=patient_id,
            created_at=timestamp,
            markdown_content=markdown_content,
            structured_data=structured_data,
        )
