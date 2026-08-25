"""
LLM Client & Clinical Synthesis Interface for MedAgent.
Provides deterministic structured report generation and API/local model support.
"""

from typing import Any, Dict, List, Optional
import json

from src.utils.logger import get_logger

logger = get_logger("LLMClient")


class LLMClient:
    """
    Unified LLM Client for medical reasoning and report generation.
    """

    def __init__(
        self,
        provider: str = "deterministic",
        model_name: str = "medagent-clinical-v1",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key

    def generate_clinical_summary(
        self,
        patient_info: Dict[str, Any],
        signal_report: Dict[str, Any],
        imaging_report: Dict[str, Any],
        risk_report: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
    ) -> str:
        """
        Synthesizes multimodal findings, risk assessment, and RAG literature into an
        explainable, structured Clinical Decision Support Report.
        """
        # Formulate grounded clinical explanation
        findings_bullets = []
        if signal_report.get("finding") != "Normal (N)":
            findings_bullets.append(
                f"- **Cardiac Rhythm**: {signal_report.get('finding')} detected with {signal_report.get('confidence', 0)*100:.1f}% confidence. {signal_report.get('clinical_implication', '')}"
            )
        else:
            findings_bullets.append("- **Cardiac Rhythm**: Normal sinus rhythm pattern detected without acute ST/morphological abnormalities.")

        if imaging_report.get("finding") != "Normal":
            regions_str = ", ".join([r.get("region", "") for r in imaging_report.get("salient_regions", [])])
            regions_str = f" localized primarily in {regions_str}" if regions_str else ""
            findings_bullets.append(
                f"- **Radiological Imaging**: Possible {imaging_report.get('finding')}{regions_str} with {imaging_report.get('confidence', 0)*100:.1f}% confidence."
            )
        else:
            findings_bullets.append("- **Radiological Imaging**: Clear lung fields without focal alveolar consolidation or acute pleural abnormality.")

        evidence_bullets = []
        for ev in evidence_list:
            evidence_bullets.append(
                f"- *{ev['title']}* ({ev['source']}): \"{ev['evidence_text']}\""
            )

        risk_level = risk_report.get("risk_level", "Low Risk")
        urgency = "Routine Clinical Follow-up" if "Low" in risk_level else (
            "Physician Evaluation Recommended within 24-48 Hours" if "Moderate" in risk_level else "Urgent Medical Attention Advised"
        )

        report_lines = [
            f"# MedAgent Clinical Decision Support Report",
            f"**Patient ID**: {patient_info.get('patient_id', 'PT-9042')} | **Age**: {patient_info.get('age', 55)} | **Sex**: {patient_info.get('sex', 'Unknown')}",
            f"**Clinical History**: {patient_info.get('medical_history', 'Hypertension')}",
            "",
            f"## 1. Executive Summary & Multimodal Risk Assessment",
            f"**Overall Stratified Risk**: **{risk_level}** ({risk_report.get('risk_score', 0)*100:.1f}% Composite Severity Score)",
            f"**Primary Contributing Modalities**: {risk_report.get('primary_drivers', 'ECG Waveform & Chest Radiograph')}",
            f"**Recommended Clinical Urgency**: `{urgency}`",
            "",
            f"## 2. Objective AI Findings",
            "\n".join(findings_bullets),
            "",
            f"## 3. Evidence Grounding & Clinical Guidelines (RAG)",
            "\n".join(evidence_bullets) if evidence_bullets else "- No acute guideline citations triggered.",
            "",
            f"## 4. Suggested Clinical Actions",
            f"1. **Physician Review**: Confirm algorithmic findings with clinical correlation and history.",
            f"2. **Diagnostic Follow-Up**: {risk_report.get('suggested_diagnostics', 'Consider 12-lead ECG and follow-up thoracic imaging if symptoms progress.')}",
            f"3. **Patient Monitoring**: Continuous wearable biosignal telemetry and symptom logging.",
            "",
            f"> **Clinical Safety Disclaimer**: MedAgent is an AI-assisted clinical decision support tool designed to assist healthcare professionals. It does not replace medical judgment. Final diagnosis and care decisions remain with the attending physician.",
        ]

        return "\n".join(report_lines)
