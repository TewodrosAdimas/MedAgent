"""
Clinical Report Generation Agent for MedAgent.
Assembles, structures, and formats multimodal diagnostic findings, RAG citations,
and clinician recommendations into a finalized Clinical Decision Support Report.
"""

from typing import Any, Dict, List, Optional
from src.llm.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("ReportAgent")


class ReportAgent:
    """
    Agent responsible for generating structured, explainable clinical reports.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client if llm_client is not None else LLMClient()

    def generate_report(
        self,
        patient_info: Dict[str, Any],
        signal_report: Dict[str, Any],
        imaging_report: Dict[str, Any],
        risk_report: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Synthesizes all agent outputs into a standardized clinical report.
        """
        logger.info(f"ReportAgent generating clinical report for patient {patient_info.get('patient_id', 'PT-001')}")

        markdown_report = self.llm_client.generate_clinical_summary(
            patient_info=patient_info,
            signal_report=signal_report,
            imaging_report=imaging_report,
            risk_report=risk_report,
            evidence_list=evidence_list,
        )

        return {
            "patient_id": patient_info.get("patient_id", "PT-001"),
            "risk_level": risk_report.get("risk_level", "Low Risk"),
            "markdown_report": markdown_report,
            "signal_summary": signal_report,
            "imaging_summary": imaging_report,
            "risk_summary": risk_report,
            "evidence_citations": evidence_list,
        }
