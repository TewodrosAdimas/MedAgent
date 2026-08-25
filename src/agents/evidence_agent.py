"""
Evidence Retrieval Agent for MedAgent.
Retrieves and grounds multimodal AI findings against verified clinical practice guidelines,
medical literature, and peer-reviewed studies using RAG.
"""

from typing import Any, Dict, List, Optional
from src.llm.rag.retrieval import MedicalRetriever
from src.utils.logger import get_logger

logger = get_logger("EvidenceAgent")


class EvidenceAgent:
    """
    Agent responsible for clinical knowledge grounding and medical literature citations.
    """

    def __init__(self, retriever: Optional[MedicalRetriever] = None):
        self.retriever = retriever if retriever is not None else MedicalRetriever()

    def retrieve_guidelines(
        self,
        signal_finding: Optional[str] = None,
        imaging_finding: Optional[str] = None,
        patient_history: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Queries the clinical vector store to find supporting medical guidelines.
        """
        logger.info(f"EvidenceAgent retrieving guidelines for Signal='{signal_finding}', Imaging='{imaging_finding}'")
        evidence = self.retriever.retrieve_evidence(
            signal_finding=signal_finding,
            imaging_finding=imaging_finding,
            patient_history=patient_history,
            top_k=top_k,
        )
        return evidence
