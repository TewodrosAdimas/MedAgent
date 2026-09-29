"""
Evidence Retrieval Agent for MedAgent (Medical RAG).

Queries the curated clinical practice guidelines and medical literature
to provide grounded citations and evidence-backed diagnostic context.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.llm.rag.knowledge_base import ClinicalGuideline
from src.llm.rag.retriever import MedicalKnowledgeRetriever


@dataclass
class EvidenceResult:
    retrieved_guidelines: List[ClinicalGuideline]
    formatted_evidence: str
    citations: List[str]


class EvidenceAgent:
    """
    Agent responsible for grounding clinical observations in validated guidelines.
    """

    def __init__(self, retriever: Optional[MedicalKnowledgeRetriever] = None):
        self.retriever = retriever or MedicalKnowledgeRetriever()

    def retrieve_evidence(
        self,
        signal_findings: str,
        imaging_findings: str,
        patient_context: Optional[str] = None,
        top_k: int = 2,
    ) -> EvidenceResult:
        """
        Retrieves evidence relevant to the multimodal patient case.
        """
        combined_query = f"{signal_findings} {imaging_findings} {patient_context or ''}"
        matches = self.retriever.retrieve(query=combined_query, top_k=top_k)

        guidelines = [g for g, _ in matches]
        citations = [f"{g.title} ({g.citation})" for g in guidelines]
        formatted_text = self.retriever.format_evidence_for_prompt(matches)

        return EvidenceResult(
            retrieved_guidelines=guidelines,
            formatted_evidence=formatted_text,
            citations=citations,
        )
