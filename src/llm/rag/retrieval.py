"""
Medical RAG Retrieval Engine for MedAgent.
Retrieves relevant medical guidelines, evidence, and citations given patient findings.
"""

from typing import Any, Dict, List, Optional
from src.llm.rag.vector_store import MedicalVectorStore
from src.utils.logger import get_logger

logger = get_logger("MedicalRetriever")


class MedicalRetriever:
    """
    High-level clinical knowledge retrieval interface for Evidence Agent.
    """

    def __init__(self, vector_store: Optional[MedicalVectorStore] = None):
        self.vector_store = vector_store if vector_store is not None else MedicalVectorStore()

    def retrieve_evidence(
        self,
        signal_finding: Optional[str] = None,
        imaging_finding: Optional[str] = None,
        patient_history: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves supporting medical guidelines based on multimodal patient findings.
        """
        query_parts = []
        if signal_finding and signal_finding != "Normal":
            query_parts.append(f"ECG cardiac finding {signal_finding}")
        if imaging_finding and imaging_finding != "Normal":
            query_parts.append(f"Chest radiograph finding {imaging_finding}")
        if patient_history:
            query_parts.append(f"Patient history {patient_history}")

        if not query_parts:
            query_parts = ["Normal physiological health and preventive screening guidelines"]

        query = " ".join(query_parts)
        results = self.vector_store.similarity_search(query, top_k=top_k)

        formatted_evidence = []
        for doc, score in results:
            formatted_evidence.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "evidence_text": doc["text"],
                "source": doc["source"],
                "doi": doc.get("doi", ""),
                "relevance_score": round(score, 4),
            })

        return formatted_evidence
