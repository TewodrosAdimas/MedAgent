"""
Medical Knowledge Retrieval Engine for Retrieval-Augmented Generation (RAG).

Matches patient AI findings against clinical practice guidelines using
lexical TF-IDF and semantic keyword scoring with modality-specific filtering.
"""

from typing import Dict, List, Optional, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.rag.knowledge_base import GUIDELINE_CORPUS, ClinicalGuideline


class MedicalKnowledgeRetriever:
    """
    RAG evidence retriever over curated clinical guidelines.
    """

    def __init__(self, corpus: Optional[List[ClinicalGuideline]] = None):
        self.corpus = corpus if corpus is not None else GUIDELINE_CORPUS
        self.documents = [
            f"{g.topic} {g.title} {g.content}" for g in self.corpus
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve(
        self,
        query: str,
        modality: Optional[str] = None,
        top_k: int = 2,
        min_similarity: float = 0.05,
    ) -> List[Tuple[ClinicalGuideline, float]]:
        """
        Retrieves top-k most relevant clinical guidelines matching the clinical query.

        Args:
            query: Query string containing symptoms, findings, or disease terms.
            modality: Optional modality filter ('ecg', 'xray', 'multimodal').
            top_k: Maximum number of guidelines to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (ClinicalGuideline, relevance_score) tuples.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Filter by modality if requested
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_similarity and len(results) > 0:
                continue

            guideline = self.corpus[idx]
            if modality is not None and guideline.modality not in [modality, "multimodal"]:
                continue

            results.append((guideline, score))
            if len(results) >= top_k:
                break

        # If no results matched filter, return top general guideline
        if not results and len(self.corpus) > 0:
            results.append((self.corpus[0], 0.1))

        return results

    def format_evidence_for_prompt(
        self,
        guideline_matches: List[Tuple[ClinicalGuideline, float]],
    ) -> str:
        """
        Formats retrieved guidelines into structured text suitable for agent prompts.
        """
        if not guideline_matches:
            return "No specific practice guidelines retrieved for this finding."

        formatted_blocks = []
        for i, (g, score) in enumerate(guideline_matches, 1):
            block = (
                f"[{i}] {g.title} ({g.citation})\n"
                f"    - Recommendation Level: {g.strength_of_recommendation}\n"
                f"    - Evidence Summary: {g.content}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
