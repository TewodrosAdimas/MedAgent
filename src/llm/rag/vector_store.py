"""
Clinical Vector Store for MedAgent RAG.
Stores and indexes verified medical guidelines, clinical practice recommendations,
and literature citations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.llm.rag.embeddings import MedicalEmbeddings
from src.utils.logger import get_logger

logger = get_logger("VectorStore")

# Pre-populated Clinical Knowledge Base
CLINICAL_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "AHA-ARRHYTHMIA-01",
        "title": "AHA/ACC Guidelines on Ventricular Arrhythmias",
        "category": "Cardiology",
        "text": "Premature ventricular contractions (PVCs) appearing frequently or in patients with underlying cardiac structural disease or hypertension indicate elevated cardiovascular risk and warrant 12-lead ECG, echocardiographic evaluation, and Holter monitoring.",
        "source": "AHA/ACC/HRS 2017 Guidelines for Management of Patients With Ventricular Arrhythmias",
        "doi": "10.1161/CIR.0000000000000549",
    },
    {
        "id": "AHA-SUPRAVENTRICULAR-02",
        "title": "AHA/ACC Guidelines for Supraventricular Tachycardia",
        "category": "Cardiology",
        "text": "Supraventricular ectopic beats (SVEBs) and atrial premature contractions require clinical correlation with symptoms, resting heart rate, and electrolyte balance. Isolated occurrences in asymptomatic patients are often benign, but increased burden warrants ambulatory monitoring.",
        "source": "ACC/AHA/HRS 2015 Guideline for the Management of Adult Patients With Supraventricular Tachycardia",
        "doi": "10.1161/CIR.0000000000000311",
    },
    {
        "id": "ACR-PNEUMONIA-01",
        "title": "ACR Appropriateness Criteria for Acute Respiratory Illness",
        "category": "Pulmonology / Radiology",
        "text": "Focal alveolar opacities, consolidations, or air bronchograms on chest radiography in patients with respiratory symptoms, fever, or hypoxemia are primary diagnostic indicators of community-acquired pneumonia requiring prompt antimicrobial and supportive therapy.",
        "source": "American College of Radiology Appropriateness Criteria for Acute Respiratory Illness",
        "doi": "10.1016/j.jacr.2018.09.023",
    },
    {
        "id": "ACC-CARDIOMEGALY-01",
        "title": "Radiological Evaluation of Cardiomegaly & Heart Failure",
        "category": "Cardiopulmonary",
        "text": "A cardiothoracic ratio (CTR) greater than 0.50 on standard posteroanterior chest radiographs indicates cardiomegaly, strongly associated with chronic hypertension, dilated cardiomyopathy, or pericardial effusion, warranting echocardiography and BNP measurement.",
        "source": "AHA/ACC/HFSA 2022 Guideline for the Management of Heart Failure",
        "doi": "10.1161/CIR.0000000000001063",
    },
    {
        "id": "FLEISCHNER-EFFUSION-01",
        "title": "Fleischner Society Guidelines on Pleural Disease",
        "category": "Pulmonology / Radiology",
        "text": "Blunting of the lateral or posterior costophrenic angle indicates pleural effusion (typically >175 mL fluid). In conjunction with cardiac abnormalities or elevated systemic venous pressure, transudative congestive heart failure etiology should be investigated.",
        "source": "Fleischner Society Statement on Imaging of Pleural Disease",
        "doi": "10.1148/radiol.2018170882",
    },
    {
        "id": "MULTIMODAL-INTERACTION-01",
        "title": "Cardiopulmonary Interaction in Acute Clinical Deterioration",
        "category": "Multimodal Medicine",
        "text": "Concomitant cardiac arrhythmias (e.g. PVCs or sinus tachycardia) in the presence of acute pulmonary infiltrates or pleural effusions exponentially increases the risk of acute cardiopulmonary decompensation and hypoxic myocardial strain, mandating urgent physician review.",
        "source": "European Society of Cardiology (ESC) / European Respiratory Society (ERS) Consensus",
        "doi": "10.1093/eurheartj/ehz897",
    },
]


class MedicalVectorStore:
    """
    In-memory vector database indexed with clinical knowledge.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embeddings = MedicalEmbeddings(embedding_dim=embedding_dim)
        self.documents: List[Dict[str, Any]] = []
        self.doc_vectors: Optional[np.ndarray] = None
        self._populate_default_knowledge()

    def _populate_default_knowledge(self):
        """Initializes the vector store with verified clinical practice guidelines."""
        texts = [f"{doc['title']}. {doc['text']}" for doc in CLINICAL_KNOWLEDGE_BASE]
        self.embeddings.fit(texts)
        self.doc_vectors = self.embeddings.embed_documents(texts)
        self.documents = list(CLINICAL_KNOWLEDGE_BASE)
        logger.info(f"MedicalVectorStore initialized with {len(self.documents)} clinical guideline documents.")

    def add_documents(self, new_docs: List[Dict[str, Any]]):
        """Adds custom medical documents to the vector store."""
        all_docs = self.documents + new_docs
        texts = [f"{doc['title']}. {doc['text']}" for doc in all_docs]
        self.embeddings.fit(texts)
        self.doc_vectors = self.embeddings.embed_documents(texts)
        self.documents = all_docs

    def similarity_search(
        self,
        query: str,
        top_k: int = 2,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs cosine similarity search against clinical knowledge base.

        Returns:
            List of (document_dict, similarity_score) tuples sorted by relevance.
        """
        if self.doc_vectors is None or len(self.documents) == 0:
            return []

        query_vector = self.embeddings.embed_query(query)  # (embed_dim,)
        scores = np.dot(self.doc_vectors, query_vector)   # Cosine similarity since vectors are L2-normalized

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.documents[idx], float(scores[idx])) for idx in top_indices]
        return results
