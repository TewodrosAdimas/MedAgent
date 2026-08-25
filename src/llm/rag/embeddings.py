"""
Medical Text Embedding Module for MedAgent.
Provides normalized dense vector embeddings for clinical guidelines and literature.
"""

from typing import List, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class MedicalEmbeddings:
    """
    Lightweight, high-speed dense embedding engine for clinical texts.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.reducer = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fits vocabulary and dimensionality reduction on clinical corpus."""
        tfidf = self.vectorizer.fit_transform(texts)
        # Ensure n_components does not exceed n_features or n_samples
        n_comp = min(self.embedding_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_comp < 2:
            n_comp = 2
        self.reducer = TruncatedSVD(n_components=n_comp, random_state=42)
        self.reducer.fit(tfidf)
        self.is_fitted = True

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of medical documents."""
        if not self.is_fitted:
            self.fit(texts)
        tfidf = self.vectorizer.transform(texts)
        dense = self.reducer.transform(tfidf)
        # L2 Normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D vector."""
        if not self.is_fitted:
            self.fit([query, query + " medical reference"])
        tfidf = self.vectorizer.transform([query])
        dense = self.reducer.transform(tfidf)
        norm = np.linalg.norm(dense) + 1e-8
        return (dense[0] / norm).astype(np.float32)
