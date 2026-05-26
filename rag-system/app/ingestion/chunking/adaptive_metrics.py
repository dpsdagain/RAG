"""Adaptive chunk quality metrics: ICC and DCC.

Computes Intrachunk Cohesion (ICC) and Document Contextual Coherence (DCC)
to evaluate the semantic quality of generated chunks.
"""
from __future__ import annotations

import numpy as np


def compute_icc(sentence_embeddings: list[list[float]]) -> float:
    """Compute Intrachunk Cohesion: mean pairwise cosine similarity.

    Args:
        sentence_embeddings: Embeddings of sentences within a chunk.

    Returns:
        ICC score in [0, 1]. Higher = more cohesive.
    """
    if len(sentence_embeddings) < 2:
        return 1.0

    embs = np.array(sentence_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embs / norms

    # Pairwise cosine similarity matrix
    sim_matrix = normalized @ normalized.T
    n = len(sentence_embeddings)

    # Mean of upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise_sims = sim_matrix[mask]

    return float(np.mean(pairwise_sims)) if len(pairwise_sims) > 0 else 1.0


def compute_dcc(chunk_embedding: list[float], doc_embedding: list[float]) -> float:
    """Compute Document Contextual Coherence: cosine similarity.

    Args:
        chunk_embedding: Embedding of the chunk.
        doc_embedding: Embedding of the full document.

    Returns:
        DCC score in [-1, 1]. Higher = more contextually relevant.
    """
    a = np.array(chunk_embedding, dtype=np.float32)
    b = np.array(doc_embedding, dtype=np.float32)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(dot / (norm_a * norm_b))


def evaluate_chunk_quality(icc: float, dcc: float) -> str:
    """Evaluate overall chunk quality based on ICC and DCC scores.

    Args:
        icc: Intrachunk Cohesion score.
        dcc: Document Contextual Coherence score.

    Returns:
        Quality label: 'good', 'fair', or 'poor'.
    """
    if icc < 0.2:
        return "poor"
    elif icc < 0.4 or dcc < 0.3:
        return "fair"
    else:
        return "good"
