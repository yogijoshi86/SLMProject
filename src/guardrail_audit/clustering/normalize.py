"""L2 normalization + reusable vector math (Days 6, 11)."""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization: x' = x / ||x||_2. Safe against zero vectors."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / max(norm, eps)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def cosine_similarity(query: np.ndarray, centroids: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity between a 1-D query and each row of ``centroids``.

    Normalizes BOTH sides. K-means centroids fit on unit vectors are not themselves
    unit-norm, so this is the correct cosine — not a raw dot product.
    """
    q = l2_normalize(np.asarray(query, dtype=np.float64))
    c = l2_normalize(np.asarray(centroids, dtype=np.float64))
    return c @ q
