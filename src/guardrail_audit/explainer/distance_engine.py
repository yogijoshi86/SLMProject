"""Runtime cosine matching of a query embedding to prototypes (Day 11).

When the taxonomy was built with UMAP reduction (meta.umap_enabled=True),
the query embedding is projected into the same reduced space before matching.
Falls back to full-dim cosine matching for old taxonomy files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guardrail_audit.clustering.normalize import cosine_similarity, l2_normalize


@dataclass
class PrototypeMatch:
    prototype_key: str
    similarity: float
    is_ood: bool
    label: str
    failure_mode: str
    top_exemplars: list[str]
    dominant_categories: list[str]


class DistanceEngine:
    """Loads a taxonomy once and matches query embeddings to nearest prototype.

    If the taxonomy was built with UMAP (meta.umap_enabled=True), loads the
    persisted UMAP reducer and projects each query into the reduced space before
    cosine matching. This matches the geometry used during clustering.
    """

    def __init__(self, taxonomy_path: str | Path, ood_similarity_floor: float = 0.35) -> None:
        taxonomy_path = Path(taxonomy_path)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.prototypes = payload["prototypes"]
        self.keys = list(self.prototypes.keys())
        self.ood_floor = ood_similarity_floor
        meta = payload.get("meta", {})

        self.umap_enabled = meta.get("umap_enabled", False)
        self.reducer = None

        if self.umap_enabled:
            # Load the persisted UMAP reducer
            reducer_path = meta.get("reducer_path")
            if reducer_path is None:
                raise ValueError(
                    "taxonomy meta.umap_enabled=True but meta.reducer_path is missing. "
                    "Re-run build_prototypes() to regenerate the taxonomy and reducer."
                )
            # Resolve relative to taxonomy file location
            reducer_path = Path(reducer_path)
            if not reducer_path.is_absolute():
                reducer_path = taxonomy_path.parent / reducer_path.name
            if not reducer_path.exists():
                raise FileNotFoundError(
                    f"UMAP reducer not found at {reducer_path}. "
                    "Re-run 02_clustering.ipynb to regenerate it, then backup to Drive."
                )
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("joblib is required to load the UMAP reducer: pip install joblib") from exc
            self.reducer = joblib.load(reducer_path)
            # Use UMAP-space centroids
            self.centroids = np.array(
                [self.prototypes[k]["umap_centroid_vector"] for k in self.keys],
                dtype=np.float64
            )
            print(f"DistanceEngine: UMAP mode (dim={self.centroids.shape[1]}, "
                  f"reducer={reducer_path.name})")
        else:
            # Backward-compatible: full-dim centroids
            self.centroids = np.array(
                [self.prototypes[k]["centroid_vector"] for k in self.keys],
                dtype=np.float64
            )

    def _project(self, query_embedding: np.ndarray) -> np.ndarray:
        """Project query to UMAP space if enabled, otherwise return as-is."""
        if self.umap_enabled and self.reducer is not None:
            reduced = self.reducer.transform(
                query_embedding.astype(np.float64).reshape(1, -1)
            )[0]
            return l2_normalize(reduced.astype(np.float64))
        return query_embedding

    def match(self, query_embedding: np.ndarray) -> PrototypeMatch:
        query = self._project(query_embedding)
        sims = cosine_similarity(query, self.centroids)
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        key = self.keys[best]
        proto = self.prototypes[key]
        is_ood = best_sim < self.ood_floor
        return PrototypeMatch(
            prototype_key=key,
            similarity=best_sim,
            is_ood=is_ood,
            label="Uncategorized Attack Pattern" if is_ood else proto.get("label", key),
            failure_mode=proto.get("failure_mode", ""),
            top_exemplars=proto.get("top_exemplars", []),
            dominant_categories=proto.get("dominant_categories", []),
        )
