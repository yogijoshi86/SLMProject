"""Runtime cosine matching of a query embedding to prototypes (Day 11)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guardrail_audit.clustering.normalize import cosine_similarity


@dataclass
class PrototypeMatch:
    prototype_key: str
    similarity: float
    is_ood: bool               # True when similarity < ood_floor
    label: str
    failure_mode: str
    top_exemplars: list[str]
    dominant_categories: list[str]


class DistanceEngine:
    """Loads a taxonomy once and matches query embeddings to nearest prototype."""

    def __init__(self, taxonomy_path: str | Path, ood_similarity_floor: float = 0.35) -> None:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.prototypes = payload["prototypes"]
        self.keys = list(self.prototypes.keys())
        self.centroids = np.array(
            [self.prototypes[k]["centroid_vector"] for k in self.keys], dtype=np.float64
        )
        self.ood_floor = ood_similarity_floor

    def match(self, query_embedding: np.ndarray) -> PrototypeMatch:
        sims = cosine_similarity(query_embedding, self.centroids)
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
