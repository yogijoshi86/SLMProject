"""Runtime cosine matching of a query embedding to prototypes (Day 11).

When the taxonomy was built with UMAP reduction (meta.umap_enabled=True),
the query embedding is projected into the same reduced space before matching.
Falls back to full-dim cosine matching for old taxonomy files.

When SAFE prototypes are present in the taxonomy (safe_prototypes key),
the engine also computes similarity against SAFE centroids and flags
structurally ambiguous prompts — those that sit equidistant between
a SAFE and an UNSAFE centroid.
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
    # second-best UNSAFE prototype — useful for margin/uncertainty analysis
    second_prototype_key: str = ""
    second_similarity: float = 0.0
    margin: float = 0.0          # similarity - second_similarity; low = boundary case
    # SAFE prototype fields — populated when safe_prototypes present in taxonomy
    nearest_safe_key: str = ""
    nearest_safe_similarity: float = 0.0
    nearest_safe_label: str = ""
    is_ambiguous: bool = False   # True when |unsafe_sim - safe_sim| < ambiguity_threshold


# LTL property φ_ambiguous:
# G( |sim_unsafe(t) - sim_safe(t)| < δ → flag_ambiguous(t) )
# Default δ = 0.001 — calibrated to the UMAP embedding scale (all sims ~0.999x)
AMBIGUITY_THRESHOLD = 0.001


class DistanceEngine:
    """Loads a taxonomy once and matches query embeddings to nearest prototype.

    If the taxonomy was built with UMAP (meta.umap_enabled=True), loads the
    persisted UMAP reducer and projects each query into the reduced space before
    cosine matching. This matches the geometry used during clustering.

    If safe_prototypes are present in the taxonomy, also computes similarity
    against SAFE centroids and flags structurally ambiguous prompts via
    the is_ambiguous field on PrototypeMatch.
    """

    def __init__(
        self,
        taxonomy_path: str | Path,
        ood_similarity_floor: float = 0.35,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    ) -> None:
        taxonomy_path = Path(taxonomy_path)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.prototypes = payload["prototypes"]
        self.keys = list(self.prototypes.keys())
        self.ood_floor = ood_similarity_floor
        self.ambiguity_threshold = ambiguity_threshold
        meta = payload.get("meta", {})

        # Load SAFE prototypes if present
        self.safe_prototypes = payload.get("safe_prototypes", {})
        self.safe_keys = list(self.safe_prototypes.keys())
        self.safe_centroids: np.ndarray | None = None

        self.umap_enabled = meta.get("umap_enabled", False)
        self.reducer = None

        if self.umap_enabled:
            reducer_path = meta.get("reducer_path")
            if reducer_path is None:
                raise ValueError(
                    "taxonomy meta.umap_enabled=True but meta.reducer_path is missing. "
                    "Re-run build_prototypes() to regenerate the taxonomy and reducer."
                )
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
            self.centroids = np.array(
                [self.prototypes[k]["umap_centroid_vector"] for k in self.keys],
                dtype=np.float64
            )
            if self.safe_keys:
                self.safe_centroids = np.array(
                    [self.safe_prototypes[k]["umap_centroid_vector"] for k in self.safe_keys],
                    dtype=np.float64
                )
                print(f"DistanceEngine: UMAP mode (dim={self.centroids.shape[1]}, "
                      f"reducer={reducer_path.name}, "
                      f"{len(self.keys)} UNSAFE + {len(self.safe_keys)} SAFE prototypes)")
            else:
                print(f"DistanceEngine: UMAP mode (dim={self.centroids.shape[1]}, "
                      f"reducer={reducer_path.name}, no SAFE prototypes)")
        else:
            self.centroids = np.array(
                [self.prototypes[k]["centroid_vector"] for k in self.keys],
                dtype=np.float64
            )
            if self.safe_keys:
                self.safe_centroids = np.array(
                    [self.safe_prototypes[k]["centroid_vector"] for k in self.safe_keys],
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

        # ── UNSAFE prototype matching ─────────────────────────────────────
        sims = cosine_similarity(query, self.centroids)
        ranked = np.argsort(sims)[::-1]
        best   = int(ranked[0])
        second = int(ranked[1]) if len(ranked) > 1 else best

        best_sim   = float(sims[best])
        second_sim = float(sims[second])
        margin     = best_sim - second_sim

        key   = self.keys[best]
        key2  = self.keys[second]
        proto = self.prototypes[key]
        is_ood = best_sim < self.ood_floor

        # ── SAFE prototype matching (if available) ────────────────────────
        nearest_safe_key = ""
        nearest_safe_sim = 0.0
        nearest_safe_label = ""
        is_ambiguous = False

        if self.safe_centroids is not None and len(self.safe_keys) > 0:
            safe_sims = cosine_similarity(query, self.safe_centroids)
            best_safe = int(np.argmax(safe_sims))
            nearest_safe_sim = float(safe_sims[best_safe])
            nearest_safe_key = self.safe_keys[best_safe]
            nearest_safe_label = self.safe_prototypes[nearest_safe_key].get("label", nearest_safe_key)

            # φ_ambiguous: |sim_unsafe - sim_safe| < threshold → structurally ambiguous
            is_ambiguous = abs(best_sim - nearest_safe_sim) < self.ambiguity_threshold

        return PrototypeMatch(
            prototype_key=key,
            similarity=best_sim,
            is_ood=is_ood,
            label="Uncategorized Attack Pattern" if is_ood else proto.get("label", key),
            failure_mode=proto.get("failure_mode", ""),
            top_exemplars=proto.get("top_exemplars", []),
            dominant_categories=proto.get("dominant_categories", []),
            second_prototype_key=key2,
            second_similarity=second_sim,
            margin=margin,
            nearest_safe_key=nearest_safe_key,
            nearest_safe_similarity=nearest_safe_sim,
            nearest_safe_label=nearest_safe_label,
            is_ambiguous=is_ambiguous,
        )

