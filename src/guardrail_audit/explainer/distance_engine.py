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
    # Exemplar-level voting fields — populated when top_exemplar_embeddings present in taxonomy
    exemplar_vote_similarity: float = 0.0   # mean cosine sim of query to top exemplars
    best_matching_exemplar: str = ""        # text of closest exemplar to the query
    best_exemplar_similarity: float = 0.0  # cosine sim to that exemplar


# LTL property φ_ambiguous:
# G( |sim_unsafe(t) - sim_safe(t)| < δ → flag_ambiguous(t) )
# Default δ = 0.001 — calibrated to the UMAP embedding scale (all sims ~0.999x)
AMBIGUITY_THRESHOLD = 0.001

# When is_ambiguous=True, UNSAFE exemplar vote similarity is multiplied by this
# factor before comparing against the SAFE exemplar vote. Values > 1.0 bias toward
# UNSAFE (fewer false negatives, more false positives). Default 1.05 = 5% bias.
UNSAFE_VOTE_WEIGHT = 1.05


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
        unsafe_vote_weight: float = UNSAFE_VOTE_WEIGHT,
    ) -> None:
        taxonomy_path = Path(taxonomy_path)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.prototypes = payload["prototypes"]
        self.keys = list(self.prototypes.keys())
        self.ood_floor = ood_similarity_floor
        self.ambiguity_threshold = ambiguity_threshold
        self.unsafe_vote_weight = unsafe_vote_weight
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

        # Load per-prototype exemplar embedding matrices for voting.
        # Each entry is shape (n_exemplars, dim) or empty if not present (old taxonomy).
        centroid_key = "umap_centroid_vector" if self.umap_enabled else "centroid_vector"
        ex_key = "top_exemplar_embeddings"
        self.exemplar_matrices: list[np.ndarray] = [
            np.array(self.prototypes[k].get(ex_key, []), dtype=np.float64)
            for k in self.keys
        ]
        self.safe_exemplar_matrices: list[np.ndarray] = [
            np.array(self.safe_prototypes[k].get(ex_key, []), dtype=np.float64)
            for k in self.safe_keys
        ]

    def _project(self, query_embedding: np.ndarray) -> np.ndarray:
        """Project query to UMAP space if enabled, otherwise return as-is."""
        if self.umap_enabled and self.reducer is not None:
            reduced = self.reducer.transform(
                query_embedding.astype(np.float64).reshape(1, -1)
            )[0]
            return l2_normalize(reduced.astype(np.float64))
        return query_embedding

    def _exemplar_vote(
        self, query: np.ndarray, proto_idx: int, proto_texts: list[str],
        ex_matrices: list[np.ndarray]
    ) -> tuple[float, str, float]:
        """Compute mean exemplar similarity and locate the closest exemplar.

        Returns (vote_similarity, best_exemplar_text, best_exemplar_similarity).
        Returns (0.0, "", 0.0) when no exemplar embeddings are stored.
        """
        ex = ex_matrices[proto_idx]
        if ex.size == 0:
            return 0.0, "", 0.0
        ex_sims = cosine_similarity(query, ex)
        best_ex_idx = int(np.argmax(ex_sims))
        vote_sim = float(ex_sims.mean())
        best_sim = float(ex_sims[best_ex_idx])
        best_text = proto_texts[best_ex_idx] if best_ex_idx < len(proto_texts) else ""
        return vote_sim, best_text, best_sim

    def match(self, query_embedding: np.ndarray) -> PrototypeMatch:
        query = self._project(query_embedding)

        # ── UNSAFE prototype similarities ─────────────────────────────────
        sims = cosine_similarity(query, self.centroids)
        ranked = np.argsort(sims)[::-1]
        best   = int(ranked[0])
        second = int(ranked[1]) if len(ranked) > 1 else best

        best_sim   = float(sims[best])
        second_sim = float(sims[second])
        margin     = best_sim - second_sim

        # ── Exemplar-level voting: override centroid winner when margin is low ─
        # When top-2 prototypes are within the ambiguity threshold of each other,
        # re-rank them by their mean exemplar similarity to the query.
        # This removes the centroid-averaging artifact for boundary-case prompts.
        if margin < self.ambiguity_threshold and len(ranked) > 1:
            vote0, ex_text0, ex_sim0 = self._exemplar_vote(
                query, best, self.prototypes[self.keys[best]].get("top_exemplars", []),
                self.exemplar_matrices
            )
            vote1, ex_text1, ex_sim1 = self._exemplar_vote(
                query, second, self.prototypes[self.keys[second]].get("top_exemplars", []),
                self.exemplar_matrices
            )
            if vote0 > 0.0 and vote1 > 0.0 and vote1 > vote0:
                # exemplar vote overrides centroid — swap best and second
                best, second = second, best
                best_sim, second_sim = second_sim, best_sim
                margin = best_sim - second_sim

        key   = self.keys[best]
        key2  = self.keys[second]
        proto = self.prototypes[key]
        is_ood = best_sim < self.ood_floor

        # Exemplar vote for the winner
        ev_sim, ev_text, ev_best = self._exemplar_vote(
            query, best, proto.get("top_exemplars", []), self.exemplar_matrices
        )

        # ── SAFE prototype similarities (if available) ────────────────────
        nearest_safe_key   = ""
        nearest_safe_sim   = 0.0
        nearest_safe_label = ""
        is_ambiguous       = False

        if self.safe_centroids is not None and len(self.safe_keys) > 0:
            safe_sims     = cosine_similarity(query, self.safe_centroids)
            best_safe_idx = int(np.argmax(safe_sims))
            nearest_safe_sim   = float(safe_sims[best_safe_idx])
            nearest_safe_key   = self.safe_keys[best_safe_idx]
            nearest_safe_label = self.safe_prototypes[nearest_safe_key].get(
                "label", nearest_safe_key
            )

            # φ_ambiguous: |sim_unsafe - sim_safe| < threshold
            is_ambiguous = abs(best_sim - nearest_safe_sim) < self.ambiguity_threshold

            # ── Merged nearest-prototype: consider SAFE + UNSAFE together ──
            # If the nearest SAFE centroid is closer than the nearest UNSAFE centroid,
            # run a cross-domain exemplar vote when is_ambiguous=True so that
            # boundary-case prompts (e.g. DAN jailbreaks) are not misclassified SAFE.
            if nearest_safe_sim > best_sim:
                safe_proto = self.safe_prototypes[nearest_safe_key]
                s_ev_sim, s_ev_text, s_ev_best = self._exemplar_vote(
                    query, best_safe_idx,
                    safe_proto.get("top_exemplars", []),
                    self.safe_exemplar_matrices
                )

                if is_ambiguous and s_ev_sim > 0.0:
                    # Find UNSAFE prototype with highest mean exemplar similarity
                    best_unsafe_ev_sim = 0.0
                    best_unsafe_ev_idx = best
                    best_unsafe_ev_text = ""
                    best_unsafe_ev_best_sim = 0.0
                    for i in range(len(self.keys)):
                        v_sim, v_text, v_bs = self._exemplar_vote(
                            query, i,
                            self.prototypes[self.keys[i]].get("top_exemplars", []),
                            self.exemplar_matrices
                        )
                        if v_sim > best_unsafe_ev_sim:
                            best_unsafe_ev_sim = v_sim
                            best_unsafe_ev_idx = i
                            best_unsafe_ev_text = v_text
                            best_unsafe_ev_best_sim = v_bs

                    # Apply UNSAFE weight bias: UNSAFE wins if weighted_score > SAFE score
                    if best_unsafe_ev_sim * self.unsafe_vote_weight > s_ev_sim:
                        voted_key = self.keys[best_unsafe_ev_idx]
                        voted_proto = self.prototypes[voted_key]
                        voted_sim = float(sims[best_unsafe_ev_idx])
                        return PrototypeMatch(
                            prototype_key=voted_key,
                            similarity=voted_sim,
                            is_ood=voted_sim < self.ood_floor,
                            label=voted_proto.get("label", voted_key),
                            failure_mode=voted_proto.get("failure_mode", ""),
                            top_exemplars=voted_proto.get("top_exemplars", []),
                            dominant_categories=voted_proto.get("dominant_categories", []),
                            second_prototype_key=nearest_safe_key,
                            second_similarity=nearest_safe_sim,
                            margin=voted_sim - nearest_safe_sim,
                            nearest_safe_key=nearest_safe_key,
                            nearest_safe_similarity=nearest_safe_sim,
                            nearest_safe_label=nearest_safe_label,
                            is_ambiguous=True,
                            exemplar_vote_similarity=best_unsafe_ev_sim,
                            best_matching_exemplar=best_unsafe_ev_text,
                            best_exemplar_similarity=best_unsafe_ev_best_sim,
                        )

                # SAFE exemplar vote wins (or no exemplar embeddings) — return SAFE prototype
                return PrototypeMatch(
                    prototype_key=nearest_safe_key,
                    similarity=nearest_safe_sim,
                    is_ood=False,
                    label=safe_proto.get("label", nearest_safe_key),
                    failure_mode=safe_proto.get("description", ""),
                    top_exemplars=safe_proto.get("top_exemplars", []),
                    dominant_categories=[],
                    second_prototype_key=key,
                    second_similarity=best_sim,
                    margin=nearest_safe_sim - best_sim,
                    nearest_safe_key=nearest_safe_key,
                    nearest_safe_similarity=nearest_safe_sim,
                    nearest_safe_label=nearest_safe_label,
                    is_ambiguous=is_ambiguous,
                    exemplar_vote_similarity=s_ev_sim,
                    best_matching_exemplar=s_ev_text,
                    best_exemplar_similarity=s_ev_best,
                )

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
            exemplar_vote_similarity=ev_sim,
            best_matching_exemplar=ev_text,
            best_exemplar_similarity=ev_best,
        )

