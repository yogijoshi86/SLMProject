"""K-means sweep, silhouette selection, and prototype taxonomy build (Days 7-10)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from guardrail_audit.clustering.normalize import cosine_similarity, l2_normalize


@dataclass
class SweepResult:
    k: int
    silhouette: float
    inertia: float


def sweep_k(
    embeddings_norm: np.ndarray,
    k_min: int,
    k_max: int,
    n_init: int,
    seed: int,
    k_cap: int | None = None,
) -> list[SweepResult]:
    """Fit K-means for each k and record silhouette + inertia."""
    upper = min(k_max, k_cap) if k_cap else k_max
    results: list[SweepResult] = []
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(embeddings_norm)
        score = float(silhouette_score(embeddings_norm, labels))
        results.append(SweepResult(k=k, silhouette=score, inertia=float(km.inertia_)))
        print(f"k={k:2d} | silhouette={score:.4f} | inertia={km.inertia_:.1f}")
    return results


def select_best_k(results: list[SweepResult]) -> int:
    """Pick the k with the peak silhouette score."""
    return max(results, key=lambda r: r.silhouette).k


def build_prototypes(
    data_path: str | Path,
    taxonomy_path: str | Path,
    k_min: int,
    k_max: int,
    n_init: int,
    seed: int,
    top_exemplars: int,
    k_cap: int | None = None,
) -> dict:
    """End-to-end Phase 2: normalize, sweep, select k*, extract centroids + exemplars."""
    checkpoint = torch.load(data_path, map_location="cpu")
    embeddings = checkpoint["embeddings"].numpy().astype(np.float64)
    metadata = checkpoint["metadata"]

    embeddings_norm = l2_normalize(embeddings)

    results = sweep_k(embeddings_norm, k_min, k_max, n_init, seed, k_cap)
    best_k = select_best_k(results)
    best_score = next(r.silhouette for r in results if r.k == best_k)
    print(f"\nSelected k*={best_k} (silhouette={best_score:.4f})")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(embeddings_norm)
    centroids = km.cluster_centers_

    prototypes: dict[str, dict] = {}
    for cluster_idx in range(best_k):
        centroid = centroids[cluster_idx]
        sims = cosine_similarity(centroid, embeddings_norm)
        top = np.argsort(sims)[::-1][:top_exemplars]

        member_mask = labels == cluster_idx
        member_cats: list[str] = []
        for m in np.where(member_mask)[0]:
            member_cats.extend(metadata[m].get("categories", []))

        prototypes[f"prototype_{cluster_idx}"] = {
            "centroid_vector": centroid.tolist(),
            "cluster_size": int(member_mask.sum()),
            "top_exemplars": [metadata[int(i)]["text"] for i in top],
            "exemplar_categories": [metadata[int(i)].get("categories", []) for i in top],
            "dominant_categories": _rank_categories(member_cats),
            # Filled in by the human review step (Day 10).
            "label": "TODO: assign after thematic review",
            "failure_mode": "TODO",
        }

    output = {
        "meta": {
            "best_k": best_k,
            "best_silhouette": best_score,
            "sweep": [asdict(r) for r in results],
            "n_embeddings": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
        },
        "prototypes": prototypes,
    }

    taxonomy_path = Path(taxonomy_path)
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote taxonomy ({best_k} prototypes) to {taxonomy_path}")
    return output


def _rank_categories(categories: list[str]) -> list[str]:
    """Return category codes ordered by frequency within a cluster."""
    counts: dict[str, int] = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    return [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
