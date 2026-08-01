"""K-means sweep, silhouette selection, and prototype taxonomy build (Days 7-10).

Supports optional UMAP dimensionality reduction before clustering.
When use_umap=True (recommended), reduces embeddings to n_components dimensions
before K-means, producing dramatically better silhouette scores in practice
(S≈0.41 at 50-dim vs S≈0.04 at 4096-dim for Llama-Guard-3-8B embeddings).
The fitted UMAP reducer is persisted so DistanceEngine can project new query
embeddings into the same space at inference time.
"""

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
    use_umap: bool = True,
    umap_n_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
) -> dict:
    """End-to-end Phase 2: normalize, (optionally reduce), sweep, select k*, extract centroids.

    When use_umap=True (default), fits UMAP on the L2-normalised embeddings and runs
    K-means in the reduced space. This dramatically improves silhouette scores by
    mitigating the curse of dimensionality in 4096-dim hidden-state space.

    The fitted UMAP reducer is saved to <taxonomy_path_parent>/umap_reducer.pkl so
    DistanceEngine can project new query embeddings at inference time.
    """
    checkpoint = torch.load(data_path, map_location="cpu")
    all_embeddings = checkpoint["embeddings"].numpy().astype(np.float64)
    all_metadata = checkpoint["metadata"]

    train_indices = checkpoint.get("train_indices", list(range(len(all_metadata))))
    if "train_indices" in checkpoint:
        n_test = len(checkpoint.get("test_indices", []))
        print(f"Using train split: {len(train_indices)} embeddings "
              f"(held out {n_test} for A/B benchmark)")

    embeddings = all_embeddings[train_indices]
    metadata = [all_metadata[i] for i in train_indices]
    embeddings_norm = l2_normalize(embeddings)

    reducer_path: str | None = None

    if use_umap:
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for UMAP reduction. "
                "Install with: pip install umap-learn"
            ) from exc

        import joblib
        print(f"\nFitting UMAP (n_components={umap_n_components}, metric=cosine)...")
        reducer = UMAP(
            n_components=umap_n_components,
            metric="cosine",
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=seed,
        )
        embeddings_reduced = reducer.fit_transform(embeddings_norm)
        embeddings_fit = l2_normalize(embeddings_reduced.astype(np.float64))
        print(f"UMAP done: {embeddings_norm.shape} → {embeddings_fit.shape}")

        reducer_path = str(Path(taxonomy_path).parent / "umap_reducer.pkl")
        joblib.dump(reducer, reducer_path)
        print(f"Saved UMAP reducer → {reducer_path}")
    else:
        embeddings_fit = embeddings_norm

    print(f"\nK-means sweep (use_umap={use_umap})...")
    results = sweep_k(embeddings_fit, k_min, k_max, n_init, seed, k_cap)
    best_k = select_best_k(results)
    best_score = next(r.silhouette for r in results if r.k == best_k)
    print(f"\nSelected k*={best_k} (silhouette={best_score:.4f})")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(embeddings_fit)
    centroids_fit = km.cluster_centers_  # shape: (k, n_components or 4096)

    # For each centroid, also compute the corresponding full-dim centroid
    # (mean of full-dim embeddings assigned to that cluster)
    prototypes: dict[str, dict] = {}
    for cluster_idx in range(best_k):
        centroid_fit = centroids_fit[cluster_idx]

        # Nearest exemplars in the fitting space
        sims = cosine_similarity(centroid_fit, embeddings_fit)
        top = np.argsort(sims)[::-1][:top_exemplars]

        member_mask = labels == cluster_idx
        member_cats: list[str] = []
        for m in np.where(member_mask)[0]:
            member_cats.extend(metadata[m].get("categories", []))

        # Full-dim centroid for backward-compat (mean of assigned embeddings_norm)
        full_centroid = embeddings_norm[member_mask].mean(axis=0).tolist()

        prototypes[f"prototype_{cluster_idx}"] = {
            "centroid_vector": full_centroid,              # 4096-dim, backward-compat
            "umap_centroid_vector": centroid_fit.tolist(), # reduced-dim, used when umap_enabled
            "cluster_size": int(member_mask.sum()),
            "top_exemplars": [metadata[int(i)]["text"] for i in top],
            "exemplar_categories": [metadata[int(i)].get("categories", []) for i in top],
            "dominant_categories": _rank_categories(member_cats),
            "label": "TODO: assign after thematic review",
            "failure_mode": "TODO",
        }

    output = {
        "meta": {
            "best_k": best_k,
            "best_silhouette": best_score,
            "sweep": [asdict(r) for r in results],
            "n_train": int(embeddings.shape[0]),
            "n_test": len(checkpoint.get("test_indices", [])),
            "dim": int(embeddings.shape[1]),
            "umap_enabled": use_umap,
            "umap_n_components": umap_n_components if use_umap else None,
            "reducer_path": reducer_path,
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
    counts: dict[str, int] = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    return [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]


def build_safe_prototypes(
    data_path: str | Path,
    taxonomy_path: str | Path,
    k_min: int = 3,
    k_max: int = 10,
    n_init: int = 10,
    seed: int = 42,
    top_exemplars: int = 5,
    k_cap: int | None = 8,
    umap_n_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
) -> dict:
    """Cluster SAFE-classified embeddings to discover benign content prototypes.

    Reads safe_embeddings + safe_metadata from the .pt file produced by
    extract_unsafe_embeddings(record_safe=True). Uses the UNSAFE UMAP reducer
    already fitted in build_prototypes() to project SAFE embeddings into the
    same 50-dim space, then runs K-means to find k* SAFE clusters.

    The SAFE centroids are appended to the existing taxonomy JSON under
    a top-level "safe_prototypes" key so DistanceEngine can compare a new
    prompt against both UNSAFE and SAFE centroids simultaneously.
    """
    checkpoint = torch.load(data_path, map_location="cpu")

    safe_emb = checkpoint.get("safe_embeddings")
    safe_meta = checkpoint.get("safe_metadata", [])

    if safe_emb is None or len(safe_meta) == 0:
        raise ValueError(
            "No SAFE embeddings found in .pt file. "
            "Re-run 01_extraction.ipynb with record_safe=True."
        )

    safe_emb_np = safe_emb.numpy().astype(np.float64)
    print(f"SAFE embeddings loaded: {safe_emb_np.shape}")

    # Load existing taxonomy to reuse the UNSAFE UMAP reducer
    taxonomy_path = Path(taxonomy_path)
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    meta = taxonomy.get("meta", {})
    umap_enabled = meta.get("umap_enabled", False)

    embeddings_norm = l2_normalize(safe_emb_np)

    if umap_enabled:
        import joblib
        reducer_path = meta.get("reducer_path")
        if reducer_path is None:
            raise ValueError("taxonomy meta.reducer_path missing — re-run build_prototypes() first.")
        reducer_path = Path(reducer_path)
        if not reducer_path.is_absolute():
            reducer_path = taxonomy_path.parent / reducer_path.name
        reducer = joblib.load(reducer_path)
        print(f"Loaded UMAP reducer from {reducer_path.name}")

        # Project SAFE embeddings into same space as UNSAFE prototypes
        reduced = reducer.transform(embeddings_norm)
        embeddings_fit = l2_normalize(reduced.astype(np.float64))
        print(f"SAFE embeddings projected: {embeddings_norm.shape} → {embeddings_fit.shape}")
    else:
        embeddings_fit = embeddings_norm

    print(f"\nK-means sweep on SAFE embeddings...")
    results = sweep_k(embeddings_fit, k_min, k_max, n_init, seed, k_cap)
    best_k = select_best_k(results)
    best_score = next(r.silhouette for r in results if r.k == best_k)
    print(f"\nSelected k*={best_k} (silhouette={best_score:.4f})")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(embeddings_fit)
    centroids_fit = km.cluster_centers_

    safe_prototypes: dict[str, dict] = {}
    for cluster_idx in range(best_k):
        centroid_fit = centroids_fit[cluster_idx]
        sims = cosine_similarity(centroid_fit, embeddings_fit)
        top = np.argsort(sims)[::-1][:top_exemplars]

        member_mask = labels == cluster_idx
        full_centroid = embeddings_norm[member_mask].mean(axis=0).tolist()

        safe_prototypes[f"safe_prototype_{cluster_idx}"] = {
            "centroid_vector": full_centroid,
            "umap_centroid_vector": centroid_fit.tolist(),
            "cluster_size": int(member_mask.sum()),
            "top_exemplars": [safe_meta[int(i)]["text"] for i in top],
            "label": "TODO: assign after thematic review",
            "description": "TODO",
        }
        print(f"\n=== safe_prototype_{cluster_idx}  (size={member_mask.sum()}) ===")
        for j, idx in enumerate(top[:3], 1):
            print(f"  {j}. {safe_meta[int(idx)]['text'][:100]}")

    # Append SAFE prototypes to existing taxonomy
    taxonomy["safe_prototypes"] = safe_prototypes
    taxonomy["safe_meta"] = {
        "best_k": best_k,
        "best_silhouette": best_score,
        "n_safe": int(safe_emb_np.shape[0]),
        "sweep": [asdict(r) for r in results],
    }

    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"\nAppended {best_k} SAFE prototypes to {taxonomy_path}")
    return taxonomy
