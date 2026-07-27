"""Batched extraction of hidden states for UNSAFE-flagged prompts (Day 5)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from tqdm import tqdm

from guardrail_audit.data import PromptRecord, batched


def extract_unsafe_embeddings(
    guard,
    records: list[PromptRecord],
    batch_size: int,
    output_path: str | Path,
    train_ratio: float = 0.8,
) -> dict:
    """Run the guard over prompts; keep embeddings for those flagged UNSAFE.

    Applies an 80/20 train/test split on the UNSAFE-flagged results:
    - train split → used for clustering / prototype discovery
    - test split  → held out for A/B benchmark case curation

    Both splits are saved inside the same .pt file to keep things portable.
    """
    embeddings: list[torch.Tensor] = []
    metadata: list[dict] = []
    n_seen = n_unsafe = 0

    for chunk in tqdm(list(batched(records, batch_size)), desc="Extracting", unit="batch"):
        texts = [r.text for r in chunk]
        decisions, batch_emb = guard.classify_batch(texts)

        for record, decision, emb in zip(chunk, decisions, batch_emb):
            n_seen += 1
            if not decision.is_unsafe:
                continue
            n_unsafe += 1
            embeddings.append(emb)
            metadata.append({
                "index": record.index,
                "text": record.text,
                "categories": decision.categories,
                "gt_toxicity": record.gt_toxicity,
                "gt_jailbreak": record.gt_jailbreak,
            })

    if not embeddings:
        raise RuntimeError("No prompts were flagged UNSAFE; nothing to save.")

    tensor = torch.stack(embeddings)

    # Deterministic 80/20 split — no shuffling needed since guard processes
    # prompts in dataset order which is already random w.r.t. content.
    n_train = math.floor(len(metadata) * train_ratio)
    train_indices = list(range(n_train))
    test_indices  = list(range(n_train, len(metadata)))

    payload = {
        "embeddings": tensor,
        "metadata": metadata,
        "train_indices": train_indices,   # 80% — use for clustering
        "test_indices":  test_indices,    # 20% — hold out for A/B benchmark
        "stats": {
            "n_seen": n_seen,
            "n_unsafe": n_unsafe,
            "n_train": len(train_indices),
            "n_test":  len(test_indices),
            "dim": tensor.shape[1],
            "train_ratio": train_ratio,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved {tensor.shape[0]} UNSAFE embeddings "
        f"(train={len(train_indices)}, test={len(test_indices)}, "
        f"dim={tensor.shape[1]}) to {output_path}"
    )
    return payload
