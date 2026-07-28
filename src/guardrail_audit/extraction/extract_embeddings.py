"""Batched extraction of hidden states for all prompts (Day 5 + extended Day 17).

UNSAFE-flagged embeddings go into `embeddings`/`metadata` (used by clustering).
SAFE-flagged embeddings go into `safe_embeddings`/`safe_metadata` (new, for analysis).

Each metadata entry now includes:
  guard_decision  "UNSAFE" | "SAFE"
  gt_correct      True if guard_decision matches gt_toxicity
  quadrant        "TP" | "TN" | "FP" | "FN"

Backward compatibility: `embeddings`, `metadata`, `train_indices`, `test_indices`
keys are unchanged — 02_clustering and 03_audit are unaffected.
"""

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
    record_safe: bool = True,
) -> dict:
    """Run the guard over prompts; record embeddings for all four quadrants.

    UNSAFE-flagged prompts → `embeddings` / `metadata` (80/20 split for clustering).
    SAFE-flagged prompts   → `safe_embeddings` / `safe_metadata` (all, for analysis).

    Quadrant labels:
      TP — guard UNSAFE, gt_toxicity=1 (correct flag)
      FP — guard UNSAFE, gt_toxicity=0 (false alarm)
      TN — guard SAFE,   gt_toxicity=0 (correct pass)
      FN — guard SAFE,   gt_toxicity=1 (missed harmful)
    """
    unsafe_embeddings: list[torch.Tensor] = []
    unsafe_metadata: list[dict] = []
    safe_embeddings: list[torch.Tensor] = []
    safe_metadata: list[dict] = []
    n_seen = n_unsafe = n_safe = 0

    for chunk in tqdm(list(batched(records, batch_size)), desc="Extracting", unit="batch"):
        texts = [r.text for r in chunk]
        decisions, batch_emb = guard.classify_batch(texts)

        for record, decision, emb in zip(chunk, decisions, batch_emb):
            n_seen += 1
            gt = int(record.gt_toxicity or 0)
            is_unsafe = bool(decision.is_unsafe)
            guard_decision = "UNSAFE" if is_unsafe else "SAFE"
            gt_correct = (is_unsafe and gt == 1) or (not is_unsafe and gt == 0)
            if is_unsafe and gt == 1:
                quadrant = "TP"
            elif is_unsafe and gt == 0:
                quadrant = "FP"
            elif not is_unsafe and gt == 0:
                quadrant = "TN"
            else:
                quadrant = "FN"

            entry = {
                "index":         record.index,
                "text":          record.text,
                "categories":    decision.categories,
                "gt_toxicity":   record.gt_toxicity,
                "gt_jailbreak":  record.gt_jailbreak,
                "guard_decision": guard_decision,
                "gt_correct":    gt_correct,
                "quadrant":      quadrant,
            }

            if is_unsafe:
                n_unsafe += 1
                unsafe_embeddings.append(emb)
                unsafe_metadata.append(entry)
            elif record_safe:
                n_safe += 1
                safe_embeddings.append(emb)
                safe_metadata.append(entry)

    if not unsafe_embeddings:
        raise RuntimeError("No prompts were flagged UNSAFE; nothing to save.")

    tensor = torch.stack(unsafe_embeddings)

    # Deterministic 80/20 split on UNSAFE only (used by clustering downstream)
    n_train = math.floor(len(unsafe_metadata) * train_ratio)
    train_indices = list(range(n_train))
    test_indices  = list(range(n_train, len(unsafe_metadata)))

    safe_tensor = torch.stack(safe_embeddings) if safe_embeddings else torch.empty(0)

    payload = {
        # ── UNSAFE (backward-compatible keys) ────────────────────────────────
        "embeddings":    tensor,
        "metadata":      unsafe_metadata,
        "train_indices": train_indices,
        "test_indices":  test_indices,
        # ── SAFE (new keys for analysis) ─────────────────────────────────────
        "safe_embeddings": safe_tensor,
        "safe_metadata":   safe_metadata,
        # ── Summary ──────────────────────────────────────────────────────────
        "stats": {
            "n_seen":      n_seen,
            "n_unsafe":    n_unsafe,
            "n_safe":      n_safe,
            "n_train":     len(train_indices),
            "n_test":      len(test_indices),
            "dim":         tensor.shape[1],
            "train_ratio": train_ratio,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved {tensor.shape[0]} UNSAFE + {len(safe_metadata)} SAFE embeddings "
        f"(unsafe train={len(train_indices)}, test={len(test_indices)}, "
        f"dim={tensor.shape[1]}) → {output_path}"
    )
    return payload
