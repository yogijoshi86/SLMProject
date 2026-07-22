"""Batched extraction of terminal hidden states for UNSAFE-flagged prompts (Day 5)."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from guardrail_audit.data import PromptRecord, batched, to_chat
from guardrail_audit.models import LlamaGuard


def extract_unsafe_embeddings(
    guard: LlamaGuard,
    records: list[PromptRecord],
    batch_size: int,
    output_path: str | Path,
) -> dict:
    """Run Llama-Guard over prompts; keep embeddings for those it flags UNSAFE.

    Filtering is on the *model's* decision (the point of the audit), not the dataset
    ground-truth. Ground-truth labels are retained in metadata for later analysis.
    """
    embeddings: list[torch.Tensor] = []
    metadata: list[dict] = []
    n_seen = n_unsafe = 0

    for chunk in tqdm(
        list(batched(records, batch_size)),
        desc="Extracting",
        unit="batch",
    ):
        chats = [to_chat(r.text) for r in chunk]
        decisions, batch_emb = guard.classify_batch(chats)

        for record, decision, emb in zip(chunk, decisions, batch_emb):
            n_seen += 1
            if not decision.is_unsafe:
                continue
            n_unsafe += 1
            embeddings.append(emb)
            metadata.append(
                {
                    "index": record.index,
                    "text": record.text,
                    "categories": decision.categories,
                    "gt_toxicity": record.gt_toxicity,
                    "gt_jailbreak": record.gt_jailbreak,
                }
            )

    if not embeddings:
        raise RuntimeError("No prompts were flagged UNSAFE; nothing to save.")

    tensor = torch.stack(embeddings)
    payload = {
        "embeddings": tensor,
        "metadata": metadata,
        "stats": {"n_seen": n_seen, "n_unsafe": n_unsafe, "dim": tensor.shape[1]},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved {tensor.shape[0]} UNSAFE embeddings "
        f"(dim={tensor.shape[1]}, of {n_seen} seen) to {output_path}"
    )
    return payload
