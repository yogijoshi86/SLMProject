"""ToxicChat ingestion and Llama-Guard chat formatting (Day 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class PromptRecord:
    """A single prompt to audit, carrying dataset ground-truth as metadata only."""

    index: int
    text: str
    gt_toxicity: int          # dataset label — for analysis, NOT for filtering
    gt_jailbreak: int


def load_prompts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_samples: int | None = None,
) -> list[PromptRecord]:
    """Load ToxicChat prompts, dropping empty/null inputs."""
    from datasets import load_dataset

    ds = (
        load_dataset(dataset_name, dataset_config, split=split)
        if dataset_config
        else load_dataset(dataset_name, split=split)
    )

    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        records.append(
            PromptRecord(
                index=i,
                text=text,
                gt_toxicity=int(row.get("toxicity", 0) or 0),
                gt_jailbreak=int(row.get("jailbreaking", row.get("jailbreak", 0)) or 0),
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break

    if not records:
        raise RuntimeError(f"No usable prompts found in {dataset_name}:{split}")
    return records


def to_chat(text: str) -> list[dict[str, str]]:
    """Wrap a raw user prompt in the single-turn chat format Llama-Guard expects."""
    return [{"role": "user", "content": text}]


def batched(items: list, size: int) -> Iterator[list]:
    """Yield successive ``size``-length chunks of ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
