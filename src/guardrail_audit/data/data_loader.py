"""Dataset ingestion and Llama-Guard chat formatting.

Supported datasets:
  - ToxicChat (lmsys/toxic-chat, toxicchat0124) — default
  - WildChat   (allenai/WildChat-1M)            — set dataset_name to "allenai/WildChat-1M"
"""

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


def _load_toxicchat(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_samples: int | None,
) -> list[PromptRecord]:
    """Load ToxicChat (lmsys/toxic-chat, toxicchat0124)."""
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

    return records


def _load_wildchat(
    dataset_name: str,
    split: str,
    max_samples: int | None,
) -> list[PromptRecord]:
    """Load WildChat (allenai/WildChat-1M).

    WildChat stores full multi-turn conversations. We extract the first user
    turn from each conversation as the prompt. The `toxic` field is a boolean
    at the conversation level — mapped to gt_toxicity=1/0.

    Filters applied:
      - Conversation must have at least one user turn
      - First user turn must be non-empty and non-redacted
      - language == "English" (WildChat is multilingual; non-English prompts
        may produce unreliable Llama-Guard decisions)
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=False)

    records: list[PromptRecord] = []
    for i, row in enumerate(ds):
        conversation = row.get("conversation") or []
        # Extract first user turn
        user_turns = [t for t in conversation if t.get("role") == "user"]
        if not user_turns:
            continue
        text = (user_turns[0].get("content") or "").strip()
        if not text:
            continue
        # Skip non-English (optional filter — comment out to include all languages)
        if row.get("language", "English") != "English":
            continue

        gt_tox = int(bool(row.get("toxic", False)))
        records.append(
            PromptRecord(
                index=i,
                text=text,
                gt_toxicity=gt_tox,
                gt_jailbreak=0,   # WildChat has no jailbreak label
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def load_prompts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_samples: int | None = None,
) -> list[PromptRecord]:
    """Load prompts from a supported dataset.

    Automatically detects the dataset type from dataset_name:
      - "lmsys/toxic-chat" or any name containing "toxic" → ToxicChat loader
      - "allenai/WildChat" or any name containing "wild" → WildChat loader
    """
    name_lower = dataset_name.lower()

    if "wild" in name_lower:
        records = _load_wildchat(dataset_name, split, max_samples)
        print(f"Loaded {len(records)} prompts from WildChat ({dataset_name})")
    else:
        records = _load_toxicchat(dataset_name, dataset_config, split, text_column, max_samples)
        print(f"Loaded {len(records)} prompts from ToxicChat ({dataset_name})")

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
