#!/usr/bin/env python
"""Phase 1: load ToxicChat, run Llama-Guard, save UNSAFE embeddings (Days 1-5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _cli import base_parser, resolve  # noqa: E402

from guardrail_audit.data import load_prompts  # noqa: E402
from guardrail_audit.extraction import extract_unsafe_embeddings  # noqa: E402
from guardrail_audit.models import load_guard  # noqa: E402


def main() -> None:
    args = base_parser("Extract UNSAFE embeddings from Llama-Guard-3").parse_args()
    cfg = resolve(args)

    records = load_prompts(
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        split=cfg.data.split,
        text_column=cfg.data.text_column,
        max_samples=cfg.data.max_samples,
    )
    print(f"Loaded {len(records)} prompts.")

    guard = load_guard(cfg.model)
    extract_unsafe_embeddings(
        guard=guard,
        records=records,
        batch_size=cfg.extraction.batch_size,
        output_path=cfg.paths.embeddings,
    )


if __name__ == "__main__":
    main()
