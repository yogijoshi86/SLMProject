#!/usr/bin/env python
"""Phase 3: audit a single prompt end-to-end (Days 13-15)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _cli import base_parser, resolve  # noqa: E402

from guardrail_audit.explainer import (  # noqa: E402
    AuditPipeline,
    DistanceEngine,
    Explainer,
)
from guardrail_audit.models import load_guard  # noqa: E402


def main() -> None:
    parser = base_parser("Audit a prompt end-to-end")
    parser.add_argument("--text", required=True, help="Prompt to audit")
    parser.add_argument("--explain-safe", action="store_true")
    args = parser.parse_args()
    cfg = resolve(args)

    guard = load_guard(cfg.model)
    engine = DistanceEngine(
        taxonomy_path=cfg.paths.taxonomy,
        ood_similarity_floor=cfg.explainer.ood_similarity_floor,
    )
    explainer = Explainer(
        provider=cfg.explainer.provider,
        openai_model=cfg.explainer.openai_model,
        anthropic_model=cfg.explainer.anthropic_model,
        temperature=cfg.explainer.temperature,
    )
    pipeline = AuditPipeline(
        guard, engine, explainer, cfg.explainer.latency_budget_seconds
    )

    record = pipeline.audit_dict(args.text, explain_safe=args.explain_safe)
    print(json.dumps(record, indent=2))

    total = record["timings"].get("total_seconds", 0.0)
    budget = cfg.explainer.latency_budget_seconds
    if total > budget:
        print(f"\n[warn] total {total:.2f}s exceeds latency budget {budget}s")


if __name__ == "__main__":
    main()
