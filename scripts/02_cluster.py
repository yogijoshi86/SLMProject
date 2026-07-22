#!/usr/bin/env python
"""Phase 2: K-means sweep, silhouette selection, prototype taxonomy (Days 6-10)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _cli import base_parser, resolve  # noqa: E402

from guardrail_audit.clustering import build_prototypes  # noqa: E402


def main() -> None:
    args = base_parser("Cluster UNSAFE embeddings into prototypes").parse_args()
    cfg = resolve(args)

    build_prototypes(
        data_path=cfg.paths.embeddings,
        taxonomy_path=cfg.paths.taxonomy,
        k_min=cfg.clustering.k_min,
        k_max=cfg.clustering.k_max,
        n_init=cfg.clustering.n_init,
        seed=cfg.seed,
        top_exemplars=cfg.clustering.top_exemplars,
        k_cap=cfg.clustering.k_cap,
    )
    print("Next: manually fill in 'label' and 'failure_mode' in the taxonomy JSON.")


if __name__ == "__main__":
    main()
