#!/usr/bin/env python
"""Phase 4: run the paired statistical analysis on collected A/B latency logs (Day 18).

Expects a CSV with columns: participant, case_id, arm, seconds, correct
where arm is 'control' or 'treatment'. Pairs are matched on (participant, case_id).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _cli import base_parser, resolve  # noqa: E402

from guardrail_audit.evaluation import analyze_diagnostics  # noqa: E402


def main() -> None:
    parser = base_parser("Analyze A/B diagnostic latency logs")
    parser.add_argument("--logs", default=None, help="CSV path (default: cfg.paths.eval_logs)")
    args = parser.parse_args()
    cfg = resolve(args)

    import pandas as pd

    logs_path = args.logs or cfg.paths.eval_logs
    df = pd.read_csv(logs_path)

    control = df[df["arm"] == "control"].set_index(["participant", "case_id"])
    treatment = df[df["arm"] == "treatment"].set_index(["participant", "case_id"])
    common = control.index.intersection(treatment.index)
    control = control.loc[common].sort_index()
    treatment = treatment.loc[common].sort_index()

    stats = analyze_diagnostics(
        control_times=control["seconds"].tolist(),
        treatment_times=treatment["seconds"].tolist(),
        control_correct=control.get("correct", pd.Series(dtype=bool)).tolist() or None,
        treatment_correct=treatment.get("correct", pd.Series(dtype=bool)).tolist() or None,
    )

    print(f"Pairs analyzed:      {stats.n_pairs}")
    print(f"Mean control (s):    {stats.mean_control:.2f}")
    print(f"Mean treatment (s):  {stats.mean_treatment:.2f}")
    print(f"Latency reduction:   {stats.reduction_pct:.1f}%  (H1 target >= 30%)")
    print(f"Paired t-test:       t={stats.t_statistic:.3f}, p={stats.t_pvalue:.4f}")
    print(f"Wilcoxon:            W={stats.wilcoxon_statistic:.3f}, p={stats.wilcoxon_pvalue:.4f}")
    if stats.accuracy_control is not None:
        print(f"Accuracy control:    {stats.accuracy_control:.1%}")
        print(f"Accuracy treatment:  {stats.accuracy_treatment:.1%}  (target >= 85%)")

    verdict = "REJECT H0" if stats.t_pvalue < 0.05 else "FAIL TO REJECT H0"
    print(f"\nVerdict (alpha=0.05): {verdict}")


if __name__ == "__main__":
    main()
