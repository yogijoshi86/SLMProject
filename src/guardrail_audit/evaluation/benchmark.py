"""A/B evaluation helpers: benchmark packaging + paired statistical tests (Days 16-18)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class DiagnosticStats:
    mean_control: float
    mean_treatment: float
    reduction_pct: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    n_pairs: int
    accuracy_control: float | None = None
    accuracy_treatment: float | None = None


def build_control_package(case: dict) -> dict:
    """Package A: raw input + binary flag + scalar confidence only."""
    return {
        "case_id": case["case_id"],
        "failure_type": case["failure_type"],   # "false_positive" | "false_negative"
        "input_text": case["input_text"],
        "guard_decision": case["guard_decision"],
        "confidence": case.get("confidence"),
    }


def build_treatment_package(case: dict) -> dict:
    """Package B: control fields + prototype match + similarity + LLM justification."""
    return {
        **build_control_package(case),
        "matched_prototype": case["matched_prototype"],
        "prototype_label": case.get("prototype_label"),
        "similarity_score": case["similarity_score"],
        "top_exemplars": case.get("top_exemplars", []),
        "explanation": case["explanation"],
    }


def write_benchmark(cases: list[dict], output_path: str | Path) -> None:
    """Persist paired control/treatment packages for the A/B study."""
    packaged = [
        {
            "case_id": c["case_id"],
            "failure_type": c["failure_type"],
            "control": build_control_package(c),
            "treatment": build_treatment_package(c),
        }
        for c in cases
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packaged, f, indent=2)


def analyze_diagnostics(
    control_times: list[float],
    treatment_times: list[float],
    control_correct: list[bool] | None = None,
    treatment_correct: list[bool] | None = None,
) -> DiagnosticStats:
    """Paired t-test + Wilcoxon on diagnostic latency; reduction % and accuracy.

    control/treatment times must be paired per participant-case. Tests H1 (>=30%
    reduction) against H0 (no significant difference).
    """
    control = np.asarray(control_times, dtype=np.float64)
    treatment = np.asarray(treatment_times, dtype=np.float64)
    if control.shape != treatment.shape:
        raise ValueError("control and treatment time arrays must be the same length")

    mean_c = float(control.mean())
    mean_t = float(treatment.mean())
    reduction = (mean_c - mean_t) / mean_c * 100.0 if mean_c else 0.0

    t_stat, t_p = stats.ttest_rel(control, treatment)
    try:
        w_stat, w_p = stats.wilcoxon(control, treatment)
    except ValueError:  # zero differences etc.
        w_stat, w_p = float("nan"), float("nan")

    return DiagnosticStats(
        mean_control=mean_c,
        mean_treatment=mean_t,
        reduction_pct=reduction,
        t_statistic=float(t_stat),
        t_pvalue=float(t_p),
        wilcoxon_statistic=float(w_stat),
        wilcoxon_pvalue=float(w_p),
        n_pairs=int(control.shape[0]),
        accuracy_control=_accuracy(control_correct),
        accuracy_treatment=_accuracy(treatment_correct),
    )


def _accuracy(correct: list[bool] | None) -> float | None:
    if not correct:
        return None
    return float(np.mean([1.0 if c else 0.0 for c in correct]))
