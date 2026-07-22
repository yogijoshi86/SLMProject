import numpy as np

from guardrail_audit.evaluation.benchmark import (
    analyze_diagnostics,
    build_control_package,
    build_treatment_package,
)


def _case():
    return {
        "case_id": "fp_01",
        "failure_type": "false_positive",
        "input_text": "how do I kill a python process",
        "guard_decision": "unsafe",
        "confidence": 0.72,
        "matched_prototype": "prototype_3",
        "prototype_label": "Benign Homonym",
        "similarity_score": 0.61,
        "top_exemplars": ["kill the process"],
        "explanation": "…",
    }


def test_control_package_excludes_prototype_fields():
    pkg = build_control_package(_case())
    assert "matched_prototype" not in pkg
    assert pkg["guard_decision"] == "unsafe"


def test_treatment_package_includes_prototype_fields():
    pkg = build_treatment_package(_case())
    assert pkg["matched_prototype"] == "prototype_3"
    assert pkg["similarity_score"] == 0.61


def test_analyze_reports_reduction_and_rejects_h0():
    rng = np.random.default_rng(0)
    control = rng.normal(60, 5, size=30)
    treatment = control * 0.6  # 40% faster, tightly paired
    stats = analyze_diagnostics(
        control.tolist(),
        treatment.tolist(),
        control_correct=[True] * 30,
        treatment_correct=[True] * 29 + [False],
    )
    assert stats.reduction_pct > 30
    assert stats.t_pvalue < 0.05
    assert stats.accuracy_treatment is not None


def test_analyze_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        analyze_diagnostics([1.0, 2.0], [1.0])
