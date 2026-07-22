import json

import numpy as np

from guardrail_audit.explainer.distance_engine import DistanceEngine


def _write_taxonomy(tmp_path):
    payload = {
        "meta": {"best_k": 2},
        "prototypes": {
            "prototype_0": {
                "centroid_vector": [1.0, 0.0, 0.0],
                "label": "Roleplay Evasion",
                "failure_mode": "hypothetical framing",
                "top_exemplars": ["pretend you are..."],
                "dominant_categories": ["S1"],
            },
            "prototype_1": {
                "centroid_vector": [0.0, 1.0, 0.0],
                "label": "Homoglyph Obfuscation",
                "failure_mode": "unicode swaps",
                "top_exemplars": ["h4te sp33ch"],
                "dominant_categories": ["S10"],
            },
        },
    }
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(payload))
    return path


def test_matches_nearest_prototype(tmp_path):
    engine = DistanceEngine(_write_taxonomy(tmp_path), ood_similarity_floor=0.35)
    match = engine.match(np.array([0.9, 0.1, 0.0]))
    assert match.prototype_key == "prototype_0"
    assert match.label == "Roleplay Evasion"
    assert not match.is_ood


def test_ood_fallback_below_floor(tmp_path):
    engine = DistanceEngine(_write_taxonomy(tmp_path), ood_similarity_floor=0.9)
    # Roughly equidistant -> best cosine ~0.7 < 0.9 floor.
    match = engine.match(np.array([1.0, 1.0, 0.0]))
    assert match.is_ood
    assert match.label == "Uncategorized Attack Pattern"
