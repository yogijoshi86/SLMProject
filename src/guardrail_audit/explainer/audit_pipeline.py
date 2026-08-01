"""End-to-end audit pipeline: input -> guard decision -> prototype -> explanation (Days 13-15)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from guardrail_audit.explainer.distance_engine import DistanceEngine
from guardrail_audit.explainer.explainer_api import Explainer

if TYPE_CHECKING:
    from guardrail_audit.models import LlamaGuard


@dataclass
class AuditRecord:
    input_text: str
    is_unsafe: bool
    guard_categories: list[str]
    matched_prototype: str
    prototype_label: str
    similarity_score: float
    is_ood: bool
    second_prototype: str
    second_similarity: float
    margin: float
    # SAFE prototype fields — populated when safe_prototypes present in taxonomy
    nearest_safe_prototype: str = ""
    nearest_safe_label: str = ""
    nearest_safe_similarity: float = 0.0
    is_ambiguous: bool = False   # φ_ambiguous: equidistant UNSAFE + SAFE → structurally ambiguous
    explanation: str = ""
    timings: dict[str, float] = field(default_factory=dict)


class AuditPipeline:
    """Composes the guard model, distance engine, and explainer into one call."""

    def __init__(
        self,
        guard: "LlamaGuard",
        engine: DistanceEngine,
        explainer: Explainer,
        latency_budget_seconds: float = 1.5,
    ) -> None:
        self.guard = guard
        self.engine = engine
        self.explainer = explainer
        self.latency_budget = latency_budget_seconds

    def audit(self, text: str, explain_safe: bool = False) -> AuditRecord:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        decisions, embeddings = self.guard.classify_batch([text])
        decision = decisions[0]
        query_emb = embeddings[0].numpy()
        timings["guard_seconds"] = time.perf_counter() - t0

        if not decision.is_unsafe and not explain_safe:
            return AuditRecord(
                input_text=text,
                is_unsafe=False,
                guard_categories=[],
                matched_prototype="",
                prototype_label="",
                similarity_score=0.0,
                is_ood=False,
                second_prototype="",
                second_similarity=0.0,
                margin=0.0,
                nearest_safe_prototype="",
                nearest_safe_label="",
                nearest_safe_similarity=0.0,
                is_ambiguous=False,
                explanation="Input classified SAFE; no audit generated.",
                timings=timings,
            )

        t1 = time.perf_counter()
        match = self.engine.match(query_emb)
        timings["match_seconds"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        guard_decision = "UNSAFE" if decision.is_unsafe else "SAFE"
        explanation = self.explainer.explain(text, match, guard_decision=guard_decision)
        timings["explain_seconds"] = time.perf_counter() - t2
        timings["total_seconds"] = sum(timings.values())

        return AuditRecord(
            input_text=text,
            is_unsafe=decision.is_unsafe,
            guard_categories=decision.categories,
            matched_prototype=match.prototype_key,
            prototype_label=match.label,
            similarity_score=match.similarity,
            is_ood=match.is_ood,
            second_prototype=match.second_prototype_key,
            second_similarity=match.second_similarity,
            margin=match.margin,
            nearest_safe_prototype=match.nearest_safe_key,
            nearest_safe_label=match.nearest_safe_label,
            nearest_safe_similarity=match.nearest_safe_similarity,
            is_ambiguous=match.is_ambiguous,
            explanation=explanation,
            timings=timings,
        )

    def audit_dict(self, text: str, explain_safe: bool = False) -> dict:
        return asdict(self.audit(text, explain_safe=explain_safe))
