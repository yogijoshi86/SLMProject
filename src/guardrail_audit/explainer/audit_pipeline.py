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
    similarity_score: float
    is_ood: bool
    explanation: str
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
                similarity_score=0.0,
                is_ood=False,
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
            similarity_score=match.similarity,
            is_ood=match.is_ood,
            explanation=explanation,
            timings=timings,
        )

    def audit_dict(self, text: str, explain_safe: bool = False) -> dict:
        return asdict(self.audit(text, explain_safe=explain_safe))
