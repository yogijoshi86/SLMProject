"""Provider-agnostic reasoning-LLM explainer (Day 12).

Supports OpenAI (gpt-4o) and Anthropic (Claude 3.5 Sonnet). The prompt asks for a
structured audit justification tied to the matched prototype's exemplars.
"""

from __future__ import annotations

import os

from guardrail_audit.explainer.distance_engine import PrototypeMatch

SYSTEM_PROMPT = (
    "You are an expert AI Safety Audit Assistant. A Small Language Model guardrail "
    "flagged an input as [UNSAFE]. Using the matched latent prototype and its "
    "representative exemplars, produce a concise, technical audit justification."
)


def build_user_prompt(query_text: str, match: PrototypeMatch) -> str:
    exemplars = "\n".join(
        f'{i + 1}. "{ex}"' for i, ex in enumerate(match.top_exemplars[:3])
    )
    if match.is_ood:
        cluster_block = (
            f"No prototype matched above the OOD floor "
            f"(best similarity {match.similarity:.3f}). Treat as a novel/uncategorized "
            f"attack pattern and reason from the flagged input directly."
        )
    else:
        cluster_block = (
            f"Matched '{match.label}' ({match.prototype_key}) with cosine similarity "
            f"{match.similarity:.3f}.\n"
            f"Dominant policy categories: {', '.join(match.dominant_categories) or 'n/a'}.\n"
            f"Representative exemplars of this prototype:\n{exemplars}"
        )

    return f"""FLAGGED INPUT:
"{query_text}"

LATENT CLUSTER MATCH:
{cluster_block}

TASK — respond in exactly these three labeled lines:
Target Prototype Matched: <label>
Structural Strategy: <one sentence on the structural/psychological evasion tactic used>
Recommended Action: <one specific taxonomy or system-prompt refinement>
"""


class Explainer:
    """Calls the configured reasoning LLM to justify a prototype match."""

    def __init__(
        self,
        provider: str = "openai",
        openai_model: str = "gpt-4o",
        anthropic_model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.2,
    ) -> None:
        self.provider = provider.lower()
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.temperature = temperature
        self._client = None  # lazy init

    def _client_openai(self):
        from openai import OpenAI

        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _client_anthropic(self):
        from anthropic import Anthropic

        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def explain(self, query_text: str, match: PrototypeMatch) -> str:
        user_prompt = build_user_prompt(query_text, match)

        if self.provider == "openai":
            client = self._client or self._client_openai()
            self._client = client
            resp = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()

        if self.provider == "anthropic":
            client = self._client or self._client_anthropic()
            self._client = client
            resp = client.messages.create(
                model=self.anthropic_model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
            )
            return resp.content[0].text.strip()

        raise ValueError(f"Unknown explainer provider: {self.provider!r}")
