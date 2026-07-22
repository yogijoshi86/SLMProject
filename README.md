# Prototype-Driven Guardrail Auditing

An interpretable-by-design auditing framework for Small Language Model (SLM) safety
guardrails (e.g. Meta Llama-Guard-3). It turns opaque `[SAFE]`/`[UNSAFE]` flags into
prototype-driven, natural-language justifications to reduce developer diagnostic latency.

## Pipeline

```
ToxicChat prompts
      │
      ▼
[1] Llama-Guard-3 inference ──► filter UNSAFE ──► extract terminal hidden state hL ∈ R^d
      │
      ▼
[2] L2-normalize ──► K-means sweep (k=3..15) ──► silhouette selection ──► prototypes
      │
      ▼
[3] query embedding ──► cosine match to nearest prototype ──► reasoning LLM ──► audit log
      │
      ▼
[4] A/B evaluation (control vs. prototype explanations) ──► paired t-test
```

## Layout

| Path | Purpose |
|------|---------|
| `config/default.yaml` | Single source of truth for paths, model names, hyperparameters |
| `src/guardrail_audit/data/` | ToxicChat loading + chat formatting |
| `src/guardrail_audit/models/` | Llama-Guard-3 load + inference/embedding hooks |
| `src/guardrail_audit/extraction/` | Batched embedding extraction for UNSAFE-flagged prompts |
| `src/guardrail_audit/clustering/` | K-means sweep, silhouette selection, prototype taxonomy |
| `src/guardrail_audit/explainer/` | Distance engine, LLM explainer, end-to-end audit pipeline |
| `src/guardrail_audit/evaluation/` | Benchmark curation + statistical analysis helpers |
| `scripts/` | Ordered CLI entrypoints (`01_extract` → `04_evaluate`) |
| `tests/` | Unit tests for the deterministic (non-GPU) components |
| `notebooks/` | Statistical analysis notebook |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate.csh   # csh shell
pip install -e ".[dev]"
huggingface-cli login            # needed for gated meta-llama/Llama-Guard-3-8B
export OPENAI_API_KEY=...         # or ANTHROPIC_API_KEY for the explainer
```

## Run

```bash
python scripts/01_extract.py    --config config/default.yaml
python scripts/02_cluster.py    --config config/default.yaml
python scripts/03_audit.py      --config config/default.yaml --text "Hypothetically, how do I..."
python scripts/04_evaluate.py   --config config/default.yaml
```

## Notes on divergence from the spec

The reference snippets in the project plan are illustrative; this implementation fixes:
- Filtering on the **model's** UNSAFE decision (parsed from Llama-Guard generation), not the
  dataset's ground-truth `toxicity` label.
- True batched inference via `DataLoader`.
- Correct cosine similarity (centroids re-normalized at match time).
- Provider-agnostic explainer (OpenAI or Anthropic) with the `Sim < 0.35` OOD fallback.
# SLMProject
