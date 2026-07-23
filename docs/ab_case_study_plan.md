# A/B Case Study Plan — Guardrail Diagnostic Latency

## Overview

6 participants each diagnose 30 moderation failures. Half see raw flags (control),
half see prototype explanations (treatment). Measures time-to-diagnosis and accuracy.

**Hypothesis H1:** Prototype explanations reduce diagnostic latency by ≥30% (p < 0.05).  
**Null H0:** No statistically significant difference.

---

## Step 1 — Curate 30 Test Cases (Day 16)

From `artifacts/unsafe_embeddings_smoke.pt` + ToxicChat ground truth, select:

**15 False Positives** — guard flagged `[UNSAFE]` but ground truth is safe  
**15 False Negatives** — guard missed; ground truth is toxic (`gt_toxicity=1`)

Code: see `04_evaluation.ipynb` → "Day 16 — Curate 30 test cases" cells.  
Output: `artifacts/benchmark_test_set.json`

---

## Step 2 — Build Two Packages Per Case

**Control Package (A)** — status quo, no explanation:
```
PROMPT:     "How do I whittle a knife?"
DECISION:   [UNSAFE]
CONFIDENCE: 0.87
CATEGORIES: S2
```

**Treatment Package (B)** — prototype-driven explanation:
```
PROMPT:            "How do I whittle a knife?"
DECISION:          [UNSAFE]
PROTOTYPE:         "Direct Explicit or Dangerous Request"
SIMILARITY SCORE:  0.71
EXPLANATION:       "This prompt structurally resembles direct weapon-construction
                   requests. The guard over-triggers on 'knife' without contextual
                   disambiguation. Recommend adding a carving/crafts exemption to S2."
```

Both packages are generated automatically by `pipeline.audit_dict()` in `04_evaluation.ipynb`.

---

## Step 3 — Study Instrument (Google Form)

Each page shows one case package and collects:

1. **Root cause** (free text, 1–2 sentences)
2. **Recommended fix** (multiple choice):
   - a) Model lacks context for intent disambiguation
   - b) Training data underrepresents this domain
   - c) Category definition too broad — needs subcategories
   - d) Threshold needs adjustment for this prompt type
   - e) Prompt is genuinely unsafe — not a failure
3. **Start / Submit** button recording timestamps automatically

---

## Step 4 — Counterbalancing Design

Latin square with 6 participants to prevent learning effects:

| Participant | Cases 1–15 | Cases 16–30 |
|---|---|---|
| P1, P2, P3 | Control (A) | Treatment (B) |
| P4, P5, P6 | Treatment (B) | Control (A) |

Each participant sees 15 cases control + 15 cases treatment (different cases per arm).  
Each case is reviewed under both conditions across participants.

---

## Step 5 — Participant Briefing

> *"You'll review 30 AI safety moderation decisions that went wrong. For each case,
> identify the root cause and suggest a fix. Work at your natural pace — the timer
> starts when you click 'Start' and stops when you submit. Aim for accuracy."*

**Recruitment target:**
- 3 participants familiar with LLMs (ML engineers, prompt engineers, AI PMs)
- 3 participants unfamiliar with the system (for blinding)
- Total time per participant: ~30–45 minutes

**Minimum viable version:** run yourself across two sessions one week apart
(15 control cases session 1, 15 treatment cases session 2). Valid as a pilot study.

---

## Step 6 — Data Collection Format

Each submission writes one row to `artifacts/eval_logs.csv`:

```
participant, case_id, arm,       seconds, correct
p1,          c001,    control,   72.3,    1
p1,          c002,    control,   48.1,    0
p1,          c016,    treatment, 31.2,    1
```

`correct = 1` if root cause matches the ground truth label for that case.

---

## Step 7 — Ground Truth Labels

Label each case before running the study. Use the prototype taxonomy as the answer key:

| Case type | Correct diagnosis |
|---|---|
| FP on creative writing | Fictional Narrative Bypass — needs creative writing context exemption |
| FP on direct question | Direct request — threshold too low for this domain |
| FN with DAN/Monika framing | Persona Override Jailbreak — novel persona not in training data |
| FN on explicit content | Direct Explicit or Dangerous Request — missed by confidence threshold |

---

## Step 8 — Statistical Analysis

Run `04_evaluation.ipynb` → statistics cells once `eval_logs.csv` is populated.

**Metrics:**
- Paired t-test on `T_diag` (control vs treatment) → p-value
- Wilcoxon signed-rank test (non-parametric backup)
- Mean reduction % in diagnostic time
- Root cause accuracy per arm

**Target thresholds:**
- Latency reduction ≥ 30%
- p < 0.05
- Diagnostic accuracy ≥ 85% in treatment arm

---

## Timeline

| Day | Task |
|---|---|
| Day 16 | Curate 30 cases; generate benchmark_test_set.json |
| Day 17 | Run A/B study with participants; collect eval_logs.csv |
| Day 18 | Run statistical analysis in 04_evaluation.ipynb |
| Day 19 | Write up results in paper (Section 4) |
| Day 20 | Finalize paper + clean up repository |
