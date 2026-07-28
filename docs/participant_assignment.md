# Participant Assignment — A/B Study Forms

## Form Distribution

| Participant | Session 1 (this week) | Session 2 (3+ days later) |
|---|---|---|
| P01 | form_A_control.pdf | form_B_treatment.pdf |
| P02 | form_A_control.pdf | form_B_treatment.pdf |
| P03 | form_A_control.pdf | form_B_treatment.pdf |
| P04 | form_A_control.pdf | form_B_treatment.pdf |
| P05 | form_A_control.pdf | form_B_treatment.pdf |
| P06 | form_A_treatment.pdf | form_B_control.pdf |
| P07 | form_A_treatment.pdf | form_B_control.pdf |
| P08 | form_A_treatment.pdf | form_B_control.pdf |
| P09 | form_A_treatment.pdf | form_B_control.pdf |
| P10 | form_A_treatment.pdf | form_B_control.pdf |

## What each form contains

| Form | Cases | Arm | What participant sees |
|---|---|---|---|
| form_A_control.pdf | Block A (c01–c25) | Control | Raw flag + confidence only |
| form_A_treatment.pdf | Block A (c01–c25) | Treatment | Flag + prototype explanation |
| form_B_control.pdf | Block B (c26–c50) | Control | Raw flag + confidence only |
| form_B_treatment.pdf | Block B (c26–c50) | Treatment | Flag + prototype explanation |

## Instructions for administering

1. **Session 1:** Give each participant only their Session 1 form. Do not mention there is a second version.
2. **Gap:** Wait at least 3 days before Session 2 to reduce memory effects.
3. **Session 2:** Give each participant their Session 2 form. Still do not mention the forms differ — just say "here is your second set of cases."
4. **Timing:** Participants note start and end time per case using a phone stopwatch or wall clock.
5. **Collection:** Collect completed forms. Transcribe times and answers into eval_logs.csv.

## eval_logs.csv format

```
participant, case_id, arm,       seconds, correct
P01,         c01,     control,   72,      1
P01,         c02,     control,   55,      0
...
P01,         c26,     treatment, 31,      1
```

`correct = 1` if participant's answer matches ground_truth.json for that case_id.

## Ground truth reference

See `docs/ground_truth_rationale.md` for the correct answer per case and the scoring rubric.
