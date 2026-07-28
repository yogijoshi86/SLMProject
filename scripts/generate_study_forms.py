"""
Generate printable A/B study forms from benchmark_test_set.json.

Produces HTML + PDF booklets (one per arm per block):
  - form_A_control.html/pdf   : Block A, raw flag only
  - form_A_treatment.html/pdf : Block A, flag + prototype explanation
  - form_B_control.html/pdf   : Block B, raw flag only
  - form_B_treatment.html/pdf : Block B, flag + prototype explanation

PDF generation uses weasyprint (pip install weasyprint) if available,
falling back to pdfkit (pip install pdfkit + wkhtmltopdf binary) if not,
and finally leaving HTML-only if neither is installed.

Usage:
  python scripts/generate_study_forms.py

In Colab:
  !pip install -q weasyprint
  !python scripts/generate_study_forms.py
"""

import json
import os
from pathlib import Path

BENCHMARK_PATH = "artifacts/benchmark_test_set.json"
OUTPUT_DIR = "artifacts/study_forms"

CSS = """
<style>
  body { font-family: Arial, sans-serif; font-size: 11pt; margin: 0; padding: 0; }
  .cover { page-break-after: always; padding: 40px; }
  .case { page-break-after: always; padding: 30px 40px; border-top: 3px solid #333; }
  .case:last-child { page-break-after: avoid; }
  .case-header { background: #333; color: white; padding: 8px 12px; margin-bottom: 16px; }
  .case-header h2 { margin: 0; font-size: 13pt; }
  .prompt-box { background: #f5f5f5; border-left: 4px solid #666; padding: 12px; margin: 12px 0; font-family: monospace; font-size: 10pt; white-space: pre-wrap; word-wrap: break-word; }
  .decision-box { border: 2px solid #333; padding: 10px 14px; margin: 12px 0; display: inline-block; }
  .decision-unsafe { border-color: #c00; color: #c00; }
  .decision-safe { border-color: #090; color: #090; }
  .ground-truth { font-weight: bold; margin: 8px 0; }
  .prototype-box { background: #e8f4e8; border: 2px solid #4a7; padding: 12px; margin: 14px 0; }
  .prototype-box h3 { margin: 0 0 8px 0; color: #2a5; font-size: 11pt; }
  .timing { border: 1px solid #aaa; padding: 8px 12px; margin: 12px 0; display: flex; gap: 40px; }
  .timing-field { display: flex; flex-direction: column; }
  .timing-field label { font-size: 9pt; color: #666; }
  .timing-field .blank { border-bottom: 1px solid #333; width: 120px; height: 20px; }
  .questions { margin-top: 14px; }
  .question { margin-bottom: 12px; }
  .question p { font-weight: bold; margin: 0 0 6px 0; font-size: 10.5pt; }
  .option { display: flex; align-items: flex-start; gap: 8px; margin: 4px 0; font-size: 10pt; }
  .option .circle { width: 14px; height: 14px; border: 1.5px solid #333; border-radius: 50%; flex-shrink: 0; margin-top: 2px; }
  .confidence { display: flex; gap: 16px; margin-top: 4px; }
  .conf-opt { display: flex; align-items: center; gap: 6px; font-size: 10pt; }
  .conf-opt .square { width: 14px; height: 14px; border: 1.5px solid #333; flex-shrink: 0; }
  .notes { margin-top: 10px; }
  .notes label { font-size: 9pt; color: #666; }
  .notes-line { border-bottom: 1px solid #aaa; height: 20px; margin-bottom: 6px; }
  @media print {
    .case { page-break-after: always; }
  }
</style>
"""

COVER_TEMPLATE = """
<div class="cover">
  <h1 style="font-size:18pt; margin-bottom:4px;">AI Safety Guardrail Diagnostic Study</h1>
  <h2 style="font-size:14pt; color:#555; margin-top:4px;">{arm_label}</h2>
  <hr style="margin:10px 0;"/>

  <div style="background:#f0f6ff; border-left:4px solid #4a90d9; padding:12px 16px; margin:12px 0; font-size:10pt; line-height:1.7;">
    <strong style="color:#1a5fa8;">About this study</strong><br>
    Thank you for taking part in this research. We are investigating whether short, structured
    explanations can help developers diagnose why an AI safety guardrail made a wrong decision &mdash;
    either flagging a safe prompt as unsafe (false positive), or failing to catch a harmful prompt
    (false negative).<br><br>
    This booklet contains <strong>{n_cases} cases</strong>. For each case, read the prompt and the
    guard&rsquo;s decision, then identify the most likely root cause and suggest a fix.
    There are no trick questions &mdash; we want your honest, intuitive judgment.<br><br>
    Each case takes about <strong>1&ndash;2 minutes</strong>. Use a phone stopwatch and record your
    start and end time on the data collection sheet provided.
    Total session time: approximately <strong>30&ndash;45 minutes</strong>.
  </div>

  <div style="background:#fff8e1; border:1.5px solid #f0b400; padding:12px 16px; margin:12px 0; font-size:10pt; line-height:1.7;">
    <strong>Participant consent</strong><br>
    This study collects only your answers and the time spent on each case.
    <strong>No personally identifiable information is stored</strong> &mdash; responses are
    recorded under a participant ID only. Data will be used solely for academic research
    and will not be shared with any third party. You may withdraw at any time without penalty.<br><br>
    <table style="border:none; margin-top:4px;">
      <tr>
        <td style="border:none; padding:0; vertical-align:top; width:24px;">
          <div style="width:16px; height:16px; border:1.5px solid #333; margin-top:2px;">&nbsp;&nbsp;&nbsp;&nbsp;</div>
        </td>
        <td style="border:none; padding:0; font-size:9.5pt;">
          I have read the above and agree to take part. I understand that only anonymised
          timing and answer data is collected, and I can withdraw at any time.
        </td>
      </tr>
    </table>
    <div style="margin-top:10px; font-size:9.5pt;">
      Signature: <span style="display:inline-block; border-bottom:1px solid #333; width:180px;">&nbsp;</span>
      &nbsp;&nbsp;&nbsp;
      Date: <span style="display:inline-block; border-bottom:1px solid #333; width:120px;">&nbsp;</span>
    </div>
  </div>

  <h3>Participant Information</h3>
  <table style="border-collapse:collapse; width:420px;">
    <tr><td style="padding:5px 0;">Participant ID:</td><td style="border-bottom:1px solid #333; width:200px;">&nbsp;</td></tr>
    <tr><td style="padding:5px 0;">Date:</td><td style="border-bottom:1px solid #333;">&nbsp;</td></tr>
    <tr><td style="padding:5px 0;">Session (1 or 2):</td><td style="border-bottom:1px solid #333;">&nbsp;</td></tr>
  </table>

  <div style="background:#fff3cd; border:1.5px solid #e6a817; padding:10px 14px; margin:12px 0; font-size:10pt; line-height:1.7;">
    <strong>&#9888; Important &mdash; please read before you begin</strong><br>
    This booklet is your <strong>question paper only</strong>. Please <strong>do not write on it</strong>.
    Record all your answers, timings, and notes on the separate
    <strong>Data Collection Sheet</strong> provided alongside this booklet.
    Return both this booklet and your completed Data Collection Sheet when you are done.
  </div>

  <h3 style="margin-top:16px;">Answer Key</h3>
  <table style="border-collapse:collapse; font-size:10pt; line-height:1.7;">
    <tr><td style="padding-right:12px; font-weight:bold;">Q1 — Root Cause</td><td></td></tr>
    <tr><td style="padding-left:16px;">a)</td><td>Category definition too broad — keyword triggered without context</td></tr>
    <tr><td style="padding-left:16px;">b)</td><td>Novel evasion tactic — not enough training examples for this pattern</td></tr>
    <tr><td style="padding-left:16px;">c)</td><td>Confidence threshold too low / too high</td></tr>
    <tr><td style="padding-left:16px;">d)</td><td>Misinformation risk — guard may be correct to flag this</td></tr>
    <tr><td style="padding-left:16px;">e)</td><td>Prompt is genuinely unsafe — this is not a guardrail failure</td></tr>
    <tr><td style="padding-right:12px; font-weight:bold; padding-top:10px;">Q2 — Recommended Fix</td><td></td></tr>
    <tr><td style="padding-left:16px;">a)</td><td>Add domain-context examples to category training data</td></tr>
    <tr><td style="padding-left:16px;">b)</td><td>Add examples of this evasion pattern to training</td></tr>
    <tr><td style="padding-left:16px;">c)</td><td>Adjust confidence threshold for this category</td></tr>
    <tr><td style="padding-left:16px;">d)</td><td>Add an explicit policy rule blocking this content type</td></tr>
    <tr><td style="padding-left:16px;">e)</td><td>No fix needed — guard decision was correct</td></tr>
  </table>

  <p style="margin-top:16px; font-size:9.5pt; color:#555;">
    Please return this booklet together with your data collection sheet when complete.
    Thank you &mdash; your time and input are genuinely valuable to this research.
  </p>
</div>
"""

CASE_CONTROL_TEMPLATE = """
<div class="case">
  <div class="case-header">
    <h2>Case {case_id} &nbsp;|&nbsp; {failure_type_label} &nbsp;|&nbsp; {case_num} of {total}</h2>
  </div>

  <div class="timing">
    <div class="timing-field"><label>Start time</label><div class="blank"></div></div>
    <div class="timing-field"><label>End time</label><div class="blank"></div></div>
    <div class="timing-field"><label>Duration (seconds)</label><div class="blank"></div></div>
  </div>

  <p style="font-weight:bold; margin:8px 0 4px;">User Prompt:</p>
  <div class="prompt-box">{prompt}</div>

  <div style="margin:10px 0;">
    <span class="decision-box decision-{decision_class}">Guard Decision: [{decision}]</span>
    &nbsp;&nbsp;
    <span style="font-size:10pt;">Categories: {categories} &nbsp; Confidence: {confidence}</span>
  </div>

  <p class="ground-truth">Ground truth: This prompt is actually {ground_truth_label}.</p>

  <div class="questions">
    <div class="question">
      <p>Q1. What is the root cause of this guardrail failure?</p>
      <div class="option"><div class="circle"></div><span>a) Category definition too broad — triggers on keyword without context</span></div>
      <div class="option"><div class="circle"></div><span>b) Novel evasion tactic not in training data</span></div>
      <div class="option"><div class="circle"></div><span>c) Confidence threshold too low / too high</span></div>
      <div class="option"><div class="circle"></div><span>d) Prompt is generating misinformation — guard may be correct</span></div>
      <div class="option"><div class="circle"></div><span>e) Prompt is genuinely unsafe — not a failure</span></div>
    </div>

    <div class="question">
      <p>Q2. What fix would address this?</p>
      <div class="option"><div class="circle"></div><span>a) Add domain-context examples to category training data</span></div>
      <div class="option"><div class="circle"></div><span>b) Add examples of this evasion pattern to training</span></div>
      <div class="option"><div class="circle"></div><span>c) Adjust confidence threshold for this category</span></div>
      <div class="option"><div class="circle"></div><span>d) Add an explicit policy rule blocking this content type</span></div>
      <div class="option"><div class="circle"></div><span>e) No fix needed — guard decision was correct</span></div>
    </div>

    <div class="question">
      <p>Confidence in your diagnosis:</p>
      <div class="confidence">
        <div class="conf-opt"><div class="square"></div>Not confident</div>
        <div class="conf-opt"><div class="square"></div>Somewhat</div>
        <div class="conf-opt"><div class="square"></div>Confident</div>
        <div class="conf-opt"><div class="square"></div>Very confident</div>
      </div>
    </div>

    <div class="notes">
      <label>Notes (optional):</label>
      <div class="notes-line"></div>
      <div class="notes-line"></div>
    </div>
  </div>
</div>
"""

CASE_TREATMENT_TEMPLATE = """
<div class="case">
  <div class="case-header">
    <h2>Case {case_id} &nbsp;|&nbsp; {failure_type_label} &nbsp;|&nbsp; {case_num} of {total}</h2>
  </div>

  <div class="timing">
    <div class="timing-field"><label>Start time</label><div class="blank"></div></div>
    <div class="timing-field"><label>End time</label><div class="blank"></div></div>
    <div class="timing-field"><label>Duration (seconds)</label><div class="blank"></div></div>
  </div>

  <p style="font-weight:bold; margin:8px 0 4px;">User Prompt:</p>
  <div class="prompt-box">{prompt}</div>

  <div style="margin:10px 0;">
    <span class="decision-box decision-{decision_class}">Guard Decision: [{decision}]</span>
    &nbsp;&nbsp;
    <span style="font-size:10pt;">Categories: {categories} &nbsp; Confidence: {confidence}</span>
  </div>

  <p class="ground-truth">Ground truth: This prompt is actually {ground_truth_label}.</p>

  <div class="prototype-box">
    <h3>Prototype Analysis</h3>
    <p style="margin:4px 0;"><strong>Matched Prototype:</strong> {prototype_label}</p>
    <p style="margin:4px 0;"><strong>Cosine Distance:</strong> {cosine_dist}</p>
    <p style="margin:8px 0 4px;"><strong>Explanation:</strong></p>
    <p style="margin:0; font-size:10pt; line-height:1.5;">{explanation}</p>
  </div>

  <div class="questions">
    <div class="question">
      <p>Q1. What is the root cause of this guardrail failure?</p>
      <div class="option"><div class="circle"></div><span>a) Category definition too broad — triggers on keyword without context</span></div>
      <div class="option"><div class="circle"></div><span>b) Novel evasion tactic not in training data</span></div>
      <div class="option"><div class="circle"></div><span>c) Confidence threshold too low / too high</span></div>
      <div class="option"><div class="circle"></div><span>d) Prompt is generating misinformation — guard may be correct</span></div>
      <div class="option"><div class="circle"></div><span>e) Prompt is genuinely unsafe — not a failure</span></div>
    </div>

    <div class="question">
      <p>Q2. What fix would address this?</p>
      <div class="option"><div class="circle"></div><span>a) Add domain-context examples to category training data</span></div>
      <div class="option"><div class="circle"></div><span>b) Add examples of this evasion pattern to training</span></div>
      <div class="option"><div class="circle"></div><span>c) Adjust confidence threshold for this category</span></div>
      <div class="option"><div class="circle"></div><span>d) Add an explicit policy rule blocking this content type</span></div>
      <div class="option"><div class="circle"></div><span>e) No fix needed — guard decision was correct</span></div>
    </div>

    <div class="question">
      <p>Confidence in your diagnosis:</p>
      <div class="confidence">
        <div class="conf-opt"><div class="square"></div>Not confident</div>
        <div class="conf-opt"><div class="square"></div>Somewhat</div>
        <div class="conf-opt"><div class="square"></div>Confident</div>
        <div class="conf-opt"><div class="square"></div>Very confident</div>
      </div>
    </div>

    <div class="notes">
      <label>Notes (optional):</label>
      <div class="notes-line"></div>
      <div class="notes-line"></div>
    </div>
  </div>
</div>
"""


def truncate_prompt(text, max_chars=600):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated for space ...]"


def fmt_confidence(score):
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def fmt_cosine_dist(case):
    cd = case.get("treatment", {}).get("cosine_distance")
    if cd is not None:
        return f"{float(cd):.4f}"
    sim = case.get("treatment", {}).get("similarity_score")
    if sim is not None:
        return f"{1 - float(sim):.4f}"
    return "n/a"


def fmt_categories(categories):
    if not categories:
        return "n/a"
    if isinstance(categories, list):
        return ", ".join(categories) if categories else "n/a"
    return str(categories)


def build_html(cases, arm, title):
    n = len(cases)
    cover = COVER_TEMPLATE.format(
        arm_label=title,
        n_cases=n,
    )
    pages = []
    for i, case in enumerate(cases, 1):
        ctrl = case["control"]
        trt  = case["treatment"]
        failure_label = "FALSE POSITIVE" if case["failure_type"] == "false_positive" else "FALSE NEGATIVE"
        ground_truth  = "SAFE" if case["failure_type"] == "false_positive" else "UNSAFE"
        decision      = ctrl["guard_decision"]
        decision_cls  = "unsafe" if decision == "UNSAFE" else "safe"
        prompt        = truncate_prompt(ctrl["input_text"])
        confidence    = fmt_confidence(ctrl.get("confidence"))
        categories    = fmt_categories(trt.get("guard_categories", []))

        if arm == "control":
            page = CASE_CONTROL_TEMPLATE.format(
                case_id=case["case_id"],
                failure_type_label=failure_label,
                case_num=i,
                total=n,
                prompt=prompt,
                decision=decision,
                decision_class=decision_cls,
                categories=categories,
                confidence=confidence,
                ground_truth_label=ground_truth,
            )
        else:
            prototype_label = trt.get("prototype_label") or trt.get("matched_prototype", "n/a")
            explanation     = trt.get("explanation", "No explanation available.").replace("\n", "<br>")
            cosine_dist     = fmt_cosine_dist(case)
            page = CASE_TREATMENT_TEMPLATE.format(
                case_id=case["case_id"],
                failure_type_label=failure_label,
                case_num=i,
                total=n,
                prompt=prompt,
                decision=decision,
                decision_class=decision_cls,
                categories=categories,
                confidence=confidence,
                ground_truth_label=ground_truth,
                prototype_label=prototype_label,
                cosine_dist=cosine_dist,
                explanation=explanation,
            )
        pages.append(page)

    body = cover + "\n".join(pages)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{CSS}</head><body>{body}</body></html>"


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """Convert an HTML file to PDF using weasyprint or pdfkit.

    Returns True on success, False if no PDF library is available.
    """
    html_path = str(html_path)
    pdf_path  = str(pdf_path)

    # --- Try weasyprint first (pure Python, best quality) ---
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  weasyprint failed ({e}), trying pdfkit...")

    # --- Fall back to pdfkit (requires wkhtmltopdf binary) ---
    try:
        import pdfkit
        options = {
            "page-size": "A4",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
            "margin-left": "12mm",
            "margin-right": "12mm",
            "encoding": "UTF-8",
            "enable-local-file-access": "",
        }
        pdfkit.from_file(html_path, pdf_path, options=options)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  pdfkit failed ({e})")

    return False


def main():
    with open(BENCHMARK_PATH) as f:
        all_cases = json.load(f)

    # Split into two blocks for counterbalancing
    fp_cases = [c for c in all_cases if c["failure_type"] == "false_positive"]
    fn_cases = [c for c in all_cases if c["failure_type"] == "false_negative"]

    # Block A: first 13 FP + first 12 FN  (cases c01-c25 equivalent)
    # Block B: remaining FP + remaining FN  (cases c26-c50 equivalent)
    block_a = fp_cases[:13] + fn_cases[:12]
    block_b = fp_cases[13:] + fn_cases[12:]

    # Pad to 25 if needed
    while len(block_a) < 25 and len(block_b) > 25:
        block_a.append(block_b.pop(0))
    while len(block_b) < 25 and len(block_a) > 25:
        block_b.append(block_a.pop(-1))

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    forms = [
        (block_a, "control",   "Guardrail Diagnostic — Session A (Standard View)",  "form_A_control"),
        (block_b, "control",   "Guardrail Diagnostic — Session B (Standard View)",  "form_B_control"),
        (block_a, "treatment", "Guardrail Diagnostic — Session A (With Analysis)",  "form_A_treatment"),
        (block_b, "treatment", "Guardrail Diagnostic — Session B (With Analysis)",  "form_B_treatment"),
    ]

    pdf_available = None  # will be set on first attempt

    for cases, arm, title, stem in forms:
        html      = build_html(cases, arm, title)
        html_path = Path(OUTPUT_DIR) / f"{stem}.html"
        pdf_path  = Path(OUTPUT_DIR) / f"{stem}.pdf"

        html_path.write_text(html, encoding="utf-8")
        print(f"  wrote {html_path.name}  ({len(cases)} cases)")

        ok = html_to_pdf(str(html_path), str(pdf_path))
        if ok:
            print(f"  wrote {pdf_path.name}")
            pdf_available = True
        else:
            if pdf_available is None:
                pdf_available = False

    print()
    if pdf_available:
        print(f"PDF booklets saved to {OUTPUT_DIR}/")
    else:
        print("PDF generation skipped — install weasyprint or pdfkit:")
        print("  pip install weasyprint          (recommended)")
        print("  pip install pdfkit              (requires wkhtmltopdf binary)")
        print("Alternatively: open each .html in Chrome → Print → Save as PDF")

    print()
    print("Counterbalancing for 10 participants:")
    print("  P01-P05: Session 1 = form_A_control,   Session 2 = form_B_treatment")
    print("  P06-P10: Session 1 = form_A_treatment, Session 2 = form_B_control")


if __name__ == "__main__":
    main()
