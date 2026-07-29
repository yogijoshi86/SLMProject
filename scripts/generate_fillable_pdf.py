"""
Generate a fillable PDF data collection sheet for the A/B study.

Each participant gets one PDF with editable form fields for:
- Metadata (participant ID, role, familiarity)
- Session 1: 25 rows × (case type, seconds, Q1, confidence)
- Session 2: 25 rows × same fields
- Summary section

Usage:
    pip install reportlab
    python scripts/generate_fillable_pdf.py

Output:
    docs/data_collection_sheet_fillable.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import io

OUTPUT_PATH = "docs/data_collection_sheet_fillable.pdf"

# ── colours ──────────────────────────────────────────────────────────────────
HDR_BG     = colors.HexColor("#333333")
HDR_FG     = colors.white
ROW_ALT    = colors.HexColor("#f7f7f7")
SCORE_BG   = colors.HexColor("#fffde7")
TOTAL_BG   = colors.HexColor("#eeeeee")
PROTO_BG   = colors.HexColor("#e8f4e8")

W, H = A4           # 595.27 × 841.89 pt
MARGIN = 18 * mm


# ── form field helpers ────────────────────────────────────────────────────────

class FormCanvas(canvas.Canvas):
    """Canvas subclass that tracks AcroForm fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields: list[dict] = []

    def add_text_field(self, name, x, y, width, height,
                       value="", font_size=8, multiline=False):
        self.acroForm.textfield(
            name=name,
            tooltip=name,
            x=x, y=y,
            width=width, height=height,
            value=value,
            fontSize=font_size,
            borderStyle="underlined",
            borderWidth=0.5,
            borderColor=colors.HexColor("#999999"),
            fillColor=colors.white,
            textColor=colors.black,
            forceBorder=True,
        )

    def add_choice_field(self, name, x, y, width, height, options, value=""):
        """Fallback: render as a plain text field (choice widget has a reportlab bug)."""
        self.add_text_field(name, x, y, width, height, value=value)


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_label(c, x, y, text, font="Helvetica", size=7, colour=colors.HexColor("#555555")):
    c.setFont(font, size)
    c.setFillColor(colour)
    c.drawString(x, y, text)


def draw_section_header(c, y, text, page_width=W, margin=MARGIN):
    c.setFillColor(HDR_BG)
    c.rect(margin, y - 3*mm, page_width - 2*margin, 6*mm, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(HDR_FG)
    c.drawString(margin + 3*mm, y, text)
    return y - 8*mm


def draw_meta_row(c, y, fields, page_width=W, margin=MARGIN):
    """Draw a row of labelled text fields. fields = [(label, field_name, width_mm), ...]"""
    x = margin
    for label, fname, w_mm in fields:
        w_pt = w_mm * mm
        draw_label(c, x, y + 5*mm, label)
        c.add_text_field(fname, x, y - 1*mm, w_pt, 5*mm)
        x += w_pt + 6*mm
    return y - 12*mm


def draw_answer_key(c, y, page_width=W, margin=MARGIN):
    usable = page_width - 2*margin
    c.setFillColor(colors.HexColor("#f0f0f0"))
    c.rect(margin, y - 8*mm, usable, 9*mm, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.black)
    c.drawString(margin + 2*mm, y - 1*mm, "Q1 — Root Cause:")
    c.setFont("Helvetica", 7)
    c.drawString(margin + 2*mm, y - 4.5*mm,
                 "a) Over-triggered on surface wording (FP)   b) Novel evasion — guard missed it (FN)   "
                 "d) Misinformation risk   e) Guard is correct   "
                 "FP = false positive   FN = false negative   Conf: 1–4")
    return y - 11*mm


def draw_column_legend(c, y, page_width=W, margin=MARGIN):
    """Draw a compact column legend. Returns new y."""
    usable = page_width - 2 * margin
    x0     = margin

    entries = [
        ("Case",  "Case number from your booklet (1–25)"),
        ("T",     "Type: write FP (false positive) or FN (false negative) — shown on each case page"),
        ("Secs",  "How many seconds this case took (stopwatch lap time)"),
        ("Q1",    "Root cause answer: write a, b, c, d, or e"),
        ("Conf",  "Your confidence: 1=not confident  2=somewhat  3=confident  4=very confident"),
        ("✓?", "Leave blank — researcher fills this in after scoring"),
    ]

    line_h  = 4 * mm
    box_pad = 2.5 * mm
    box_h   = len(entries) * line_h + 2 * box_pad + 5 * mm

    c.setFillColor(colors.HexColor("#f5f5f5"))
    c.setStrokeColor(colors.HexColor("#cccccc"))
    c.setLineWidth(0.5)
    c.rect(x0, y - box_h, usable, box_h, fill=1, stroke=1)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawString(x0 + box_pad, y - box_pad - 3 * mm, "Column legend — what to write in each column")

    ty = y - box_pad - 3 * mm - line_h
    col_label_w = 18 * mm
    for col, desc in entries:
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(colors.black)
        c.drawString(x0 + box_pad, ty, col)
        c.setFont("Helvetica", 7.5)
        c.drawString(x0 + box_pad + col_label_w, ty, desc)
        ty -= line_h

    return y - box_h - 3 * mm


def draw_session_table(c, y, session_num, page_width=W, margin=MARGIN):
    """Draw the 25-row data entry table for one session. Returns new y.
    Columns: Case | T | Secs | Q1 | Conf | score
    """
    usable = page_width - 2*margin

    # column widths (pt) — 6 columns
    col_case  = 10*mm
    col_type  = 10*mm
    col_secs  = 20*mm
    col_ans   = 18*mm   # Q1
    col_conf  = 18*mm
    col_score = 12*mm
    total_cols = col_case + col_type + col_secs + col_ans + col_conf + col_score

    headers    = ["Case", "T", "Secs", "Q1", "Conf", "✓?"]
    col_widths = [col_case, col_type, col_secs, col_ans, col_conf, col_score]

    row_h = 6.5*mm
    hdr_h = 7*mm

    x0 = margin
    y_hdr = y - hdr_h

    # draw header background
    c.setFillColor(HDR_BG)
    c.rect(x0, y_hdr, total_cols, hdr_h, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(HDR_FG)
    cx = x0
    for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
        c.drawCentredString(cx + cw/2, y_hdr + 1.5*mm, hdr)
        cx += cw

    y_cur = y_hdr

    # options for dropdowns
    q_options = [("", ""), ("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e")]
    t_options  = [("", ""), ("FP", "FP"), ("FN", "FN")]
    conf_opts  = [("", ""), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]
    score_opts = [("", ""), ("1", "1"), ("0", "0")]

    for row in range(1, 26):
        y_row = y_cur - row_h
        bg = ROW_ALT if row % 2 == 0 else colors.white

        # row background
        c.setFillColor(bg)
        c.rect(x0, y_row, total_cols - col_score, row_h, fill=1, stroke=0)
        c.setFillColor(SCORE_BG)
        c.rect(x0 + total_cols - col_score, y_row, col_score, row_h, fill=1, stroke=0)

        # row border
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.setLineWidth(0.3)
        c.rect(x0, y_row, total_cols, row_h, fill=0, stroke=1)

        # case number
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.black)
        c.drawCentredString(x0 + col_case/2, y_row + 1.8*mm, str(row))

        # fields — Case | T | Secs | Q1 | Conf | score
        prefix = f"s{session_num}_r{row:02d}"
        pad = 1*mm
        field_h = row_h - 2*pad
        field_y = y_row + pad

        x_t   = x0 + col_case
        x_sc  = x_t  + col_type
        x_q1  = x_sc + col_secs
        x_cf  = x_q1 + col_ans
        x_ok  = x_cf + col_conf

        c.add_choice_field(f"{prefix}_type",  x_t  + pad, field_y, col_type  - 2*pad, field_h, t_options)
        c.add_text_field(  f"{prefix}_secs",  x_sc + pad, field_y, col_secs  - 2*pad, field_h)
        c.add_choice_field(f"{prefix}_q1",    x_q1 + pad, field_y, col_ans   - 2*pad, field_h, q_options)
        c.add_choice_field(f"{prefix}_conf",  x_cf + pad, field_y, col_conf  - 2*pad, field_h, conf_opts)
        c.add_choice_field(f"{prefix}_score", x_ok + pad, field_y, col_score - 2*pad, field_h, score_opts)

        y_cur = y_row

    # totals row
    y_tot = y_cur - row_h
    c.setFillColor(TOTAL_BG)
    c.rect(x0, y_tot, total_cols, row_h, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(colors.black)
    c.drawRightString(x0 + col_case + col_type - 2*mm, y_tot + 1.8*mm, "Totals →")
    prefix = f"s{session_num}_tot"
    pad = 1*mm; field_h = row_h - 2*pad
    x_sc = x0 + col_case + col_type
    x_ok = x0 + total_cols - col_score
    c.add_text_field(f"{prefix}_secs",    x_sc + pad, y_tot + pad, col_secs - 2*pad, field_h)
    c.add_text_field(f"{prefix}_mean",    x_sc + col_secs + pad, y_tot + pad,
                     col_ans + col_conf - 2*pad, field_h)
    c.add_text_field(f"{prefix}_correct", x_ok + pad, y_tot + pad, col_score - 2*pad, field_h)

    note_y = y_tot - 4*mm
    c.setFont("Helvetica-Oblique", 6.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(x0, note_y,
                 "T: FP=false positive / FN=false negative  |  "
                 "✓? filled by researcher: 1=correct 0=incorrect")
    return note_y - 6*mm


def draw_summary(c, y, page_width=W, margin=MARGIN):
    usable = page_width - 2*margin
    metrics = [
        ("Arm (control / treatment)", "sum_arm_s1", "sum_arm_s2"),
        ("Mean diagnostic time (s)",  "sum_mean_s1", "sum_mean_s2"),
        ("Total correct (/ 25)",      "sum_correct_s1", "sum_correct_s2"),
        ("Accuracy (%)",              "sum_acc_s1", "sum_acc_s2"),
        ("Mean confidence (1–4)",     "sum_conf_s1", "sum_conf_s2"),
    ]

    col_label = 70*mm
    col_val   = (usable - col_label) / 2

    row_h = 7*mm
    hdr_h = 7*mm
    x0    = margin

    # header
    c.setFillColor(HDR_BG)
    c.rect(x0, y - hdr_h, usable, hdr_h, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(HDR_FG)
    c.drawString(x0 + 2*mm, y - 5*mm, "Metric")
    c.drawCentredString(x0 + col_label + col_val/2, y - 5*mm, "Session 1")
    c.drawCentredString(x0 + col_label + col_val + col_val/2, y - 5*mm, "Session 2")

    y_cur = y - hdr_h
    for i, (label, f1, f2) in enumerate(metrics):
        y_row = y_cur - row_h
        bg = ROW_ALT if i % 2 == 0 else colors.white
        c.setFillColor(bg)
        c.rect(x0, y_row, usable, row_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.setLineWidth(0.3)
        c.rect(x0, y_row, usable, row_h, fill=0, stroke=1)
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.black)
        c.drawString(x0 + 2*mm, y_row + 2*mm, label)

        pad = 1.5*mm; fh = row_h - 2*pad
        c.add_text_field(f1, x0 + col_label + pad,           y_row + pad, col_val - 2*pad, fh)
        c.add_text_field(f2, x0 + col_label + col_val + pad, y_row + pad, col_val - 2*pad, fh)
        y_cur = y_row

    # note
    c.setFont("Helvetica-Oblique", 6.5)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(x0, y_cur - 4*mm,
                 "Primary metric: mean diagnostic time (H1: ≥30% reduction, p<0.05)  |  "
                 "Secondary: accuracy ≥85% in treatment arm")
    return y_cur - 10*mm


def draw_intro_and_consent(c, y, page_width=W, margin=MARGIN):
    """Draw study description + consent block. Returns new y."""
    usable = page_width - 2 * margin
    x0     = margin

    # ── Guard definition box ─────────────────────────────────────────────
    guard_lines = [
        ("bold",   "What is the \"guard\"?"),
        ("normal", "Throughout this study the word guard refers to Llama-Guard-3-8B — a small AI model"),
        ("normal", "fine-tuned by Meta to classify user prompts as SAFE or UNSAFE. It is used as a"),
        ("normal", "safety filter in front of larger language models to block harmful requests. It"),
        ("normal", "outputs only a binary decision (safe or unsafe) with no explanation."),
    ]
    line_h  = 4.2 * mm
    box_pad = 3   * mm
    box_h   = len(guard_lines) * line_h + 2 * box_pad

    c.setFillColor(colors.HexColor("#fff3cd"))
    c.setStrokeColor(colors.HexColor("#e6a817"))
    c.setLineWidth(1.2)
    c.rect(x0, y - box_h, usable, box_h, fill=1, stroke=1)

    ty = y - box_pad - line_h + 1 * mm
    for style, text in guard_lines:
        if style == "bold":
            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(colors.black)
        else:
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.black)
        c.drawString(x0 + 3 * mm, ty, text)
        ty -= line_h

    y -= box_h + 4 * mm

    # ── Study intro box ──────────────────────────────────────────────────
    intro_lines = [
        ("bold",   "About this study"),
        ("normal", "Thank you for taking part in this research. We are studying whether short"),
        ("normal", "explanations can help developers diagnose why an AI safety guardrail made"),
        ("normal", "a wrong decision — either flagging a safe prompt as unsafe, or missing a"),
        ("normal", "harmful one."),
        ("normal", ""),
        ("normal", "You will review 25 guardrail failure cases per session across two sessions"),
        ("normal", "(at least 3 days apart). For each case, identify the most likely root cause"),
        ("normal", "and suggest a fix. There are no trick questions — we want your honest judgment."),
        ("normal", ""),
        ("normal", "Each case takes about 1–2 minutes. Use a phone stopwatch to record your time."),
        ("normal", "Total session time: ~30–45 minutes."),
        ("normal", ""),
        ("italic",  "Each case shows you a guardrail decision that went wrong. Some cases include"),
        ("italic",  "additional analysis, some don't — just work with whatever information is on the page."),
        ("normal", ""),
        ("warn",   "Content notice: Cases are from the ToxicChat dataset (huggingface.co/datasets/lmsys/toxic-chat)."),
        ("warn",   "Some prompts contain foul or offensive language — this reflects real-world safety content."),
        ("warn",   "You may stop and withdraw at any time if you feel uncomfortable."),
    ]
    line_h   = 4.2 * mm
    box_pad  = 3   * mm
    box_h    = len(intro_lines) * line_h + 2 * box_pad

    c.setFillColor(colors.HexColor("#f0f6ff"))
    c.setStrokeColor(colors.HexColor("#4a90d9"))
    c.setLineWidth(1.2)
    c.rect(x0, y - box_h, usable, box_h, fill=1, stroke=1)

    # left accent bar
    c.setFillColor(colors.HexColor("#4a90d9"))
    c.rect(x0, y - box_h, 3, box_h, fill=1, stroke=0)

    ty = y - box_pad - line_h + 1 * mm
    for style, text in intro_lines:
        if not text:
            ty -= line_h * 0.4
            continue
        if style == "bold":
            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(colors.HexColor("#1a5fa8"))
        elif style == "italic":
            c.setFont("Helvetica-Oblique", 8)
            c.setFillColor(colors.HexColor("#6b4c00"))
        elif style == "warn":
            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(colors.HexColor("#992222"))
        else:
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.black)
        c.drawString(x0 + 5 * mm, ty, text)
        ty -= line_h

    y -= box_h + 4 * mm

    # ── Consent box ──────────────────────────────────────────────────────
    consent_lines = [
        "This study collects only your answers and the time you spend on each case.",
        "No personally identifiable information is stored — your responses are recorded",
        "under a participant ID only (e.g. P01). Your data will be used solely for",
        "academic research and will not be shared with any third party. You may",
        "withdraw at any time without penalty.",
    ]
    consent_h = (len(consent_lines) + 3) * line_h + 2 * box_pad + 10 * mm

    c.setFillColor(colors.HexColor("#fff8e1"))
    c.setStrokeColor(colors.HexColor("#f0b400"))
    c.setLineWidth(1.2)
    c.rect(x0, y - consent_h, usable, consent_h, fill=1, stroke=1)

    ty = y - box_pad - line_h + 1 * mm
    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(colors.black)
    c.drawString(x0 + 3 * mm, ty, "Participant consent")
    ty -= line_h

    c.setFont("Helvetica", 8)
    for line in consent_lines:
        c.drawString(x0 + 3 * mm, ty, line)
        ty -= line_h

    ty -= 2 * mm

    # checkbox
    cb_size = 3.5 * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(x0 + 3 * mm, ty - cb_size + 1 * mm, cb_size, cb_size, fill=0, stroke=1)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.black)
    c.drawString(x0 + 3 * mm + cb_size + 2 * mm, ty,
                 "I agree to take part. I understand only anonymised data is collected and I can withdraw at any time.")
    ty -= 6 * mm

    # signature line
    c.setFont("Helvetica", 7.5)
    c.drawString(x0 + 3 * mm, ty, "Signature:")
    c.line(x0 + 24 * mm, ty - 0.5 * mm, x0 + 80 * mm, ty - 0.5 * mm)
    c.drawString(x0 + 85 * mm, ty, "Date:")
    c.line(x0 + 96 * mm, ty - 0.5 * mm, x0 + 130 * mm, ty - 0.5 * mm)

    return y - consent_h - 4 * mm


# ── main ──────────────────────────────────────────────────────────────────────

def generate(output_path=OUTPUT_PATH):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    c = FormCanvas(output_path, pagesize=A4)
    c.setTitle("A/B Study — Participant Data Collection Sheet")
    c.setAuthor("Prototype-Driven Guardrail Auditing")

    # ── PAGE 1 ─────────────────────────────────────────────────────────────
    y = H - MARGIN

    # title
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.black)
    c.drawString(MARGIN, y, "A/B Study — Participant Data Collection Sheet")
    y -= 6*mm

    # study intro + consent
    y = draw_intro_and_consent(c, y)

    # meta row
    y = draw_meta_row(c, y, [
        ("Participant ID", "pid",       30),
        ("Role",           "role",      50),
        ("LLM familiarity (1–5)", "familiarity", 25),
        ("Date",           "date",      30),
    ])

    # answer key + column legend
    y = draw_answer_key(c, y)
    y = draw_column_legend(c, y)

    # session 1 — force new page so all 25 rows always fit
    c.showPage()
    y = H - MARGIN
    y = draw_section_header(c, y, "SESSION 1  — 25 cases")
    y = draw_meta_row(c, y, [
        ("Form received",    "s1_form",  40),
        ("Session start",    "s1_start", 25),
        ("Session end",      "s1_end",   25),
    ])
    y = draw_session_table(c, y, session_num=1)

    # session 2 — always start on a new page
    c.showPage()
    y = H - MARGIN
    y = draw_section_header(c, y, "SESSION 2  — 25 cases  (at least a couple of hours after Session 1)")
    y = draw_meta_row(c, y, [
        ("Form received",    "s2_form",  40),
        ("Session start",    "s2_start", 25),
        ("Session end",      "s2_end",   25),
    ])
    y = draw_session_table(c, y, session_num=2)

    # summary — new page if needed
    if y < 60*mm:
        c.showPage()
        y = H - MARGIN

    y = draw_section_header(c, y, "SUMMARY  (filled by researcher after scoring)")
    draw_summary(c, y)

    c.save()
    print(f"Saved fillable PDF → {output_path}")


if __name__ == "__main__":
    generate()
