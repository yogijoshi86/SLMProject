"""
Generate a fillable PDF data collection sheet for the A/B study.

Each participant gets one PDF with editable form fields for:
- Metadata (participant ID, role, familiarity)
- Session 1: 25 rows × (case type, start, end, seconds, Q1, Q2, confidence)
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
        self.acroForm.choice(
            name=name,
            tooltip=name,
            x=x, y=y,
            width=width, height=height,
            value=value,
            options=options,
            fontSize=8,
            borderStyle="solid",
            borderWidth=0.5,
            borderColor=colors.HexColor("#999999"),
            fillColor=colors.white,
        )


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
    c.rect(margin, y - 14*mm, usable, 15*mm, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.black)
    c.drawString(margin + 2*mm, y - 1*mm, "Q1 — Root Cause:")
    c.setFont("Helvetica", 7)
    c.drawString(margin + 2*mm, y - 4.5*mm,
                 "a) Category too broad   b) Novel evasion / training gap   "
                 "c) Wrong threshold   d) Misinformation risk   e) Guard is correct")
    c.setFont("Helvetica-Bold", 7)
    c.drawString(margin + 2*mm, y - 8*mm, "Q2 — Recommended Fix:")
    c.setFont("Helvetica", 7)
    c.drawString(margin + 2*mm, y - 11.5*mm,
                 "a) Add domain examples   b) Add evasion examples   "
                 "c) Adjust threshold   d) Add policy rule   e) No fix needed      "
                 "FP = false positive   FN = false negative   Conf: 1–4")
    return y - 17*mm


def draw_session_table(c, y, session_num, page_width=W, margin=MARGIN):
    """Draw the 25-row data entry table for one session. Returns new y."""
    usable = page_width - 2*margin

    # column widths (pt)
    col_case  = 10*mm
    col_type  = 8*mm
    col_time  = 14*mm   # start, end
    col_secs  = 13*mm
    col_ans   = 11*mm   # Q1, Q2
    col_conf  = 10*mm
    col_score = 10*mm
    total_cols = col_case + col_type + col_time*2 + col_secs + col_ans*2 + col_conf + col_score

    # headers
    headers = ["Case", "T", "Start", "End", "Secs", "Q1", "Q2", "Conf", "✓?"]
    col_widths = [col_case, col_type, col_time, col_time,
                  col_secs, col_ans, col_ans, col_conf, col_score]

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

        # fields
        prefix = f"s{session_num}_r{row:02d}"
        pad = 1*mm
        field_h = row_h - 2*pad
        field_y = y_row + pad

        x_t    = x0 + col_case
        x_st   = x_t   + col_type
        x_en   = x_st  + col_time
        x_sc   = x_en  + col_time
        x_q1   = x_sc  + col_secs
        x_q2   = x_q1  + col_ans
        x_cf   = x_q2  + col_ans
        x_ok   = x_cf  + col_conf

        c.add_choice_field(f"{prefix}_type",  x_t + pad,  field_y, col_type - 2*pad,  field_h, t_options)
        c.add_text_field(  f"{prefix}_start", x_st + pad, field_y, col_time - 2*pad,  field_h)
        c.add_text_field(  f"{prefix}_end",   x_en + pad, field_y, col_time - 2*pad,  field_h)
        c.add_text_field(  f"{prefix}_secs",  x_sc + pad, field_y, col_secs - 2*pad,  field_h)
        c.add_choice_field(f"{prefix}_q1",    x_q1 + pad, field_y, col_ans  - 2*pad,  field_h, q_options)
        c.add_choice_field(f"{prefix}_q2",    x_q2 + pad, field_y, col_ans  - 2*pad,  field_h, q_options)
        c.add_choice_field(f"{prefix}_conf",  x_cf + pad, field_y, col_conf - 2*pad,  field_h, conf_opts)
        c.add_choice_field(f"{prefix}_score", x_ok + pad, field_y, col_score- 2*pad,  field_h, score_opts)

        y_cur = y_row

    # totals row
    y_tot = y_cur - row_h
    c.setFillColor(TOTAL_BG)
    c.rect(x0, y_tot, total_cols, row_h, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(colors.black)
    c.drawRightString(x0 + col_case + col_type + col_time*2 - 2*mm,
                      y_tot + 1.8*mm, "Session totals →")
    prefix = f"s{session_num}_tot"
    pad = 1*mm; field_h = row_h - 2*pad
    x_sc = x0 + col_case + col_type + col_time*2
    x_ok = x0 + total_cols - col_score
    c.add_text_field(f"{prefix}_secs",     x_sc + pad, y_tot + pad, col_secs  - 2*pad, field_h)
    c.add_text_field(f"{prefix}_mean",     x_sc + col_secs + pad, y_tot + pad,
                     col_ans*2 + col_conf - 2*pad, field_h)
    c.add_text_field(f"{prefix}_correct",  x_ok + pad, y_tot + pad, col_score - 2*pad, field_h)

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

    # meta row
    y = draw_meta_row(c, y, [
        ("Participant ID", "pid",       30),
        ("Role",           "role",      50),
        ("LLM familiarity (1–5)", "familiarity", 25),
        ("Date",           "date",      30),
    ])

    # answer key
    y = draw_answer_key(c, y)

    # session 1
    y = draw_section_header(c, y, f"SESSION 1  — 25 cases")
    y = draw_meta_row(c, y, [
        ("Form received",    "s1_form",  40),
        ("Session start",    "s1_start", 25),
        ("Session end",      "s1_end",   25),
    ])
    y = draw_session_table(c, y, session_num=1)

    # check if we need page break
    needed = 7*mm * 27 + 30*mm   # approx height for session 2 + summary
    if y < needed:
        c.showPage()
        y = H - MARGIN

    # session 2
    y = draw_section_header(c, y, f"SESSION 2  — 25 cases  (min 3 days after Session 1)")
    y = draw_meta_row(c, y, [
        ("Form received",    "s2_form",  40),
        ("Session start",    "s2_start", 25),
        ("Session end",      "s2_end",   25),
    ])
    y = draw_session_table(c, y, session_num=2)

    if y < 60*mm:
        c.showPage()
        y = H - MARGIN

    # summary
    y = draw_section_header(c, y, "SUMMARY  (filled by researcher after scoring)")
    draw_summary(c, y)

    c.save()
    print(f"Saved fillable PDF → {output_path}")


if __name__ == "__main__":
    generate()
