from __future__ import annotations

import csv
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


BASE = Path(os.environ.get("PGRF_BASE_DIR", "/path/to/pgrf_manifest"))
STAGE = BASE / "pgrf_longitudinal_count_ge2_v1"
DATA = Path(os.environ.get("PGRF_FIGURE_DATA_DIR", STAGE / "patient_clustered_bootstrap_v1"))
OUT = Path(os.environ.get("PGRF_FIGURE_OUT_DIR", STAGE / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

BACKBONE_ORDER = ["ST-MEM", "MERL", "CLEAR-HUG", "ECG-FM", "MELP"]
BACKBONE_LABELS = {
    "ST-MEM": "ST-MEM",
    "MERL": "MERL",
    "ECG-FM": "ECG-FM",
    "MELP": "MELP",
    "CLEAR-HUG": "CLEAR-HUG",
}
ENDPOINT_ORDER = ["In-hospital", "30-day", "1-year"]
PALETTE = {
    "ST-MEM": colors.HexColor("#2F5DAA"),
    "MERL": colors.HexColor("#4E8A5A"),
    "ECG-FM": colors.HexColor("#B36B2C"),
    "MELP": colors.HexColor("#7A5AA6"),
    "CLEAR-HUG": colors.HexColor("#C44E52"),
}
TEXT = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#5B677A")
GRID = colors.HexColor("#D8DEE8")
AXIS = colors.HexColor("#303642")
LIGHT_BORDER = colors.HexColor("#EDF0F5")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def set_font(c: canvas.Canvas, size=7.5, bold=False):
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)


def pp(x) -> float:
    return 100.0 * float(x)


def interp_color(v, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # Light blue to deep blue, print-friendly.
    c0 = (0xEC / 255, 0xF3 / 255, 0xFF / 255)
    c1 = (0x2F / 255, 0x5D / 255, 0xAA / 255)
    r = c0[0] + t * (c1[0] - c0[0])
    g = c0[1] + t * (c1[1] - c0[1])
    b = c0[2] + t * (c1[2] - c0[2])
    return colors.Color(r, g, b)


def draw_panel_a(c, rows, x, y, w, h):
    by_backbone = {r["backbone"]: r for r in rows}
    set_font(c, 8.2, True)
    c.setFillColor(TEXT)
    c.drawString(x, y + h - 0.02 * inch, "(a) Patient-level clustered bootstrap")

    plot_left = x + 0.78 * inch
    plot_right = x + w - 0.12 * inch
    plot_top = y + h - 0.34 * inch
    plot_bottom = y + 0.44 * inch
    xmin, xmax = 0.0, 3.5

    def sx(v):
        return plot_left + (v - xmin) / (xmax - xmin) * (plot_right - plot_left)

    # Grid.
    for tick in [0, 1, 2, 3]:
        tx = sx(tick)
        c.setStrokeColor(colors.HexColor("#9AA4B2") if tick == 0 else GRID)
        c.setLineWidth(0.55 if tick == 0 else 0.35)
        c.line(tx, plot_bottom, tx, plot_top)
        set_font(c, 6.8)
        c.setFillColor(MUTED)
        c.drawCentredString(tx, plot_bottom - 0.15 * inch, f"{tick}")

    # Axis line.
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.7)
    c.line(plot_left, plot_bottom, plot_right, plot_bottom)
    set_font(c, 7.0)
    c.setFillColor(MUTED)
    c.drawCentredString((plot_left + plot_right) / 2, y + 0.05 * inch, "Delta Macro-AUPRC (percentage points)")

    row_top = plot_top - 0.04 * inch
    row_bottom = plot_bottom + 0.20 * inch
    row_gap = (row_top - row_bottom) / (len(BACKBONE_ORDER) - 1)
    for i, bb in enumerate(BACKBONE_ORDER):
        r = by_backbone[bb]
        yy = row_top - i * row_gap
        delta = pp(r["delta"])
        lo = pp(r["ci_low"])
        hi = pp(r["ci_high"])
        col = PALETTE[bb]

        set_font(c, 7.4, True)
        c.setFillColor(TEXT)
        c.drawRightString(plot_left - 0.12 * inch, yy - 2.4, BACKBONE_LABELS[bb])

        c.setStrokeColor(col)
        c.setLineWidth(1.8)
        c.line(sx(lo), yy, sx(hi), yy)
        c.setLineWidth(1.2)
        c.line(sx(lo), yy - 0.055 * inch, sx(lo), yy + 0.055 * inch)
        c.line(sx(hi), yy - 0.055 * inch, sx(hi), yy + 0.055 * inch)
        c.setFillColor(col)
        c.circle(sx(delta), yy, 0.045 * inch, stroke=0, fill=1)

        set_font(c, 6.7)
        c.setFillColor(TEXT)
        label = f"+{delta:.2f} [{lo:.2f}, {hi:.2f}]"
        c.drawString(min(sx(hi) + 0.05 * inch, plot_right - 0.52 * inch), yy + 0.075 * inch, label)


def draw_panel_b(c, rows, x, y, w, h):
    set_font(c, 8.2, True)
    c.setFillColor(TEXT)
    c.drawString(x, y + h - 0.02 * inch, "(b) Endpoint-specific AUPRC gains")

    # Prepare values.
    data = {}
    vals = []
    for r in rows:
        bb = r["backbone"]
        ep = r["endpoint_pretty"]
        v = pp(r["delta_auprc"])
        ci_low = pp(r["delta_auprc_ci_low"])
        data[(bb, ep)] = (v, ci_low)
        vals.append(v)
    vmin, vmax = min(vals), max(vals)

    heat_left = x + 0.76 * inch
    heat_top = y + h - 0.57 * inch
    cell_w = 0.56 * inch
    cell_h = 0.31 * inch
    gap = 0.045 * inch

    # Column labels.
    set_font(c, 6.8, True)
    c.setFillColor(MUTED)
    c.drawCentredString(heat_left + 1.5 * (cell_w + gap) - gap / 2, heat_top + 0.25 * inch, "Mortality horizon")

    set_font(c, 6.8, True)
    c.setFillColor(TEXT)
    for j, ep in enumerate(ENDPOINT_ORDER):
        cx = heat_left + j * (cell_w + gap) + cell_w / 2
        c.drawCentredString(cx, heat_top + 0.09 * inch, ep)

    for i, bb in enumerate(BACKBONE_ORDER):
        yy = heat_top - (i + 1) * cell_h - i * gap
        set_font(c, 7.0, True)
        c.setFillColor(TEXT)
        c.drawRightString(heat_left - 0.10 * inch, yy + cell_h / 2 - 2.4, BACKBONE_LABELS[bb])

        for j, ep in enumerate(ENDPOINT_ORDER):
            xx = heat_left + j * (cell_w + gap)
            v, ci_low = data[(bb, ep)]
            fill = interp_color(v, vmin, vmax)
            c.setFillColor(fill)
            c.setStrokeColor(LIGHT_BORDER)
            c.setLineWidth(0.55)
            c.roundRect(xx, yy, cell_w, cell_h, 2.5, stroke=1, fill=1)

            text_col = colors.white if v > (vmin + 0.58 * (vmax - vmin)) else TEXT
            c.setFillColor(text_col)
            set_font(c, 6.7, True)
            c.drawCentredString(xx + cell_w / 2, yy + cell_h / 2 - 2.5, f"+{v:.2f}")

    # Legend.
    leg_y = y + 0.15 * inch
    set_font(c, 6.4)
    c.setFillColor(MUTED)
    c.drawString(heat_left, leg_y, "Cell values are Delta AUPRC in percentage points.")


def main():
    macro_rows = read_csv(DATA / "patient_clustered_macro_bootstrap.csv")
    endpoint_rows = read_csv(DATA / "patient_clustered_endpoint_bootstrap.csv")

    W, H = 7.20 * inch, 3.35 * inch
    out = OUT / "figure2.pdf"
    c = canvas.Canvas(str(out), pagesize=(W, H))
    c.setTitle("Patient-level clustered bootstrap and endpoint-specific AUPRC gains")
    c.setAuthor("Anonymous")

    # Subtle panel separator.
    c.setStrokeColor(colors.HexColor("#E5E9F0"))
    c.setLineWidth(0.55)
    c.line(3.60 * inch, 0.26 * inch, 3.60 * inch, H - 0.22 * inch)

    draw_panel_a(c, macro_rows, 0.26 * inch, 0.23 * inch, 3.18 * inch, 2.88 * inch)
    draw_panel_b(c, endpoint_rows, 3.78 * inch, 0.23 * inch, 3.12 * inch, 2.88 * inch)

    c.showPage()
    c.save()
    print(out)


if __name__ == "__main__":
    main()


