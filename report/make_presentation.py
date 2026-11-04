from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "GOPIM_study_presentation_15slides.pptx"

SW, SH = 13.333, 7.5
NAVY = "101D35"
NAVY2 = "192B49"
INK = "18243A"
MUTED = "62718A"
PAPER = "F5F7FB"
WHITE = "FFFFFF"
TEAL = "18A999"
BLUE = "4C78A8"
ORANGE = "F28E2B"
GREEN = "54A24B"
RED = "D65F5F"
PALE_TEAL = "E5F5F2"
PALE_BLUE = "E9EFF7"
PALE_ORANGE = "FFF2E5"
GRID = "DCE3ED"
FONT = "Aptos"


def color(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def add_shape(slide, kind, x, y, w, h, fill, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else kind
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background() if line is None else None
    if line is not None:
        shape.line.color.rgb = color(line)
        shape.line.width = Pt(1)
    if radius:
        shape.adjustments[0] = 0.12
    return shape


def text(
    slide,
    value,
    x,
    y,
    w,
    h,
    size=16,
    fill=INK,
    bold=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
    font=FONT,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for i, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(3)
        paragraph.font.name = font
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(fill)
    return box


def rich_text(slide, runs, x, y, w, h, size=16, fill=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0)
    frame.margin_right = Inches(0)
    frame.margin_top = Inches(0)
    frame.margin_bottom = Inches(0)
    p = frame.paragraphs[0]
    for value, bold, run_color in runs:
        r = p.add_run()
        r.text = value
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color(run_color or fill)
    return box


def add_base(prs, title, section, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(PAPER)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, SW, 0.12, TEAL)
    text(slide, section.upper(), 0.65, 0.34, 3.0, 0.25, 9, TEAL, True)
    text(slide, title, 0.65, 0.70, 12.0, 0.55, 25, NAVY, True)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0.65, 1.38, 12.02, 0.012, GRID)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0.65, 7.12, 12.02, 0.012, GRID)
    text(
        slide,
        "REDUCED-SCALE GOPIM STUDY  •  SINGLE SEED  •  ANALYTICAL MODEL",
        0.65,
        7.20,
        9.5,
        0.17,
        8,
        MUTED,
    )
    text(slide, f"{page:02d}", 12.1, 7.18, 0.55, 0.2, 9, MUTED, True, PP_ALIGN.RIGHT)
    return slide


def card(slide, x, y, w, h, title, body, accent=TEAL, body_size=13):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, WHITE, GRID, True)
    add_shape(slide, MSO_SHAPE.RECTANGLE, x, y + 0.18, 0.055, h - 0.36, accent)
    text(slide, title, x + 0.22, y + 0.20, w - 0.42, 0.35, 15, NAVY, True)
    text(slide, body, x + 0.22, y + 0.66, w - 0.44, h - 0.82, body_size, MUTED)


def connector(slide, x1, y1, x2, y2, stroke=GRID, width=1.5, arrow=True):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = color(stroke)
    line.line.width = Pt(width)
    return line


def load_training():
    with (RESULTS / "update_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}


def accuracy(row, key):
    return 100 * json.loads(row[key])["acc"]


def load_pipeline(dataset):
    with (RESULTS / f"{dataset}_pipeline.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        return {r["scheme"]: r for r in csv.DictReader(f)}


def add_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(NAVY)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 0.18, SH, TEAL)
    add_shape(slide, MSO_SHAPE.OVAL, 10.15, 0.32, 2.82, 2.82, NAVY2)
    add_shape(slide, MSO_SHAPE.OVAL, 10.78, 0.95, 1.56, 1.56, TEAL)
    add_shape(slide, MSO_SHAPE.OVAL, 11.17, 1.34, 0.78, 0.78, NAVY)

    text(slide, "GOPIM • REDUCED-SCALE STUDY", 0.85, 0.72, 6.9, 0.3, 11, TEAL, True)
    text(
        slide,
        "Selective Vertex Updates\n& Pipeline Evaluation",
        0.85,
        1.45,
        8.7,
        1.65,
        33,
        WHITE,
        True,
    )
    text(
        slide,
        "GCN training on Cora and CiteSeer, plus an analytical comparison "
        "of crossbar allocation strategies",
        0.88,
        3.55,
        7.5,
        0.8,
        17,
        "C9D4E7",
    )
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.88, 4.76, 5.2, 0.52, NAVY2, radius=True)
    text(
        slide,
        "Deepit Gupta  •  Lakshay Bansal  •  Pratyush  •  Sanjeev",
        1.08,
        4.92,
        4.85,
        0.18,
        11,
        WHITE,
        True,
    )
    text(
        slide,
        "A transparent software/model study—not a full NeuroSim reproduction",
        0.88,
        6.53,
        8.6,
        0.25,
        10,
        "AFC0D0",
    )
    for i, (lab, val) in enumerate(
        [("GRAPHS", "2"), ("EPOCHS", "50"), ("SEED", "42")]
    ):
        x = 9.1 + (i % 2) * 1.65
        y = 4.10 + (i // 2) * 1.1
        text(slide, val, x, y, 1.25, 0.52, 29, WHITE, True, PP_ALIGN.CENTER)
        text(slide, lab, x, y + 0.52, 1.25, 0.22, 8, "AFC0D0", True, PP_ALIGN.CENTER)
    return slide


def slide_motivation(prs):
    slide = add_base(prs, "Why this study?", "Motivation", 2)
    card(
        slide,
        0.72,
        1.78,
        3.78,
        2.0,
        "GCNs are irregular",
        "Combination and neighborhood aggregation have different work patterns; "
        "graph degree skew can create load imbalance.",
        BLUE,
    )
    card(
        slide,
        4.78,
        1.78,
        3.78,
        2.0,
        "PIM targets data movement",
        "Resistive crossbars can perform matrix operations near stored weights. "
        "GOPIM combines pipelining, selective updates, and resource allocation.",
        TEAL,
    )
    card(
        slide,
        8.84,
        1.78,
        3.78,
        2.0,
        "Our scope is narrower",
        "We test small-graph training and a pulse-time analytical model—not the "
        "full upstream NeuroSim simulator or paper-scale OGB workloads.",
        ORANGE,
    )
    text(slide, "Study questions", 0.82, 4.32, 3.0, 0.35, 18, NAVY, True)
    questions = [
        ("01", "Can selective updates reduce modeled update work?"),
        ("02", "Does loss-aware selection help at the same update budget?"),
        ("03", "Do our simplified pipeline estimates match paper trends?"),
    ]
    for i, (num, q) in enumerate(questions):
        y = 4.9 + i * 0.57
        add_shape(slide, MSO_SHAPE.OVAL, 0.86, y, 0.34, 0.34, PALE_TEAL)
        text(slide, num, 0.86, y + 0.08, 0.34, 0.14, 8, TEAL, True, PP_ALIGN.CENTER)
        text(slide, q, 1.38, y + 0.03, 10.7, 0.28, 14, INK)
    return slide


def slide_roadmap(prs):
    slide = add_base(prs, "The study in one view", "Overview", 3)
    stages = [
        ("01", "Train", "Cora + CiteSeer\n2-layer GCN"),
        ("02", "Select", "Degree ISU vs.\nloss-aware ISU"),
        ("03", "Model", "Crossbar timing\nand array energy"),
        ("04", "Compare", "Accuracy, update work,\nand makespan"),
    ]
    for i, (num, head, body) in enumerate(stages):
        x = 0.88 + i * 3.08
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.15, 2.45, 2.05, WHITE, GRID, True)
        add_shape(slide, MSO_SHAPE.OVAL, x + 0.2, 2.40, 0.47, 0.47, [PALE_BLUE, PALE_TEAL, PALE_ORANGE, "EEEAF8"][i])
        text(slide, num, x + 0.2, 2.54, 0.47, 0.16, 9, [BLUE, TEAL, ORANGE, "7961A8"][i], True, PP_ALIGN.CENTER)
        text(slide, head, x + 0.83, 2.40, 1.4, 0.35, 18, NAVY, True)
        text(slide, body, x + 0.23, 3.18, 2.0, 0.55, 13, MUTED, False, PP_ALIGN.CENTER)
        if i < 3:
            connector(slide, x + 2.5, 3.13, x + 2.95, 3.13, TEAL, 1.5)
    text(slide, "What this work contributes", 0.93, 4.82, 4.4, 0.32, 18, NAVY, True)
    card(
        slide,
        0.9,
        5.32,
        5.48,
        1.05,
        "Reproducible small-graph comparison",
        "Same seed, split, model, and epoch budget across training policies.",
        BLUE,
        11,
    )
    card(
        slide,
        6.75,
        5.32,
        5.48,
        1.05,
        "Transparent model boundary",
        "Report where analytical estimates differ from paper-grade simulation.",
        TEAL,
        11,
    )
    return slide


def slide_gcn_stages(prs):
    slide = add_base(prs, "GCN training has distinct compute stages", "Background", 4)
    text(
        slide,
        "The GOPIM paper organizes an L-layer training iteration into 4L stages. "
        "This is the conceptual pipeline—not a claim that the local GCN code runs on NeuroSim.",
        0.88,
        1.72,
        11.6,
        0.62,
        14,
        MUTED,
    )
    items = [
        ("CO", "Combination", "Feature × weight\nmatrix operation", BLUE),
        ("AG", "Aggregation", "Neighborhood\ninformation flow", TEAL),
        ("LC", "Layer error", "Back-propagate\nlayer errors", ORANGE),
        ("GC", "Gradient", "Compute weight\ngradients", GREEN),
    ]
    for i, (abbr, title, detail, accent) in enumerate(items):
        x = 0.88 + i * 3.08
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.88, 2.35, 1.8, WHITE, GRID, True)
        add_shape(slide, MSO_SHAPE.OVAL, x + 0.78, 3.12, 0.78, 0.78, accent)
        text(slide, abbr, x + 0.78, 3.36, 0.78, 0.25, 16, WHITE, True, PP_ALIGN.CENTER)
        text(slide, title, x + 0.18, 4.06, 1.99, 0.27, 14, NAVY, True, PP_ALIGN.CENTER)
        text(slide, detail, x + 0.18, 4.39, 1.99, 0.4, 10, MUTED, False, PP_ALIGN.CENTER)
        if i < 3:
            connector(slide, x + 2.38, 3.78, x + 2.96, 3.78, MUTED, 1.3)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 1.05, 5.47, 11.1, 0.78, PALE_BLUE, radius=True)
    text(
        slide,
        "Pipeline goal: overlap micro-batches and allocate replicas to balance stage service times.",
        1.35,
        5.72,
        10.5,
        0.24,
        13,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    text(
        slide,
        "Our local timing comparison is analytical and simplified; it does not execute this full hardware pipeline.",
        1.02,
        6.54,
        11.3,
        0.24,
        10,
        MUTED,
    )
    return slide


def slide_hardware(prs):
    slide = add_base(prs, "Modeled crossbar system: configuration and scale", "Model setup", 6)
    metrics = [
        ("65,536", "virtual tiles"),
        ("8", "PEs per tile"),
        ("32", "crossbars per PE"),
        ("16,777,216", "crossbars total"),
    ]
    for i, (value, label) in enumerate(metrics):
        x = 0.84 + i * 3.05
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.86, 2.65, 1.2, WHITE, GRID, True)
        text(slide, value, x + 0.12, 2.10, 2.4, 0.36, 19 if i < 3 else 16, NAVY, True, PP_ALIGN.CENTER)
        text(slide, label, x + 0.12, 2.58, 2.4, 0.2, 10, MUTED, False, PP_ALIGN.CENTER)
    text(slide, "Per-crossbar and on-chip assumptions", 0.9, 3.53, 5.5, 0.32, 17, NAVY, True)
    parts = [
        ("64 × 64", "crossbar dimensions", BLUE),
        ("2-bit", "modeled cell precision", TEAL),
        ("29.31 ns", "read pulse", ORANGE),
        ("50.88 ns", "write pulse", GREEN),
        ("128 KiB", "global buffer", "7961A8"),
    ]
    for i, (value, desc, accent) in enumerate(parts):
        x = 0.9 + (i % 3) * 3.92
        y = 4.12 + (i // 3) * 1.08
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, 3.45, 0.86, WHITE, GRID, True)
        add_shape(slide, MSO_SHAPE.RECTANGLE, x, y + 0.13, 0.055, 0.6, accent)
        text(slide, value, x + 0.2, y + 0.13, 3.0, 0.27, 14, NAVY, True)
        text(slide, desc, x + 0.2, y + 0.49, 3.0, 0.2, 9, MUTED)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 8.75, 5.20, 3.7, 0.86, PALE_ORANGE, radius=True)
    text(slide, "16 GiB modeled array capacity", 8.98, 5.39, 3.25, 0.22, 12, NAVY, True, PP_ALIGN.CENTER)
    text(
        slide,
        "These are virtual resource counts for analytical calculations—not instantiated tiles or a full circuit simulation.",
        0.92,
        6.54,
        11.5,
        0.24,
        10,
        MUTED,
    )
    return slide


def slide_method(prs):
    slide = add_base(prs, "What we implemented", "Method", 5)
    text(slide, "TRAINING EXPERIMENT", 0.82, 1.76, 3.2, 0.24, 10, TEAL, True)
    boxes = [
        (0.82, "Planetoid graphs", "Cora + CiteSeer"),
        (3.15, "2-layer GCN", "64 hidden channels"),
        (5.48, "50 epochs", "Seed 42 • CPU"),
        (7.81, "3 policies", "Full / degree / loss-aware"),
    ]
    for x, title, body in boxes:
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.23, 2.0, 1.12, WHITE, GRID, True)
        text(slide, title, x + 0.13, 2.50, 1.75, 0.27, 13, NAVY, True, PP_ALIGN.CENTER)
        text(slide, body, x + 0.12, 2.91, 1.76, 0.24, 9, MUTED, False, PP_ALIGN.CENTER)
    for x in [2.91, 5.24, 7.57]:
        connector(slide, x, 2.78, x + 0.2, 2.78, TEAL, 1.5)

    text(slide, "ANALYTICAL CROSSBAR MODEL", 0.82, 3.94, 4.0, 0.24, 10, BLUE, True)
    arch = [
        ("64 × 64", "crossbar"),
        ("29.31 ns", "read pulse"),
        ("50.88 ns", "write pulse"),
        ("128 KiB", "global buffer"),
    ]
    for i, (value, label) in enumerate(arch):
        x = 0.82 + i * 2.33
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 4.42, 2.0, 1.0, PALE_BLUE, radius=True)
        text(slide, value, x + 0.1, 4.64, 1.8, 0.3, 16, NAVY, True, PP_ALIGN.CENTER)
        text(slide, label, x + 0.1, 5.06, 1.8, 0.2, 9, MUTED, False, PP_ALIGN.CENTER)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 10.2, 4.42, 2.15, 1.0, PALE_ORANGE, radius=True)
    text(slide, "3 allocators", 10.31, 4.64, 1.94, 0.3, 15, NAVY, True, PP_ALIGN.CENTER)
    text(slide, "serial • static • greedy", 10.31, 5.06, 1.94, 0.2, 8, MUTED, False, PP_ALIGN.CENTER)
    text(
        slide,
        "Model boundary: pulse-time floor + first-order array energy; peripheral, "
        "buffer/interconnect, and idle costs are omitted.",
        0.84,
        6.15,
        11.5,
        0.5,
        12,
        MUTED,
    )
    return slide


def slide_experimental_protocol(prs):
    slide = add_base(prs, "Training experiment: controlled comparison", "Method • training", 7)
    settings = [
        ("Datasets", "Cora and CiteSeer\nPlanetoid node classification"),
        ("Model", "2 GCN layers\n64 hidden channels"),
        ("Optimization", "Adam • lr 0.01\nDropout 0.5"),
        ("Run protocol", "50 epochs • seed 42\nCPU execution"),
    ]
    for i, (head, detail) in enumerate(settings):
        x = 0.88 + i * 3.1
        card(slide, x, 1.92, 2.62, 1.42, head, detail, [BLUE, TEAL, ORANGE, GREEN][i], 12)
    text(slide, "Three training conditions", 0.92, 3.85, 4.0, 0.32, 18, NAVY, True)
    conditions = [
        ("Full update", "Reference control", BLUE),
        ("Degree ISU", "Top-degree vertices prioritized", ORANGE),
        ("Loss-aware ISU", "Loss + degree ranking, same budget", GREEN),
    ]
    for i, (name, detail, accent) in enumerate(conditions):
        y = 4.45 + i * 0.56
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 1.0, y, 0.23, 0.23, accent, radius=True)
        text(slide, name, 1.43, y - 0.01, 2.2, 0.28, 12, NAVY, True)
        text(slide, detail, 3.78, y - 0.01, 7.7, 0.28, 12, MUTED)
    text(
        slide,
        "Report final-epoch test accuracy and modeled vertex-update count. One seed only; no significance testing.",
        0.98,
        6.46,
        11.45,
        0.3,
        11,
        TEAL,
        True,
    )
    return slide


def slide_isu(prs):
    slide = add_base(prs, "Selective updating: same budget, different ranking", "Policy", 8)
    card(
        slide,
        0.8,
        1.82,
        5.55,
        2.12,
        "Degree ISU • paper-inspired",
        "Rank vertices by graph degree. For these sparse graphs (average degree ≤ 8), "
        "select the top 80% for every-epoch updates; defer the remaining 20% to "
        "every 20 epochs.",
        BLUE,
        14,
    )
    card(
        slide,
        6.72,
        1.82,
        5.55,
        2.12,
        "Loss-aware ISU • our extension",
        "Rank labeled vertices using prior-epoch training loss and degree:\n"
        "score = 0.75 × loss rank + 0.25 × degree rank.\n"
        "Keep the same selected-vertex budget as degree ISU.",
        TEAL,
        13,
    )
    text(slide, "UPDATE SCHEDULE (SCHEMATIC)", 0.86, 4.50, 4.1, 0.24, 10, MUTED, True)
    text(slide, "Selected 80%", 0.88, 5.00, 1.5, 0.28, 12, INK, True)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.45, 4.95, 7.8, 0.38, PALE_TEAL, radius=True)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.45, 4.95, 6.3, 0.38, TEAL, radius=True)
    text(slide, "every epoch", 10.48, 4.99, 1.45, 0.25, 11, TEAL, True)
    text(slide, "Deferred 20%", 0.88, 5.77, 1.5, 0.28, 12, INK, True)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.45, 5.72, 7.8, 0.38, PALE_ORANGE, radius=True)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.45, 5.72, 1.56, 0.38, ORANGE, radius=True)
    text(slide, "every 20 epochs", 10.48, 5.76, 1.55, 0.25, 11, ORANGE, True)
    text(
        slide,
        "The extension changes which vertices are prioritized—not the configured update budget.",
        0.88,
        6.47,
        11.6,
        0.28,
        11,
        MUTED,
    )
    return slide


def draw_group_bars(
    slide, x, y, w, h, series, datasets, maximum, ticks, formatter, value_formatter=None
):
    chart_left = x + 0.55
    chart_top = y + 0.18
    chart_w = w - 0.75
    chart_h = h - 0.85
    for tick in ticks:
        yy = chart_top + chart_h * (1 - tick / maximum)
        add_shape(slide, MSO_SHAPE.RECTANGLE, chart_left, yy, chart_w, 0.008, GRID)
        text(slide, formatter(tick), x, yy - 0.09, 0.45, 0.18, 8, MUTED, False, PP_ALIGN.RIGHT)
    colors = [BLUE, ORANGE, GREEN]
    group_w = chart_w / len(datasets)
    bar_w = 0.34
    gap = 0.09
    for g, dataset in enumerate(datasets):
        center = chart_left + group_w * (g + 0.5)
        for i, values in enumerate(series):
            value = values[g]
            bh = chart_h * value / maximum
            bx = center + (i - 1) * (bar_w + gap) - bar_w / 2
            by = chart_top + chart_h - bh
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bar_w, bh, colors[i], radius=True)
            label = value_formatter(value) if value_formatter else formatter(value)
            text(slide, label, bx - 0.12, by - 0.24, bar_w + 0.24, 0.18, 8, INK, True, PP_ALIGN.CENTER)
        text(slide, dataset, center - 0.7, chart_top + chart_h + 0.17, 1.4, 0.2, 10, MUTED, True, PP_ALIGN.CENTER)


def slide_training_results(prs, training):
    slide = add_base(prs, "Training results: fewer updates; accuracy is mixed", "Results • training", 9)
    cora, cite = training["cora"], training["citeseer"]
    datasets = ["Cora", "CiteSeer"]
    acc = [
        [accuracy(r, "full_update_final_test") for r in [cora, cite]],
        [accuracy(r, "isu_final_test") for r in [cora, cite]],
        [accuracy(r, "loss_aware_isu_final_test") for r in [cora, cite]],
    ]
    full = [int(cora["full_update_total_vertex_updates"]), int(cite["full_update_total_vertex_updates"])]
    update = [
        [1.0, 1.0],
        [int(r["isu_total_vertex_updates"]) / int(r["full_update_total_vertex_updates"]) for r in [cora, cite]],
        [int(r["loss_aware_isu_total_vertex_updates"]) / int(r["full_update_total_vertex_updates"]) for r in [cora, cite]],
    ]
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.78, 1.72, 5.85, 4.76, WHITE, GRID, True)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 6.82, 1.72, 5.75, 4.76, WHITE, GRID, True)
    text(slide, "Final test accuracy", 1.05, 1.98, 4.2, 0.3, 16, NAVY, True)
    text(slide, "Modeled vertex updates", 7.08, 1.98, 4.5, 0.3, 16, NAVY, True)
    draw_group_bars(
        slide,
        1.0,
        2.48,
        5.35,
        3.15,
        acc,
        datasets,
        100,
        [0, 25, 50, 75, 100],
        lambda v: f"{v:.0f}",
        lambda v: f"{v:.1f}",
    )
    draw_group_bars(
        slide,
        7.02,
        2.48,
        5.25,
        3.15,
        update,
        datasets,
        1.2,
        [0, 0.4, 0.8, 1.2],
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.3f}",
    )
    labels = [("Full update", BLUE), ("Degree ISU", ORANGE), ("Loss-aware ISU", GREEN)]
    for i, (label, col) in enumerate(labels):
        x = 1.14 + i * 1.72
        add_shape(slide, MSO_SHAPE.OVAL, x, 5.92, 0.15, 0.15, col)
        text(slide, label, x + 0.22, 5.90, 1.45, 0.2, 9, MUTED)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 7.18, 5.75, 5.0, 0.46, PALE_TEAL, radius=True)
    text(slide, "Both ISU variants: ~19.2% fewer modeled updates", 7.36, 5.87, 4.64, 0.2, 11, TEAL, True, PP_ALIGN.CENTER)
    text(
        slide,
        "One seed (42), 50 epochs. Accuracy differences are descriptive, not statistically significant.",
        0.9,
        6.67,
        11.6,
        0.22,
        10,
        MUTED,
    )
    return slide


def slide_update_counts(prs, training):
    slide = add_base(prs, "Selective policies reduce modeled update work", "Results • update work", 10)
    rows = [("Cora", training["cora"]), ("CiteSeer", training["citeseer"])]
    for i, (dataset, row) in enumerate(rows):
        y = 1.92 + i * 2.12
        full = int(row["full_update_total_vertex_updates"])
        isu = int(row["isu_total_vertex_updates"])
        loss = int(row["loss_aware_isu_total_vertex_updates"])
        reduction = 100 * (1 - isu / full)
        text(slide, dataset, 0.96, y, 1.5, 0.34, 18, NAVY, True)
        text(slide, f"Full  {full:,}", 2.6, y + 0.02, 2.5, 0.28, 12, BLUE, True)
        text(slide, f"Degree ISU  {isu:,}", 5.48, y + 0.02, 2.7, 0.28, 12, ORANGE, True)
        text(slide, f"Loss-aware  {loss:,}", 8.5, y + 0.02, 2.8, 0.28, 12, GREEN, True)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.62, y + 0.58, 7.8, 0.34, GRID, radius=True)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 2.62, y + 0.58, 7.8 * isu / full, 0.34, TEAL, radius=True)
        text(slide, f"{reduction:.1f}% fewer updates", 10.68, y + 0.61, 1.68, 0.22, 10, TEAL, True)
        text(slide, "Bar length: selective update count relative to full update", 2.63, y + 1.03, 7.8, 0.2, 9, MUTED)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.98, 6.30, 11.3, 0.50, PALE_TEAL, radius=True)
    text(
        slide,
        "Both selective policies have identical counts by design; loss-aware ISU changes selection, not budget.",
        1.2,
        6.45,
        10.88,
        0.2,
        11,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    return slide


def slide_allocator_model(prs):
    slide = add_base(prs, "Three allocation strategies, one analytical workload", "Model • allocation", 11)
    approaches = [
        (
            "Serial",
            "Run stages sequentially; reuse the full modeled crossbar budget.",
            BLUE,
            "Reference makespan",
        ),
        (
            "Static equal-replica",
            "Divide resources evenly across stages; fixed allocation proxy.",
            ORANGE,
            "SlimGNN-like proxy",
        ),
        (
            "GOPIM greedy",
            "Allocate replicas to the current modeled bottleneck stage.",
            GREEN,
            "GOPIM-inspired policy",
        ),
    ]
    for i, (name, detail, accent, tag) in enumerate(approaches):
        y = 1.88 + i * 1.42
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.92, y, 7.2, 1.08, WHITE, GRID, True)
        add_shape(slide, MSO_SHAPE.RECTANGLE, 0.92, y + 0.15, 0.06, 0.78, accent)
        text(slide, name, 1.2, y + 0.19, 2.5, 0.3, 15, NAVY, True)
        text(slide, detail, 3.55, y + 0.16, 4.15, 0.52, 11, MUTED)
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 8.48, y + 0.18, 3.7, 0.62, [PALE_BLUE, PALE_ORANGE, PALE_TEAL][i], radius=True)
        text(slide, tag, 8.7, y + 0.37, 3.26, 0.2, 11, accent, True, PP_ALIGN.CENTER)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.98, 6.34, 11.35, 0.48, PALE_ORANGE, radius=True)
    text(
        slide,
        "The static strategy is a simplified proxy; neither baseline reproduces every detail of the cited systems.",
        1.18,
        6.47,
        10.95,
        0.2,
        10,
        NAVY,
        True,
        PP_ALIGN.CENTER,
    )
    return slide


def slide_pipeline(prs):
    slide = add_base(prs, "Pipeline model: no speedup in this setup", "Results • analytical model", 12)
    schemes = [
        ("serial", "Serial", BLUE),
        ("slimgnn_like_static_equal_replication", "Static equal-replica", ORANGE),
        ("gopim_greedy_bottleneck", "GOPIM greedy", GREEN),
    ]
    chart_x, chart_y, chart_w, chart_h = 0.86, 1.88, 7.28, 4.75
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, chart_x, chart_y, chart_w, chart_h, WHITE, GRID, True)
    text(slide, "Makespan / serial (lower is better)", 1.12, 2.10, 5.3, 0.32, 15, NAVY, True)
    rows = {dataset: load_pipeline(dataset) for dataset in ("cora", "citeseer")}
    values = {}
    for dataset, label in [("cora", "Cora"), ("citeseer", "CiteSeer")]:
        serial = float(rows[dataset]["serial"]["makespan_seconds"])
        values[label] = [float(rows[dataset][key]["makespan_seconds"]) / serial for key, _, _ in schemes]

    left, top, plot_w, plot_h = 1.42, 2.77, 6.15, 3.0
    max_value = 2.5
    for tick in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        yy = top + plot_h * (1 - tick / max_value)
        add_shape(slide, MSO_SHAPE.RECTANGLE, left, yy, plot_w, 0.008, GRID)
        text(slide, f"{tick:.1f}", 0.95, yy - 0.08, 0.37, 0.18, 8, MUTED, False, PP_ALIGN.RIGHT)
    bw, gap = 0.5, 0.16
    for g, dataset in enumerate(["Cora", "CiteSeer"]):
        center = left + plot_w * (g + 0.5) / 2
        for i, (_, _, col) in enumerate(schemes):
            val = values[dataset][i]
            bh = plot_h * val / max_value
            bx = center + (i - 1) * (bw + gap) - bw / 2
            by = top + plot_h - bh
            add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bw, bh, col, radius=True)
            text(slide, f"{val:.2f}×", bx - 0.13, by - 0.25, bw + 0.26, 0.2, 9, INK, True, PP_ALIGN.CENTER)
        text(slide, dataset, center - 0.6, 5.89, 1.2, 0.2, 10, MUTED, True, PP_ALIGN.CENTER)
    for i, (_, label, col) in enumerate(schemes):
        xx = 1.16 + i * 2.05
        add_shape(slide, MSO_SHAPE.OVAL, xx, 6.20, 0.14, 0.14, col)
        text(slide, label, xx + 0.22, 6.17, 1.7, 0.22, 8, MUTED)

    card(
        slide,
        8.48,
        1.88,
        4.0,
        1.32,
        "Cora",
        "GOPIM greedy: 1.61× serial makespan",
        ORANGE,
        13,
    )
    card(
        slide,
        8.48,
        3.42,
        4.0,
        1.32,
        "CiteSeer",
        "GOPIM greedy: 2.24× serial makespan",
        ORANGE,
        13,
    )
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 8.48, 4.98, 4.0, 1.37, PALE_ORANGE, radius=True)
    text(slide, "Energy caveat", 8.73, 5.18, 3.4, 0.28, 14, NAVY, True)
    text(
        slide,
        "Equal across allocators by construction; this model cannot measure energy savings.",
        8.73,
        5.57,
        3.45,
        0.56,
        11,
        MUTED,
    )
    return slide


def slide_interpretation(prs):
    slide = add_base(prs, "Why these results are not a paper reproduction", "Interpretation", 13)
    factors = [
        ("Workloads", "Local: Cora + CiteSeer. Paper: OGB benchmarks and Cora."),
        ("Model scale", "Local: 2 layers / 64 hidden channels. Paper Cora: 3 / 256."),
        ("Timing fidelity", "Pulse-time floor; no full peripheral, buffer, or interconnect model."),
        ("Baselines", "Serial reuses the full local resource budget; static replication is only a proxy."),
    ]
    for i, (head, body) in enumerate(factors):
        y = 1.82 + i * 1.04
        add_shape(slide, MSO_SHAPE.OVAL, 0.88, y + 0.06, 0.44, 0.44, [PALE_BLUE, PALE_TEAL, PALE_ORANGE, "EEEAF8"][i])
        text(slide, f"{i+1:02d}", 0.88, y + 0.18, 0.44, 0.16, 9, [BLUE, TEAL, ORANGE, "7961A8"][i], True, PP_ALIGN.CENTER)
        text(slide, head, 1.55, y, 2.0, 0.3, 15, NAVY, True)
        text(slide, body, 3.62, y, 8.25, 0.48, 13, MUTED)
        if i < len(factors) - 1:
            add_shape(slide, MSO_SHAPE.RECTANGLE, 1.55, y + 0.74, 10.45, 0.01, GRID)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.92, 6.12, 11.58, 0.55, NAVY, radius=True)
    text(
        slide,
        "Interpret the slowdown as a limitation of this analytical model—not as evidence against GOPIM.",
        1.17,
        6.29,
        11.05,
        0.2,
        12,
        WHITE,
        True,
        PP_ALIGN.CENTER,
    )
    return slide


def slide_conclusion(prs):
    slide = add_base(prs, "Takeaways and next steps", "Conclusion", 14)
    cards = [
        ("01", "Update work fell", "Both selective policies used about 19.2% fewer modeled vertex updates.", TEAL),
        ("02", "Accuracy was mixed", "Loss-aware ISU was not consistently better than degree ISU.", BLUE),
        ("03", "Pipeline result was negative", "Greedy allocation was slower in this simplified model; energy was fixed by design.", ORANGE),
    ]
    for i, (num, head, body, accent) in enumerate(cards):
        x = 0.82 + i * 4.12
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.92, 3.75, 2.15, WHITE, GRID, True)
        text(slide, num, x + 0.24, 2.17, 0.65, 0.45, 23, accent, True)
        text(slide, head, x + 0.24, 2.86, 3.22, 0.34, 15, NAVY, True)
        text(slide, body, x + 0.24, 3.35, 3.22, 0.55, 12, MUTED)
    text(slide, "What would strengthen the study?", 0.91, 4.55, 5.0, 0.32, 17, NAVY, True)
    steps = [
        ("More seeds", "Report variance and confidence intervals"),
        ("Matched workloads", "Add paper benchmarks and model settings"),
        ("Higher-fidelity model", "Include peripherals, memory movement, and energy"),
    ]
    for i, (head, body) in enumerate(steps):
        x = 0.95 + i * 4.05
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 5.16, 3.62, 0.91, PALE_BLUE, radius=True)
        text(slide, head, x + 0.18, 5.35, 3.25, 0.25, 12, NAVY, True)
        text(slide, body, x + 0.18, 5.70, 3.25, 0.2, 9, MUTED)
    text(
        slide,
        "Bottom line: useful local comparison and a reproducible model—not a full hardware-performance claim.",
        0.93,
        6.48,
        11.5,
        0.28,
        11,
        TEAL,
        True,
    )
    return slide


def slide_references(prs):
    slide = add_base(prs, "Selected references", "References", 15)
    refs = [
        (
            "[1]",
            "S. Yang et al., “GOPIM: GCN-Oriented Pipeline Optimization for PIM Accelerators,” "
            "HPCA, 2025. doi:10.1109/HPCA61900.2025.00056.",
        ),
        (
            "[2]",
            "T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” "
            "ICLR, 2017. arXiv:1609.02907.",
        ),
        (
            "[3]",
            "A. Shafiee et al., “ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog "
            "Arithmetic in Crossbars,” ISCA, 2016. doi:10.1109/ISCA.2016.12.",
        ),
        (
            "[4]",
            "M. Yan et al., “HyGCN: A GCN Accelerator with Hybrid Architecture,” HPCA, 2020. "
            "doi:10.1109/HPCA47549.2020.00012.",
        ),
        (
            "[5]",
            "W. Hu et al., “Open Graph Benchmark: Datasets for Machine Learning on Graphs,” "
            "NeurIPS, 2020. arXiv:2005.00687.",
        ),
        (
            "[6]",
            "M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” "
            "ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.",
        ),
    ]
    for i, (num, citation) in enumerate(refs):
        y = 1.8 + i * 0.78
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.85, y, 11.55, 0.62, WHITE, GRID, True)
        text(slide, num, 1.08, y + 0.17, 0.55, 0.2, 11, TEAL, True)
        text(slide, citation, 1.76, y + 0.13, 10.35, 0.39, 10, INK)
    text(
        slide,
        "Results shown in this deck come from the included experiment CSVs and are limited to one seed.",
        0.92,
        6.68,
        11.4,
        0.22,
        9,
        MUTED,
    )
    return slide


def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    prs.core_properties.title = (
        "A Reduced-Scale Study of GOPIM: Selective Vertex Updates and Analytical Pipeline Evaluation"
    )
    prs.core_properties.subject = "GCN selective updating and analytical PIM pipeline comparison"
    prs.core_properties.author = "Deepit Gupta; Lakshay Bansal; Pratyush; Sanjeev"
    prs.core_properties.keywords = "GOPIM, GCN, PIM, ISU, Cora, CiteSeer"

    training = load_training()
    add_cover(prs)
    slide_motivation(prs)
    slide_roadmap(prs)
    slide_gcn_stages(prs)
    slide_method(prs)
    slide_hardware(prs)
    slide_experimental_protocol(prs)
    slide_isu(prs)
    slide_training_results(prs, training)
    slide_update_counts(prs, training)
    slide_allocator_model(prs)
    slide_pipeline(prs)
    slide_interpretation(prs)
    slide_conclusion(prs)
    slide_references(prs)

    prs.save(OUTPUT)
    print(f"Created {OUTPUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
