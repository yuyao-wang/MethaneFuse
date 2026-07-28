#!/usr/bin/env python3
"""Draw the README hero overview for MethaneFuse."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "methanefuse_overview.png"

W, H = 3200, 960
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

BG = "#f5f8fb"
INK = "#102a43"
MUTED = "#52606d"
TEAL = "#0f766e"
TEAL_DARK = "#0b5345"
TEAL_SOFT = "#dff4ed"
BLUE = "#2563eb"
BLUE_SOFT = "#dbeafe"
GOLD = "#b45309"
GOLD_SOFT = "#fef3c7"
CORAL = "#c2410c"
CORAL_SOFT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_SOFT = "#ede9fe"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_card(base: Image.Image, box, radius=34, fill=WHITE, outline="#d7e0ea"):
    x1, y1, x2, y2 = box
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x1 + 8, y1 + 14, x2 + 8, y2 + 14), radius=radius, fill=(16, 42, 67, 28))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(shadow)
    draw = ImageDraw.Draw(base)
    rounded(draw, box, radius, fill, outline, 2)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text(draw, xy, text, size, fill=INK, bold=False, anchor=None, **kwargs):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor, **kwargs)


def draw_centered(draw, x, y, text, size, fill=INK, bold=False):
    draw.text((x, y), text, font=font(size, bold), fill=fill, anchor="mm")


def wrap_lines(draw, text, fnt, max_width):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = word if not line else f"{line} {word}"
        if text_size(draw, test, fnt)[0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_wrapped(draw, xy, text, size, max_width, fill=MUTED, bold=False, line_gap=10, anchor="la"):
    fnt = font(size, bold)
    lines = wrap_lines(draw, text, fnt, max_width)
    x, y = xy
    line_h = text_size(draw, "Ag", fnt)[1] + line_gap
    if anchor == "mm":
        total_h = line_h * len(lines) - line_gap
        y -= total_h / 2
        for line in lines:
            draw.text((x, y), line, font=fnt, fill=fill, anchor="ma")
            y += line_h
    else:
        for line in lines:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += line_h


def draw_pill(draw, box, text, fill, fg, outline=None, size=30):
    rounded(draw, box, 28, fill, outline, 2 if outline else 1)
    x1, y1, x2, y2 = box
    draw_centered(draw, (x1 + x2) / 2, (y1 + y2) / 2, text, size, fg, True)


def draw_arrow(draw, start, end, color=TEAL, width=8):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 26
    spread = 0.55
    p1 = (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread))
    p2 = (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread))
    draw.polygon((end, p1, p2), fill=color)


def draw_satellite(draw, cx, cy, color):
    rounded(draw, (cx - 30, cy - 20, cx + 30, cy + 20), 8, color)
    rounded(draw, (cx - 90, cy - 17, cx - 42, cy + 17), 5, "#e0f2fe", color, 3)
    rounded(draw, (cx + 42, cy - 17, cx + 90, cy + 17), 5, "#e0f2fe", color, 3)
    draw.line((cx - 40, cy, cx - 30, cy), fill=color, width=5)
    draw.line((cx + 30, cy, cx + 40, cy), fill=color, width=5)
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=WHITE)


def draw_sensor_card(draw, box, label, sublabel, color, soft, active=True):
    x1, y1, x2, y2 = box
    rounded(draw, box, 24, soft, color if active else "#cbd5e1", 3)
    draw_satellite(draw, x1 + 105, (y1 + y2) // 2, color if active else "#94a3b8")
    draw_text(draw, (x1 + 210, y1 + 28), label, 34, INK if active else "#64748b", True)
    draw_text(draw, (x1 + 210, y1 + 74), sublabel, 24, MUTED if active else "#94a3b8", True)
    dot = color if active else "#cbd5e1"
    draw.ellipse((x2 - 60, y1 + 42, x2 - 32, y1 + 70), fill=dot)


def draw_token_row(draw, x, y, labels):
    colors = [TEAL, BLUE, GOLD, PURPLE]
    for i, label in enumerate(labels):
        bx = x + i * 113
        rounded(draw, (bx, y, bx + 88, y + 64), 16, "#eef5fb", colors[i], 3)
        draw_centered(draw, bx + 44, y + 32, label, 24, colors[i], True)


def draw_metric(draw, x, y, label, value, color, soft):
    rounded(draw, (x, y, x + 210, y + 118), 22, soft, color, 3)
    draw_centered(draw, x + 105, y + 39, label, 26, color, True)
    draw_centered(draw, x + 105, y + 81, value, 36, INK, True)


def draw_background(draw):
    for i in range(0, W, 160):
        draw.line((i, 0, i + 420, H), fill="#edf2f7", width=2)
    for cx, cy, r, fill in [
        (2780, 170, 110, "#e0f2fe"),
        (3050, 810, 150, "#dff4ed"),
        (160, 790, 130, "#ffedd5"),
    ]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)


def main():
    img = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw_background(draw)

    draw_text(draw, (120, 82), "MethaneFuse", 86, INK, True)
    draw_text(draw, (125, 172), "Partial multi-sensor methane plume detection", 38, TEAL_DARK, True)
    draw_wrapped(
        draw,
        (126, 222),
        "Learn from whichever satellite observations are available, then adapt to scale-controlled plume queries.",
        30,
        1180,
        fill=MUTED,
        bold=True,
        line_gap=8,
    )
    draw_pill(draw, (2290, 88, 3090, 154), "Sentinel-2 | Landsat 8/9 | EMIT | Sentinel-5P", "#ffffff", TEAL_DARK, "#b6d9d2", 28)

    left = (105, 300, 755, 820)
    s1 = (845, 300, 1465, 820)
    s2 = (1555, 300, 2175, 820)
    out = (2265, 300, 3095, 820)
    for box in [left, s1, s2, out]:
        shadowed_card(img, box)

    draw_text(draw, (155, 365), "Partial observations", 39, INK, True)
    draw_text(draw, (157, 414), "Each plume event may have a different sensor set", 25, MUTED, True)
    sensor_y = [462, 552, 642, 732]
    sensors = [
        ("Sentinel-2", "optical time series", TEAL, TEAL_SOFT, True),
        ("Landsat 8/9", "thermal + optical context", BLUE, BLUE_SOFT, True),
        ("EMIT", "hyperspectral methane cues", GOLD, GOLD_SOFT, True),
        ("Sentinel-5P", "coarse methane column", PURPLE, PURPLE_SOFT, False),
    ]
    for y, spec in zip(sensor_y, sensors):
        draw_sensor_card(draw, (150, y, 710, y + 74), *spec)

    draw_pill(draw, (900, 350, 1112, 408), "Stage 1", TEAL_SOFT, TEAL_DARK, "#a7d6ce", 30)
    draw_text(draw, (900, 462), "Sensor-native pretraining", 38, INK, True)
    draw_wrapped(draw, (902, 512), "Tokenize each sensor in its native temporal resolution and learn methane-aware representations.", 27, 500, MUTED, True)
    draw_token_row(draw, 900, 625, ["S2", "L8", "EM", "S5P"])
    rounded(draw, (930, 720, 1390, 778), 22, "#eefaf7", TEAL, 3)
    draw_centered(draw, 1160, 749, "Masked sensor-set fusion", 30, TEAL_DARK, True)

    draw_pill(draw, (1610, 350, 1822, 408), "Stage 2", BLUE_SOFT, BLUE, "#b8cdfd", 30)
    draw_text(draw, (1610, 462), "Query-level adaptation", 38, INK, True)
    draw_wrapped(draw, (1612, 512), "Freeze the pretrained encoder and train lightweight sensor-aware experts for downstream plume tasks.", 27, 500, MUTED, True)
    rounded(draw, (1610, 634, 2120, 694), 22, "#f8fafc", "#cbd5e1", 3)
    draw_centered(draw, 1865, 664, "Frozen encoder", 29, INK, True)
    rounded(draw, (1610, 720, 1840, 778), 22, PURPLE_SOFT, PURPLE, 3)
    draw_centered(draw, 1725, 749, "LoRA experts", 28, PURPLE, True)
    rounded(draw, (1888, 720, 2120, 778), 22, CORAL_SOFT, CORAL, 3)
    draw_centered(draw, 2004, 749, "Task heads", 28, CORAL, True)

    draw_text(draw, (2320, 365), "Methane monitoring outputs", 39, INK, True)
    draw_text(draw, (2322, 414), "Detection-ready predictions from incomplete coverage", 25, MUTED, True)
    rounded(draw, (2320, 468, 3040, 590), 30, "#eefaf7", TEAL, 4)
    draw_text(draw, (2370, 508), "Plume classification", 34, TEAL_DARK, True)
    draw_text(draw, (2370, 555), "scale-controlled query footprint", 25, MUTED, True)
    draw.ellipse((2920, 500, 2988, 568), fill=TEAL)
    draw.line((2940, 534, 2957, 552, 2984, 516), fill=WHITE, width=8)

    rounded(draw, (2320, 620, 3040, 704), 26, "#fff7ed", CORAL, 3)
    draw_text(draw, (2370, 654), "Segmentation masks + examples", 31, CORAL, True)
    draw_text(draw, (2370, 690), "for plume-localized analysis", 23, MUTED, True)

    draw_metric(draw, 2320, 742, "F1", "84.87", TEAL, TEAL_SOFT)
    draw_metric(draw, 2575, 742, "AUROC", "93.62", BLUE, BLUE_SOFT)
    draw_metric(draw, 2830, 742, "FPR", "14.87", CORAL, CORAL_SOFT)

    draw_arrow(draw, (770, 560), (830, 560), TEAL, 9)
    draw_arrow(draw, (1480, 560), (1540, 560), BLUE, 9)
    draw_arrow(draw, (2190, 560), (2250, 560), TEAL, 9)

    rounded(draw, (118, 872, 3082, 922), 25, "#ffffff", "#d9e2ec", 2)
    draw_centered(draw, W / 2, 897, "Built for real monitoring: missing acquisitions, cloud cover, mixed missions, and transient emissions", 26, MUTED, True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.save(OUT, quality=98, optimize=True)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
