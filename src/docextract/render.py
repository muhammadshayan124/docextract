"""Draw a page from text lines and degrade it like a scan.

Severity levels (0 = clean render):
  1  slight skew, mild sensor noise, light JPEG
  2  stronger skew, blur, lower contrast, a shadow gradient, JPEG q60
  3  heavy skew, blur, low contrast and brightness drift, specks, a low-resolution round-trip, JPEG q35 (a bad fax/phone photo)

`handwriting_style=True` draws the characters with a script font and per-glyph jitter. It is a stress condition built from a
font, NOT real handwriting; the README says so, and results for it must not be read as handwriting-recognition accuracy.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

PAGE = (1240, 1754)  # A4 at 150 dpi
MARGIN = 90
MONO_CANDIDATES = ["DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "LiberationMono-Regular.ttf"]
SCRIPT_CANDIDATES = ["Inkfree.ttf", "segoesc.ttf", "Purisa.ttf"]
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation"), Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]


def find_font(candidates: list[str], size: int) -> ImageFont.ImageFont:
    for d in FONT_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)  # still works, but is a different (proportional) face: results then differ slightly


def draw_page(lines: list[str], font_size: int = 30, handwriting_style: bool = False, seed: int = 0) -> Image.Image:
    img = Image.new("L", PAGE, 255)
    draw = ImageDraw.Draw(img)
    font = find_font(SCRIPT_CANDIDATES if handwriting_style else MONO_CANDIDATES, font_size + (4 if handwriting_style else 0))
    rng = random.Random(seed)
    y = MARGIN
    step = int(font_size * 1.55)
    for line in lines:
        if handwriting_style and line:
            x = MARGIN
            for ch in line:
                glyph = Image.new("L", (font_size * 2, font_size * 2), 255)
                ImageDraw.Draw(glyph).text((font_size // 2, font_size // 2 + rng.randint(-3, 3)), ch, font=font, fill=rng.randint(0, 60))
                glyph = glyph.rotate(rng.uniform(-6, 6), fillcolor=255, resample=Image.BICUBIC)
                box = (x - font_size // 2, y - font_size // 2, x + 3 * font_size // 2, y + 3 * font_size // 2)
                img.paste(ImageChops.darker(img.crop(box), glyph), box[:2])  # ink only ever darkens the page
                x += int(font.getlength(ch)) + rng.randint(-1, 2)
        else:
            draw.text((MARGIN, y), line, font=font, fill=0)
        y += step
    return img


def degrade(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    if severity <= 0:
        return img
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-1, 1)) * {1: 0.6, 2: 1.5, 3: 2.5}[severity]
    out = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    if severity >= 3:  # low-resolution round-trip
        out = out.resize((int(PAGE[0] * 0.6), int(PAGE[1] * 0.6)), Image.BILINEAR).resize(PAGE, Image.BILINEAR)
    blur = {1: 0.0, 2: 0.8, 3: 1.4}[severity]
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(out, dtype=np.float32)
    contrast = {1: 1.0, 2: 0.8, 3: 0.65}[severity]
    a = 255 - (255 - a) * contrast  # dark ink fades toward grey
    if severity >= 2:  # lighting gradient / shadow
        grad = np.linspace(0, {2: 25, 3: 55}[severity], a.shape[1], dtype=np.float32)[None, :]
        a = a - grad * rng.choice([-1, 1]) * 0.5
        a = a + (rng.uniform(-10, 10) if severity == 3 else 0)
    a = a + rng.normal(0, {1: 6, 2: 14, 3: 25}[severity], a.shape)
    if severity >= 3:  # specks
        for _ in range(250):
            y, x = rng.integers(0, a.shape[0]), rng.integers(0, a.shape[1])
            a[y:y + rng.integers(1, 4), x:x + rng.integers(1, 4)] = rng.choice([0, 255])
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality={1: 85, 2: 60, 3: 35}[severity])
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def scan(lines: list[str], severity: int, seed: int = 0, handwriting_style: bool = False) -> Image.Image:
    return degrade(draw_page(lines, handwriting_style=handwriting_style, seed=seed), severity, seed)
