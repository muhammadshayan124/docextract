"""OCR (RapidOCR: PaddleOCR detection + recognition models run by ONNX Runtime, CPU only, models ship inside the wheel).

The detector returns text boxes, and in table-style layouts a label and its value come back as separate boxes on the
same visual row. `group_rows` merges boxes whose vertical centres are close, left to right, so downstream code sees
one string per printed row plus the minimum recognition confidence on that row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Row:
    text: str
    conf: float  # minimum confidence of the boxes on the row: a row is only as reliable as its weakest piece
    y: float


_engine = None


def run_ocr(image) -> list[Box]:
    """image: PIL image or ndarray. Returns the boxes RapidOCR found (possibly none)."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    # use_cls=False: the angle classifier flips short, near-symmetric numbers upside down ("0.00006" came back as "90000'0"). These pages are
    # upright (small skew only); a document that can arrive rotated by 180 degrees needs a page-level orientation check instead.
    result, _ = _engine(arr, use_cls=False)
    boxes = []
    for pts, text, conf in result or []:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append(Box(text, min(xs), min(ys), max(xs), max(ys), float(conf)))
    return boxes


def group_rows(boxes: list[Box], tol: float = 0.6) -> list[Row]:
    """Boxes whose vertical centres differ by less than `tol` x the median box height share a row."""
    if not boxes:
        return []
    med = float(np.median([b.h for b in boxes])) or 1.0
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.yc):
        if rows and abs(b.yc - np.mean([x.yc for x in rows[-1]])) < tol * med:
            rows[-1].append(b)
        else:
            rows.append([b])
    out = []
    for r in rows:
        r.sort(key=lambda b: b.x0)
        out.append(Row(" ".join(b.text.strip() for b in r), min(b.conf for b in r), float(np.mean([b.yc for b in r]))))
    return out


def ocr_rows(image) -> list[Row]:
    return group_rows(run_ocr(image))
