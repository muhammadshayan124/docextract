"""The experiment: synthetic documents x scan severities, OCR once per page, then every extraction mode on the cached OCR.

Caches (under `results/cache/`) make the run resumable and keep model use sequential: OCR for all pages first, then the LLM over
its sample, then the cheap modes. A cached LLM answer is keyed by the document id, severity and the exact OCR text it saw.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from docextract import evaluate, ocr, pipeline, render, synth
from docextract.llm import LLMRecord, OllamaExtractor
from docextract.ocr import Row

SEVERITIES = (0, 1, 2, 3)


def page_key(doc_id: str, severity: int, hand: bool) -> str:
    return f"{doc_id}-s{severity}{'-hand' if hand else ''}"


def ocr_cached(cache: Path, doc: synth.Doc, severity: int, hand: bool = False, save_images: Path | None = None) -> tuple[list[Row], float]:
    key = page_key(doc.doc_id, severity, hand)
    f = cache / "ocr" / f"{key}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return [Row(**r) for r in d["rows"]], d["seconds"]
    img: Image.Image = render.scan(doc.lines, severity, seed=int(doc.doc_id[-5:]), handwriting_style=hand)
    if save_images is not None:
        save_images.mkdir(parents=True, exist_ok=True)
        img.save(save_images / f"{key}.png")
    t0 = time.time()
    rows = ocr.ocr_rows(img)
    secs = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"rows": [asdict(r) for r in rows], "seconds": secs}), encoding="utf-8")
    return rows, secs


def llm_cached(cache: Path, extractor: OllamaExtractor, key: str, rows: list[Row]) -> tuple[LLMRecord, float]:
    text = "\n".join(r.text for r in rows)
    h = hashlib.sha1(text.encode()).hexdigest()[:10]
    f = cache / "llm" / f"{key}-{h}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return LLMRecord.model_validate(d["record"]), d["seconds"]
    t0 = time.time()
    rec = extractor.extract(text)
    secs = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"record": rec.model_dump(), "seconds": secs}), encoding="utf-8")
    return rec, secs


def score_all(docs: list[synth.Doc], severities, cache: Path, mode: str, extractor: OllamaExtractor | None, llm_keys: set[str] | None,
              hand: bool = False) -> list[evaluate.DocScore]:
    scores = []
    for d in docs:
        for s in severities:
            rows, _ = ocr_cached(cache, d, s, hand)
            key = page_key(d.doc_id, s, hand)
            rec = None
            if mode in ("llm", "hybrid") and extractor is not None and (llm_keys is None or key in llm_keys):
                rec, _ = llm_cached(cache, extractor, key, rows)
            elif mode == "llm":
                continue  # the LLM modes are only scored on pages the LLM actually processed
            res = pipeline.run(rows, mode, rec)
            text = " ".join(r.text for r in rows)
            scores.append(evaluate.score_doc(d.doc_id, d.layout, s, res, d.truth, evaluate.cer(text, d.lines)))
    return scores
