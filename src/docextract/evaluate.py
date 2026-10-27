"""Scoring extractions against ground truth, and the accuracy / review-burden trade-off.

The headline numbers are FIELD-level accuracy (each of 13 scalar fields on each document is one trial) and DOCUMENT-level
"every field correct" rate. Both come with intervals (Wilson for fields; a bootstrap over documents for the document-level
rate, because fields within one document are not independent).

The operational question is not "how accurate is the extractor" but "how much must a person check to reach a target error rate":
`tradeoff` sweeps the acceptance threshold and reports, for each, the share of fields auto-accepted (coverage), the accuracy among
them, and the errors that would slip through to the final record if a person fixed everything routed to review.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from docextract.pipeline import SCALAR, Result
from docextract.synth import CURRENCIES

HEADLINE_MATCH = 90  # rapidfuzz ratio


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def field_correct(name: str, got, truth: dict) -> bool:
    want = truth[name]
    if got is None:
        return False
    if name in ("currency", "report_date"):
        return str(got) == str(want)
    if name == "news_count":
        return int(got) == int(want)
    dp = 5 if name == "change_percent" else CURRENCIES[truth["currency"]][2]
    return round(float(got), dp) == round(float(want), dp)


def headline_scores(got: list[str], want: list[str]) -> tuple[int, int, int]:
    """(matched, n_got, n_want): a printed headline counts once if a distinct extracted one matches it at >= 90 similarity."""
    used: set[int] = set()
    matched = 0
    for w in want:
        for i, g in enumerate(got):
            if i not in used and fuzz.ratio(re.sub(r"\s+", " ", g).lower(), re.sub(r"\s+", " ", w).lower()) >= HEADLINE_MATCH:
                used.add(i)
                matched += 1
                break
    return matched, len(got), len(want)


def cer(ocr_text: str, true_lines: list[str]) -> float:
    squash = lambda t: re.sub(r"[.·'*]{3,}", ".", re.sub(r"\s+", "", t))  # noqa: E731 - leader dots: OCR miscounts them; not a readability signal
    ref, hyp = squash("".join(true_lines)), squash(ocr_text)
    return Levenshtein.distance(hyp, ref) / max(len(ref), 1)


@dataclass
class DocScore:
    doc_id: str
    layout: str
    severity: int
    correct: dict[str, bool]
    conf: dict[str, float]
    routes: dict[str, str]
    headline: tuple[int, int, int]
    cer: float | None = None

    @property
    def all_correct(self) -> bool:
        return all(self.correct.values())


def score_doc(doc_id: str, layout: str, severity: int, result: Result, truth: dict, ocr_cer: float | None = None) -> DocScore:
    correct = {f: field_correct(f, result.fields[f].value, truth) for f in SCALAR}
    return DocScore(doc_id, layout, severity, correct, {f: result.fields[f].conf for f in SCALAR}, {f: result.fields[f].route for f in SCALAR},
                    headline_scores(result.headlines, [n["headline"] for n in truth["news"]]), ocr_cer)


def summarise(scores: list[DocScore]) -> dict:
    n_fields = sum(len(s.correct) for s in scores)
    k = sum(sum(s.correct.values()) for s in scores)
    lo, hi = wilson(k, n_fields)
    docs = np.array([s.all_correct for s in scores], dtype=float)
    rng = np.random.default_rng(0)
    boots = [docs[rng.integers(0, len(docs), len(docs))].mean() for _ in range(1000)] if len(docs) else [0.0]
    hm = sum(s.headline[0] for s in scores)
    hg = sum(s.headline[1] for s in scores)
    hw = sum(s.headline[2] for s in scores)
    p, r = (hm / hg if hg else 0.0), (hm / hw if hw else 0.0)
    return {
        "docs": len(scores), "field_accuracy": k / n_fields if n_fields else 0.0, "field_ci95": (lo, hi),
        "doc_all_correct": float(docs.mean()) if len(docs) else 0.0, "doc_ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "headline_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "mean_cer": float(np.mean([s.cer for s in scores if s.cer is not None])) if any(s.cer is not None for s in scores) else None,
    }


def per_field(scores: list[DocScore]) -> dict[str, float]:
    return {f: float(np.mean([s.correct[f] for s in scores])) for f in SCALAR}


def tradeoff(scores: list[DocScore], thresholds=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)) -> list[dict]:
    """For each acceptance threshold: coverage, accuracy of what is auto-accepted, and the error rate of the FINAL records
    if every routed-to-review field is corrected by a person (so only auto-accepted mistakes remain)."""
    flat = [(s.conf[f], s.correct[f]) for s in scores for f in SCALAR]
    conf = np.array([c for c, _ in flat])
    ok = np.array([o for _, o in flat])
    out = []
    for t in thresholds:
        auto = conf >= t
        cov = float(auto.mean())
        out.append({"threshold": t, "auto_coverage": cov, "review_burden": 1 - cov,
                    "accuracy_of_auto": float(ok[auto].mean()) if auto.any() else float("nan"),
                    "final_error_rate": float((~ok & auto).sum() / len(ok))})
    return out


def threshold_for(curve: list[dict], max_final_error: float) -> dict | None:
    """The cheapest operating point (lowest review burden) whose final error rate is at most the target."""
    ok = [c for c in curve if c["final_error_rate"] <= max_final_error]
    return min(ok, key=lambda c: c["review_burden"]) if ok else None
