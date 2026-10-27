"""Validation: the part that decides what a person has to look at.

OCR and language models both make confident mistakes, so "the model said so" cannot be the acceptance test. The records
have internal structure, and a misread digit usually breaks it: bid must not exceed ask, the day's low/high must bracket
the open and the bid, the 52-week range must bracket the day's range, the change must equal bid - previous close, and the
percentage must equal change / previous close. Each violated constraint implicates specific fields; a field implicated by
violations loses confidence in proportion, and low-confidence fields are routed to review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docextract.synth import CURRENCIES

NUM = ["bid", "ask", "open", "previous_close", "high", "low", "change", "change_percent", "high_52week", "low_52week"]


@dataclass(frozen=True)
class Violation:
    name: str
    fields: tuple[str, ...]
    detail: str


def decimals(raw: str) -> int | None:
    m = re.search(r"[-−]?\d+\.(\d+)\s*[,;]?\s*$", raw.strip())
    return len(m.group(1)) if m else None


def violations(rec: dict, raw: dict[str, str] | None = None) -> list[Violation]:
    """Constraint violations for whatever fields are present. Missing fields are skipped, never guessed."""
    v: list[Violation] = []
    g = rec.get
    cur = g("currency")
    dp = CURRENCIES[cur][2] if cur in CURRENCIES else None
    unit = 10 ** -dp if dp else None

    def le(a: str, b: str, name: str, tol: float = 0.0):
        if g(a) is not None and g(b) is not None and g(a) > g(b) + tol:
            v.append(Violation(name, (a, b), f"{a} {g(a)} > {b} {g(b)}"))

    le("bid", "ask", "bid<=ask")
    le("low", "open", "low<=open")
    le("open", "high", "open<=high")
    le("low", "bid", "low<=bid")
    le("bid", "high", "bid<=high")
    le("low", "ask", "low<=ask")
    le("ask", "high", "ask<=high")
    le("low_52week", "low", "52wk low<=low")
    le("high", "high_52week", "high<=52wk high")
    if None not in (g("change"), g("bid"), g("previous_close")) and unit:
        if abs(g("change") - (g("bid") - g("previous_close"))) > 1.5 * unit:
            v.append(Violation("change=bid-prev", ("change", "bid", "previous_close"), f"change {g('change')} vs bid-prev {g('bid') - g('previous_close'):.{dp}f}"))
    if None not in (g("change_percent"), g("change"), g("previous_close")) and g("previous_close"):
        expect = g("change") / g("previous_close") * 100
        if abs(g("change_percent") - expect) > 2e-4 + 2e-4 * abs(expect):
            v.append(Violation("pct=change/prev", ("change_percent", "change", "previous_close"), f"pct {g('change_percent')} vs {expect:.5f}"))
    if cur in CURRENCIES:
        lo, hi, _ = CURRENCIES[cur]
        for f in ("bid", "ask", "open", "previous_close", "high", "low", "high_52week", "low_52week"):
            if g(f) is not None and not (lo * 0.85 <= g(f) <= hi * 1.15):
                v.append(Violation("plausible range", (f,), f"{f} {g(f)} outside the usual {cur} range {lo}-{hi}"))
    elif cur is not None:
        v.append(Violation("known currency", ("currency",), f"unknown currency code {cur!r}"))
    if raw and dp:
        for f in NUM:
            if f in raw and f not in ("change_percent",) and (d := decimals(raw[f])) is not None and d != dp:
                v.append(Violation("decimal places", (f,), f"{f} printed with {d} decimals, {cur} uses {dp}"))
    if g("report_date"):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", g("report_date"))
        if not m or not (1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31):
            v.append(Violation("date", ("report_date",), f"invalid date {g('report_date')!r}"))
    if g("news_count") is not None and g("headlines") is not None and g("news_count") != len(g("headlines")):
        v.append(Violation("news count", ("news_count",), f"header says {g('news_count')}, found {len(g('headlines'))} headlines"))
    return v


def suspicion(vs: list[Violation]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in vs:
        for f in x.fields:
            out[f] = out.get(f, 0) + 1
    return out


def field_confidence(ocr_conf: float | None, suspicion_count: int, agreement: bool | None) -> float:
    """OCR row confidence, halved for every violated constraint touching the field, and cut hard if two extractors disagree."""
    c = ocr_conf if ocr_conf is not None else 0.0
    if agreement is False:
        c *= 0.4
    return c * (0.5**suspicion_count)
