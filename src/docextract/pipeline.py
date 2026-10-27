"""From OCR rows to a validated record with per-field routing.

Modes:
  rules    layout-agnostic parsing only
  llm      the local language model only
  hybrid   rules first; where the validator finds a violation, try the LLM's value for the implicated field and keep it only if the
           total number of violations goes down (so the LLM can repair a misread digit but cannot override a value that checks out)

Every field leaves with a confidence and a route: `auto` (accepted without a person) or `review` (a person must check it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from docextract import parse, validate
from docextract.llm import LLMRecord
from docextract.ocr import Row

SCALAR = ["currency", "report_date", "news_count", *parse.NUM_FIELDS]


@dataclass
class FieldResult:
    value: object
    source: str  # rules | llm | none
    conf: float
    suspicion: int
    route: str = "review"


@dataclass
class Result:
    fields: dict[str, FieldResult]
    headlines: list[str]
    violations: list[validate.Violation]
    ocr_conf: float = 0.0
    notes: list[str] = field(default_factory=list)


def llm_to_values(rec: LLMRecord) -> tuple[dict, dict[str, str]]:
    vals: dict = {}
    raw: dict[str, str] = {}
    for f in parse.NUM_FIELDS:
        tok = getattr(rec, f)
        if tok is not None:
            n = parse.clean_number(tok)
            if n is not None:
                vals[f] = n
                raw[f] = tok
    if rec.currency:
        vals["currency"] = rec.currency.strip().upper()[:3]
    if rec.report_date:
        vals["report_date"] = rec.report_date.strip()
    if rec.news_count is not None:
        vals["news_count"] = rec.news_count
    return vals, raw


def _count(vals: dict, raw: dict, headlines: list[str]) -> int:
    return len(validate.violations({**vals, "headlines": headlines}, raw))


def run(rows: list[Row], mode: str = "rules", llm_record: LLMRecord | None = None, threshold: float = 0.5) -> Result:
    p = parse.parse_rows(rows)
    rules_vals: dict = dict(p.fields)
    for k, v in (("currency", p.currency), ("report_date", p.report_date), ("news_count", p.news_count)):
        if v is not None:
            rules_vals[k] = v
    llm_vals, llm_raw = llm_to_values(llm_record) if llm_record is not None else ({}, {})
    raw = dict(p.raw)
    headlines = list(p.headlines)
    notes: list[str] = []

    if mode == "llm":
        vals, raw, source = dict(llm_vals), dict(llm_raw), {k: "llm" for k in llm_vals}
        headlines = list(llm_record.headlines) if llm_record else []
    else:
        vals, source = dict(rules_vals), {k: "rules" for k in rules_vals}
        if mode == "hybrid" and llm_record is not None:
            if not headlines and llm_record.headlines:
                headlines = list(llm_record.headlines)
            for f in SCALAR:  # a field the rules could not read at all is filled from the LLM
                if f not in vals and f in llm_vals:
                    vals[f], source[f], raw[f] = llm_vals[f], "llm", llm_raw.get(f, "")
            sus = validate.suspicion(validate.violations({**vals, "headlines": headlines}, raw))
            for f in sorted(sus, key=lambda k: -sus[k]):  # most-implicated first
                if f in llm_vals and llm_vals[f] != vals.get(f):
                    trial = {**vals, f: llm_vals[f]}
                    if _count(trial, {**raw, f: llm_raw.get(f, raw.get(f, ""))}, headlines) < _count(vals, raw, headlines):
                        notes.append(f"{f}: LLM value {llm_vals[f]!r} replaced rules value {vals.get(f)!r} (fewer violations)")
                        vals[f], source[f] = llm_vals[f], "llm"
                        raw[f] = llm_raw.get(f, raw.get(f, ""))

    vs = validate.violations({**vals, "headlines": headlines}, raw)
    sus = validate.suspicion(vs)
    fields: dict[str, FieldResult] = {}
    for f in SCALAR:
        if f not in vals:
            fields[f] = FieldResult(None, "none", 0.0, sus.get(f, 0), "review")
            continue
        agree = None
        if llm_record is not None and f in llm_vals and f in rules_vals:
            agree = llm_vals[f] == rules_vals[f]
        ocr_conf = p.conf.get(f)
        if ocr_conf is None:  # header fields: use the mean row confidence
            ocr_conf = sum(r.conf for r in rows) / len(rows) if rows else 0.0
        c = validate.field_confidence(ocr_conf, sus.get(f, 0), agree)
        fields[f] = FieldResult(vals[f], source.get(f, "rules"), c, sus.get(f, 0), "auto" if c >= threshold else "review")
    return Result(fields, headlines, vs, sum(r.conf for r in rows) / len(rows) if rows else 0.0, notes)


def reroute(result: Result, threshold: float) -> Result:
    """Same extraction, different acceptance threshold (used to draw the accuracy-coverage curve)."""
    for fr in result.fields.values():
        fr.route = "auto" if fr.conf >= threshold and fr.value is not None else "review"
    return result
