"""Rule-based extraction from OCR rows, built to survive layout changes and common OCR confusions.

It does not assume a layout. For every row it takes the LAST number as the value and whatever precedes it as the label,
then maps the label to a schema field by fuzzy matching (so 'Prev. close', 'Previous close ....' and '"previous_close":' all
land on the same field, and a label damaged by one OCR error still matches). Numbers are repaired for the usual
look-alike confusions only when the token is mostly digits, so a word is never turned into a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from docextract.ocr import Row
from docextract.synth import CURRENCIES

NUM_FIELDS = ["bid", "ask", "open", "previous_close", "high", "low", "change", "change_percent", "high_52week", "low_52week"]
ALIASES = {
    "bid": ["bid"], "ask": ["ask", "offer"], "open": ["open", "opening"], "previous_close": ["previous close", "prev close", "previous_close", "prior close"],
    "high": ["high", "day high"], "low": ["low", "day low"], "change": ["change", "net change"],
    "change_percent": ["change %", "change percent", "change_percent", "pct change", "percent change"],
    "high_52week": ["52-week high", "52 week high", "high_52week", "52wk high"], "low_52week": ["52-week low", "52 week low", "low_52week", "52wk low"],
}
_CANON = {re.sub(r"[^a-z0-9%]", "", a.lower()): f for f, als in ALIASES.items() for a in als}
CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "i": "1", "|": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", ",": "."})
NUM_TAIL = re.compile(r"(?P<num>(?:[-−–]\s?)?[0-9OoDlIi|SsBZz][0-9OoDlIi|SsBZz.,]*)\s*[,;]?\s*$")


@dataclass
class Parsed:
    fields: dict[str, float]  # parsed numeric fields
    raw: dict[str, str]  # the text the number came from, for audit
    conf: dict[str, float]  # row confidence per field
    currency: str | None
    report_date: str | None
    news_count: int | None
    headlines: list[str]


def clean_number(token: str) -> float | None:
    """Repair look-alikes in a mostly-numeric token and parse it; None if it does not look like one number."""
    t = token.strip().replace(" ", "").replace("−", "-").replace("–", "-")
    neg = t.startswith("-")
    body = t.lstrip("-")
    digits = sum(c.isdigit() for c in body)
    if not body or digits < max(1, len(body) // 2):
        return None
    body = body.translate(CONFUSIONS)
    parts = body.split(".")
    if len(parts) > 2:  # stray dots from noise: keep the first one as the decimal point
        body = parts[0] + "." + "".join(parts[1:])
    try:
        v = float(body)
    except ValueError:
        return None
    return -v if neg else v


def match_field(label: str) -> str | None:
    key = re.sub(r"[^a-z0-9%]", "", label.lower())
    if not key:
        return None
    if key in _CANON:
        return _CANON[key]
    ranked = process.extract(key, list(_CANON), scorer=fuzz.ratio, limit=2, score_cutoff=66)
    if not ranked:
        return None
    # accept a near match only if it clearly beats the runner-up (a one-character OCR error on 'bid', not a guess between two labels)
    if len(ranked) > 1 and _CANON[ranked[0][0]] != _CANON[ranked[1][0]] and ranked[0][1] - ranked[1][1] < 8:
        return None
    return _CANON[ranked[0][0]]


def split_row(text: str) -> tuple[str, str] | None:
    """(label, number-token) from a row, or None if the row does not end in a number."""
    m = NUM_TAIL.search(text.strip())
    if not m:
        return None
    label = text[: m.start()].strip().strip("\"',:{}.= ")
    label = re.sub(r"[.\s]{3,}$", "", label).strip(" :.\"'")
    token = m.group("num").rstrip(",;. ")  # a trailing comma is JSON punctuation, not a decimal point
    return (label, token) if label and token else None


DATE = re.compile(r"(2[0O]\d[0-9O])[-/.]([0-1]?[0-9O])[-/.]([0-3]?[0-9O])")


def _digits(s: str) -> str:
    return s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")


_FULLWIDTH = str.maketrans({"（": "(", "）": ")", "：": ":", "，": ","})
KNOWN_CURRENCIES = tuple(CURRENCIES)


def _norm(text: str) -> str:
    return text.translate(_FULLWIDTH).strip()


def read_currency(token: str) -> str | None:
    """The 3-letter code from the title row. OCR mixes case ('AuD', 'NoK') and swaps digits for letters, so normalise, then accept an exact known
    code or a unique known code one edit away; anything else is left unread (and goes to a person) rather than guessed."""
    code = token.upper().replace("0", "O").replace("1", "I").replace("5", "S")
    if code in KNOWN_CURRENCIES:
        return code
    near = [c for c in KNOWN_CURRENCIES if sum(a != b for a, b in zip(code, c, strict=True)) == 1]
    return near[0] if len(near) == 1 else None


def parse_rows(rows: list[Row]) -> Parsed:
    fields: dict[str, float] = {}
    raw: dict[str, str] = {}
    conf: dict[str, float] = {}
    currency = date = None
    news_count = None
    headlines: list[str] = []
    in_news = False
    news_buf = ""
    for r in rows:
        text = _norm(r.text)
        if not text:
            continue
        if currency is None and re.search(r"snapshot", text, re.I) and (m := re.match(r"^([A-Za-z0-9]{3})\b", text)):
            currency = read_currency(m.group(1))
        if date is None and (m := DATE.search(text)) and re.search(r"snapshot|daily", text, re.I):
            date = f"{_digits(m.group(1))}-{int(_digits(m.group(2))):02d}-{int(_digits(m.group(3))):02d}"
        if re.match(r"^\s*2\.?\s*Relevant\s*news", text, re.I):
            in_news = True
            if (m := re.search(r"\(\s*([0-9OolIi])\s*\)", text)):
                news_count = int(_digits(m.group(1)))
            continue
        if re.match(r"^\s*3\.?\s*Summary", text, re.I):
            in_news = False
            continue
        if in_news:
            # Items are delimited by their closing "... UTC)", NOT by a leading dash (OCR often drops it) or by line breaks (long items wrap).
            news_buf = (news_buf + " " + text.lstrip("-– ")).strip()
            if re.search(r"UTC\s*\)", news_buf):
                headlines.append(re.sub(r"\s*\(.*$", "", news_buf).strip())
                news_buf = ""
            continue
        sp = split_row(text)
        if not sp:
            continue
        label, tok = sp
        f = match_field(label)
        if f and f not in fields:
            v = clean_number(tok)
            if v is not None:
                fields[f], raw[f], conf[f] = v, text, r.conf
    return Parsed(fields, raw, conf, currency, date, news_count, headlines)
