"""Synthetic market-snapshot reports with known ground truth.

The public repository uses only synthetic documents: a generated page has a hidden ground-truth record, so every
extracted field can be scored exactly, and no real report content is ever needed in the repo. The records are
internally consistent on purpose (bid <= ask, low <= open <= high, change = bid - previous close, the 52-week range
brackets the day's range), because that consistency is what the validator later uses to catch OCR mistakes.

Four page layouts are generated so the extractor cannot overfit one format:
  kv      key: value
  dots    Key ........ value
  table   Key      value  (aligned columns, no separators)
  json    "key": value,  (the shape of an embedded JSON block)
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass, field

CURRENCIES = {  # code: (low, high, decimals)
    "AUD": (0.60, 0.74, 5), "GBP": (1.20, 1.38, 5), "EUR": (1.04, 1.16, 5), "CHF": (0.84, 0.98, 5),
    "CAD": (0.71, 0.76, 5), "JPY": (138.0, 158.0, 3), "NZD": (0.56, 0.64, 5), "NOK": (0.085, 0.100, 5),
}
FIELDS = ["bid", "ask", "open", "previous_close", "high", "low", "change", "change_percent", "high_52week", "low_52week"]
LAYOUTS = ("kv", "dots", "table", "json")
HEADLINE_STEMS = [
    "{c} steadies as traders wait for central bank minutes", "{c} slips on softer risk appetite", "Weekly outlook: {c} range-bound ahead of data",
    "{c} firms after upbeat retail figures", "Dollar strength weighs on {c}", "{c} holds gains as yields ease",
    "Analysts expect {c} volatility around policy meeting", "{c} edges higher in thin Asian trade", "Market wrap: {c} unchanged after mixed signals",
]
SOURCES = ["Market Wire", "FX Daily", "Currency Desk", "Global Markets Feed"]
SUMMARIES = [
    "The pair traded in a narrow range and closed close to its opening level.",
    "Moves were small, with participants waiting for the next scheduled data release.",
    "Price action was muted; the session range stayed well inside the 52-week band.",
]


@dataclass
class Doc:
    doc_id: str
    layout: str
    truth: dict  # the ground-truth record
    lines: list[str] = field(default_factory=list)  # the text of the page, line by line


def _fmt(x: float, dp: int) -> str:
    return f"{x:.{dp}f}"


def make_record(rng: random.Random, doc_id: str) -> dict:
    cur = rng.choice(list(CURRENCIES))
    lo, hi, dp = CURRENCIES[cur]
    unit = 10 ** -dp
    prev = round(rng.uniform(lo, hi), dp)
    day_move = rng.gauss(0, (hi - lo) * 0.004)
    opn = round(prev + rng.gauss(0, (hi - lo) * 0.002), dp)
    bid = round(prev + day_move, dp)
    ask = round(bid + rng.choice([1, 2, 3, 4]) * unit, dp)
    high = round(max(opn, bid, ask) + abs(rng.gauss(0, (hi - lo) * 0.002)), dp)
    low = round(min(opn, bid, ask) - abs(rng.gauss(0, (hi - lo) * 0.002)), dp)
    change = round(bid - prev, dp)
    pct = round(change / prev * 100, 5)
    hi52 = round(high + abs(rng.gauss(0, (hi - lo) * 0.08)) + unit, dp)
    lo52 = round(low - abs(rng.gauss(0, (hi - lo) * 0.08)) - unit, dp)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    n_news = rng.randint(1, 3)
    news = [{"headline": rng.choice(HEADLINE_STEMS).format(c=cur), "source": rng.choice(SOURCES),
             "published_at": f"{date} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"} for _ in range(n_news)]
    return {"doc_id": doc_id, "currency": cur, "report_date": date, "bid": bid, "ask": ask, "open": opn, "previous_close": prev, "high": high, "low": low,
            "change": change, "change_percent": pct, "high_52week": hi52, "low_52week": lo52, "news_count": n_news, "news": news,
            "summary": rng.choice(SUMMARIES)}


def _market_lines(rec: dict, layout: str) -> list[str]:
    dp = CURRENCIES[rec["currency"]][2]
    pairs = [(f, _fmt(rec[f], 5 if f == "change_percent" else dp)) for f in FIELDS]
    label = {"bid": "Bid", "ask": "Ask", "open": "Open", "previous_close": "Previous close", "high": "High", "low": "Low", "change": "Change",
             "change_percent": "Change %", "high_52week": "52-week high", "low_52week": "52-week low"}
    if layout == "kv":
        return [f"{label[f]}: {v}" for f, v in pairs]
    if layout == "dots":
        return [f"{label[f]} " + "." * (26 - len(label[f])) + f" {v}" for f, v in pairs]
    if layout == "table":
        return [f"{label[f]:<22}{v:>12}" for f, v in pairs]
    return ['{'] + [f'  "{f}": {v}' + ("," if i < len(pairs) - 1 else "") for i, (f, v) in enumerate(pairs)] + ['}']


def make_doc(seed: int, layout: str | None = None) -> Doc:
    rng = random.Random(seed)
    rec = make_record(rng, f"doc-{seed:05d}")
    layout = layout or rng.choice(LAYOUTS)
    lines = [f"{rec['currency']} - Market Snapshot Report", f"Daily currency snapshot - {rec['report_date']}", "", "1. Market data"]
    lines += _market_lines(rec, layout)
    lines += ["", f"2. Relevant news ({rec['news_count']})"]
    for n in rec["news"]:
        lines += textwrap.wrap(f"- {n['headline']} ({n['source']}, {n['published_at']} UTC)", 66, subsequent_indent="  ")
    lines += ["", "3. Summary", *textwrap.wrap(rec["summary"], 66)]
    return Doc(rec["doc_id"], layout, rec, lines)


def make_corpus(n: int, seed: int = 0) -> list[Doc]:
    """n documents, layouts balanced, deterministic in `seed`."""
    return [make_doc(seed * 100_000 + i, LAYOUTS[i % len(LAYOUTS)]) for i in range(n)]
