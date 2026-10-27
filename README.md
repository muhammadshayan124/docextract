# docextract

Scanned reports in, **verified JSON out**: synthetic scans at four quality levels, OCR, rule-based and small-LLM field extraction, validation that
routes doubtful fields to a person, and honest measurement of how accurate it is and **how much a human has to check to reach a target error rate**.

> **Scope, stated plainly.**
> - The public repository uses **synthetic documents only** (a generator with hidden ground truth), so every extracted field is scored exactly and no real report is needed.
> - The scans are **rendered and degraded by code**, not photographed. The handwriting-style pages use a script font with per-glyph jitter: a stress test, **not real
>   handwriting**, and its results must not be read as handwriting-recognition accuracy.
> - The language model is **Qwen3-1.7B (1.4 GB) on a CPU through Ollama**. A vision-language model or TrOCR would be the next step for real handwriting; neither was used.

## The task

A one-page "market snapshot report": a title with a currency code, a date, ten market-data fields (bid, ask, open, previous close, high, low, change, change %, 52-week high
and low), a count of news items and their headlines, and a summary. Four layouts (`key: value`, dot leaders, aligned columns, a JSON-like block) so the extractor cannot
overfit one format. Records are internally consistent by construction (bid <= ask, low <= open <= high, change = bid - previous close, the 52-week range brackets the day).

## Pipeline

```
page image --OCR (RapidOCR / PaddleOCR models via ONNX, CPU)--> rows + per-row confidence
   --> rules: last number on the row is the value, the text before it is the label, fuzzy-matched to a field; look-alike repair (O/0, l/1, S/5 ...) only on mostly-numeric tokens
   --> LLM (optional): schema-constrained JSON, numbers as strings copied digit-for-digit, "do not calculate, correct or guess"
   --> validation: bid <= ask, low/high bracket open and bid, 52-week range brackets the day, change = bid - previous close, pct = change / previous close, plausible range and
       decimal places per currency, date validity, headline count = header count
   --> per-field confidence = OCR row confidence x 0.5^(violated constraints touching the field) x 0.4 if rules and LLM disagree
   --> route: auto-accept, or send to review
```

`hybrid` mode starts from the rules and lets the LLM replace a value **only if that reduces the number of violated constraints**, so the model can repair a misread digit
but cannot override a value that checks out. `docextract review` is an interactive CLI that shows a person only the fields the pipeline will not trust.

## How the evaluation was run (and why you can trust the numbers below)

1. **Development corpus (seed 0).** I built the pipeline, ran it on 80 synthetic documents x 4 scan severities, and looked at what failed. The first run scored 0.843 field
   accuracy (rules) and **0 of 80 pages fully correct at severity 2**, which was the signal that something systematic was wrong. Two causes were my parser (OCR drops the leading
   dash of list items, drops spaces, uses full-width parentheses, and mixes letter case in the currency code, none of which my clean-text unit tests had), and one was the OCR engine:
   its angle classifier **flipped short numbers upside down** (`0.00006` came back as `90000'0`), so it is disabled (a document that can arrive rotated by 180 degrees needs a
   page-level orientation check instead). The first run's summary is kept in `results/dev_run1/`.
2. **The parser was frozen in a commit (`f3c289a`) before the held-out corpus existed.** After the fixes, the dev corpus (40 documents) scores 100% at severities 0-2.
3. **Held-out corpus (seed 1): 80 new documents x 4 severities, generated and OCR'd only after the freeze.** Everything under "Results" is from this corpus, scored once.

Dev and held-out agree (severity 3: 60.6% dev, 61.8% held-out), which is the evidence against having overfit the parser to the dev pages. It is **not** evidence about other
document layouts: both corpora come from the same generator and templates.

## Results (held-out corpus)

<!-- results:start -->
**Same pages for every mode** (the LLM sample), field accuracy over 13 scalar fields per page, with 95% Wilson interval:

| mode | pages | field accuracy | every field right (page) | headline F1 |
|---|---|---|---|---|
| Rules | 48 | 0.899 (0.873-0.920) | 0.75 (0.62-0.85) | 0.91 |
| LLM only (Qwen3-1.7B) | 48 | 0.670 (0.632-0.706) | 0.00 (0.00-0.00) | 0.01 |
| Hybrid (rules + LLM repair) | 48 | 0.905 (0.880-0.926) | 0.75 (0.62-0.85) | 0.88 |

**Rules mode on all pages, by scan severity** (0 clean ... 3 bad fax/phone photo):

| severity | pages | field accuracy | every field right | OCR character error rate |
|---|---|---|---|---|
| 0 | 80 | 1.000 (0.996-1.000) | 1.00 | 0.011 |
| 1 | 80 | 1.000 (0.996-1.000) | 1.00 | 0.013 |
| 2 | 80 | 1.000 (0.996-1.000) | 1.00 | 0.022 |
| 3 | 80 | 0.618 (0.588-0.647) | 0.00 | 0.092 |

**By layout** (rules, all severities):

| layout | pages | field accuracy |
|---|---|---|
| kv | 80 | 0.890 (0.870-0.908) |
| dots | 80 | 0.910 (0.891-0.926) |
| table | 80 | 0.926 (0.908-0.940) |
| json | 80 | 0.892 (0.872-0.910) |

**Handwriting-style stress test** (20 pages in a script font with per-glyph jitter; *not* real handwriting): field accuracy 0.615 (0.555-0.672).

**Review burden vs residual error, Rules** (a person corrects every field routed to review; the final error is what remains among auto-accepted fields):

| threshold | auto-accepted | needs review | accuracy of auto-accepted | final error rate |
|---|---|---|---|---|
| 0.00 | 100.0% | 0.0% | 0.8990 | 10.10% |
| 0.20 | 93.6% | 6.4% | 0.9606 | 3.69% |
| 0.40 | 91.3% | 8.7% | 0.9719 | 2.56% |
| 0.50 | 85.6% | 14.4% | 0.9869 | 1.12% |
| 0.60 | 85.6% | 14.4% | 0.9869 | 1.12% |
| 0.70 | 84.8% | 15.2% | 0.9868 | 1.12% |
| 0.80 | 83.0% | 17.0% | 0.9865 | 1.12% |
| 0.90 | 74.7% | 25.3% | 0.9957 | 0.32% |
| 0.95 | 63.8% | 36.2% | 1.0000 | 0.00% |
| 0.98 | 28.0% | 72.0% | 1.0000 | 0.00% |

To keep the final error at or below 1.0%: threshold 0.90 needs a person to review 25.3% of fields.

To keep the final error at or below 0.5%: threshold 0.90 needs a person to review 25.3% of fields.

**Review burden vs residual error, Hybrid (rules + LLM repair)** (a person corrects every field routed to review; the final error is what remains among auto-accepted fields):

| threshold | auto-accepted | needs review | accuracy of auto-accepted | final error rate |
|---|---|---|---|---|
| 0.00 | 100.0% | 0.0% | 0.9054 | 9.46% |
| 0.20 | 94.1% | 5.9% | 0.9608 | 3.69% |
| 0.40 | 91.5% | 8.5% | 0.9755 | 2.24% |
| 0.50 | 85.6% | 14.4% | 0.9906 | 0.80% |
| 0.60 | 85.6% | 14.4% | 0.9906 | 0.80% |
| 0.70 | 84.8% | 15.2% | 0.9905 | 0.80% |
| 0.80 | 82.9% | 17.1% | 0.9903 | 0.80% |
| 0.90 | 74.5% | 25.5% | 0.9978 | 0.16% |
| 0.95 | 63.8% | 36.2% | 1.0000 | 0.00% |
| 0.98 | 28.0% | 72.0% | 1.0000 | 0.00% |

To keep the final error at or below 1.0%: threshold 0.50 needs a person to review 14.4% of fields.

To keep the final error at or below 0.5%: threshold 0.90 needs a person to review 25.5% of fields.
<!-- results:end -->

### What the numbers say

- **Severities 0-2 are solved here; severity 3 is where it gets hard.** Rules extraction gets all 3,120 fields right on the 240 held-out pages at severities 0-2 and 61.8% of fields at
  severity 3 (a bad fax or phone photo; OCR character error 9.2% against 1.1% clean). So this benchmark discriminates only at severity 3, and the perfect scores at 0-2 are a
  statement about this generator, not a promise about real scans.
- **The small language model is much worse than rules on the same 48 pages** (0.670 vs 0.899 field accuracy; no page fully correct). It copies the printed numbers faithfully where it
  extracts them (bid 0.85, ask 0.94, open 0.94), but returns null for the currency (17% correct), the date (2%) and the 52-week fields (38-40%), and ignores the instruction to leave the
  source and time out of headline text (headline F1 0.01). It also costs about 38 s per page on this CPU, against effectively nothing for rules.
- **Hybrid accuracy equals rules accuracy** (0.905 vs 0.899; the intervals overlap, so this is not a difference), **but the operating point differs.** To keep the error rate of the
  final records at or below 1%, rules need a person to review **25.3%** of fields, hybrid **14.4%**. Treat that as indicative: the threshold was read off the same 48 pages (624 fields),
  and the 1.12% vs 0.80% residual errors rest on a handful of severity-3 fields.
- **The confidence signal is useful.** Fields the constraint checks leave untouched and the OCR is sure about are almost always right: 99.6% accurate among the 74.7% of fields auto-accepted at
  threshold 0.9. All residual errors are on severity-3 pages.
- **The handwriting-style stress pages score 0.615**, close to severity 3. That says a script font with jitter defeats this OCR+rules pipeline; it says nothing about real handwriting.


## Run it

```bash
pip install -e ".[dev]"
pytest                                      # 27 tests (one runs real OCR on a clean page)
docextract preview                          # writes one page at every severity, plus the handwriting-style page
docextract run --docs 80 --llm-sample 48 --out results/heldout --seed 1   # OCR all pages, then the LLM on a balanced sample; resumable
python scripts/fill_readme.py               # fill the results block from results/heldout/summary.json
docextract review --pages 3 --severity 2    # human verification, interactively
```

## Limitations

- **Synthetic documents.** Real reports have more layout variety, fonts, stamps and tables; accuracy there is unmeasured.
- **Rendered "scans".** Noise, blur, skew, contrast loss, JPEG compression and a low-resolution round trip approximate a poor fax or phone photo but are not one.
- **A small LLM.** The LLM modes are evaluated on a sample of pages (the LLM is the slow part on CPU), and all modes are compared on that same sample.
- **Constraint-based validation is specific to this schema.** It works here because the fields obey arithmetic relations; documents without such redundancy need a different confidence signal.
- **Review is simulated in the evaluation** (a reviewer who fixes every routed field correctly); the interactive CLI is the real workflow, but its error rate was not measured.

MIT licence.
