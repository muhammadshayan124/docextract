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

## Results

<!-- results:start -->
_Pending the experiment run._
<!-- results:end -->

## Run it

```bash
pip install -e ".[dev]"
pytest                                      # 23 tests (one runs real OCR on a clean page)
docextract preview                          # writes one page at every severity, plus the handwriting-style page
docextract run --docs 80 --llm-sample 48    # OCR all pages, then the LLM on a balanced sample; resumable (results/cache)
docextract tables                           # Markdown tables for this README
docextract review --pages 3 --severity 2    # human verification, interactively
```

## Limitations

- **Synthetic documents.** Real reports have more layout variety, fonts, stamps and tables; accuracy there is unmeasured.
- **Rendered "scans".** Noise, blur, skew, contrast loss, JPEG compression and a low-resolution round trip approximate a poor fax or phone photo but are not one.
- **A small LLM.** The LLM modes are evaluated on a sample of pages (the LLM is the slow part on CPU), and all modes are compared on that same sample.
- **Constraint-based validation is specific to this schema.** It works here because the fields obey arithmetic relations; documents without such redundancy need a different confidence signal.
- **Review is simulated in the evaluation** (a reviewer who fixes every routed field correctly); the interactive CLI is the real workflow, but its error rate was not measured.

MIT licence.
