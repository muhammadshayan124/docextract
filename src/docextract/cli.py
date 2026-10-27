from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docextract import evaluate, experiment, pipeline, report, synth
from docextract.llm import OllamaExtractor

app = typer.Typer(help="Scanned reports to verified JSON.", no_args_is_help=True)
console = Console(highlight=False)


@app.command()
def preview(seed: int = 7, out: Annotated[Path, typer.Option(help="Directory for the example pages")] = Path("results/examples")) -> None:
    """Write one document at every scan severity (plus a handwriting-style page) so you can see what the pipeline is up against."""
    from docextract import render

    d = synth.make_doc(seed, "table")
    out.mkdir(parents=True, exist_ok=True)
    for s in experiment.SEVERITIES:
        render.scan(d.lines, s, seed=seed).save(out / f"severity{s}.png")
    render.scan(d.lines, 0, seed=seed, handwriting_style=True).save(out / "handwriting_style.png")
    (out / "truth.json").write_text(json.dumps(d.truth, indent=2), encoding="utf-8")
    console.print(f"wrote {out}")


@app.command()
def run(
    docs: Annotated[int, typer.Option(help="Number of synthetic documents (layouts balanced)")] = 80,
    llm_sample: Annotated[int, typer.Option(help="Pages (balanced over severities) that also go through the LLM; 0 disables the LLM")] = 48,
    hand: Annotated[int, typer.Option(help="Extra handwriting-style documents (rules only)")] = 20,
    model: str = "qwen3:1.7b",
    out: Annotated[Path, typer.Option(help="Results directory (also holds the resumable caches)")] = Path("results"),
    seed: int = 0,
) -> None:
    """OCR every page once, run the extraction modes, and write summary.json."""
    cache = out / "cache"
    corpus = synth.make_corpus(docs, seed)
    # OCR everything first (CPU, no language model running), then the LLM sample, so the two never run together
    for d in corpus:
        for s in experiment.SEVERITIES:
            experiment.ocr_cached(cache, d, s)
    console.print(f"OCR done for {docs * len(experiment.SEVERITIES)} pages")
    result: dict = {"docs": docs, "seed": seed, "modes": {}, "by_severity": {}}
    keys = None
    extractor = None
    if llm_sample:
        per = max(1, llm_sample // len(experiment.SEVERITIES))
        keys = {experiment.page_key(d.doc_id, s, False) for s in experiment.SEVERITIES for d in corpus[:per]}
        extractor = OllamaExtractor(model)
    modes = ["rules"] + (["llm", "hybrid"] if extractor else [])
    all_scores: dict[str, list[evaluate.DocScore]] = {}
    for mode in modes:
        sev = experiment.SEVERITIES
        sc = experiment.score_all(corpus, sev, cache, mode, extractor, keys)
        all_scores[mode] = sc
        console.print(f"{mode}: {len(sc)} pages, field accuracy {evaluate.summarise(sc)['field_accuracy']:.3f}")
    sample_ids = {(s.doc_id, s.severity) for s in all_scores.get("llm", [])}
    result["sample_pages"] = len(sample_ids)
    for mode, sc in all_scores.items():
        subset = [s for s in sc if (s.doc_id, s.severity) in sample_ids] if sample_ids else sc
        result["modes"][mode] = {"all_pages": evaluate.summarise(sc), "llm_sample_pages": evaluate.summarise(subset), "per_field": evaluate.per_field(subset),
                                 "tradeoff": evaluate.tradeoff(subset)}
        result["by_severity"][mode] = {str(sv): evaluate.summarise([s for s in sc if s.severity == sv]) for sv in experiment.SEVERITIES}
    result["by_layout"] = {lay: evaluate.summarise([s for s in all_scores["rules"] if s.layout == lay]) for lay in synth.LAYOUTS}
    if hand:
        hcorpus = synth.make_corpus(hand, seed + 99)
        hs = experiment.score_all(hcorpus, (0,), cache, "rules", None, None, hand=True)
        result["handwriting_style"] = {"docs": hand, "rules": evaluate.summarise(hs), "per_field": evaluate.per_field(hs)}
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    report.figures(result, all_scores, out)
    console.print(f"wrote {out / 'summary.json'}")


@app.command()
def tables(results: Annotated[Path, typer.Option()] = Path("results")) -> None:
    """Print the Markdown tables for the README from results/summary.json."""
    console.print(report.markdown(json.loads((results / "summary.json").read_text(encoding="utf-8"))))


@app.command()
def review(
    pages: Annotated[int, typer.Option(help="How many pages to review")] = 3,
    seed: int = 0,
    severity: int = 2,
    cache: Annotated[Path, typer.Option()] = Path("results/cache"),
) -> None:
    """Interactive human verification on synthetic pages: the pipeline shows every field it will not trust, you confirm or correct it,
    and the log records how many corrections were needed."""
    corpus = synth.make_corpus(pages, seed)
    log = []
    for d in corpus:
        rows, _ = experiment.ocr_cached(cache, d, severity)
        res = pipeline.run(rows, "rules")
        console.print(f"\n[bold]{d.doc_id}[/bold]  ({sum(f.route == 'review' for f in res.fields.values())} fields need review)")
        for name, fr in res.fields.items():
            if fr.route == "review":
                shown = "" if fr.value is None else str(fr.value)
                answer = typer.prompt(f"  {name} = {shown!r} (conf {fr.conf:.2f}, {fr.suspicion} violated constraints); Enter to accept, or type the right value", default=shown, show_default=False)
                log.append({"doc": d.doc_id, "field": name, "proposed": shown, "final": answer, "changed": answer != shown, "truth": str(d.truth.get(name))})
    changed = sum(x["changed"] for x in log)
    console.print(f"\nreviewed {len(log)} fields, corrected {changed}")


if __name__ == "__main__":
    app()
