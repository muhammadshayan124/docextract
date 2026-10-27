"""Markdown tables and figures from results/summary.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from docextract import evaluate  # noqa: E402

MODE_LABEL = {"rules": "Rules", "llm": "LLM only (Qwen3-1.7B)", "hybrid": "Hybrid (rules + LLM repair)"}


def _fmt(s: dict) -> str:
    lo, hi = s["field_ci95"]
    return f"{s['field_accuracy']:.3f} ({lo:.3f}-{hi:.3f})"


def markdown(res: dict) -> str:
    out = ["**Same pages for every mode** (the LLM sample), field accuracy over 13 scalar fields per page, with 95% Wilson interval:", "",
           "| mode | pages | field accuracy | every field right (page) | headline F1 |", "|---|---|---|---|---|"]
    for mode, m in res["modes"].items():
        s = m["llm_sample_pages"]
        out.append(f"| {MODE_LABEL[mode]} | {s['docs']} | {_fmt(s)} | {s['doc_all_correct']:.2f} ({s['doc_ci95'][0]:.2f}-{s['doc_ci95'][1]:.2f}) | {s['headline_f1']:.2f} |")
    out += ["", "**Rules mode on all pages, by scan severity** (0 clean ... 3 bad fax/phone photo):", "",
            "| severity | pages | field accuracy | every field right | OCR character error rate |", "|---|---|---|---|---|"]
    for sv, s in res["by_severity"]["rules"].items():
        cer = "" if s["mean_cer"] is None else f"{s['mean_cer']:.3f}"
        out.append(f"| {sv} | {s['docs']} | {_fmt(s)} | {s['doc_all_correct']:.2f} | {cer} |")
    out += ["", "**By layout** (rules, all severities):", "", "| layout | pages | field accuracy |", "|---|---|---|"]
    out += [f"| {k} | {v['docs']} | {_fmt(v)} |" for k, v in res["by_layout"].items()]
    if "handwriting_style" in res:
        h = res["handwriting_style"]["rules"]
        out += ["", f"**Handwriting-style stress test** ({h['docs']} pages in a script font with per-glyph jitter; *not* real handwriting): field accuracy {_fmt(h)}."]
    for mode in ("rules", "hybrid"):
        if mode not in res["modes"]:
            continue
        curve = res["modes"][mode]["tradeoff"]
        out += ["", f"**Review burden vs residual error, {MODE_LABEL[mode]}** (a person corrects every field routed to review; the final error is what remains among auto-accepted fields):", "",
                "| threshold | auto-accepted | needs review | accuracy of auto-accepted | final error rate |", "|---|---|---|---|---|"]
        for c in curve:
            acc = "-" if c["accuracy_of_auto"] != c["accuracy_of_auto"] else f"{c['accuracy_of_auto']:.4f}"
            out.append(f"| {c['threshold']:.2f} | {c['auto_coverage']:.1%} | {c['review_burden']:.1%} | {acc} | {c['final_error_rate']:.2%} |")
        for target in (0.01, 0.005):
            p = evaluate.threshold_for(curve, target)
            out.append("")
            out.append(f"To keep the final error at or below {target:.1%}: " + (f"threshold {p['threshold']:.2f} needs a person to review {p['review_burden']:.1%} of fields."
                                                                                if p else "no threshold in the sweep reaches that (everything would need review)."))
    return "\n".join(out)


def figures(res: dict, all_scores: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, m in res["modes"].items():
        pts = [(c["review_burden"], c["final_error_rate"]) for c in m["tradeoff"]]
        ax.plot(*zip(*pts), marker="o", label=MODE_LABEL[mode])
    ax.set_xlabel("share of fields a person must review")
    ax.set_ylabel("error rate of the final records")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("Cost of reaching a target error rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for mode, by in res["by_severity"].items():
        ax.plot([int(k) for k in by], [v["field_accuracy"] for v in by.values()], marker="o", label=MODE_LABEL[mode])
    ax.set_xlabel("scan severity (0 clean ... 3 bad fax/phone photo)")
    ax.set_ylabel("field accuracy")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_severity.png", dpi=130)
    plt.close(fig)
