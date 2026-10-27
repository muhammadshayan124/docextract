from __future__ import annotations

import numpy as np
import pytest

from docextract import evaluate, parse, pipeline, render, synth, validate
from docextract.llm import LLMRecord
from docextract.ocr import Box, Row, group_rows

# -- synthetic documents --------------------------------------------------------------------------


def rows_of(doc: synth.Doc, conf: float = 0.97) -> list[Row]:
    return [Row(line, conf, float(i)) for i, line in enumerate(doc.lines) if line.strip()]


def test_synthetic_records_are_internally_consistent():
    for seed in range(60):
        d = synth.make_doc(seed)
        assert validate.violations({**d.truth, "headlines": [n["headline"] for n in d.truth["news"]]}) == [], seed


def test_corpus_is_deterministic_and_layouts_are_balanced():
    a, b = synth.make_corpus(8, seed=3), synth.make_corpus(8, seed=3)
    assert [x.lines for x in a] == [x.lines for x in b]
    assert [x.layout for x in a] == list(synth.LAYOUTS) * 2
    assert synth.make_corpus(4, seed=1)[0].truth != synth.make_corpus(4, seed=2)[0].truth


def test_rendering_is_deterministic_and_degradation_changes_the_page():
    lines = synth.make_doc(1).lines
    a, b = render.scan(lines, 2, seed=5), render.scan(lines, 2, seed=5)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and a.size == render.PAGE
    clean = np.asarray(render.scan(lines, 0, seed=5), dtype=float)
    for sev in (1, 2, 3):
        assert np.abs(np.asarray(render.scan(lines, sev, seed=5), dtype=float) - clean).mean() > 1.0
    assert np.asarray(render.scan(lines, 0, seed=1, handwriting_style=True)).min() < 255


# -- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", synth.LAYOUTS)
def test_rules_recover_every_field_from_a_clean_page_in_every_layout(layout):
    for seed in range(15):
        d = synth.make_doc(seed, layout)
        p = parse.parse_rows(rows_of(d))
        for f in parse.NUM_FIELDS:
            assert evaluate.field_correct(f, p.fields.get(f), d.truth), (layout, seed, f, p.fields.get(f), d.truth[f])
        assert p.currency == d.truth["currency"] and p.report_date == d.truth["report_date"] and p.news_count == d.truth["news_count"]
        assert evaluate.headline_scores(p.headlines, [n["headline"] for n in d.truth["news"]])[0] == d.truth["news_count"]


def test_number_repair_only_touches_mostly_numeric_tokens():
    assert parse.clean_number("O.6977l") == pytest.approx(0.69771)
    assert parse.clean_number("-0,0I274") == pytest.approx(-0.01274)
    assert parse.clean_number("156.9.37") == pytest.approx(156.937)  # stray dot
    assert parse.clean_number("Bid") is None and parse.clean_number("ask:") is None and parse.clean_number("") is None


def test_label_matching_is_fuzzy_but_not_reckless():
    assert parse.match_field("Previous close") == "previous_close" and parse.match_field("Prev. close") == "previous_close"
    assert parse.match_field("52-week high") == "high_52week" and parse.match_field("High") == "high"
    assert parse.match_field("Change %") == "change_percent" and parse.match_field("Change") == "change"
    assert parse.match_field("8id") == "bid"  # one OCR error
    assert parse.match_field("Summary text here") is None


def test_split_row_handles_every_layout():
    assert parse.split_row("Bid: 0.69777") == ("Bid", "0.69777")
    assert parse.split_row("Previous close ........ 0.6981") == ("Previous close", "0.6981")
    assert parse.split_row('  "high_52week": 0.7277,') == ("high_52week", "0.7277")
    assert parse.split_row("Change %  -0.04727") == ("Change %", "-0.04727")
    assert parse.split_row("1. Market data") is None  # no trailing number


def test_row_grouping_joins_label_and_value_boxes():
    boxes = [Box("Bid", 90, 100, 140, 130, 0.99), Box("0.69777", 600, 102, 760, 131, 0.95), Box("Ask", 90, 150, 140, 180, 0.98), Box("0.69781", 600, 151, 760, 181, 0.4)]
    rows = group_rows(boxes)
    assert [r.text for r in rows] == ["Bid 0.69777", "Ask 0.69781"] and rows[1].conf == 0.4 and rows[0].conf == 0.95


# -- validation and routing -----------------------------------------------------------------------


def corrupt(doc: synth.Doc, field: str, new: float):
    vals = dict(doc.truth)
    vals[field] = new
    return vals


def test_a_misread_digit_is_caught_and_the_right_field_is_implicated():
    d = synth.make_doc(11, "kv")
    bad = corrupt(d, "bid", d.truth["bid"] + 0.5 * (10 ** -CURR_DP(d)) * 40)  # bid far above ask
    vs = validate.violations({**bad, "headlines": [n["headline"] for n in d.truth["news"]]})
    sus = validate.suspicion(vs)
    assert vs and sus["bid"] >= 1
    assert sus.get("bid", 0) == max(sus.values())  # the corrupted field is the most implicated


def CURR_DP(d):  # noqa: N802
    return synth.CURRENCIES[d.truth["currency"]][2]


def test_consistent_records_get_full_confidence_and_inconsistent_ones_lose_it():
    assert validate.field_confidence(0.9, 0, None) == 0.9
    assert validate.field_confidence(0.9, 2, None) == pytest.approx(0.225)
    assert validate.field_confidence(0.9, 0, False) == pytest.approx(0.36)
    assert validate.field_confidence(None, 0, None) == 0.0


def test_pipeline_clean_page_is_fully_auto_and_correct():
    d = synth.make_doc(5, "dots")
    res = pipeline.run(rows_of(d), "rules")
    assert all(f.route == "auto" for f in res.fields.values()) and res.violations == []
    assert evaluate.score_doc(d.doc_id, d.layout, 0, res, d.truth).all_correct


def test_low_ocr_confidence_routes_to_review_even_when_the_value_is_right():
    d = synth.make_doc(5, "kv")
    res = pipeline.run(rows_of(d, conf=0.3), "rules")
    assert all(f.route == "review" for f in res.fields.values())
    assert evaluate.score_doc(d.doc_id, d.layout, 0, res, d.truth).all_correct  # right, but not trusted


def test_hybrid_lets_the_llm_repair_a_misread_value_only_when_it_reduces_violations():
    d = synth.make_doc(21, "kv")
    rows = rows_of(d)
    dp = CURR_DP(d)
    bad_bid = f"{d.truth['bid'] + 40 * 10 ** -dp:.{dp}f}"
    rows = [Row(f"Bid: {bad_bid}", 0.97, r.y) if r.text.startswith("Bid") else r for r in rows]
    rules_only = pipeline.run(rows, "rules")
    assert not evaluate.field_correct("bid", rules_only.fields["bid"].value, d.truth) and rules_only.fields["bid"].route == "review"
    good = LLMRecord(bid=f"{d.truth['bid']:.{dp}f}", ask=f"{d.truth['ask']:.{dp}f}")
    fixed = pipeline.run(rows, "hybrid", good)
    assert evaluate.field_correct("bid", fixed.fields["bid"].value, d.truth) and fixed.fields["bid"].source == "llm" and fixed.notes
    wrong_llm = LLMRecord(bid="9.99999")  # makes things worse: must be ignored
    kept = pipeline.run(rows, "hybrid", wrong_llm)
    assert kept.fields["bid"].source == "rules"


def test_llm_values_are_parsed_from_strings_and_failures_stay_failures():
    vals, raw = pipeline.llm_to_values(LLMRecord(bid="0.6977l", ask="n/a", currency=" aud ", news_count=2))
    assert vals["bid"] == pytest.approx(0.69771) and "ask" not in vals and vals["currency"] == "AUD" and vals["news_count"] == 2


# -- evaluation -----------------------------------------------------------------------------------


def test_field_correctness_rules():
    t = {"currency": "AUD", "bid": 0.69777, "change_percent": -0.04727, "news_count": 2, "report_date": "2026-10-12"}
    assert evaluate.field_correct("bid", 0.69777, t) and not evaluate.field_correct("bid", 0.69778, t)
    assert evaluate.field_correct("change_percent", -0.04727, t) and not evaluate.field_correct("change_percent", None, t)
    assert evaluate.field_correct("news_count", 2, t) and evaluate.field_correct("currency", "AUD", t) and not evaluate.field_correct("report_date", "2026-10-13", t)


def test_headline_matching_is_fuzzy_one_to_one():
    want = ["AUD steadies as traders wait for central bank minutes", "AUD slips on softer risk appetite"]
    assert evaluate.headline_scores(want, want) == (2, 2, 2)
    assert evaluate.headline_scores(["AUD steadies as traders wait for central bank minutes"] * 2, want)[0] == 1  # no double counting
    assert evaluate.headline_scores(["AUD steadies as traders wait for central bank minute5"], want)[0] == 1
    assert evaluate.headline_scores([], want) == (0, 0, 2)


def test_cer_and_wilson():
    assert evaluate.cer("abc def", ["abc", "def"]) == 0.0
    assert evaluate.cer("abX def", ["abc", "def"]) == pytest.approx(1 / 6)
    lo, hi = evaluate.wilson(90, 100)
    assert lo < 0.9 < hi and evaluate.wilson(0, 0) == (0.0, 1.0)


def make_scores(n_docs=40, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_docs):
        correct = {f: bool(rng.random() < 0.9) for f in pipeline.SCALAR}
        # confidence is informative: wrong fields tend to have low confidence
        conf = {f: float(np.clip(rng.normal(0.9 if c else 0.4, 0.1), 0, 1)) for f, c in correct.items()}
        scores.append(evaluate.DocScore(f"d{i}", "kv", 0, correct, conf, dict.fromkeys(correct, "auto"), (2, 2, 2)))
    return scores


def test_tradeoff_curve_is_monotone_and_reaches_the_target():
    curve = evaluate.tradeoff(make_scores())
    cov = [c["auto_coverage"] for c in curve]
    assert cov == sorted(cov, reverse=True) and curve[0]["auto_coverage"] == 1.0
    err = [c["final_error_rate"] for c in curve]
    assert err == sorted(err, reverse=True)  # a stricter threshold can only remove auto-accepted mistakes
    pick = evaluate.threshold_for(curve, 0.02)
    assert pick is not None and pick["final_error_rate"] <= 0.02 and pick["review_burden"] < 1.0
    assert evaluate.threshold_for(curve, -1) is None


def test_summary_numbers():
    s = evaluate.summarise(make_scores())
    assert 0.85 < s["field_accuracy"] < 0.95 and s["field_ci95"][0] < s["field_accuracy"] < s["field_ci95"][1]
    assert 0 <= s["doc_all_correct"] <= 1 and s["headline_f1"] == 1.0


# -- real OCR smoke test (one clean page; skipped if the OCR engine is unavailable) -----------------


def test_real_ocr_reads_a_clean_page():
    pytest.importorskip("rapidocr_onnxruntime")
    from docextract import ocr

    d = synth.make_doc(2, "kv")
    rows = ocr.ocr_rows(render.scan(d.lines, 0, seed=2))
    res = pipeline.run(rows, "rules")
    s = evaluate.score_doc(d.doc_id, d.layout, 0, res, d.truth)
    assert sum(s.correct.values()) >= 11 and s.correct["currency"]  # at least 11 of 13 on a clean page


# -- robustness to what real OCR does to a page (regressions found on the development set) -------------


def test_headlines_are_split_by_their_closing_time_not_by_dashes_or_line_breaks():
    rows = [Row(t, 0.9, float(i)) for i, t in enumerate([
        "2. Relevant news (2)",
        "- Market wrap: AUD unchanged after mixed signals (Currency Desk,", "2026-07-2117:59UTC)",  # wrapped, spaces lost
        "AUD edges higher in thin Asian trade (Currency Desk, 2026-07-21", "01:55 UTC)",  # the leading dash was dropped by OCR
        "3. Summary", "The pair traded in a narrow range.",
    ])]
    p = parse.parse_rows(rows)
    assert p.headlines == ["Market wrap: AUD unchanged after mixed signals", "AUD edges higher in thin Asian trade"] and p.news_count == 2


def test_section_header_survives_lost_spaces_and_fullwidth_parentheses():
    p = parse.parse_rows([Row("2.Relevantnews（2)", 0.9, 0.0)])
    assert p.news_count == 2


def test_currency_is_normalised_and_guessed_only_when_unambiguous():
    assert parse.read_currency("AuD") == "AUD" and parse.read_currency("NoK") == "NOK" and parse.read_currency("jpy") == "JPY"
    assert parse.read_currency("AUO") == "AUD"  # one edit from exactly one known code
    assert parse.read_currency("XYZ") is None
    p = parse.parse_rows([Row("NzD - Market Snapshot Report", 0.9, 0.0)])
    assert p.currency == "NZD"


def test_cer_ignores_how_many_leader_dots_there_were():
    assert evaluate.cer("Bid.........0.5", ["Bid ............ 0.5"]) == 0.0
    assert evaluate.cer("Bid 0.6", ["Bid 0.5"]) > 0
