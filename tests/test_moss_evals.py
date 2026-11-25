"""Offline unit tests for the moss-evals harness (no network, no credentials)."""
import json
import math

from moss_evals import calibrate, dataset as ds, metrics, report


def test_metrics_hand_computed():
    ranked, rel = ["a", "b", "c", "d"], {"c": 1, "x": 1}
    assert metrics.hit_at_k(ranked, rel, 2) == 0.0
    assert metrics.hit_at_k(ranked, rel, 3) == 1.0
    assert metrics.recall_at_k(ranked, rel, 4) == 0.5
    assert math.isclose(metrics.mrr_at_k(ranked, rel, 10), 1 / 3)
    # one relevant at rank 3, ideal has two relevant at ranks 1-2
    dcg = 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(metrics.ndcg_at_k(ranked, rel, 10), dcg / idcg)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0, 1] * 50
    a, b = metrics.bootstrap_ci(vals), metrics.bootstrap_ci(vals)
    assert a == b and a["lo"] <= a["mean"] <= a["hi"] and math.isclose(a["mean"], 0.5)


def test_paired_bootstrap_detects_real_improvement():
    base = [0.0] * 40 + [1.0] * 10
    cand = [1.0] * 40 + [1.0] * 10
    pb = metrics.paired_bootstrap(base, cand)
    assert pb["delta"] > 0.7 and pb["p_value"] < 0.01


def test_normalize_dedupes_case_whitespace_and_trailing_punctuation():
    # regression: French "valides ?" vs "VALIDES " produced two cases before the fix
    assert ds.normalize_query("Combien ... valides ?") == ds.normalize_query("COMBIEN ... VALIDES ")
    assert ds.normalize_query("ＡＰＩ  limit？") == "api limit"


def test_detect_script_and_tags():
    assert ds.detect_script("गिफ्ट वाउचर कितने महीनों") == "hi"
    assert ds.detect_script("ഓക്ലൻഡ് സിബിഡി") == "ml"
    assert ds.detect_script("フォークリフト") == "ja"
    assert "stratum:multilingual" in ds.tag_case("ഓക്ലൻഡ് സിബിഡി")
    assert "stratum:adversarial" in ds.tag_case("Is SKU MNG-4402 cold stored?")


def test_build_regression_dataset_dedupes_and_orders_by_frequency():
    ev = [{"query": "Refund window?", "expected_ids": ["r1"], "source": "s1"},
          {"query": "refund   window", "expected_ids": ["r1", "r2"], "source": "s2"},
          {"query": "Other", "expected_ids": ["o1"]},
          {"query": "no label"}]
    cases = ds.build_regression_dataset(ev)
    assert len(cases) == 2
    assert cases[0]["occurrences"] == 2 and cases[0]["qrels"] == {"r1": 1, "r2": 1} and cases[0]["sources"] == ["s1", "s2"]


def test_gate_min_threshold_and_paired_regression():
    qrels = {f"q{i}": {"t": 1} for i in range(40)}
    good = {"ranked": {q: ["t"] for q in qrels}}
    bad = {"ranked": {q: (["x"] if i < 20 else ["t"]) for i, q in enumerate(qrels)}}
    base_tbl = report.per_case_table(good, qrels, 5)
    cur_tbl = report.per_case_table(bad, qrels, 5)
    gates = report.gate(cur_tbl, {"hit@5": {"min": 0.8, "max_drop": 0.02}}, base_tbl)
    assert not gates[0]["passed"]
    assert len(report.newly_failing(cur_tbl, base_tbl, "hit@5")) == 20
    assert report.gate(base_tbl, {"hit@5": {"min": 0.8}})[0]["passed"]


def test_junit_is_wellformed_xml():
    import xml.etree.ElementTree as ET
    gates = [{"gate": "hit@5", "passed": False, "detail": "mean=0.5 <below> & 'x'"}]
    root = ET.fromstring(report.to_junit("moss_evals", gates, ["reg-1"]))
    assert root.find(".//failure") is not None


def test_calibration_kappa():
    assert calibrate.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    cases = {"a": {"facts": ["NZ$79"]}, "b": {"facts": ["12 minutes"]}}
    judged = {"a": {"answer": "It costs nz$79 per year", "correct": True}, "b": {"answer": "about ten", "correct": True}}
    out = calibrate.calibrate(cases, judged)
    assert out["confusion"] == {"both_true": 1, "judge_only": 1, "det_only": 0, "both_false": 0}


def test_facts_present_alternatives_and_degrees():
    assert calibrate.facts_present("between 2°C and 8°C", ["2c|2", "8c|8"])
    assert calibrate.facts_present("1,500 ms", ["1,500|1500"])
    assert not calibrate.facts_present("24 hours", ["24", "422"])
