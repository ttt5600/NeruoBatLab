"""Plant each failure check_finding.py exists to catch, and check it is caught.

    python3 -m pytest harness/test_check_finding.py -q
"""
import importlib.util
import json
import sys
from pathlib import Path

import yaml

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_finding", H / "check_finding.py")
cf = importlib.util.module_from_spec(spec)
sys.modules["check_finding"] = cf
spec.loader.exec_module(cf)

SOURCE = {"m | b": {"a_minus_b": {"diff": 0.014654, "lo": -0.004618, "hi": 0.035907}},
          "acc": {"per_layer": [0.80, 0.8599062]}}


def finding(tmp_path, claim, evidence, numbers, provenance="pairs.json"):
    (tmp_path / "pairs.json").write_text(json.dumps(SOURCE))
    p = tmp_path / "999-test.yaml"
    p.write_text(yaml.safe_dump({"id": "999", "claim": claim, "evidence": evidence,
                                 "provenance": provenance, "numbers": numbers}))
    cf.ANALYSIS_DIRS = [tmp_path]
    return cf.check(p)


def entry(value, key):
    return {"value": value, "file": "pairs.json", "key": key}


def test_keyed_numbers_pass(tmp_path):
    r = finding(tmp_path, "m beats b by +0.0147 [-0.0046, +0.0359] at 0.8599.", "",
                [entry("+0.0147", ["m | b", "a_minus_b", "diff"]),
                 entry("-0.0046", ["m | b", "a_minus_b", "lo"]),
                 entry("+0.0359", ["m | b", "a_minus_b", "hi"]),
                 entry("0.8599", ["acc", "per_layer", 1])])
    assert r["pass"], r


def test_wrong_value_fails(tmp_path):
    r = finding(tmp_path, "gap +0.0157.", "", [entry("+0.0157", ["m | b", "a_minus_b", "diff"])])
    assert not r["pass"] and "does not round" in r["entries"][0]["error"]


def test_sign_flip_fails(tmp_path):
    r = finding(tmp_path, "gap -0.0147.", "", [entry("-0.0147", ["m | b", "a_minus_b", "diff"])])
    assert not r["pass"]


def test_bad_key_fails(tmp_path):
    r = finding(tmp_path, "gap +0.0147.", "", [entry("+0.0147", ["m | b", "diff"])])
    assert not r["pass"] and "does not resolve" in r["entries"][0]["error"]


def test_unbacked_claim_number_fails(tmp_path):
    r = finding(tmp_path, "gap +0.0147, acc 0.8599.", "",
                [entry("+0.0147", ["m | b", "a_minus_b", "diff"])])
    assert not r["pass"] and r["unbacked_claim"] == ["0.8599"]


def test_number_parser_edges():
    assert cf.NUM.findall("acc 0.8599. Then +0.01, −0.020; sklearn 1.7.2 and L3") == \
        ["0.8599", "+0.01", "−0.020"]


def test_literal_must_appear_in_text(tmp_path):
    # an entry for a number the prose no longer says (the text was edited, the block was not)
    r = finding(tmp_path, "gap +0.015.", "", [entry("+0.0147", ["m | b", "a_minus_b", "diff"])])
    assert not r["pass"]


def test_evidence_residue_is_reported_not_failed(tmp_path):
    r = finding(tmp_path, "no numbers here.", "derived 0.0003 and 0.8599", [])
    assert r["pass"]
    assert {x["value"] for x in r["residue"]} == {"0.0003", "0.8599"}
    assert {x["value"]: x["occurs_in_cited_files"] for x in r["residue"]}["0.8599"] is True


def test_template_citation_is_flagged(tmp_path):
    r = finding(tmp_path, "x.", "", [], provenance="pairs.json, <tag>_calltype.json")
    assert any("template" in u for u in r["unresolved_citations"])
