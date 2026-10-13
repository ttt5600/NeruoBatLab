#!/usr/bin/env python3
"""Check that a finding's numbers come from the JSON files it cites. Run before the critic.

    python3 harness/check_finding.py 073            # by id (or a path to the yaml)
    python3 harness/check_finding.py 073 074 --json

A finding backs its numbers with a `numbers:` block, one entry per quoted literal:

    numbers:
    - value: '+0.0147'                      # exactly as written in claim/evidence
      file: calltype_pairs.json             # basename, looked up in the analysis dirs
      key: [hf-whisper-small-enc | aves-base-bio, a_minus_b, diff]
      # optional: scale: 100  (the text shows a percent)

Each entry is looked up exactly: the file must exist, the key path must resolve, the value there
must round to the literal at the precision it was written (sign included), and the literal must
appear in the claim or evidence. That is the strong check.

Every decimal in the CLAIM must be backed this way, or the finding fails. Decimals in the evidence
that have no entry are listed as residue for the critic to verify by hand. For each one the script
also says whether the literal occurs anywhere in the cited files, and prints how often a DECOY
number (the same literal shifted 5-50 steps in its last digit) also occurs anywhere. On real
findings that decoy rate was 0.84-0.90 (2026-10-13): the cited files hold ~25,000 numbers, so
"occurs somewhere" is nearly worthless as evidence, and only the keyed entries count as checked.

Exit 0: every entry resolves and every claim number is backed. Exit 1: otherwise. Exit 2: usage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "knowledge" / "findings"
DEFAULT_DIRS = [Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis",
                ROOT / "savio_artifacts/analysis"]
ANALYSIS_DIRS = [Path(p) for p in os.environ["FINDING_ANALYSIS_DIRS"].split(os.pathsep)] \
    if os.environ.get("FINDING_ANALYSIS_DIRS") else DEFAULT_DIRS
# decimals with 2+ places; the lookbehind keeps out version strings and ids (sklearn 1.7.2, L3)
NUM = re.compile(r"(?<![\w.])[+\-−]?\d*\.\d{2,}(?!\d|\.\d)")  # a sentence-final "0.86." still counts


def find_yaml(arg: str) -> Path:
    p = Path(arg)
    if p.suffix == ".yaml" and p.exists():
        return p
    hits = sorted(FINDINGS.glob(f"{arg.zfill(3)}-*.yaml"))
    if len(hits) != 1:
        sys.exit(f"finding {arg!r}: {len(hits)} matches in {FINDINGS}")
    return hits[0]


def locate(name: str) -> list[Path]:
    base = Path(name).name
    return [Path(h) for d in ANALYSIS_DIRS for h in sorted(glob.glob(str(d / base)))]


def precision(lit: str) -> int:
    return len(lit.split(".")[1]) if "." in lit else 0


def as_float(lit: str) -> float:
    return float(lit.replace("−", "-"))


def rounds_to(x: float, lit: str) -> bool:
    return abs(x - as_float(lit)) <= 0.5 * 10 ** -precision(lit) + 1e-12


def resolve(doc, key):
    cur = doc
    for k in key:
        if isinstance(cur, list):
            cur = cur[int(k)]
        elif isinstance(cur, dict):
            if k not in cur and str(k) in cur:
                k = str(k)
            cur = cur[k]
        else:
            raise KeyError(f"{k!r}: parent is a {type(cur).__name__}, not a dict or list")
    return cur


def all_numbers(doc, out):
    if isinstance(doc, bool):
        return
    if isinstance(doc, (int, float)):
        out.append(float(doc))
    elif isinstance(doc, dict):
        for v in doc.values():
            all_numbers(v, out)
    elif isinstance(doc, list):
        for v in doc:
            all_numbers(v, out)


def check(path: Path) -> dict:
    d = yaml.safe_load(path.read_text())
    claim, evidence = str(d.get("claim", "")), str(d.get("evidence", ""))
    text = claim + "\n" + evidence
    rep = {"finding": path.name, "entries": [], "unbacked_claim": [], "residue": [],
           "cited_files": [], "unresolved_citations": [], "decoy_rate": None}

    backed = set()
    cache: dict[Path, object] = {}
    for e in d.get("numbers") or []:
        lit, fname, key = str(e.get("value", "")), e.get("file", ""), list(e.get("key") or [])
        r = {"value": lit, "file": fname, "key": key, "ok": False}
        files = locate(fname) if fname else []
        if not lit or not fname or not key:
            r["error"] = "entry needs value, file and key"
        elif len(files) != 1:
            r["error"] = f"{len(files)} files named {Path(fname).name} in {[str(x) for x in ANALYSIS_DIRS]}"
        else:
            try:
                doc = cache.setdefault(files[0], json.loads(files[0].read_text()))
                got = resolve(doc, key)
                if isinstance(got, bool) or not isinstance(got, (int, float)):
                    raise TypeError(f"key resolves to {type(got).__name__}, not a number")
                got = float(got) * float(e.get("scale", 1))
                r["source_value"] = got
                if not rounds_to(got, lit):
                    r["error"] = f"source has {got:.6g}, which does not round to {lit}"
                elif lit not in text:
                    r["error"] = "literal does not appear in the claim or evidence"
                else:
                    r["ok"] = True
                    backed.add(lit)
            except (KeyError, IndexError, ValueError, TypeError) as ex:
                r["error"] = f"key path does not resolve: {ex}"
        rep["entries"].append(r)

    # citations named in provenance (and by entries), to scan for residue hints
    names = set(re.findall(r"[\w\-*.<>]+\.json", str(d.get("provenance", ""))))
    names |= {e.get("file", "") for e in d.get("numbers") or [] if e.get("file")}
    pool: list[float] = []
    for n in sorted(names):
        if "<" in n:
            rep["unresolved_citations"].append(n + " (template, not a file)")
            continue
        hits = locate(n)
        if not hits:
            rep["unresolved_citations"].append(n)
        for h in hits:
            rep["cited_files"].append(str(h))
            try:
                all_numbers(json.loads(h.read_text()), pool)
            except (OSError, ValueError):
                rep["unresolved_citations"].append(f"{h} (unreadable)")
    pool_abs = np.abs(np.array(pool)) if pool else np.array([np.nan])

    def anywhere(lit: str) -> bool:
        return bool((np.abs(pool_abs - abs(as_float(lit))) <= 0.5 * 10 ** -precision(lit) + 1e-12).any())

    rep["unbacked_claim"] = [lit for lit in NUM.findall(claim) if lit not in backed]
    seen = set()
    for lit in NUM.findall(evidence):
        if lit in backed or lit in seen:
            continue
        seen.add(lit)
        i = evidence.find(lit)
        rep["residue"].append({"value": lit, "occurs_in_cited_files": anywhere(lit),
                               "context": evidence[max(0, i - 60): i + len(lit) + 10].replace("\n", " ")})

    lits = NUM.findall(text)
    if lits and pool:
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(500):
            lit = lits[rng.integers(len(lits))]
            p = precision(lit)
            decoy = abs(as_float(lit)) + rng.choice([-1, 1]) * rng.integers(5, 51) * 10 ** -p
            hits += anywhere(f"{decoy:.{p}f}")
        rep["decoy_rate"] = hits / 500

    rep["pass"] = all(r["ok"] for r in rep["entries"]) and not rep["unbacked_claim"]
    return rep


def show(rep: dict) -> None:
    ok = [r for r in rep["entries"] if r["ok"]]
    bad = [r for r in rep["entries"] if not r["ok"]]
    print(f"== {rep['finding']}: {'PASS' if rep['pass'] else 'FAIL'}")
    print(f"  keyed numbers: {len(ok)} verified, {len(bad)} failed")
    for r in bad:
        print(f"    FAIL {r['value']}  {r['file']} {r['key']}: {r['error']}")
    if rep["unbacked_claim"]:
        print(f"  claim numbers with no keyed entry (each needs one): {rep['unbacked_claim']}")
    if rep["unresolved_citations"]:
        print(f"  provenance names not found: {rep['unresolved_citations']}")
    if rep["residue"]:
        dr = rep["decoy_rate"]
        print(f"  evidence residue for the critic: {len(rep['residue'])} numbers not keyed. "
              f"'occurs somewhere' is weak: a decoy occurs {dr:.0%} of the time"
              if dr is not None else f"  evidence residue: {len(rep['residue'])} (no cited files to scan)")
        for r in rep["residue"]:
            mark = "somewhere" if r["occurs_in_cited_files"] else "NOWHERE  "
            print(f"    {mark} {r['value']:>9}  ...{r['context']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("findings", nargs="+", help="ids (073) or yaml paths")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    reps = [check(find_yaml(f)) for f in a.findings]
    if a.json:
        print(json.dumps(reps, indent=1, default=str))
    else:
        for r in reps:
            show(r)
    return 0 if all(r["pass"] for r in reps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
