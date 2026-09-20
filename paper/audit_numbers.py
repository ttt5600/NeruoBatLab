#!/usr/bin/env python
"""Refuse any number in the manuscript that is not traceable to results.json.

Usage:  audit_numbers.py DRAFT.md [DRAFT2.md ...]  [--whitelist paper/whitelist.txt]

Every decimal in the prose is checked against the registry. A match means some registry entry
rounds to that literal at the precision written. Anything else is reported with its line, and the
exit code is nonzero, so this can sit in a pre-commit hook or CI and stop a stale number from
reaching a reviewer.

The point is not that unmatched numbers are wrong -- sample sizes, years and section numbers are
fine -- it is that each one has to be either registry-backed or explicitly whitelisted, so no number
enters the paper by being typed once and never checked again.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# a number, allowing thousands separators; the lookarounds keep us out of dates (2026-09-13),
# hyphenated identifiers (111021-000), ranges (0-50 ms) and finding ids (019)
NUM = re.compile(r"(?<![\w.,\-])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])"
                 r"|(?<![\w.,\-])[-+]?\d+(?:\.\d+)?(?![\w,]|\-\d)")
# markdown/latex scaffolding that legitimately contains bare integers
SKIP_LINE = re.compile(r"^\s*(\|[-: |]+\||#{1,6}\s|```|\[\^|<!--)")


def registry_values():
    r = json.loads((HERE / "results.json").read_text())
    vals, disp = [], {}
    for k, e in r["entries"].items():
        v = e["value"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append((k, float(v)))
        disp.setdefault(str(e["display"]).strip(), []).append(k)
    return vals, disp, r


def matches(lit: str, vals, disp):
    """A literal is backed if it equals a registry display string, or rounds to a registry value."""
    s = lit.lstrip("+")
    if lit in disp:
        return disp[lit]
    if s in disp:
        return disp[s]
    try:
        x = float(lit)
    except ValueError:
        return None
    dec = len(lit.split(".")[1]) if "." in lit else 0
    hits = []
    for k, v in vals:
        if round(v, dec) == round(x, dec):
            hits.append(k)
        # also allow a percentage rendering of a fraction, and ms/s rescalings
        elif dec <= 2 and round(v * 100, dec) == round(x, dec):
            hits.append(k + " (as %)")
    return hits or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--whitelist", default=str(HERE / "whitelist.txt"))
    ap.add_argument("--quiet", action="store_true", help="only print the unmatched")
    a = ap.parse_args()

    vals, disp, reg = registry_values()
    wl = set()
    wlp = Path(a.whitelist)
    if wlp.exists():
        for ln in wlp.read_text().splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                wl.add(ln)

    n_ok = n_wl = 0
    bad = []
    for f in a.files:
        p = Path(f)
        if not p.exists():
            print(f"[skip] {f} not found")
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            # typographic minus / en-dash read as a sign, or a negative number silently loses it
            line = line.replace("\u2212", "-").replace("\u2013", "-")
            if SKIP_LINE.match(line):
                continue
            for m in NUM.finditer(line):
                lit = m.group(0).replace(",", "")
                if lit in wl:
                    n_wl += 1
                    continue
                hit = matches(lit, vals, disp)
                if hit:
                    n_ok += 1
                else:
                    bad.append((str(p), i, lit, line.strip()[:100]))

    print(f"registry: {reg['n_entries']} entries, generated {reg['generated']}")
    print(f"checked {len(a.files)} file(s): {n_ok} registry-backed, {n_wl} whitelisted, "
          f"{len(bad)} UNMATCHED")
    if bad:
        print("\nunmatched numbers -- each must be registry-backed or added to whitelist.txt:")
        last = None
        for f, i, lit, ctx in bad:
            if f != last:
                print(f"\n  {f}")
                last = f
            print(f"    line {i:4d}  {lit:>10s}   {ctx}")
        return 1
    print("\nall numbers traceable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
