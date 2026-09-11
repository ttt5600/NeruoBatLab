#!/usr/bin/env python3
"""Validate the findings store and render its two outputs.

  index.md    the full, human-readable table of everything, grouped by status
  CONTEXT.md  a compact digest meant to be pasted or loaded at the start of a session,
              so a cold agent starts with what is already known and -- just as important --
              what has already been refuted, instead of rediscovering it

Run after editing anything under findings/. Validation is strict on purpose: a findings store
that tolerates a claim without a method or a comparison without an interval is just a pile of
numbers, and this project has already been burned by exactly that.
"""
import json, sys, datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

ROOT = Path(__file__).parent
SCHEMA = json.loads((ROOT / "schema.json").read_text())
STATUSES = ["confirmed", "refuted", "superseded", "open", "retracted"]


def load():
    out, errs = [], []
    for p in sorted((ROOT / "findings").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
        except Exception as e:
            errs.append(f"{p.name}: unparseable ({e})"); continue
        for k in SCHEMA["required"]:
            if not d.get(k):
                errs.append(f"{p.name}: missing required field '{k}'")
        if d.get("status") not in STATUSES:
            errs.append(f"{p.name}: status '{d.get('status')}' not in {STATUSES}")
        if d.get("id") and not str(d["id"]).isdigit():
            errs.append(f"{p.name}: id must be digits")
        d["_file"] = p.name
        out.append(d)
    ids = [d.get("id") for d in out]
    for i in set(ids):
        if ids.count(i) > 1:
            errs.append(f"duplicate id {i}")
    known = set(ids)
    for d in out:
        for ref in (d.get("supersedes") or []) + ([d["superseded_by"]] if d.get("superseded_by") else []):
            if ref not in known:
                errs.append(f"{d['_file']}: references unknown id {ref}")
    return out, errs


def fmt(d, compact=False):
    L = [f"### [{d['id']}] {d['title']}  ·  **{d['status'].upper()}**",
         "", d["claim"], ""]
    if d.get("evidence"):
        L += [f"**Evidence.** {d['evidence']}", ""]
    if not compact:
        L += [f"**Method.** {d['method']}", ""]
        if d.get("caveats"):
            L += [f"**Caveats.** {d['caveats']}", ""]
        L += [f"**Provenance.** {d['provenance']}", ""]
        if d.get("supersedes"):
            L += [f"Supersedes: {', '.join(d['supersedes'])}", ""]
    elif d.get("caveats"):
        L += [f"**Caveats.** {d['caveats']}", ""]
    if d.get("superseded_by"):
        L += [f"> Superseded by [{d['superseded_by']}].", ""]
    return "\n".join(L)


def main():
    ds, errs = load()
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print("  " + e)
        sys.exit(1)
    print(f"{len(ds)} findings validated")
    today = datetime.date.today().isoformat()

    # index.md -- everything
    parts = [f"# Findings index\n\nGenerated {today} by `build.py`. Do not edit by hand; "
             f"edit `findings/*.yaml`.\n\n{len(ds)} findings.\n"]
    for st in STATUSES:
        sel = [d for d in ds if d["status"] == st]
        if not sel:
            continue
        parts.append(f"\n## {st.upper()} ({len(sel)})\n")
        for d in sorted(sel, key=lambda x: x["id"]):
            parts.append(fmt(d))
    (ROOT / "index.md").write_text("\n".join(parts))

    # CONTEXT.md -- the cold-start digest
    live = [d for d in ds if d["status"] in ("confirmed", "open")]
    dead = [d for d in ds if d["status"] in ("refuted", "retracted")]
    C = [f"# Project context: zebra finch HuBERT\n",
         f"Generated {today}. {len(ds)} findings: {len(live)} live, {len(dead)} closed.\n",
         "Read the REFUTED section. Several of these ideas look obviously correct and are not; "
         "they have each cost a day.\n",
         "\n## What is established\n"]
    for d in sorted([d for d in ds if d["status"] == "confirmed"], key=lambda x: x["id"]):
        C.append(fmt(d, compact=True))
    op = [d for d in ds if d["status"] == "open"]
    if op:
        C.append("\n## Open questions\n")
        for d in sorted(op, key=lambda x: x["id"]):
            C.append(fmt(d, compact=True))
    if dead:
        C.append("\n## Refuted — do not retry without new evidence\n")
        for d in sorted(dead, key=lambda x: x["id"]):
            C.append(f"- **[{d['id']}] {d['title']}** — {d['claim']}"
                     + (f" _{d.get('evidence','')}_" if d.get("evidence") else ""))
    (ROOT / "CONTEXT.md").write_text("\n".join(C))
    print(f"wrote index.md ({len(ds)} findings) and CONTEXT.md ({len(live)} live, {len(dead)} closed)")


if __name__ == "__main__":
    main()
