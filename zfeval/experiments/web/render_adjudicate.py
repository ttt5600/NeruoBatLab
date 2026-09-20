#!/usr/bin/env python
"""Inject the clip payload into the adjudication template."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = Path.home() / ".claude/jobs/63c218d9/tmp"
cards = json.loads((TMP / "adjudication_cards.json").read_text())
tpl = (HERE / "adjudicate_template.html").read_text()
marker = "/*__CARDS__*/[]"
if marker not in tpl:
    sys.exit("template marker missing")
out = tpl.replace(marker, json.dumps(cards, separators=(",", ":")))
dest = TMP / "adjudicate.html"
dest.write_text(out)
print(f"{len(cards)} cards -> {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
