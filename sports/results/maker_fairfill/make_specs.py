"""Write the 12 maker_fairfill edgelab specs (pre-registered in notes/maker_fairfill.md) and register them."""
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
fam = {"nba": ("nba_totals", "nba_totals_open"), "cfb": ("cfb_lines", "kalshi_cfb_lines"),
       "nfl": ("nfl_lines", "kalshi_nfl_lines"), "cbb": ("cbb_totals", "kalshi_cbb_totals")}
pre = ("maker_fairfill (notes/maker_fairfill.md, pre-registered 2026-11-15 before any trade tape). Same bets as the "
       "primary K50 taker rule (CBB: cbb_t1). At T post a bid on the model side of the bought rung at "
       "floor_cent(side_mid - x), capped at ask-1c; rest to start-5min; FILL iff a tape print strictly through b; "
       "cost b + maker fee 0.0175p(1-p) ceil/100ct; graded as the taker bet.")
out = ROOT / "edgelab" / "specs"
for lg, (f, ds) in fam.items():
    for x in (0, 1, 2):
        s = dict(id=f"mf_{lg}_x{x}", family=f, dataset=ds, hypothesis=pre + f" League {lg}, x = {x}c.",
                 decision_time="same T as the K50/T1 taker bet; rest until start - 5 min",
                 split=f"vault:{ds}",
                 legacy="new 2026-11-15 (agent maker_fairfill). PASS dev only if filled ROI p < family bar, "
                        "n_filled>=15, CLV>0; GO fwd if filled ROI>0, n_filled>=15; KILL if ROI CI upper<0 or "
                        "(AS gap CI upper<0 and ROI<0).",
                 rule="edgelab.rules.forward:unwired", price_model="maker_tape", min_n=100,
                 kill=dict(kind="clv_negative_significant", after_n=100, p=0.05), fee=dict(rounding="exact"))
        p = out / f"mf_{lg}_x{x}.yaml"
        p.write_text(yaml.safe_dump(s, sort_keys=False))
        r = subprocess.run([sys.executable, "-m", "edgelab", "register", str(p)], cwd=ROOT, capture_output=True, text=True)
        print(p.name, (r.stdout + r.stderr).strip().splitlines()[-1:])
