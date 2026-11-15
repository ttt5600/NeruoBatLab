"""Sandbox test of the OT1b pick path + adapter (never touches scored logs). notes/ot1b_monotone.md.
1) ot1b() on the 11-15 forced test snapshot with ONE rung's quote edited so the rule fires (offline: book GET stubbed).
2) adapter ot1b_fwd on a temp jsonl: that pick as status bet + a fake settle row -> vault.resolve_outcomes + schema.enforce.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(S / "scripts"), str(S)]
import ot1_forward as o  # noqa: E402
import options_implied_tails as oit  # noqa: E402
from edgelab import schema, vault  # noqa: E402
from edgelab.adapters import ot1b_fwd  # noqa: E402

D = Path(os.path.expanduser("~/.claude/quant-sports/data/kx_forward"))
q = pd.DataFrame([json.loads(l) for l in open(D / "ot1_quotes_ot1btest.jsonl")])
q = q[q.event.str.startswith("KXBTCD")].copy()
T = int(pd.Timestamp(q["T"].iloc[0]).timestamp())
close = int(pd.Timestamp("2026-11-15T22:00:00Z").timestamp())
# rebuild sm from the snapshot is not possible offline (trades not stored) -> fit a tiny synthetic smile around spot
spot = float(q.spot.iloc[0])
sm = (spot, (close + 3600 * 8 - T) / (365 * 86400), np.linspace(-0.08, 0.08, 9), np.full(9, 0.45))
tauK = (close - T) / (365 * 86400)
df = pd.DataFrame({"series": "KXBTCD", "ticker": q.ticker, "event_ticker": q.event, "strike_type": q.strike_type,
                   "floor": q.floor, "cap": q["cap"], "close_ts": close, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
                   "yes_bid_size": q.yes_bid_size, "yes_ask_size": q.yes_ask_size})
df["p_yes"] = [oit.p_yes(r, lambda k: oit.p_above_crypto(k, sm, tauK)) for r in df.to_dict("records")]
for c, v in {"spot": spot, "date": pd.Timestamp("2026-11-15"), "asset_class": "crypto", "T": T, "inputs_as_of": T - 5,
             "quote_ts": T, "mark_yes_bid": np.nan, "mark_yes_ask": np.nan}.items():
    df[c] = v
grid = o.p_above_grid(sm, tauK)
df["_pb"] = [oit.p_yes(r, lambda k: o.p_above_mono(k, grid)) for r in df.to_dict("records")]
cand = df[(df._pb > 0.002) & (df._pb < 0.02) & df.strike_type.isin(["greater"])].sort_values("floor").iloc[0]
i = df.index[df.ticker == cand.ticker][0]
df.loc[i, ["yes_bid", "yes_ask"]] = [0.06, 0.07]          # NO ask .94 -> edge ~ (1-p) - .94 - fee > .03
df = df.drop(columns="_pb")
o.book_depth = lambda t, s, a: {"book_best_ask": a, "depth_0c": 10.0, "depth_1c": 20.0, "depth_2c": 30.0,
                                "book_time": "stub"}
quotes = [{"event": "KXBTCD-26NOV1517", "ticker": t} for t in df.ticker]
pk = o.ot1b(df, sm, tauK, "KXBTCD-26NOV1517", "KXBTCD", "2026-11-15", True, T, T + 1, (T - 5) * 1000, quotes, [])
print("ot1b picks:", [(p["ticker"], p["status"], p["side"], p["ask"], round(p["edge"], 4), round(p["p_yes"], 4),
                       round(p["p_yes_ot1"], 4)) for p in pk])
print("quotes with p_yes_b:", sum("p_yes_b" in x for x in quotes), "of", len(quotes))
assert pk, "no OT1b pick produced"
p = dict(pk[0], status="bet")
tmp = Path(tempfile.mkdtemp()) / "ot1.jsonl"
with open(tmp, "w") as f:
    f.write(json.dumps(p, default=o._js) + "\n")
    f.write(json.dumps({"kind": "pick", **{k: v for k, v in p.items() if k != "kind"}, "ticker": "OTHER-OT1-ONLY",
                        "key": "OTHER-OT1-ONLY"}, default=o._js) + "\n")   # OT1 row must be ignored by ot1b adapter
    f.write(json.dumps({"kind": "settle", "date": "2026-11-15", "event": p["event"],
                        "results": {p["ticker"]: "no", "OTHER-OT1-ONLY": "yes"}}) + "\n")
ot1b_fwd.qs_or_repo = lambda rel: tmp
frame, _ = ot1b_fwd.load({})
print("adapter frame rows", len(frame), frame.attrs["loader_report"])
bets = ot1b_fwd.rows(frame, {})
hidden = frame[["bet_id", "result"]].copy()
hidden.attrs = {"split": "forward", "outcome": ot1b_fwd.OUTCOME}
b = vault.resolve_outcomes("kalshi_opt_tails_mono", bets, hidden, frame)
r = schema.enforce(b)
print(r[["bet_id", "side", "price", "decided_at", "inputs_as_of", "price_time", "event_start", "outcome"]].to_string())
