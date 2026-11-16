"""One-off patch: add CB2/CB3 to scripts/crypto_barrier_forward.py (run from sports/). Asserts each anchor is unique."""
from pathlib import Path

p = Path("scripts/crypto_barrier_forward.py")
s = p.read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep('''--force-now writes to *_manual.jsonl only. No orders, no keys, no portfolio endpoints.
"""''', '''--force-now writes to *_manual.jsonl only. No orders, no keys, no portfolio endpoints.

CB2 / CB3 (added 2026-11-16, agent crypto_barrier_v2, prereg notes/crypto_barrier_v2.md; models
scripts/crypto_barrier_v2_model.py): same snapshot. Deribit option book summaries (BTC, ETH) are read right after DVOL,
before any Kalshi GET. Quote rows gain ps (smile barrier P), sigma_b, fires_cb2, fires_cb3; CB2/CB3 picks (field `rule`)
and a v2 `day` row go to crypto_barrier_v2.jsonl (CB1's crypto_barrier.jsonl keeps exactly its old pick semantics).
Any CB2 failure is logged (cb2_error) and never blocks CB1 / CB3.
"""''')
rep('''import crypto_barrier_model as M  # noqa: E402
''', '''import crypto_barrier_model as M  # noqa: E402
import crypto_barrier_v2_model as V  # noqa: E402
''')
rep('''def decide_day(now: float, uday: str) -> dict:
    t_start = time.time()
    inputs = {}
    for cur in ("BTC", "ETH"):
        sig, sig_end = dvol_last(cur, now)
        inputs[cur] = {"sigma": sig, "dvol_end": sig_end}
''', '''def deribit_smile(cur: str) -> tuple[dict, float]:
    t = time.time()
    j = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency", {"currency": cur, "kind": "option"})
    return V.smile_from_summary(j["result"], t), t


def decide_day(now: float, uday: str) -> dict:
    t_start = time.time()
    inputs = {}
    for cur in ("BTC", "ETH"):
        sig, sig_end = dvol_last(cur, now)
        inputs[cur] = {"sigma": sig, "dvol_end": sig_end}
    smiles, cb2_err = {}, {}
    for cur in ("BTC", "ETH"):
        try:
            smiles[cur], t_sm = deribit_smile(cur)
            inputs[cur].update(smile_t=t_sm, smile_expiries=len(smiles[cur]))
        except Exception as e:  # noqa: BLE001
            cb2_err[cur] = repr(e)[:200]
''')
rep('''    prior = {r["ticker"] for r in read(path("crypto_barrier")) if r.get("kind") == "pick"}
    quotes, picks, nser = [], [], 0
''', '''    prior = {r["ticker"] for r in read(path("crypto_barrier")) if r.get("kind") == "pick"}
    prior_v2 = {(r["rule"], r["ticker"]) for r in read(path("crypto_barrier_v2")) if r.get("kind") == "pick"}
    quotes, picks, picks_v2, nser = [], [], [], 0
''')
rep('''                row["fires"] = bool(dec)
                quotes.append(row)
''', '''                row["fires"] = bool(dec)
                dec2 = dec3 = None
                if cur in smiles:
                    try:
                        row["ps"], row["sigma_b"] = V.p_ms(kind, S, B, smiles[cur], tau)
                        dec2 = M.decide(row["yes_bid"], row["yes_ask"], row["ps"], p2, tau * 365) if trial else None
                    except Exception as e:  # noqa: BLE001
                        row["cb2_error"] = repr(e)[:200]
                if trial:
                    dec3 = V.decide_cb3(kind, row["yes_bid"], row["yes_ask"], p1, tau * 365)
                row["fires_cb2"], row["fires_cb3"] = bool(dec2), bool(dec3)
                quotes.append(row)
                for rule, dv in (("CB2", dec2), ("CB3", dec3)):
                    if dv and (rule, m["ticker"]) not in prior_v2:
                        late = (k_start - (pd.Timestamp(uday, tz="UTC").timestamp() + DECIDE_H * 3600)) / 60
                        picks_v2.append({**{k: row.get(k) for k in ("series", "event", "ticker", "asset", "barrier", "strike",
                                                                    "spot", "sigma", "sigma_b", "tau_d", "d", "p0", "p1",
                                                                    "p2", "ps", "yes_bid", "yes_ask", "quote_time",
                                                                    "decided_at", "window_end")},
                                         "kind": "pick", "rule": rule, "date": uday, "side": dv["side"],
                                         "ask": dv["ask"], "edge": dv["edge"], "p_cons": dv.get("p_cons"),
                                         "fee": M.fee(dv["ask"]), "inputs_as_of": iso(inputs_as_of),
                                         "cluster": "crypto_month|" + pd.Timestamp(wend - 1, unit="s", tz="UTC")
                                         .tz_convert("America/New_York").strftime("%Y-%m"),
                                         "status": "bet" if 0 <= late < BET_WINDOW_MIN else "off_time", "late_min": late})
''')
rep('''                       + [v["hist_end"] for v in inputs.values()])''',
    '''                       + [v["hist_end"] for v in inputs.values()] + [v.get("smile_t", 0) for v in inputs.values()])''')
rep('''    append(path("crypto_barrier"), [day])
    return day
''', '''    append(path("crypto_barrier"), [day])
    append(path("crypto_barrier_v2"), picks_v2 + [{
        "kind": "day", "uday": uday, "status": "decided", "finished": iso(time.time()), "inputs_as_of": iso(inputs_as_of),
        "cb2_error": cb2_err or None, "smile_expiries": {c: x.get("smile_expiries") for c, x in inputs.items()},
        "scored_cb2": sum(1 for r in quotes if r.get("ps") is not None),
        "fires_cb2": sum(1 for r in quotes if r.get("fires_cb2")), "fires_cb3": sum(1 for r in quotes if r.get("fires_cb3")),
        "picks": {k: sum(1 for p in picks_v2 if p["rule"] == k) for k in ("CB2", "CB3")},
        "bets": {k: sum(1 for p in picks_v2 if p["rule"] == k and p["status"] == "bet") for k in ("CB2", "CB3")}}])
    day["v2"] = {"fires_cb2": sum(1 for r in quotes if r.get("fires_cb2")),
                 "fires_cb3": sum(1 for r in quotes if r.get("fires_cb3")), "picks_v2": len(picks_v2),
                 "cb2_error": cb2_err or None}
    return day
''')
rep('''def cmd_settle():
    rs = read(path("crypto_barrier"))
    done = {r["ticker"] for r in rs if r.get("kind") == "settle"}
    todo = [r for r in rs if r.get("kind") == "pick" and r["ticker"] not in done]
    out = []
    for r in todo:
        m = (kget(f"/markets/{r['ticker']}") or {}).get("market", {})
        if m.get("result") in ("yes", "no") and m.get("status") in ("settled", "finalized", "determined"):
            out.append({"kind": "settle", "ticker": r["ticker"], "result": m["result"], "status_mkt": m.get("status"),
                        "close_time": m.get("close_time"), "settled_seen": iso(time.time())})
    append(path("crypto_barrier"), out)
    if out:
        print(f"settled {len(out)} of {len(todo)} open picks")
''', '''def cmd_settle():
    for name in ("crypto_barrier", "crypto_barrier_v2"):
        rs = read(path(name))
        done = {r["ticker"] for r in rs if r.get("kind") == "settle"}
        todo = sorted({r["ticker"] for r in rs if r.get("kind") == "pick" and r["ticker"] not in done})
        out = []
        for t in todo:
            m = (kget(f"/markets/{t}") or {}).get("market", {})
            if m.get("result") in ("yes", "no") and m.get("status") in ("settled", "finalized", "determined"):
                out.append({"kind": "settle", "ticker": t, "result": m["result"], "status_mkt": m.get("status"),
                            "close_time": m.get("close_time"), "settled_seen": iso(time.time())})
        append(path(name), out)
        if out:
            print(f"{name}: settled {len(out)} of {len(todo)} open picked markets")
''')
p.write_text(s)
print("patched")
