"""P2 TAILPOOL (winners_transfer, pre-registered in notes/winners_transfer.md before running).
Read-only public Kalshi GETs. For each series: settled markets closing in the last D days; trades in the decision window
(final 24 h if market life <= 3 d, else final 7 d). TAIL pool = sum(count * p_loser) where p_loser <= .15;
TAIL risk = sum(count * (1 - p_w)) where the <=.15 side WON (p_w = price of the winning cheap side);
FULL pool = sum(count * p_loser) all trades. Secondary (added before running): EARLY = trades in the first half of the window.
Usage: python tailpool.py OUT.json SERIES:DAYS ..."""
import json, sys, time, requests
from datetime import datetime, timezone
API = "https://api.elections.kalshi.com/trade-api/v2"
GAP = 0.12; last = [0.0]; nreq = [0]
def get(path, params):
    for a in range(8):
        w = GAP - (time.time() - last[0])
        if w > 0: time.sleep(w)
        last[0] = time.time(); nreq[0] += 1
        try:
            r = requests.get(API + path, params=params, timeout=30, headers={"User-Agent": "research-script/0.1"})
            if r.status_code == 429: time.sleep(min(5 * 2 ** a, 60)); continue
            if r.status_code >= 500: time.sleep(2 * (a + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            if a == 7: raise
            time.sleep(1.5 * (a + 1))
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def series_pool(series, days, now):
    t0 = now - days * 86400
    mk, cur = [], None
    while True:
        p = {"series_ticker": series, "status": "settled", "min_close_ts": int(t0), "limit": 1000}
        if cur: p["cursor"] = cur
        d = get("/markets", p); mk += d.get("markets", []); cur = d.get("cursor")
        if not cur or not d.get("markets"): break
    out = dict(series=series, days=days, markets=len(mk), traded=0, events=len({m["event_ticker"] for m in mk}),
               tail=0.0, tail_risk=0.0, full=0.0, early_tail=0.0, early_full=0.0, contracts=0.0, n_trades=0,
               tail_contracts=0.0)
    for m in mk:
        if float(m.get("volume_fp") or 0) <= 0 or m.get("result") not in ("yes", "no"): continue
        out["traded"] += 1
        c, o = ts(m["close_time"]), ts(m["open_time"])
        win = 86400 if (c - o) <= 3 * 86400 else 7 * 86400
        ws = c - win; mid = c - win / 2
        cur = None
        while True:
            p = {"ticker": m["ticker"], "min_ts": int(ws), "max_ts": int(c), "limit": 1000}
            if cur: p["cursor"] = cur
            d = get("/markets/trades", p); tr = d.get("trades", [])
            for t in tr:
                n = float(t["count_fp"]); y = float(t["yes_price_dollars"]); no = float(t["no_price_dollars"])
                pl = y if m["result"] == "no" else no      # price paid by the side that lost
                pw = no if m["result"] == "no" else y      # price paid by the side that won
                early = ts(t["created_time"]) < mid
                out["contracts"] += n; out["n_trades"] += 1; out["full"] += n * pl
                if early: out["early_full"] += n * pl
                if pl <= 0.15:
                    out["tail"] += n * pl; out["tail_contracts"] += n
                    if early: out["early_tail"] += n * pl
                if pw <= 0.15: out["tail_risk"] += n * (1 - pw)
            cur = d.get("cursor")
            if not cur or not tr: break
    f = 30.0 / days
    for k in ("tail", "tail_risk", "full", "early_tail", "early_full", "contracts", "tail_contracts"):
        out[k + "_per30d"] = round(out[k] * f, 2)
    out["events_per30d"] = round(out["events"] * f, 2)
    return out
if __name__ == "__main__":
    outp = sys.argv[1]; now = time.time(); res = {}
    try: res = json.load(open(outp))
    except Exception: pass
    for arg in sys.argv[2:]:
        s, d = arg.split(":")
        if s in res: continue
        t = time.time(); r = series_pool(s, int(d), now); r["secs"] = round(time.time() - t, 1); r["reqs_total"] = nreq[0]
        res[s] = r; json.dump(res, open(outp, "w"), indent=1)
        print(s, r["markets"], r["traded"], "tail/30d", r["tail_per30d"], "risk", r["tail_risk_per30d"], "full", r["full_per30d"], r["secs"], "s", flush=True)
