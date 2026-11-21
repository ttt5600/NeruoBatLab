"""pool_census: TAKER-REACHABLE share of the loser-paid pools (lead request 2026-11-21, after gas_daily_v2 found KXAAAGASD's
pool collected by the quoting market maker). Exact trade tape (GET /markets/trades, `taker_side`) on a sample of each
series' measured markets, inside the market's decision window (candle_pool.plan).
A losing buyer's money is reachable by a TAKER only when the loser was the resting (maker) side, i.e. the trade's
taker_side == the WINNING side (the taker sold the loser side into a resting bid / bought the winner at the ask).
phi_tail = sum(count*p_l | p_l <= .15, taker on winner) / sum(count*p_l | p_l <= .15); phi_mirror likewise for p_l >= .85.
Sample: markets in descending candle TAIL $ (then MIRROR $), at most PPM pages per market (newest first; truncated
tapes flagged), at most PPS pages per series. Usage: python tape_phi.py MARKETS_JSONL SERIES1,SERIES2,... OUT.json"""
import json, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget

PPM, PPS = 3, 15   # tail: top-3 markets x <= 3 pages; mirror: top-2 markets x <= 3 pages

def tape(tk, ws, we, res, maxp):
    o = collections.Counter(); pages = 0; cur = None; trunc = False
    while True:
        if pages >= maxp:
            trunc = True; break
        p = {"ticker": tk, "min_ts": int(ws), "max_ts": int(we), "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets/trades", p); pages += 1
        if d is None: break
        tr = d.get("trades", [])
        for t in tr:
            n = float(t["count_fp"]); y = float(t["yes_price_dollars"]); no = float(t["no_price_dollars"])
            pl = y if res == "no" else no
            win = "no" if res == "no" else "yes"
            reach = t.get("taker_side") == win
            o["n"] += 1; o["has_taker"] += int("taker_side" in t)
            if pl <= 0.15:
                o["tail"] += n * pl; o["tail_reach"] += n * pl * reach
            if pl >= 0.85:
                o["mirror"] += n * pl; o["mirror_reach"] += n * pl * reach
            o["full"] += n * pl; o["full_reach"] += n * pl * reach
        cur = d.get("cursor")
        if not cur or not tr: break
    return o, pages, trunc

def main():
    mfile, series, outp = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    res = json.load(open(outp)) if Path(outp).exists() else {}
    rows = collections.defaultdict(list)
    for line in open(mfile):
        x = json.loads(line)
        if x["series"] in series: rows[x["series"]].append(x)
    for s in series:
        mirror_only = s in res and res[s].get("phi_mirror") is None and "--mirror" in sys.argv
        if (s in res and not mirror_only) or not rows[s]: continue
        ms = rows[s]
        mir = lambda x: x["mirror"] if x.get("per") == 60 else x["mirror_rng"]
        order = [] if mirror_only else [x for x in sorted(ms, key=lambda x: -x["tail"])[:3] if x["tail"] > 0]
        order += [x for x in sorted(ms, key=lambda x: -mir(x))[:2] if mir(x) > 0]
        seen, agg, used, ntr, cand_tail, cand_mirror = set(), collections.Counter(), 0, 0, 0.0, 0.0
        if mirror_only:
            agg.update({k: v for k, v in res[s].items() if k in ("tail", "tail_reach", "n", "has_taker", "full", "full_reach")})
        for x in order:
            if x["ticker"] in seen or used >= PPS: continue
            seen.add(x["ticker"])
            o, pg, tr = tape(x["ticker"], x["ws"], x["we"], x["result"], min(PPM, PPS - used))
            used += pg; ntr += int(tr); agg.update(o)
            if not tr: cand_tail += x["tail"]; cand_mirror += x["mirror"]
        r = {k: round(v, 2) for k, v in agg.items()}
        r.update(markets=len(seen), pages=used, truncated=ntr,
                 phi_tail=round(agg["tail_reach"] / agg["tail"], 3) if agg["tail"] else None,
                 phi_mirror=round(agg["mirror_reach"] / agg["mirror"], 3) if agg["mirror"] else None,
                 phi_full=round(agg["full_reach"] / agg["full"], 3) if agg["full"] else None)
        res[s] = r; json.dump(res, open(outp, "w"), indent=1)
        print(s, r["markets"], r["pages"], "trunc", ntr, "phi_tail", r["phi_tail"], "phi_mirror", r["phi_mirror"], "n", r.get("n"), flush=True)

if __name__ == "__main__":
    main()
