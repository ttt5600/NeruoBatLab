"""pool_census stage 2: candle-based loser-paid pools per market (approximation of P2 tailpool.py; notes/pool_census.md).
Read-only batch GET /markets/candlesticks through scripts/kx_throttle (max 100 markets, max 10,000 candles per call).
Decision window per market = P2's: final 24 h if (close - open) <= 3 d else final 7 d (clipped at open).
Resolution (segments):
  * markets living <= 3 h (crypto 15-min / hourly): 1-min candles over their whole life;
  * SPORTS series living > 3 h: hourly candles up to S = the hour boundary at or before close - 4 h, then 1-min candles
    from S to close (in-game action: hourly means hid the whole tail on KXNBAGAME, validate_extra.log: exact $150,564 vs
    hourly $0);
  * everything else: hourly (validated vs exact tape on 19 P2 series, validate_p2.json).
Per candle h: L_h = loser price = mean_dollars (result no) or 1 - mean_dollars (result yes) [volume-weighted mean];
  tail += vol_h * L_h if L_h <= .15 ; mirror += vol_h * L_h if L_h >= .85 ; full += vol_h * L_h ; contracts += vol_h
  'rng' variant: the share of the candle's loser-price range [low, high] below .15 (above .85) carries the volume.
Far-rung quote state over the window (lead 11-21): per candle close, far = cheap side buyable at <= .15
(yes_ask <= .15 or yes_bid >= .85); far_c = far candles, far2_c = far AND two-sided; far_spr_hist = spread (cents) counts.
Decision-time quote = first candle of the window: a0 / b0 (yes ask / bid open), far0 (cheap side buyable at <= .15),
two_sided0 (bid > 0 and ask < 1), spread0 / mid0 when two-sided.
Usage (library): run(markets, out_jsonl) ; resumable (skips tickers already in out_jsonl)."""
import json, math, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget

MAXC, MAXM = 10000, 100
SPORTS = {s["ticker"] for s in json.load(open(HERE / "series_all.json")) if s.get("category") == "Sports"}

def ts(s): return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def segments(m):
    """[(lo, hi, per, last)] ; a candle belongs to a segment if lo < end <= hi (+ per if last)."""
    c, o = ts(m["close_time"]), ts(m["open_time"])
    life = c - o
    if life <= 3 * 3600:
        return [((o // 60) * 60, c, 60, True)]
    w = 86400 if life <= 3 * 86400 else 7 * 86400
    ws = max(c - w, (o // 3600) * 3600)
    if m["event_ticker"].split("-")[0] in SPORTS:
        S = max(ws, ((c - 4 * 3600) // 3600) * 3600)
        segs = [(ws, S, 3600, False)] if S > ws else []
        return segs + [(S, c, 60, True)]
    return [(ws, c, 3600, True)]

def plan(m):
    """window start / end / finest resolution (used by tape_phi and validation)."""
    sg = segments(m)
    return sg[0][0], sg[-1][1], min(x[2] for x in sg)

def score(m, seg_candles):
    res = m["result"]
    sg = segments(m)
    r = dict(ticker=m["ticker"], event=m["event_ticker"], series=m["event_ticker"].split("-")[0], result=res,
             close=m["close_time"], ws=sg[0][0], we=sg[-1][1], per=min(x[2] for x in sg), tail=0.0, mirror=0.0, full=0.0,
             contracts=0.0, tail_ct=0.0, mirror_ct=0.0, tail_rng=0.0, mirror_rng=0.0, n_candles=0, spread0=None, mid0=None,
             a0=None, b0=None, t0=None, far0=False, two_sided0=False, far_c=0, far2_c=0, far_spr_hist={})
    first = None
    for (lo, hi, per, last), candles in zip(sg, seg_candles):
        for c in sorted(candles, key=lambda c: c["end_period_ts"]):
            e = c["end_period_ts"]
            if e <= lo or e > hi + (per if last else 0): continue
            r["n_candles"] += 1
            if first is None: first = c
            ac, bc = f((c.get("yes_ask") or {}).get("close_dollars")), f((c.get("yes_bid") or {}).get("close_dollars"))
            if (ac is not None and 0 < ac <= 0.15) or (bc is not None and bc >= 0.85):   # rung is FAR at this candle's close
                r["far_c"] += 1
                if ac is not None and bc is not None and 0 < bc and ac < 1 and ac >= bc:
                    r["far2_c"] += 1
                    k = str(int(round((ac - bc) * 100)))
                    r["far_spr_hist"][k] = r["far_spr_hist"].get(k, 0) + 1
            v = f(c.get("volume_fp")) or 0.0
            if v <= 0: continue
            p = c.get("price") or {}
            mean = f(p.get("mean_dollars"))
            if mean is None: continue
            lo_p, hi_p = f(p.get("low_dollars")), f(p.get("high_dollars"))
            if lo_p is None or hi_p is None: lo_p = hi_p = mean
            if res == "no": L, Llo, Lhi = mean, lo_p, hi_p
            else: L, Llo, Lhi = 1 - mean, 1 - hi_p, 1 - lo_p
            r["full"] += v * L; r["contracts"] += v
            if L <= 0.15: r["tail"] += v * L; r["tail_ct"] += v
            if L >= 0.85: r["mirror"] += v * L; r["mirror_ct"] += v
            if Lhi <= Llo + 1e-9:
                if Llo <= 0.15: r["tail_rng"] += v * Llo
                if Llo >= 0.85: r["mirror_rng"] += v * Llo
            else:
                s = min(1.0, max(0.0, (0.15 - Llo) / (Lhi - Llo)))
                if s > 0: r["tail_rng"] += v * s * (Llo + min(Lhi, 0.15)) / 2
                s2 = min(1.0, max(0.0, (Lhi - 0.85) / (Lhi - Llo)))
                if s2 > 0: r["mirror_rng"] += v * s2 * (max(Llo, 0.85) + Lhi) / 2
    if first is not None:
        a, b = f((first.get("yes_ask") or {}).get("open_dollars")), f((first.get("yes_bid") or {}).get("open_dollars"))
        r["a0"], r["b0"], r["t0"] = a, b, first["end_period_ts"]
        if a is not None and b is not None and 0 < b and a < 1 and a >= b:
            r["spread0"] = round(a - b, 4); r["mid0"] = round((a + b) / 2, 4)
        r["far0"] = bool((a is not None and 0 < a <= 0.15) or (b is not None and b >= 0.85))
        r["two_sided0"] = bool(a is not None and b is not None and 0 < b and a < 1)
    for k in ("tail", "mirror", "full", "contracts", "tail_ct", "mirror_ct", "tail_rng", "mirror_rng"):
        r[k] = round(r[k], 4)
    return r

def batches(ms):
    """Items = (market, segment index). Group by resolution, sort by segment start, grow a union window while the
    requested candles (periods x markets) <= MAXC and markets <= MAXM."""
    by_per = defaultdict(list)
    for m in ms:
        for i, (lo, hi, per, last) in enumerate(segments(m)):
            by_per[per].append((lo, hi, i, m))
    out = []
    for per, lst in by_per.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        cur, lo, hi = [], None, None
        for a, b, i, m in lst:
            nlo = a if lo is None else min(lo, a); nhi = b if hi is None else max(hi, b)
            nper = (nhi - nlo) // per + 2
            if cur and (len(cur) + 1 > MAXM or nper * (len(cur) + 1) > MAXC):
                out.append((per, lo, hi, cur)); cur, lo, hi = [], None, None
                nlo, nhi = a, b
            cur.append((i, m)); lo, hi = nlo, nhi
        if cur: out.append((per, lo, hi, cur))
    return out

def run(markets, outp, log_every=50, chunk=400):
    done = set()
    if Path(outp).exists():
        for line in open(outp):
            try: done.add(json.loads(line)["ticker"])
            except Exception: pass
    todo = [m for m in markets if m["ticker"] not in done and m.get("result") in ("yes", "no") and (f(m.get("volume_fp")) or 0) > 0]
    todo.sort(key=lambda m: m["close_time"])
    ncall = 0; t0 = time.time()
    with open(outp, "a") as fo:
        for k in range(0, len(todo), chunk):
            part = todo[k:k + chunk]
            got = defaultdict(dict)    # ticker -> seg index -> candles
            for per, lo, hi, items in batches(part):
                tks = sorted({m["ticker"] for _, m in items})
                d = kget("/markets/candlesticks", {"market_tickers": ",".join(tks), "start_ts": lo, "end_ts": hi,
                                                   "period_interval": per // 60})
                ncall += 1
                resp = {}
                if d is not None:
                    for x in d.get("markets", []):
                        resp[x.get("market_ticker")] = x.get("candlesticks", [])
                for i, m in items:
                    if m["ticker"] in resp: got[m["ticker"]][i] = resp[m["ticker"]]
                if ncall % log_every == 0:
                    print(ncall, "calls", round(time.time() - t0), "s", flush=True)
            for m in part:
                sg = segments(m)
                g = got.get(m["ticker"], {})
                r = score(m, [g.get(i, []) for i in range(len(sg))])
                r["missing"] = len(g) < len(sg)
                r["volume_total"] = f(m.get("volume_fp"))
                fo.write(json.dumps(r) + "\n")
            fo.flush()
    return ncall
