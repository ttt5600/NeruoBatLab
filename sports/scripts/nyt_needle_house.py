"""NYT House needle vs Kalshi (user 2026-11-03, election night). Tier C judgment, routed through bet.py.

    nyt_needle_house.py [--live] [--once] [--edge 0.25] [--kelly 0.25] [--interval 60]

Each loop: NYT results-house.json -> per-race party win_probability (the needle). Kalshi HOUSE<ST><N> series,
markets -<suffix> D/R. Fires when needle p - ask >= edge on YES or NO side. Size = kelly * full-Kelly fraction
of bankroll (Kalshi cash + stake already placed by this bot), clipped to bet.py's tier-C cap ($12.50). IOC taker
at the observed ask (limit), one fill per market. bet.py enforces KILL file + all loss caps. Default is DRY.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kalshi_exec as K  # noqa: E402
import needle_adj as A  # noqa: E402

SEEN: dict = {}  # race -> (votes_counted, first time seen at that count)
DEM_SHIFT = 0.0
BLEND = 1.0  # weight on the needle; rest on the market mid (user 2026-11-03: $50 on 2c contracts -> be careful)


def blend(p: float, yb, ya, yes: bool) -> float:
    """Shrink the needle toward the market mid: the stale-needle longshots showed the model can be badly wrong."""
    if yb is None or ya is None:
        return p
    mid = (yb + ya) / 2 if yes else 1 - (yb + ya) / 2
    return BLEND * p + (1 - BLEND) * mid  # set from --dem-shift; added to the projected Dem margin (blue-wave tilt, judgment)

NYT = "https://static01.nyt.com/elections-assets/pages/data/2026-11-03/results-house.json"
KB = "https://api.elections.kalshi.com/trade-api/v2"
SER = re.compile(r"^(?:KX)?HOUSE(?:PARTY-)?([A-Z]{2})-?(\d{1,2}|AL)$")
STATE = K.DEPLOY / "data" / "nyt_needle_house_state.json"
LOG = K.DEPLOY / "data" / "nyt_needle_house_log.jsonl"
RACE_CAP_FRAC = 0.10   # user 2026-11-03 upset at $90 bets: halved from 0.20
TOTAL_CAP_FRAC = 0.95  # max total House cost basis, of bankroll
PARTY = {"DEM": "D", "GOP": "R", "REP": "R"}


def get(url: str, tries: int = 5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            raise
    raise RuntimeError(f"429 x{tries}: {url}")


def needles() -> dict:
    """(ST, district) -> {'D': p, 'R': p, 'counted': n, 'updated': iso}"""
    out = {}
    for r in get(NYT)["races"]:
        t = r.get("top_reporting_unit") or {}
        st, seat = t.get("state_postal"), r.get("seat_number")
        if not st:
            continue
        won = (r.get("outcome") or {}).get("won") or []
        if r.get("race_call_status") == "called" and len(won) == 1:  # AP call: Kalshi often lags it by minutes
            cm = r.get("candidate_metadata") or {}
            wp_ = ((cm.get(won[0]) or {}).get("party") or {}).get("nyt_id")
            wp_ = PARTY.get(wp_ or next(((c.get("nyt_model_estimates") or {}).get("party_caucus") for c in
                                          t.get("candidates") or [] if c.get("nyt_id") == won[0]), None))
            if wp_:
                d = "AL" if seat in (None, "", "0", "AL") or st in AT_LARGE else str(int(seat))
                lo = "R" if wp_ == "D" else "D"
                out[(st, d)] = {wp_: 0.99, lo: 0.01, "counted": 10**9, "call": "called:" + ",".join(
                    (r.get("outcome") or {}).get("called_by") or []), "updated": r.get("updated_at", "").replace("Z", "+00:00")}
                continue
        if (r.get("model_settings") or {}).get("model_status") != "active":
            continue
        est = t.get("nyt_model_estimates") or {}
        ps = {}
        for c in t.get("candidates") or []:
            e = c.get("nyt_model_estimates") or {}
            pa = PARTY.get(e.get("party_caucus"))
            if pa and e.get("win_probability") is not None:
                ps[pa] = ps.get(pa, 0.0) + float(e["win_probability"])
        if not ps:
            continue
        mod = A.race_model(r)  # tilt only where the rebuild reproduces NYT (FL-13 broke it: 0.006 -> 1.000)
        if DEM_SHIFT and mod and "D" in ps and "R" in ps and abs(A.p_dem(mod, 0) - ps["D"]) < 0.03:
            pd = A.p_dem(mod, DEM_SHIFT)
            ps = {"D": pd, "R": 1 - pd, "nyt_D": ps["D"]}
        d = "AL" if seat in (None, "", "0", "AL") or st in AT_LARGE else str(int(seat))
        out[(st, d)] = dict(ps, counted=est.get("votes_counted") or 0, updated=est.get("nyt_updated_at"),
                            call=r.get("race_call_status"), ap_total=t.get("total_votes") or 0, eevp=t.get("eevp") or 0)
    return out


CNN = "https://politics.api.cnn.io/results/national-races/2026-HG.json"  # 20 s cache; own decision desk
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}


def cnn_calls() -> dict:
    """(ST, district) -> winning party letter for races CNN has called (status W, exactly one winner)."""
    out = {}
    try:
        rows = get(CNN)
    except Exception as e:
        print("cnn fetch error:", e, flush=True)
        return out
    for r in rows:
        if r.get("status") != "W":
            continue
        w = [c for c in r.get("candidates") or [] if c.get("winner")]
        if len(w) != 1 or w[0].get("candidatePartyCode") not in ("D", "R"):
            continue
        st = r.get("stateAbbreviation")
        dn = str(r.get("ecKey", "")).rsplit("-", 1)[-1]
        if not st or not dn.isdigit():
            continue
        out[(st, "AL" if st in AT_LARGE else str(int(dn)))] = w[0]["candidatePartyCode"]
    return out


_series: list[str] = []
PM: dict = {}


def poly_mids() -> dict:
    """((ST, district), party) -> Polymarket mid. Second opinion: buy only where Polymarket also says Kalshi is cheap
    (NYT ran ~20-40 pts R of BOTH exchanges all night and the exchanges were right)."""
    out, off = {}, 0
    while off < 1000:
        ev = get(f"https://gamma-api.polymarket.com/events?tag_slug=house-elections&closed=false&limit=100&offset={off}")
        if not ev:
            break
        for e in ev:
            for m in e.get("markets") or []:
                q = m.get("question") or ""
                mm = re.search(r"win the ([A-Z]{2})-(\d+|AL) House seat", q)
                pa = "R" if "Republican" in q else "D" if "Democratic" in q else None
                bb, ba = f(m.get("bestBid")), f(m.get("bestAsk"))
                if not mm or not pa or m.get("closed") or bb is None or ba is None or ba - bb > 0.10:
                    continue
                st, d = mm.groups()
                out[((st, "AL" if d == "AL" or st in AT_LARGE else str(int(d))), pa)] = (bb + ba) / 2
        off += 100
    return out


def kalshi_markets() -> list[dict]:
    global _series
    if not _series:
        _series = [x["ticker"] for x in get(KB + "/series?category=Elections")["series"] if SER.match(x["ticker"])]
    ms = []
    for s in _series:
        st, d = SER.match(s).groups()
        d = d if d == "AL" else str(int(d))
        for m in get(f"{KB}/markets?series_ticker={s}&status=open&limit=200").get("markets", []):
            suf = m["ticker"].rsplit("-", 1)[-1]
            if suf in ("D", "R"):
                m["_key"], m["_party"] = (st, d), suf
                ms.append(m)
        time.sleep(0.15)
    # KXHOUSERACE-{ST}{NN}-26-{D|R}: ~60 districts with no per-district series (missed until 06:55 UTC)
    for m in get(f"{KB}/markets?series_ticker=KXHOUSERACE&status=open&limit=500").get("markets", []):
        g = re.match(r"KXHOUSERACE-([A-Z]{2})(\d{2}|AL)-26-(D|R)$", m["ticker"])
        if g:
            st, d, pa = g.groups()
            m["_key"], m["_party"] = (st, "AL" if d == "AL" or st in AT_LARGE else str(int(d))), pa
            ms.append(m)
    return ms


def iso(t: str) -> datetime:
    """fromisoformat that takes any fraction length (py3.10 rejects '06:52:35.75+00:00' from NYT/CNN)."""
    t = t.replace("Z", "+00:00")
    m = re.match(r"(.*T\d\d:\d\d:\d\d)(?:\.(\d+))?(.*)$", t)
    if m:
        t = m.group(1) + ("." + (m.group(2) + "000000")[:6] if m.group(2) else "") + m.group(3)
    return datetime.fromisoformat(t)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kelly(p: float, q: float) -> float:
    return max(0.0, (p - q) / (1 - q))


def cash() -> float:
    import bet as B
    c = B.client()
    code, bal = c.request("GET", "/portfolio/balance")
    return (bal.get("balance") or 0) / 100.0 if code == 200 else 0.0


def held(ticker: str) -> bool:
    """IOC can be accepted with zero fill; only a real position closes the race."""
    import bet as B
    code, p = B.client().request("GET", f"/portfolio/positions?ticker={ticker}")
    return code == 200 and any(abs(float(x.get("position_fp") or x.get("position") or 0)) > 0
                               for x in p.get("market_positions", []) if x.get("ticker") == ticker)


def house_positions() -> dict:
    import bet as B
    return {p["ticker"]: (float(p.get("position_fp") or 0), float(p.get("market_exposure_dollars") or 0), p.get("position_fp"))
            for p in B.client().positions() if float(p.get("position_fp") or 0) and "HOUSE" in p["ticker"]}


BUCKET_FRAC, BUCKET_SELL_FRAC = 0.50, 0.60  # same-winner races are correlated (one NYT turnout model): cap the bucket


def book(N: dict, M: list[dict]) -> dict:
    """Mark held House positions at the price we could SELL at now. Returns bank (cash + marked value), per-ticker
    rows and per-direction value (direction = the party our position needs to win)."""
    pos = house_positions()
    rows, dirv = {}, {"D": 0.0, "R": 0.0}
    for m in M:
        if m["ticker"] not in pos:
            continue
        k, basis, kstr = pos[m["ticker"]]
        yb, ya = f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars"))
        bid = yb if k > 0 else (None if ya is None else round(1 - ya, 4))
        d = m["_party"] if k > 0 else ("R" if m["_party"] == "D" else "D")
        val = abs(k) * (bid or 0.0)
        n = N.get(m["_key"])
        p = None if not n or m["_party"] not in n else (n[m["_party"]] if k > 0 else 1 - n[m["_party"]])
        # exits judge on the RAW needle (user 2026-11-03: "DO NOT BE AFRAID TO HOLD", "trust the arbitrage");
        # the blend is buy-side sizing only
        rows[m["ticker"]] = dict(m=m, k=k, kstr=kstr, basis=basis, bid=bid, yb=yb, ya=ya, dir=d, val=val, p=p)
        dirv[d] += val
    # bank = cash + Kalshi's portfolio_value (ALL positions: Senate, governor, seat counts too). Cash + House-only
    # marks read $381 on an ~$620 account at 07:18 UTC and the bucket rule dumped six favorites at the bid.
    try:
        import bet as B_
        _, bal = B_.client().request("GET", "/portfolio/balance")
        bank = float(bal["balance_dollars"]) + float(bal.get("portfolio_value") or 0) / 100
    except Exception:
        bank = cash() + sum(r["val"] for r in rows.values())
    return {"bank": bank, "rows": rows, "dirv": dirv}


def exits(N: dict, M: list[dict], a, now) -> None:
    """Risk-adjusted selling (user 2026-11-03: "risk adjust when considering selling"). Each position is trimmed to its
    quarter-Kelly target AT THE BID; target 0 when bid >= needle - buffer (edge gone) -> sell all; and a same-winner
    bucket above 60% of bank is scaled back to 50%. Buys size at the ask, so buy target < sell target: no churn."""
    import bet as B
    bk = book(N, M)
    bank = bk["bank"]
    for t, r in bk["rows"].items():
        p, bid = r["p"], r["bid"]
        pm = PM.get((r["m"]["_key"], r["dir"]))  # hold while Polymarket still prices it above our bid (CO-3: the needle
        if p is not None and pm is not None:      # sold a market-backed D buy at 75c on a stale-looking 30% needle)
            p = max(p, pm)
        if p is None or bid is None or bid < 0.02:
            continue
        target = 0.0 if bid + a.exit_buffer >= p else min(RACE_CAP_FRAC * bank, a.kelly * kelly(p, bid) * bank)
        dv = bk["dirv"][r["dir"]]
        if dv > BUCKET_SELL_FRAC * bank:
            target = min(target, r["val"] * BUCKET_FRAC * bank / dv)
        excess = r["val"] - target
        if target > 0 and (a.no_trim or excess < max(5.0, 0.3 * r["val"]) or bid < 0.05):  # $46.69 fees by 04:55
            if not (dv > BUCKET_SELL_FRAC * bank):
                continue  # hold: only edge-gone or the correlated-bucket cap sells  # trimming a 1-2c position only pays the spread: hold it
        cnt = r["kstr"].lstrip("-") if target == 0 else str(int(excess / bid))
        if float(cnt) < 1:
            continue
        m, k = r["m"], r["k"]
        body = {"ticker": t, "side": "ask" if k > 0 else "bid", "count": cnt,
                "price": f"{(r['yb'] if k > 0 else r['ya']):.2f}", "time_in_force": "immediate_or_cancel",
                "self_trade_prevention_type": "taker_at_cross", "client_order_id": f"needle-exit-{now.strftime('%H%M%S')}-{t}"[:60]}
        why = "edge gone" if target == 0 else ("bucket" if dv > BUCKET_SELL_FRAC * bank else "over kelly")
        row = dict(ts=now.isoformat(timespec="seconds"), event="exit", ticker=t, held=k, sell=cnt, why=why,
                   needle=round(p, 4), exit_bid=bid, value=round(r["val"], 2), target=round(target, 2), bank=round(bank, 2))
        if a.live:
            code, out = B.client().create_order(body)
            o = (out or {}).get("order", out) or {}
            row.update(http=code, order_id=o.get("order_id"), fill=o.get("fill_count_fp", o.get("fill_count")))
        print(f"  SELL {'LIVE' if a.live else 'DRY'} {t} {'YES' if k > 0 else 'NO'} x{cnt}/{abs(k):g} @ {bid:.2f} [{why}] "
              f"needle {p:.3f} value ${r['val']:.2f} -> target ${target:.2f} -> {row.get('http')} fill {row.get('fill')}", flush=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


def exit_alerts(N: dict, M: list[dict], buf: float) -> None:
    import bet as B
    pos = {p["ticker"]: float(p.get("position_fp") or 0) for p in B.client().positions()
           if float(p.get("position_fp") or 0) and "HOUSE" in p["ticker"]}
    for m in M:
        n, k = N.get(m["_key"]), pos.get(m["ticker"])
        if not k or not n or m["_party"] not in n:
            continue
        p = n[m["_party"]] if k > 0 else 1 - n[m["_party"]]
        yb, ya = f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars"))
        bid = yb if k > 0 else (None if ya is None else round(1 - ya, 4))
        if bid is not None and (bid + buf >= p or p < 0.5 * bid):
            msg = f"{m['ticker']} {'YES' if k > 0 else 'NO'} x{abs(k):g}: needle {p:.3f} vs exit bid {bid:.2f}"
            print("  EXIT? (alert only, not selling)", msg, flush=True)
            K.notify_once(f"needle-exit-{m['ticker']}-{round(bid, 2)}", "House needle: consider selling", msg, {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--edge", type=float, default=0.25)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-age-min", type=float, default=20)
    ap.add_argument("--min-counted", type=int, default=1)
    ap.add_argument("--dem-shift", type=float, default=0.0)
    ap.add_argument("--call-edge", type=float, default=0.05)  # AP-called races: winner at 0.99
    ap.add_argument("--exit-buffer", type=float, default=0.03)
    ap.add_argument("--frozen-min", type=float, default=20)
    ap.add_argument("--max-ap-lag", type=float, default=0.05)
    ap.add_argument("--poly-gate", action="store_true")  # off = trust a FRESH NYT count over the exchanges (CO-3, AK)
    ap.add_argument("--min-price", type=float, default=0.08)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--stop-bank", type=float, default=0.0)
    ap.add_argument("--min-eevp", type=float, default=0.0)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--price-cap-k", type=float, default=0.4)
    ap.add_argument("--edge-d", type=float, default=None)
    ap.add_argument("--edge-r", type=float, default=None)
    a = ap.parse_args()
    global DEM_SHIFT
    DEM_SHIFT = a.dem_shift
    global BLEND
    BLEND = a.blend
    a.edge_d = a.edge if a.edge_d is None else a.edge_d
    a.edge_r = a.edge if a.edge_r is None else a.edge_r
    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "staked": 0.0}
    if LOG.exists():  # seed the frozen-count guard across restarts: earliest time each race sat at its last count
        for line in LOG.read_text().splitlines():
            try:
                r = json.loads(line)
                mm = SER.match(r["ticker"].rsplit("-", 2)[0])
                ck = (mm.group(1), mm.group(2) if mm.group(2) == "AL" else str(int(mm.group(2))))
                ts = iso(r["ts"])
            except Exception:
                continue
            if "counted" in r and (ck not in SEEN or SEEN[ck][0] != r["counted"]):
                SEEN[ck] = (r["counted"], ts)
    while True:
        now = datetime.now(timezone.utc)
        try:
            N, M = needles(), kalshi_markets()
            PM.clear(); PM.update(poly_mids())
            for ck, w in cnn_calls().items():  # CNN call where AP/NYT hasn't called yet -> treat like a call
                if not str((N.get(ck) or {}).get("call", "")).startswith("called"):
                    lo = "R" if w == "D" else "D"
                    N[ck] = {w: 0.99, lo: 0.01, "counted": 10**9, "call": "called:cnn", "updated": now.isoformat()}
                    print(f"  CNN CALL ahead of AP: {'-'.join(ck)} -> {w}", flush=True)
        except Exception as e:  # keep running through transient fetch errors
            print(now.isoformat(timespec="seconds"), "fetch error:", e, flush=True)
            time.sleep(a.interval)
            continue
        bank = None
        sigs = []
        for m in M:
            n = N.get(m["_key"])
            if not n or m["_party"] not in n:
                continue
            age = (now - iso(n["updated"])).total_seconds() / 60 if n.get("updated") else 1e9
            called = str(n.get("call", "")).startswith("called")
            if not called and (age > a.max_age_min or n["counted"] < a.min_counted):
                continue
            ck = m["_key"]  # frozen-count guard: nyt_updated_at refreshes even when no votes come in (TX-9 sat at
            if ck not in SEEN or SEEN[ck][0] != n["counted"]:  # 130,429 for 50 min while Kalshi moved)
                SEEN[ck] = (n["counted"], now)
            if not called and (now - SEEN[ck][1]).total_seconds() > a.frozen_min * 60:
                continue
            ap_tot = n.get("ap_total") or 0  # model behind the AP count = the market knows more (CA-22: 62% lag)
            if not called and ap_tot and (ap_tot - n["counted"]) / ap_tot > a.max_ap_lag:
                continue
            if not called and (n.get("eevp") or 0) < a.min_eevp:  # early needle = mostly NYT's prior (MT-1 at 6%)
                continue
            thr = a.call_edge if called else a.edge
            p = n[m["_party"]]
            ya, yb = f(m.get("yes_ask_dollars")), f(m.get("yes_bid_dollars"))
            for side, prob, ask in (("yes", blend(p, yb, ya, True), ya),
                                    ("no", blend(1 - p, yb, ya, False), None if yb is None else round(1 - yb, 4))):
                if ask is None or not (0.01 if called else a.min_price) <= ask <= 0.97:
                    continue  # 1-5c needle longshots lost all night (WI-1, PA-1, TX-9, CA-22): market was right
                wins = m["_party"] if side == "yes" else ("R" if m["_party"] == "D" else "D")
                if not called and a.poly_gate:  # optional: size on the lower of needle-blend and Polymarket's price
                    pm = PM.get((m["_key"], wins))
                    if pm is None:
                        continue
                    prob = min(prob, pm)
                e = prob - ask
                t_ = thr if called else (a.edge_d if wins == "D" else a.edge_r)  # lean left: NYT ran R of market all night
                if e >= t_:
                    sigs.append((e, m, side, prob, ask, n, age))
        exits(N, M, a, now)
        bk = book(N, M)
        pos = house_positions()
        held_races = {m["_key"]: m["ticker"] for m in M if m["ticker"] in pos}
        sigs.sort(key=lambda s: -s[0])
        best = {}
        for s_ in sigs:  # one leg per race: the YES-R and NO-D legs are the same bet
            best.setdefault(s_[1]["_key"], s_)
        sigs = list(best.values())
        print(f"{now.isoformat(timespec='seconds')} needles={len(N)} kalshi={len(M)} signals={len(sigs)}", flush=True)
        if bk["bank"] < a.stop_bank:  # drawdown stop: no new buys, sells/trims above still run
            print(f"  DRAWDOWN STOP: bank ${bk['bank']:.2f} < ${a.stop_bank:.0f} -- no new buys", flush=True)
            sigs = []
        for e, m, side, prob, ask, n, age in sigs:
            key = "-".join(m["_key"])
            if m["_key"] in held_races and held_races[m["_key"]] != m["ticker"]:
                continue  # never hold both legs of one race
            hk = pos.get(m["ticker"])
            if hk and (hk[0] > 0) != (side == "yes"):
                continue
            row = dict(ts=now.isoformat(timespec="seconds"), ticker=m["ticker"], side=side, needle=round(prob, 4),
                       ask=ask, edge=round(e, 4), counted=n["counted"], call=n.get("call"), age_min=round(age, 1))
            bank = bk["bank"]
            # count what we already hold at max(marked value, cost): bid-marked alone made the bot re-buy the
            # spread every pass (NC-03 / AZ-08 top-ups 07:37 UTC) and fees were already $78 on the night
            have = max(bk["rows"][m["ticker"]]["val"], bk["rows"][m["ticker"]]["basis"]) if m["ticker"] in bk["rows"] else 0.0
            d = m["_party"] if side == "yes" else ("R" if m["_party"] == "D" else "D")
            room = min(TOTAL_CAP_FRAC * bank - sum(r["val"] for r in bk["rows"].values()),
                       BUCKET_FRAC * bank - bk["dirv"][d])
            # price-scaled cap (user: "$50 on 1 cent" is wrong): <= price_cap_k * price of bank -> 1c ~$3, 10c ~$30
            stake = min(RACE_CAP_FRAC * bank, a.price_cap_k * ask * bank, a.kelly * kelly(prob, ask) * bank) - have
            stake = min(stake, room)
            row["stake"] = round(stake, 2)
            print(f"  SIGNAL {m['ticker']} {side.upper()} needle {prob:.3f} ask {ask:.2f} edge {e:+.2f} "
                  f"counted {n['counted']} age {age:.0f}m -> ${stake:.2f}", flush=True)
            if stake < 2.0:
                bk["dirv"][d] += 0
                row["action"] = "too_small"
            else:
                # sweep the book: the limit is the worst price that still clears the entry edge (thin books, IOC)
                lim = max(ask, min(0.97, ask + 0.05, round(prob - (a.call_edge if str(n.get("call", "")).startswith("called") else a.edge), 2)))
                cmd = [sys.executable, str(HERE / "bet.py"), "place", m["ticker"], side, "--units", f"{stake / 50:.4f}",
                       "--price", f"{lim:.2f}", "--taker", "--tier", "E", "--strategy", "nyt_needle_house"]
                if not a.live:
                    cmd.append("--dry")
                r = subprocess.run(cmd, capture_output=True, text=True)
                row["action"], row["bet_out"] = ("live" if a.live else "dry"), (r.stdout + r.stderr).strip()[-400:]
                print("   ", row["bet_out"], flush=True)
                bk["dirv"][d] += stake  # later signals this pass see the bucket fill
                if a.live and "placed" in r.stdout and held(m["ticker"]):
                    state["done"][key] = row
                    state["staked"] += stake
                    STATE.write_text(json.dumps(state, indent=1))
            with LOG.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
