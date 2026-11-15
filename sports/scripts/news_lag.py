"""news_lag dev: NFL starting-QB availability resolved by the KO-90 min inactive list vs Kalshi KXNFLGAME prices.

    ../.venv_quant/bin/python scripts/news_lag.py events    # news side only (nflverse), both seasons -> results/news_lag/events.parquet
    ../.venv_quant/bin/python scripts/news_lag.py prices    # Kalshi 1-min candles, DEV (2025 season) games only, read-only kget
    ../.venv_quant/bin/python scripts/news_lag.py score     # pre-registered T1/T2 + descriptive D1/D2 (notes/news_lag.md)

Pre-registration: notes/news_lag.md (frozen before any price was fetched).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "news_lag"
RAW = OUT / "raw"
CANDLES = OUT / "candles_dev.parquet"
MKTS = OUT / "kalshi_nfl_markets.parquet"


def fee(p):
    return 0.07 * p * (1 - p)


def schedule() -> pd.DataFrame:
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.season.isin([2025, 2026])].copy()
    ko = pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    g["ko"] = (ko - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)  # traps: no //1e9
    return g


def cmd_events() -> int:
    g = schedule()
    rows = []
    for season in (2025, 2026):
        inj = pd.read_parquet(RAW / f"injuries_{season}.parquet")
        snaps = pd.read_parquet(RAW / f"snap_counts_{season}.parquet")
        ros = pd.read_parquet(RAW / f"roster_weekly_{season}.parquet")
        pfr = ros.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id").set_index("gsis_id").pfr_id
        qs = snaps[snaps.position == "QB"].copy()
        qs["wk"] = qs.week
        q = inj[(inj.position == "QB") & inj.report_status.isin(["Questionable", "Doubtful"])].copy()
        for _, r in q.iterrows():
            team, wk = r.team, int(r.week)
            gm = g[(g.season == season) & (g.week == wk) & ((g.home_team == team) | (g.away_team == team))]
            if len(gm) != 1:
                rows.append({"season": season, "week": wk, "team": team, "player": r.full_name, "drop": "no_game"})
                continue
            gm = gm.iloc[0]
            pid = pfr.get(r.gsis_id)
            base = {"season": season, "week": wk, "game_id": gm.game_id, "ko": int(gm.ko), "team": team,
                    "opp": gm.away_team if gm.home_team == team else gm.home_team, "home": gm.home_team == team,
                    "player": r.full_name, "gsis": r.gsis_id, "pfr": pid, "status": r.report_status,
                    "game_type": gm.game_type}
            if pid is None:
                rows.append({**base, "drop": "no_pfr"})
                continue
            mine = qs[(qs.pfr_player_id == pid) & (qs.wk < wk) & (qs.team == team)]
            if mine.empty:
                rows.append({**base, "drop": "no_prev_game"})
                continue
            prev = mine.sort_values("wk").iloc[-1]
            team_prev = qs[(qs.team == team) & (qs.game_id == prev.game_id)]
            starter = prev.offense_pct >= 0.70 and prev.offense_snaps >= team_prev.offense_snaps.max()
            now = qs[(qs.pfr_player_id == pid) & (qs.game_id == gm.game_id)]
            played = bool((now.offense_snaps > 0).any())
            started_qb = gm.home_qb_id if gm.home_team == team else gm.away_qb_id
            rows.append({**base, "prev_game": prev.game_id, "prev_pct": float(prev.offense_pct), "starter": bool(starter),
                         "played": played, "started_nflverse": started_qb == r.gsis_id,
                         "drop": None if starter else "not_starter"})
    ev = pd.DataFrame(rows)
    ev["benefit"] = np.where(ev.played.fillna(False).astype(bool), ev.team, ev.opp)
    ev["t2"] = ((ev.status == "Questionable") & ~ev.played.fillna(True).astype(bool)) | ((ev.status == "Doubtful") & ev.played.fillna(False).astype(bool))
    ev["split"] = np.where(ev.season == 2025, "dev", "holdout")
    ev.to_parquet(OUT / "events.parquet")
    k = ev[ev["drop"].isna()]
    print("drops:", ev["drop"].value_counts(dropna=False).to_dict())
    print(k.groupby(["split", "status", "played"]).size())
    print("T2 by split:", k.groupby("split").t2.sum().to_dict())
    print("played vs started mismatches (kept):")
    print(k[k.played != k.started_nflverse][["game_id", "player", "status", "played", "started_nflverse"]])
    return 0


def kmarkets() -> pd.DataFrame:
    """All KXNFLGAME markets (historical + live settled), cached."""
    if MKTS.exists():
        return pd.read_parquet(MKTS)
    from kx_throttle import kget
    out = []
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": "KXNFLGAME", "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            j = kget(path, p)
            if not j:
                break
            for m in j.get("markets", []):
                out.append({"ticker": m["ticker"], "event": m["event_ticker"], "team_sub": m.get("yes_sub_title"),
                            "result": m.get("result"), "occ": m.get("occurrence_datetime") or m.get("expected_expiration_time"),
                            "open_time": m.get("open_time"), "src": path})
            cur = j.get("cursor")
            if not cur:
                break
    m = pd.DataFrame(out).drop_duplicates("ticker")
    m.to_parquet(MKTS)
    return m


KTEAM = {"LA": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}


def match_markets(ev: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Map (game_id, benefit team) -> ticker. Kalshi event = KXNFLGAME-YYMONDD<AWAY><HOME>, ticker suffix = team code."""
    m = m.copy()
    m["suffix"] = m.ticker.str.split("-").str[-1].map(lambda s: KTEAM.get(s, s))
    m["key"] = m.event.str.split("-").str[1]
    g = schedule().set_index("game_id")
    out = []
    for gid in ev.game_id.unique():
        r = g.loc[gid]
        d = pd.Timestamp(r.gameday).strftime("%y%b%d").upper()
        cand = m[m.key.str.startswith(d)]
        teams = {r.away_team, r.home_team}
        cand = cand[cand.suffix.isin(teams)]
        evs = cand.event.unique()
        if len(evs) != 1:
            out.append({"game_id": gid, "event": None, "n_events": len(evs)})
            continue
        row = {"game_id": gid, "event": evs[0], "n_events": 1}
        for _, x in cand.iterrows():
            row[f"tk_{x.suffix}"] = x.ticker
        out.append(row)
    return pd.DataFrame(out)


def candles(ticker: str, s0: int, e0: int) -> pd.DataFrame:
    from kx_throttle import kget
    for path in (f"/historical/markets/{ticker}/candlesticks", f"/series/KXNFLGAME/markets/{ticker}/candlesticks"):
        j = kget(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1})
        if j and j.get("candlesticks"):
            rows = []
            for c in j["candlesticks"]:
                yb, ya = c.get("yes_bid", {}), c.get("yes_ask", {})

                def px(d):
                    if d.get("close_dollars") is not None:
                        return float(d["close_dollars"])
                    if d.get("close") is not None:
                        v = d["close"]
                        return float(v) if isinstance(v, str) else v / 100
                    return np.nan
                rows.append({"end": int(c["end_period_ts"]), "bid": px(yb), "ask": px(ya),
                             "vol": float(c.get("volume_fp", c.get("volume", 0)) or 0)})
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["end", "bid", "ask", "vol"])


def cmd_prices() -> int:
    ev = pd.read_parquet(OUT / "events.parquet")
    g = schedule()
    dev_games = g[(g.season == 2025)]
    m = kmarkets()
    print("kalshi markets", len(m))
    mm = match_markets(dev_games.rename(columns={})[["game_id"]], m)
    mm.to_parquet(OUT / "market_map_dev.parquet")
    print("matched", int((mm.n_events == 1).sum()), "of", len(mm), "dev games")
    done = set()
    parts = []
    if CANDLES.exists():
        old = pd.read_parquet(CANDLES)
        parts.append(old)
        done = set(old.ticker.unique())
    gi = g.set_index("game_id")
    todo = []
    for _, r in mm[mm.n_events == 1].iterrows():
        ko = int(gi.loc[r.game_id].ko)
        for c in [c for c in mm.columns if c.startswith("tk_")]:
            tk = r.get(c)
            if isinstance(tk, str) and tk not in done:
                todo.append((r.game_id, tk, ko))
    print("to fetch", len(todo))
    for i, (gid, tk, ko) in enumerate(todo):
        c = candles(tk, ko - 26 * 3600, ko + 60)
        c["ticker"], c["game_id"] = tk, gid
        parts.append(c)
        if i % 50 == 49:
            pd.concat(parts).to_parquet(CANDLES)
            print(i + 1, "fetched", flush=True)
    pd.concat(parts).to_parquet(CANDLES)
    return 0


def quotes(c: pd.DataFrame) -> pd.DataFrame:
    c = c[(c.bid > 0) & (c.ask < 1) & (c.bid <= c.ask)].sort_values("end")
    return c.assign(mid=(c.bid + c.ask) / 2)


def mid_at(q: pd.DataFrame, t: int) -> float:
    i = np.searchsorted(q.end.values, t, side="right") - 1
    return float(q.mid.values[i]) if i >= 0 else np.nan


def boot(v, g, n=10000, seed=0):
    v, g = np.asarray(v, float), np.asarray(g)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(seed)
    ms = np.empty(n)
    for b in range(n):
        pick = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[u] for u in pick])
        ms[b] = v[ii].mean()
    return float(v.mean()), [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))], float((ms <= 0).mean())


def cmd_score() -> int:
    ev = pd.read_parquet(OUT / "events.parquet")
    cd = pd.read_parquet(CANDLES)
    mm = pd.read_parquet(OUT / "market_map_dev.parquet")
    g = schedule().set_index("game_id")
    Q = {tk: quotes(c) for tk, c in cd.groupby("ticker")}
    res = {}
    # ---- D1: news-agnostic |mid move| in 10-min windows, home-team market
    rows = []
    for _, r in mm[mm.n_events == 1].iterrows():
        gm = g.loc[r.game_id]
        tk = r.get(f"tk_{gm.home_team}")
        if not isinstance(tk, str) or tk not in Q or len(Q[tk]) < 30:
            continue
        q, ko = Q[tk], int(gm.ko)
        rec = {"game_id": r.game_id, "ko": ko}
        for s in (151, 121, 91, 61, 31):
            a, b = mid_at(q, ko - s * 60), mid_at(q, ko - s * 60 + 600)
            rec[f"w{s}"] = abs(b - a)
        rec["prof"] = [mid_at(q, ko - m * 60) for m in range(180, -1, -1)]
        rows.append(rec)
    d1 = pd.DataFrame(rows).dropna(subset=["w151", "w121", "w91", "w61"])
    d1["plac"] = d1[["w151", "w121", "w61"]].mean(axis=1)
    res["D1"] = {"n_games": len(d1), **{f"mean_abs_w{s}": float(d1[f"w{s}"].mean()) for s in (151, 121, 91, 61, 31)},
                 "w91_minus_placebo": boot(d1.w91 - d1.plac, d1.game_id)}
    prof = np.vstack(d1.prof.values)
    dm = np.abs(np.diff(prof, axis=1))  # minute m -> m-1
    mins = np.arange(179, -1, -1)
    res["D1"]["per_minute_abs_move_c"] = {int(m): round(float(np.nanmean(dm[:, i])) * 100, 3) for i, m in enumerate(mins) if m % 5 == 0 or 85 <= m <= 95}
    # ---- T1/T2 + D2
    k = ev[(ev.split == "dev") & ev["drop"].isna()].copy()
    bets, path = [], []
    for gid, grp in k.groupby("game_id"):
        bens = grp.benefit.unique()
        if len(bens) > 1:
            res.setdefault("dropped_conflict", []).append(gid)
            continue
        r0 = grp.iloc[0]
        mrow = mm[mm.game_id == gid]
        tk = mrow.iloc[0].get(f"tk_{bens[0]}") if len(mrow) else None
        if not isinstance(tk, str) or tk not in Q:
            bets.append({"game_id": gid, "nobet": "no_market"})
            continue
        q, ko = Q[tk], int(r0.ko)
        td = ko - 87 * 60
        e = q[(q.end >= td) & (q.end <= td + 600) & (q.end < ko) & (q.ask - q.bid <= 0.10)]
        pts = {lab: mid_at(q, ko - m) for lab, m in (("-24h", 86400), ("-6h", 21600), ("-3h", 10800), ("-2h", 7200),
                                                       ("-95m", 5700), ("-91m", 5460), ("-85m", 5100), ("-80m", 4800),
                                                       ("-60m", 3600), ("-30m", 1800), ("KO", 0))}
        path.append({"game_id": gid, "player": r0.player, "played": bool(r0.played), "t2": bool(grp.t2.any()), **pts})
        if e.empty:
            bets.append({"game_id": gid, "nobet": "no_entry"})
            continue
        ask = float(e.ask.iloc[0])
        settle = 1.0 if cd[cd.ticker == tk].shape[0] and False else np.nan
        bets.append({"game_id": gid, "ko": ko, "ticker": tk, "player": r0.player, "status": r0.status,
                     "played": bool(r0.played), "t2": bool(grp.t2.any()), "entry_end": int(e.end.iloc[0]),
                     "ask": ask, "bid": float(e.bid.iloc[0]), "mid_td": mid_at(q, td), "mid_ko": pts["KO"],
                     "net_clv": pts["KO"] - ask - fee(ask), "gross": pts["KO"] - mid_at(q, td)})
    b = pd.DataFrame(bets)
    pth = pd.DataFrame(path)
    m = kmarkets().set_index("ticker")
    ok = b.dropna(subset=["net_clv"]) if "net_clv" in b else b.iloc[0:0]
    ok = ok.assign(settle=ok.ticker.map(lambda t: 1.0 if m.result.get(t) == "yes" else 0.0))
    ok["roi"] = (ok.settle - ok.ask - fee(ok.ask)) / ok.ask
    for name, sub in (("T1", ok), ("T2", ok[ok.t2])):
        if len(sub) < 2:
            res[name] = {"n": len(sub)}
            continue
        sub = sub.sort_values("ko")
        h = len(sub) // 2
        res[name] = {"n": len(sub), "net_clv": boot(sub.net_clv, sub.game_id), "gross_drift": boot(sub.gross, sub.game_id),
                     "roi": boot(sub.roi, sub.game_id), "halves": [float(sub.net_clv.iloc[:h].mean()), float(sub.net_clv.iloc[h:].mean())],
                     "mean_spread": float((sub.ask - sub.bid).mean())}
    res["nobet"] = b["nobet"].value_counts().to_dict() if "nobet" in b else {}
    # D2: benefit-direction path relative to mid at KO-91m; share of (-24h -> KO) move before t0
    if len(pth):
        rel = pth[["-24h", "-6h", "-3h", "-2h", "-95m", "-91m", "-85m", "-80m", "-60m", "-30m", "KO"]].sub(pth["-91m"], axis=0)
        res["D2_path_rel_-91m_c"] = {c: round(float(rel[c].mean()) * 100, 2) for c in rel.columns}
        tot = pth.KO - pth["-24h"]
        pre = pth["-91m"] - pth["-24h"]
        res["D2_share_pre_t0_of_sum"] = float(pre.sum() / tot.sum()) if tot.sum() else None
        res["D2_n"] = len(pth)
    b.to_csv(OUT / "bets_dev.csv", index=False)
    pth.to_csv(OUT / "paths_dev.csv", index=False)
    d1.drop(columns="prof").to_csv(OUT / "d1_dev.csv", index=False)
    (OUT / "dev_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_forward() -> int:
    """Descriptive read of the forward logger: per news change, Kalshi mids of that game's markets at -2,0,+1,+3,+10 min."""
    F = OUT / "forward"
    kal, news = [], []
    for f in sorted(F.glob("kalshi_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            for m in d.get("m", []):
                kal.append((d["t"], m[0], float(m[1] or 0), float(m[2] or 0)))
    for f in sorted(F.glob("news_*.jsonl")):
        news += [json.loads(x) for x in open(f)]
    k = pd.DataFrame(kal, columns=["t", "tk", "yb", "ya"])
    k = k[(k.yb > 0) & (k.ya < 1) & (k.yb <= k.ya)].assign(mid=lambda x: (x.yb + x.ya) / 2).sort_values("t")
    sched = [n for n in news if n["src"] == "schedule"]
    games = {g["id"]: g for s in sched for g in s["games"]}
    rows = []
    for n in news:
        if n["src"] not in ("espn_inj", "espn_roster", "espn_box") or n.get("game") not in games:
            continue
        g = games[n["game"]]
        teams = [g["home"][1], g["away"][1]]
        series = "KXNFLGAME" if g["lg"] == "nfl" else "KXNBAGAME"
        sub = k[k.tk.str.startswith(series) & k.tk.str.split("-").str[-1].isin(teams)]
        d = pd.Timestamp(g["ko"], unit="s", tz="UTC").tz_convert("America/New_York").strftime("%y%b%d").upper()
        sub = sub[sub.tk.str.split("-").str[1].str.startswith(d)]
        rec = {"src": n["src"], "lg": g["lg"], "game": g["name"], "t": n["t"], "min_to_ko": round((g["ko"] - n["t"]) / 60, 1),
               "team": n.get("team"), "http": n.get("http"), "n_dnp": len(n.get("dnp", []) or [])}
        for tk, q in sub.groupby("tk"):
            for lab, off in (("m2", -120), ("m0", 0), ("p1", 60), ("p3", 180), ("p10", 600)):
                qq = q[q.t <= n["t"] + off]
                rec[f"{tk.split('-')[-1]}_{lab}"] = float(qq.mid.iloc[-1]) if len(qq) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "forward_events.csv", index=False)
    print(f"{len(k)} kalshi quotes, {len(news)} news rows, {len(out)} game news changes -> results/news_lag/forward_events.csv")
    if len(out):
        print(out.groupby(["lg", "src"]).size())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "events"
    sys.exit({"events": cmd_events, "prices": cmd_prices, "score": cmd_score, "forward": cmd_forward}[cmd]())
