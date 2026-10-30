"""L36: SEALED test of the frozen CFB totals model vs the ESPN OPENER at the REAL opening price. No orders.

    python scripts/l36_cfb_sealed.py fetch-games      # ESPN FBS scoreboards 2022-2026 (results only; nothing printed)
    python scripts/l36_cfb_sealed.py names            # ESPN team_id -> SBR model name (from 2022 results overlap)
    python scripts/l36_cfb_sealed.py fetch-lines      # ESPN core odds for every 2025 + 2026 game (not printed)
    python scripts/l36_cfb_sealed.py score [--ledger] # ONE-SHOT sealed scoring (pre-registered in notes/l36_cfb_totals_sealed.md)

Pre-registration: notes/l36_cfb_totals_sealed.md (committed before `score` was ever run).
Model = scripts/cfb_simple_model.py, imported, hyperparameters untouched. Only the RESULTS source changes:
SBR 2007-2021 (as in dev) + ESPN 2022-2026 (SBR 2022 stops on 11-15; ESPN 2022 is the full season).

Collector bug found here (2026-10-30): the ESPN site scoreboard with groups=80 AND limit=1000 now silently returns only
the ~25 featured games (2024 file = 548 games). limit=300 returns the full FBS slate. This script fetches with
limit=300 into games_full_<season>.parquet and does not touch sports/espn.py or the existing games_<season> files.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cfb_simple_model as csm  # noqa: E402
from sports import espn, http  # noqa: E402

http.MIN_INTERVAL = 0.5          # <= 2 requests / s per host (fetch() already backs off on 429/5xx)
ROOT = Path(__file__).resolve().parents[1]
ESPN = ROOT / "data" / "espn" / "cfb"
NAMEMAP = ROOT / "data" / "lab" / "l36_espn_to_sbr_names.json"
OUT_JSON = ROOT / "data" / "lab" / "l36_cfb_sealed_results.json"
BETS_CSV = ROOT / "results" / "l36_cfb_sealed" / "bets.csv"
ESPN_SEASONS = [2022, 2023, 2024, 2025, 2026]
SEALED = [2025, 2026]
BOOK_PRIORITY = ["DraftKings", "ESPN BET"]   # then any other provider, alphabetical; see prereg
NBOOT = 4000


# ------------------------------------------------------------------------------------------- fetch
def scoreboard(day: date) -> list[dict]:
    p = {"dates": day.strftime("%Y%m%d"), "groups": 80, "limit": 300}
    return http.fetch(espn.SITE.format(path=espn.LEAGUES["cfb"]) + "/scoreboard", p,
                      cache=espn._settled(day)).get("events", [])


def fetch_games():
    for y in ESPN_SEASONS:
        lo, hi = espn.WINDOWS["cfb"](y)
        hi = min(hi, datetime.now(timezone.utc).date())
        rows, seen, d = [], set(), lo
        while d <= hi:
            for e in scoreboard(d):
                if e["id"] not in seen:
                    seen.add(e["id"]); rows.append(espn.parse_event("cfb", e))
            d += timedelta(days=1)
        g = pd.DataFrame(rows)
        g = g[g["season"] == y].sort_values("start_utc").reset_index(drop=True)
        for c in ("event_id", "home_team_id", "away_team_id", "venue_id"):
            g[c] = g[c].astype("string")
        g.to_parquet(ESPN / f"games_full_{y}.parquet", index=False)
        print(y, "games", len(g), "final", int((g["status"] == "STATUS_FINAL").sum()))   # counts only


def fetch_lines():
    for y in SEALED:
        g = pd.read_parquet(ESPN / f"games_full_{y}.parquet")
        g = g[g["status"] == "STATUS_FINAL"]
        done = (pd.to_datetime(g["start_utc"], utc=True) < pd.Timestamp.now(tz="UTC") - espn.SETTLED_AFTER).to_list()
        rows = []
        for i, (eid, c) in enumerate(zip(g["event_id"].astype(str), done)):
            rows.extend(espn.lines("cfb", eid, cache=c))
            if (i + 1) % 200 == 0:
                print(f"  {y}: {i + 1}/{len(g)}", flush=True)
        L = pd.DataFrame(rows)
        L["event_id"] = L["event_id"].astype("string"); L["provider_id"] = L["provider_id"].astype("string")
        L.to_parquet(ESPN / f"lines_full_{y}.parquet", index=False)
        print(y, "line rows", len(L), "events with any line", L["event_id"].nunique())   # counts only


# ------------------------------------------------------------------------------------------- results
def espn_results(y: int) -> pd.DataFrame:
    g = pd.read_parquet(ESPN / f"games_full_{y}.parquet")
    g = g[(g["status"] == "STATUS_FINAL") & g["home_score"].notna() & g["away_score"].notna()].copy()
    g["date"] = pd.to_datetime(g["start_utc"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return g


def build_namemap() -> dict:
    """ESPN team_id -> SBR name, matched on 2022 games (date +-1 day, exact final scores). Results only."""
    s = csm.parse_season(2022)
    s["home"] = s["home"].replace(csm.ALIAS); s["away"] = s["away"].replace(csm.ALIAS)
    s = s[s["rot"] < 100000]
    e = espn_results(2022)
    votes: dict[str, dict[str, int]] = {}
    key = {}
    for r in s.itertuples():
        key.setdefault((r.home_pts, r.away_pts), []).append(r)
    for r in e.itertuples():
        cands = []
        for (hp, ap, flip) in ((r.home_score, r.away_score, False), (r.away_score, r.home_score, True)):
            for m in key.get((int(hp), int(ap)), []):
                if abs((m.date - r.date).days) <= 1:
                    cands.append((m, flip))
        if len(cands) != 1:   # ambiguous (same date, same score) -> no vote
            continue
        m, flip = cands[0]
        eh, ea = (r.home_team_id, r.away_team_id) if not flip else (r.away_team_id, r.home_team_id)
        votes.setdefault(str(eh), {}).setdefault(m.home, 0); votes[str(eh)][m.home] += 1
        votes.setdefault(str(ea), {}).setdefault(m.away, 0); votes[str(ea)][m.away] += 1
    mp = {}
    for tid, v in votes.items():
        best = max(v.items(), key=lambda kv: kv[1])
        if best[1] >= 3 and best[1] >= 0.8 * sum(v.values()):
            mp[tid] = best[0]
    # one SBR name per ESPN id; drop SBR names claimed twice
    inv = pd.Series(mp).value_counts()
    mp = {k: v for k, v in mp.items() if inv[v] == 1}
    return mp


def names():
    mp = build_namemap()
    NAMEMAP.parent.mkdir(parents=True, exist_ok=True)
    NAMEMAP.write_text(json.dumps(mp, indent=0, sort_keys=True))
    sbr21 = csm.load_all()
    fbs21 = sbr21.attrs["fbs"][2021]
    print("mapped ESPN ids:", len(mp), " SBR 2021 FBS teams:", len(fbs21),
          " 2021 FBS teams with an ESPN id:", len(fbs21 & set(mp.values())))
    print("2021 SBR FBS teams NOT mapped:", sorted(fbs21 - set(mp.values())))
    for y in ESPN_SEASONS:
        e = espn_results(y)
        c = pd.concat([e["home_team_id"], e["away_team_id"]]).value_counts()
        fbs_ids = set(c[c >= 6].index.astype(str)) & fbs_group_ids(y)
        un = sorted({(t, a) for t, a in zip(pd.concat([e.home_team_id, e.away_team_id]).astype(str),
                                             pd.concat([e.home, e.away])) if t in fbs_ids and t not in mp})
        print(y, "final games", len(e), "FBS ids (>=6 games)", len(fbs_ids), "unmapped FBS ids", len(un), un)


def combined_games() -> pd.DataFrame:
    """SBR 2007-2021 + ESPN 2022-2026 in the cfb_simple_model.load_all row format, same FBS/week post-processing."""
    mp = json.loads(NAMEMAP.read_text())
    sb = pd.concat([csm.parse_season(y) for y in range(2007, 2022)], ignore_index=True)
    sb["event_id"] = pd.NA
    parts = [sb]
    for y in ESPN_SEASONS:
        e = espn_results(y)
        nm = lambda tid, ab: mp.get(str(tid), f"ESPN_{tid}")
        parts.append(pd.DataFrame(dict(
            season=y, date=e["date"].to_numpy(), rot=np.arange(1, len(e) + 1),
            away=[nm(t, a) for t, a in zip(e.away_team_id, e.away)],
            home=[nm(t, a) for t, a in zip(e.home_team_id, e.home)],
            away_pts=e["away_score"].astype(int).to_numpy(), home_pts=e["home_score"].astype(int).to_numpy(),
            neutral=e["neutral"].astype(bool).to_numpy(), event_id=e["event_id"].astype(str).to_numpy(),
            home_tid=e["home_team_id"].astype(str).to_numpy(), away_tid=e["away_team_id"].astype(str).to_numpy())))
    g = pd.concat(parts, ignore_index=True)
    # identical to cfb_simple_model.load_all from here
    g["home"] = g["home"].replace(csm.ALIAS); g["away"] = g["away"].replace(csm.ALIAS)
    g = g[g["rot"] < 100000].copy()
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["points"] = g["home_pts"] + g["away_pts"]
    fbs = {}
    for s, d in g.groupby("season"):
        c = pd.concat([d["home"], d["away"]]).value_counts()
        fbs[s] = set(c[c >= (3 if s == 2020 else 6)].index)
    fbs[2020] = {t for t in fbs[2020] if t in fbs[2019] or t in fbs[2021]}
    # AMENDMENT (pre-scoring, results-free): the ESPN groups=80 feed lists games of teams that are FBS TODAY
    # (e.g. Delaware's / Sacramento St's FCS seasons), so for ESPN seasons FBS = count rule AND ESPN's FBS group list
    # for that season (core /seasons/{y}/types/2/groups/80/teams).
    for y in ESPN_SEASONS:
        grp = fbs_group_ids(y)
        d = g[g["season"] == y]
        names_in = {n for n, t in zip(pd.concat([d.home, d.away]), pd.concat([d.home_tid, d.away_tid])) if t in grp}
        fbs[y] = fbs[y] & names_in
    g["fbs_game"] = [(r.home in fbs[r.season]) and (r.away in fbs[r.season]) for r in g.itertuples()]
    g["h_id"] = [r.home if r.home in fbs[r.season] else "FCS" for r in g.itertuples()]
    g["a_id"] = [r.away if r.away in fbs[r.season] else "FCS" for r in g.itertuples()]
    first = g.groupby("season")["date"].transform("min")
    g["week"] = ((g["date"] - first).dt.days // 7 + 1).astype(int)
    g = g.sort_values(["date", "rot"]).reset_index(drop=True)
    g.attrs["fbs"] = fbs
    return g


def fbs_group_ids(y: int) -> set:
    import re
    r = http.fetch(espn._core("cfb") + f"/seasons/{y}/types/2/groups/80/teams", {"limit": 300}, cache=True)
    return {re.search(r"/teams/(\d+)", i["$ref"]).group(1) for i in r.get("items", [])}


def predictions() -> pd.DataFrame:
    csm.SEASONS = list(range(2007, 2027))
    csm.TEST = SEALED
    return csm.build_predictions(combined_games())


# ------------------------------------------------------------------------------------------- market
def pick_open(L: pd.DataFrame) -> pd.DataFrame:
    """One row per event: first provider in BOOK_PRIORITY (then alphabetical) with total_open, both open prices
    and total_close all present."""
    ok = L[L["total_open"].notna() & L["over_odds_open"].notna() & L["under_odds_open"].notna()
           & L["total_close"].notna()].copy()
    rank = {b: i for i, b in enumerate(BOOK_PRIORITY)}
    ok["_r"] = [rank.get(p, len(rank)) for p in ok["provider"]]
    ok = ok.sort_values(["event_id", "_r", "provider"])
    return ok.groupby("event_id", as_index=False).first().drop(columns="_r")


def payout(odds):
    odds = np.asarray(odds, float)
    return np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))


def implied(odds):
    odds = np.asarray(odds, float)
    return np.where(odds > 0, 100.0 / (odds + 100.0), np.abs(odds) / (np.abs(odds) + 100.0))


def boot(vals, clusters, n=NBOOT, seed=0):
    p, lo, hi = csm.boot_p(vals, clusters, n=n, seed=seed)
    return p, lo, hi


def score(ledger: bool):
    P = predictions()
    P = P[P["season"].isin(SEALED)].copy()
    P = P.drop(columns=[c for c in P.columns if c.endswith(("_open", "_close"))])   # SBR-only columns (NaN here)
    L = pd.concat([pd.read_parquet(ESPN / f"lines_full_{y}.parquet") for y in SEALED], ignore_index=True)
    L["event_id"] = L["event_id"].astype(str)
    L["provider"] = L["provider"].replace({"Draft Kings": "DraftKings"})   # ESPN spells it "Draft Kings" in 2025+
    O = pick_open(L)
    print("providers chosen per event:", O["provider"].value_counts().to_dict())
    P["event_id"] = P["event_id"].astype(str)
    assert P["event_id"].is_unique
    d = P.merge(O[["event_id", "provider", "total_open", "over_odds_open", "under_odds_open", "total_close",
                   "over_odds_close", "under_odds_close"]], on="event_id", how="inner", validate="one_to_one")
    n_fbs = int(P["fbs_game"].sum())
    d = d[d["fbs_game"]]
    d = d[d["total_open"].between(25, 100) & ((d["total_close"] - d["total_open"]).abs() <= 14)]
    n_mkt = len(d)
    gap = d["pred_total"] - d["total_open"]
    B = d[gap.abs() >= csm.TH_T].copy()
    B["side"] = np.sign(B["pred_total"] - B["total_open"])         # +1 over, -1 under
    B["price"] = np.where(B["side"] > 0, B["over_odds_open"], B["under_odds_open"])
    B["other"] = np.where(B["side"] > 0, B["under_odds_open"], B["over_odds_open"])
    r = B["side"] * (B["points"] - B["total_open"])
    B["result"] = np.sign(r)
    B["roi"] = np.where(r > 0, payout(B["price"]), np.where(r < 0, -1.0, 0.0))
    B["roi110"] = np.where(r > 0, 100 / 110, np.where(r < 0, -1.0, 0.0))
    B["clv"] = B["side"] * (B["total_close"] - B["total_open"])
    pi, po = implied(B["price"]), implied(B["other"])
    B["p_novig"] = pi / (pi + po)
    B["cl"] = B["season"].astype(str) + "_" + B["week"].astype(str)
    BETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    B[["season", "week", "date", "event_id", "home", "away", "neutral", "provider", "pred_total", "total_open",
       "total_close", "over_odds_open", "under_odds_open", "side", "price", "points", "result", "roi", "clv",
       "p_novig"]].to_csv(BETS_CSV, index=False)

    def summ(x: pd.DataFrame, col: str) -> dict:
        v = x[col].to_numpy(float)
        if len(v) == 0:
            return dict(n=0)
        p, lo, hi = boot(v, x["cl"].to_numpy())
        out = dict(n=int(len(v)), mean=float(v.mean()), lo=lo, hi=hi, p=p)
        for s in SEALED:
            vs = x.loc[x["season"] == s, col]
            out[f"s{s}"] = float(vs.mean()) if len(vs) else None
            out[f"n{s}"] = int(len(vs))
        return out

    res = {"P1_roi_open_price": summ(B, "roi"), "P2_total_clv_pts": summ(B, "clv"),
           "P3_roi_open_price_wk1_4": summ(B[B["week"] <= 4], "roi")}
    p1 = res["P1_roi_open_price"]
    p1["passed"] = bool(p1["n"] and p1["p"] < 0.05 and p1["mean"] > 0
                        and all((p1[f"s{s}"] or -1) > 0 for s in SEALED))
    dec = B[B["result"] != 0]
    diag = dict(
        n_fbs_games=n_fbs, n_fbs_with_market=n_mkt, n_bets=int(len(B)), n_over=int((B["side"] > 0).sum()),
        hit_rate_ex_push=float((dec["result"] > 0).mean()) if len(dec) else None,
        mean_novig_p=float(dec["p_novig"].mean()) if len(dec) else None,
        avg_price=float(B["price"].mean()), avg_payout=float(payout(B["price"]).mean()),
        roi110_ref=summ(B, "roi110"),
        mae_model=float((d["pred_total"] - d["points"]).abs().mean()),
        mae_open=float((d["total_open"] - d["points"]).abs().mean()),
        mae_close=float((d["total_close"] - d["points"]).abs().mean()),
        bias_model=float((d["pred_total"] - d["points"]).mean()),
        by_season={int(s): dict(n=int(len(x)), roi=float(x.roi.mean()), clv=float(x.clv.mean()))
                   for s, x in B.groupby("season")},
        providers=B["provider"].value_counts().to_dict())
    print(json.dumps(dict(trials=res, diag=diag), indent=1, default=str))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(trials=res, diag=diag), indent=1, default=str))
    if ledger:
        ts = datetime.now(timezone.utc).isoformat()
        with csm.LEDGER.open("a") as fh:
            fh.write(json.dumps(dict(ts=ts, strategy="l36_cfb_totals_P1", dataset="l36_cfb_sealed", metric="roi_open_price",
                                     hypothesis="pre-registered in notes/l36_cfb_totals_sealed.md", split="SEALED 2025+2026wk1-9",
                                     source="lead L36: CFB totals vs ESPN opener", status="ok", trial_number=1,
                                     study_trials=1, bar=0.05, passed=p1["passed"], bets=p1["n"], mean=p1["mean"],
                                     ci_lo=p1["lo"], ci_hi=p1["hi"], p=p1["p"],
                                     per_season={str(s): p1[f"s{s}"] for s in SEALED},
                                     clv=res["P2_total_clv_pts"]["mean"], clv_ci=[res["P2_total_clv_pts"]["lo"], res["P2_total_clv_pts"]["hi"]])) + "\n")
        print("ledger: appended 1")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"fetch-games": fetch_games, "names": names, "fetch-lines": fetch_lines,
     "score": lambda: score("--ledger" in sys.argv)}.get(cmd, lambda: print(__doc__))()
