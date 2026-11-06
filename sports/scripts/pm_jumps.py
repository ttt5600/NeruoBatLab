"""Overreaction study (user hypothesis 11-06): do large short-window Polymarket moves that look immaterial revert?

    python scripts/pm_jumps.py enum [dev|sealed]   # gamma events (cached) -> binary markets with conditionId
    python scripts/pm_jumps.py tape [dev|sealed]   # data-api /v2/trades, window [S-14d, S], compact parquet per market
    python scripts/pm_jumps.py jumps [dev|sealed]  # jump events + pre-decision proxies + executable fade entry/exit
    python scripts/pm_jumps.py describe            # DEV ONLY: drift / fade return by proxy, event-clustered bootstrap

Split (edgelab/vault.yaml pm_jumps, set before any tape or outcome was read): key = UTC date of S.
  dev    = the 92 days sampled by smallmarket_calibration (seed 20261027) in 2025-05-01..2026-04-30
  sealed = every day 2026-05-01..2026-10-31, categories mentions + culture_tech EXCLUDED (pm_maker_no sealed holdout)
S = min(closedTime, endDate, gameStartTime): sports use only the PRE-GAME tape.
Prices: YES-equivalent of every taker print (NO prints -> 1 - price). Jump = |p(t) - p(last print <= t - W)| >= X.
Decision = jump print time + LAT. Fade entry = first taker print AFTER the decision that bought the fade side
(its price is an ask someone actually paid), within ENTRY_MAX; fee from the market's fee flag (sports 0.05 p(1-p),
crypto 0.07; otherwise 0, the 2025-26 schedule). Public read-only data; no keys, no orders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallmarket_calibration as sm  # noqa: E402
from pm_maker_no import dev_days  # noqa: E402

sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pm_jumps"
(OUT / "tape").mkdir(parents=True, exist_ok=True)
DATA_API = "https://data-api.polymarket.com/v2/trades"
sm.RATE["data-api.polymarket.com"] = 0.12
SEAL0, SEAL1 = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")
CATS = ("politics_world", "finance_econ", "culture_tech", "mentions", "other", "sports")
SEALED_EXCL = ("mentions", "culture_tech")
VOL_MIN = 10_000.0          # lifetime $ volume (sampling frame only; ex post, never a decision input)
PER_CAT_EVENTS = {"sports": 700, "politics_world": 900, "finance_econ": 900, "culture_tech": 900, "mentions": 500,
                  "other": 300}
MAX_MKTS_PER_EVENT = 6      # top by lifetime volume
TAPE_DAYS = 14
MAX_PAGES = 40
LAT = 60                    # s between the jump print and our decision
ENTRY_MAX = 15 * 60         # s: no fade-side print within this after decision -> no entry
COOLDOWN = 6 * 3600         # one jump per market per 6 h
DEFS = {"W15_X05": (15 * 60, 0.05), "W60_X10": (60 * 60, 0.10)}
HORIZ = {"1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}


def _days(split):
    if split in ("dev", "devlow"):
        return dev_days()
    return list(pd.date_range(SEAL0, SEAL1 - pd.Timedelta(days=1)))


def cmd_enum(split="dev"):
    rows = []
    days = _days(split)
    for i, evs in enumerate(sm.pmap(sm.pm_day, days, workers=3)):
        for e in evs:
            cat = sm.pm_category([t.get("label", "") for t in (e.get("tags") or [])])
            if cat not in CATS or (split == "sealed" and cat in SEALED_EXCL) or \
                    (split == "devlow" and cat not in ("politics_world", "finance_econ")):
                continue
            for m in e.get("markets") or []:
                try:
                    outs = json.loads(m.get("outcomes") or "[]")
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                    op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                except Exception:
                    continue
                if len(toks) != 2 or len(op) != 2 or not m.get("conditionId"):
                    continue
                rows.append({"event_id": str(e["id"]), "ticker": str(m["id"]), "cond": m["conditionId"],
                             "question": m.get("question"), "outcome0": outs[0] if outs else None, "cat": cat,
                             "token0": str(toks[0]), "token1": str(toks[1]),
                             "end": m.get("endDate") or e.get("endDate"), "closed_time": m.get("closedTime"),
                             "game_start": m.get("gameStartTime"), "volume": float(m.get("volumeNum") or 0),
                             "neg_risk": bool(m.get("negRisk")), "fees_enabled": bool(m.get("feesEnabled")),
                             "y_yes": 1.0 if op[0] == 1 else (0.0 if op[0] == 0 else np.nan), "split": split})
        if i % 20 == 0:
            print(f"  {i}/{len(days)} rows {len(rows)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    for c in ("end", "closed_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="mixed")
    df["game_start"] = pd.to_datetime(df["game_start"], utc=True, errors="coerce", format="mixed")
    df["S"] = df[["end", "closed_time", "game_start"]].min(axis=1)
    if split == "sealed":
        df = df[(df.S >= SEAL0) & (df.S < SEAL1)]
    else:
        df = df[(df.S >= sm.DEV0) & (df.S < sm.DEV1)]
    if (df.cat == "sports").any():   # pre-game tape only: need a game start
        df = df[(df.cat != "sports") | df.game_start.notna()]
    if split == "devlow":   # AUDIT complement: the ex-post-volume frame excluded these (lifetime $1k..10k)
        df = df[df.volume.between(1000, VOL_MIN - 1e-9) & df.y_yes.notna()]
    else:
        df = df[(df.volume >= VOL_MIN) & df.y_yes.notna()]
    rng = np.random.default_rng(20261106)
    keep = []
    for cat, g in df.groupby("cat"):
        evs = g.event_id.unique()
        evs = rng.permutation(evs)[:PER_CAT_EVENTS[cat]]
        gg = g[g.event_id.isin(evs)].sort_values("volume", ascending=False).groupby("event_id").head(MAX_MKTS_PER_EVENT)
        keep.append(gg)
    df = pd.concat(keep)
    df["date"] = df.S.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df.to_parquet(OUT / f"markets_{split}.parquet", index=False)
    print(df.groupby("cat").agg(markets=("ticker", "size"), events=("event_id", "nunique"),
                                med_vol=("volume", "median")))


def tape(r):
    path = OUT / "tape" / f"{r.ticker}.parquet"
    if path.exists():
        return
    S = int(r.S.timestamp())
    t0 = S - TAPE_DAYS * 86400
    out, cursor, trunc = [], None, False
    for page in range(MAX_PAGES):
        params = {"condition": r.cond, "limit": 1000} if cursor is None else {"condition": r.cond, "limit": 1000, "cursor": cursor}
        d = sm.get(DATA_API, params, cache=False)
        if not d or not d.get("data"):
            break
        for x in d["data"]:
            ts = int(x["timestamp"])
            if t0 <= ts <= S:
                tok = str(x.get("token_id"))
                if tok not in (r.token0, r.token1):
                    continue
                is0 = tok == r.token0
                pr = float(x["price"])
                out.append((ts, pr if is0 else 1 - pr, float(x["size"]), float(x["size"]) * pr,
                            (x["side"] == "BUY") == is0, str(x.get("transaction_hash", ""))))
        oldest = min(int(x["timestamp"]) for x in d["data"])
        cursor = (d.get("pagination") or {}).get("next_cursor")
        if oldest <= t0 or not cursor:
            break
    else:
        trunc = True
    tp = pd.DataFrame(out, columns=["ts", "p", "size", "usd", "buy_yes", "tx"])
    tp = tp.sort_values("ts", kind="stable").reset_index(drop=True)
    tp.attrs["trunc"] = trunc
    tp["trunc"] = trunc
    tp.to_parquet(path, index=False)


def cmd_tape(split="dev"):
    m = pd.read_parquet(OUT / f"markets_{split}.parquet")
    rs = [r for r in m.itertuples() if not (OUT / "tape" / f"{r.ticker}.parquet").exists()]
    print(f"markets {len(m)}, to pull {len(rs)}", flush=True)
    for i in range(0, len(rs), 200):
        sm.pmap(tape, rs[i:i + 200], workers=6)
        print(f"  {min(i + 200, len(rs))}/{len(rs)}", flush=True)


def _fee(p, r):
    if not r.fees_enabled:
        return 0.0
    return (0.05 if r.cat == "sports" else 0.07) * p * (1 - p)


def _first(ts, by, i0, t_ref, want_yes, S):
    """first print after index i0, within ENTRY_MAX of t_ref and before S, whose taker bought YES == want_yes"""
    for q in range(i0 + 1, min(len(ts), i0 + 5000)):
        if ts[q] - t_ref > ENTRY_MAX or ts[q] >= S:
            return None
        if by[q] == want_yes:
            return q
    return None


def _trade(row, ts, p, by, size, S, dec, want_yes, r, pre=""):
    """position bought at dec on side want_yes: executable entry (an ask someone paid), marks, executable exits"""
    i_dec = np.searchsorted(ts, dec, side="right") - 1
    e = _first(ts, by, i_dec, dec, want_yes, S)
    if e is not None:
        pe = p[e] if want_yes else 1 - p[e]
        row.update({pre + "entry_ts": int(ts[e]), pre + "entry_p": pe, pre + "entry_size": float(size[e]),
                    pre + "fee": _fee(pe, r)})
    for h, hs in HORIZ.items():
        th = dec + hs
        if th >= S:
            row[pre + f"m_{h}"] = np.nan; row[pre + f"x_{h}"] = np.nan; continue
        ih = np.searchsorted(ts, th, side="right") - 1
        row[pre + f"m_{h}"] = p[ih]
        x = _first(ts, by, ih, th, not want_yes, S)        # someone bought the other side = our side's bid
        row[pre + f"x_{h}"] = np.nan if x is None else (p[x] if want_yes else 1 - p[x])


def jumps_for(r):
    path = OUT / "tape" / f"{r.ticker}.parquet"
    if not path.exists():
        return []
    tp = pd.read_parquet(path)
    if len(tp) < 20:
        return []
    ts = tp.ts.to_numpy(np.int64); p = tp.p.to_numpy(float); usd = tp.usd.to_numpy(float)
    by = tp.buy_yes.to_numpy(bool)
    S = int(r.S.timestamp())
    cov0 = int(ts[0])                      # tape coverage start (truncated markets start later)
    cum_usd = np.cumsum(usd)
    size = tp["size"].to_numpy(float)
    base = {"ticker": r.ticker, "event_id": r.event_id, "cat": r.cat, "date": r.date, "S": S, "y_yes": r.y_yes,
            "neg_risk": r.neg_risk, "trunc": bool(tp.trunc.iloc[0])}
    out = []
    for name, (W, X) in DEFS.items():
        last = -10 ** 12
        for j in range(len(ts)):
            t = ts[j]
            if t - cov0 < W + 86400 or t - last < COOLDOWN:   # need a full window + 1 day of history for baselines
                continue
            k = np.searchsorted(ts, t - W, side="right") - 1
            if k < 0:
                continue
            d = p[j] - p[k]
            if abs(d) < X or not (0.03 <= p[k] <= 0.97):
                continue
            last = t
            sgn = 1 if d > 0 else -1                           # jump direction in YES
            dec = t + LAT
            if dec >= S:
                continue
            i_dec = np.searchsorted(ts, dec, side="right") - 1
            w_usd = cum_usd[j] - cum_usd[k]                    # $ in the jump window (k, j]
            i_day = np.searchsorted(ts, t - W - 86400, side="right")
            base_usd = (cum_usd[k] - (cum_usd[i_day - 1] if i_day > 0 else 0.0)) / (86400 / W)  # avg $ per W, prior day
            i5 = np.searchsorted(ts, t + 300, side="right") - 1
            row = dict(base, kind="jump", **{"def": name, "t_jump": t, "decided_at": dec, "tts_h": (S - dec) / 3600,
                       "p0": p[k], "p1": p[j], "p_dec": p[i_dec], "sgn": sgn, "dp": d, "w_usd": w_usd, "n_w": j - k,
                       "base_usd_W": base_usd, "n_day": k - i_day + 1,
                       "vol_ratio": w_usd / base_usd if base_usd > 0 else np.inf,
                       "give5": -(p[i5] - p[j]) * sgn / abs(d)})
            want_yes = sgn < 0                                 # fade: buy NO after an up-jump, YES after a down-jump
            _trade(row, ts, p, by, size, S, dec, want_yes, r)
            if t + 300 + LAT < S:                              # same fade decided after watching 5 min (give5 known)
                _trade(row, ts, p, by, size, S, t + 300 + LAT, want_yes, r, pre="w5_")
            out.append(row)
    # placebo: 4 random prints per market (>= 1 day of tape), random side, same entry/exit machinery
    rng = np.random.default_rng(int(r.ticker) % (2 ** 32))
    ok = np.flatnonzero((ts - cov0 >= 86400) & (p >= 0.03) & (p <= 0.97) & (ts + LAT < S))
    for j in (rng.choice(ok, size=min(4, len(ok)), replace=False) if len(ok) else []):
        sgn = int(rng.choice([-1, 1]))
        dec = int(ts[j]) + LAT
        i_dec = np.searchsorted(ts, dec, side="right") - 1
        row = dict(base, kind="placebo", **{"def": "placebo", "t_jump": int(ts[j]), "decided_at": dec,
                   "tts_h": (S - dec) / 3600, "p0": p[j], "p1": p[j], "p_dec": p[i_dec], "sgn": sgn, "dp": 0.0})
        _trade(row, ts, p, by, size, S, dec, sgn < 0, r)
        out.append(row)
    return out


def cmd_jumps(split="dev"):
    m = pd.read_parquet(OUT / f"markets_{split}.parquet")
    rows = []
    for x in sm.pmap(jumps_for, list(m.itertuples()), workers=8):
        rows += x
    j = pd.DataFrame(rows)
    traps.check_epoch_seconds(j.decided_at, "decided_at")
    assert (j.entry_ts.dropna() > j.loc[j.entry_ts.notna(), "decided_at"]).all(), "entry before decision"
    w = j.w5_entry_ts.notna()
    assert (j.loc[w, "w5_entry_ts"] > j.loc[w, "t_jump"] + 300 + LAT).all(), "w5 entry before its decision"
    j["won_fade"] = np.where(j.sgn > 0, 1 - j.y_yes, j.y_yes)
    j.to_parquet(OUT / f"jumps_{split}.parquet", index=False)
    print(j.groupby(["def", "cat"]).size().unstack(0))
    print("with entry:", j.entry_p.notna().mean().round(3), "trunc markets:", j.trunc.mean().round(3))


def _roi(j, pre=""):
    c = j[pre + "entry_p"] + j[pre + "fee"]
    out = pd.DataFrame({"res": (j["won_fade"] - c) / c, "event_id": j.event_id})
    out["x24"] = (j[pre + "x_24h"] - j[pre + "fee"] - c) / c
    for h in HORIZ:   # mark drift toward the fade side, cents (includes bid/ask bounce; not a P&L)
        out[f"d_{h}"] = -j.sgn * (j[pre + f"m_{h}"] - j.p_dec) * 100
    out["d_res"] = -j.sgn * (j.y_yes - j.p_dec) * 100
    return out


def _boot(v, cl, B=2000, seed=0):
    ok = v.notna().to_numpy()
    v, cl = v.to_numpy(float)[ok], cl.to_numpy()[ok]
    if len(v) < 10:
        return (len(v), np.nan, np.nan, np.nan, 0)
    codes, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, v); n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(codes), (B, len(codes)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return (len(v), v.mean(), np.quantile(bs, .025), np.quantile(bs, .975), len(codes))


def cmd_describe():
    j = pd.read_parquet(OUT / "jumps_dev.parquet")
    assert set(pd.to_datetime(j.date).dt.year) <= {2025, 2026} and pd.to_datetime(j.date).max() < pd.Timestamp("2026-05-01")
    pl = j[j.kind == "placebo"]
    J = j[j.kind == "jump"]
    bins = [0, .1, .2, .35, .5, .65, .8, .9, 1.0001]
    plr = _roi(pl[pl.entry_p.notna()]).assign(b=pd.cut(pl.loc[pl.entry_p.notna(), "entry_p"], bins))
    pl_bin = plr.groupby("b", observed=False).res.mean()

    def cell(label, g, pre=""):
        g = g[g[pre + "entry_p"].notna()]
        if len(g) == 0:
            return None
        r = _roi(g, pre)
        b = pd.cut(g[pre + "entry_p"], bins)
        r["exc"] = r.res - b.map(pl_bin).astype(float).to_numpy()   # minus placebo ROI at the same entry price
        n, m, lo, hi, ne = _boot(r.res, r.event_id)
        _, mx, lox, hix, _ = _boot(r.exc, r.event_id)
        _, m24, lo24, hi24, _ = _boot(r.x24, r.event_id)
        d = {h: r[f"d_{h}"].mean() for h in ("1h", "6h", "24h", "res")}
        return {"cell": label, "n": n, "events": ne, "roi_res": m, "lo": lo, "hi": hi, "excess_vs_placebo": mx,
                "x_lo": lox, "x_hi": hix, "roi_x24": m24, "x24_lo": lo24, "x24_hi": hi24,
                **{f"drift_{k}_c": v for k, v in d.items()}, "med_entry": g[pre + "entry_p"].median(),
                "med_entry_usd": (g[pre + "entry_size"] * g[pre + "entry_p"]).median()}

    rows = [cell("PLACEBO random time/side", pl)]
    for dn, g in J.groupby("def"):
        rows.append(cell(f"{dn} ALL", g))
        rows.append(cell(f"{dn} entry rate {g.entry_p.notna().mean():.2f}", g.iloc[:0]) or {"cell": f"{dn} entry found {g.entry_p.notna().mean():.2f} of {len(g)}"})
        for lab, msk in {"thin n_w<=3": g.n_w <= 3, "busy n_w>=10": g.n_w >= 10,
                         "low $ <1k": g.w_usd < 1000, "heavy $ >=10k": g.w_usd >= 10000,
                         "vol_ratio<2": g.vol_ratio < 2, "vol_ratio>=5": g.vol_ratio >= 5,
                         "tts<6h": g.tts_h < 6, "tts 6-48h": g.tts_h.between(6, 48), "tts>=48h": g.tts_h >= 48,
                         "up": g.sgn > 0, "down": g.sgn < 0, "p0 .2-.8": g.p0.between(.2, .8),
                         "IMMATERIAL thin & tts>=48h": (g.n_w <= 3) & (g.tts_h >= 48),
                         "MATERIAL heavy & tts<6h": (g.w_usd >= 10000) & (g.tts_h < 6)}.items():
            rows.append(cell(f"{dn} {lab}", g[msk]))
        for c, gc in g.groupby("cat"):
            rows.append(cell(f"{dn} cat={c}", gc))
        gw = g[g.give5 >= 0.3]
        rows.append(cell(f"{dn} give5>=0.3 (entry after 5m)", gw, pre="w5_"))
        rows.append(cell(f"{dn} give5<0.3 (entry after 5m)", g[g.give5 < 0.3], pre="w5_"))
    # selection check: does conditioning on a fade-side print existing pick reverting jumps? (mark drift, all jumps)
    for dn, g in J.groupby("def"):
        for lab, gg in (("WITH entry print", g[g.entry_p.notna()]), ("NO entry print", g[g.entry_p.isna()])):
            dres = -gg.sgn * (gg.y_yes - gg.p_dec) * 100
            d24 = -gg.sgn * (gg.m_24h - gg.p_dec) * 100
            n, m, lo, hi, ne = _boot(dres, gg.event_id)
            _, m24, lo24, hi24, _ = _boot(d24, gg.event_id)
            print(f"SELECTION {dn} {lab}: n {n} events {ne} drift_to_res {m:.2f}c [{lo:.2f},{hi:.2f}] "
                  f"drift_24h {m24:.2f}c [{lo24:.2f},{hi24:.2f}]")
    t = pd.DataFrame([r for r in rows if r])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    cols = ["cell", "n", "events", "roi_res", "lo", "hi", "excess_vs_placebo", "x_lo", "x_hi", "roi_x24", "x24_lo",
            "x24_hi", "drift_1h_c", "drift_24h_c", "drift_res_c", "med_entry", "med_entry_usd"]
    print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
    t.to_csv(OUT / "describe_dev.csv", index=False)


def maker_fill(g, tp):
    """resting fade order posted at decided_at at the jump print's level: up-jump -> offer YES at a = p1 (= NO bid
    1 - a); down-jump -> YES bid at a = p1. Filled at the first print strictly THROUGH a after decided_at and before S
    (a YES print above a for an up-jump, below a for a down-jump); no cancel. Returns (fill_ts, fade-side price)."""
    ts = tp.ts.to_numpy(np.int64); p = tp.p.to_numpy(float)
    out = []
    for r in g.itertuples():
        a = round(r.p1, 3)
        m = (ts > r.decided_at) & (ts < r.S) & ((p > a + 1e-9) if r.sgn > 0 else (p < a - 1e-9))
        i = np.flatnonzero(m)
        out.append((int(ts[i[0]]) if len(i) else np.nan, (1 - a) if r.sgn > 0 else a))
    return out


FRAME_DEF = "W15_X05"


def cmd_frame(split="dev"):
    """edgelab input (dataset pm_jumps): one row per detected FRAME_DEF jump with ONLY pre-decision columns plus
    `result` (hidden by the vault) and p_close (last YES print <= S - 1 h, CLV mark); trades_{split}.parquet = the
    YES-equivalent tape (ts, p) of those markets from the first decision on (maker fills are modeled from it)."""
    j = pd.read_parquet(OUT / f"jumps_{split}.parquet")
    j = j[(j.kind == "jump") & (j["def"] == FRAME_DEF)].copy()
    keep = ["ticker", "event_id", "cat", "date", "t_jump", "decided_at", "S", "tts_h", "p0", "p1", "p_dec", "sgn",
            "dp", "w_usd", "n_w", "base_usd_W", "vol_ratio", "neg_risk", "y_yes"]
    f = j[keep].copy()
    f["result"] = np.where(f.y_yes == 1, "yes", "no")
    f = f.drop(columns="y_yes")
    trs, pcl = [], {}
    for tk, g in f.groupby("ticker"):
        tp = pd.read_parquet(OUT / "tape" / f"{tk}.parquet")
        S = int(g.S.iloc[0])
        i = np.searchsorted(tp.ts.to_numpy(), S - 3600, side="right") - 1
        pcl[tk] = float(tp.p.iloc[i]) if i >= 0 else np.nan
        x = tp[(tp.ts > g.decided_at.min()) & (tp.ts < S)][["ts", "p"]].assign(ticker=tk)
        trs.append(x)
    f["p_close"] = f.ticker.map(pcl)
    for c in ("t_jump", "decided_at", "S"):
        traps.check_epoch_seconds(f[c], c)
    tr = pd.concat(trs, ignore_index=True)
    traps.check_epoch_seconds(tr.ts, "trade ts")
    f["date"] = pd.to_datetime(f["date"])
    f.to_parquet(OUT / f"frame_{split}.parquet", index=False)
    tr.to_parquet(OUT / f"trades_{split}.parquet", index=False)
    print(f"frame {split}: {len(f)} jumps, {f.ticker.nunique()} markets; tape rows {len(tr)}")


def cmd_maker(split="dev"):
    assert split in ("dev", "devlow"), "descriptive maker check is dev-only"
    j = pd.read_parquet(OUT / f"jumps_{split}.parquet")
    J = j[j.kind.isin(["jump", "placebo"])].copy()
    J.loc[J.kind == "placebo", "p1"] = J.loc[J.kind == "placebo", "p_dec"]
    res = {}
    for tk, g in J.groupby("ticker"):
        tp = pd.read_parquet(OUT / "tape" / f"{tk}.parquet")
        for idx, v in zip(g.index, maker_fill(g, tp)):
            res[idx] = v
    J["mk_fill_ts"] = [res[i][0] for i in J.index]
    J["mk_p"] = [res[i][1] for i in J.index]
    J.to_parquet(OUT / f"maker_{split}.parquet", index=False)
    J = J[J.mk_fill_ts.notna() & J.mk_p.between(0.01, 0.99)]
    J["roi"] = (J.won_fade - J.mk_p) / J.mk_p
    J["pnl"] = J.won_fade - J.mk_p
    rows = []
    for dn, g in J.groupby("def"):
        cells = {"ALL": g.index == g.index}
        if dn != "placebo":
            cells.update({"thin n_w<=3": g.n_w <= 3, "low $ <1k": g.w_usd < 1000, "heavy $>=10k": g.w_usd >= 10000,
                          "tts>=48h": g.tts_h >= 48, "tts<6h": g.tts_h < 6, "up": g.sgn > 0, "down": g.sgn < 0,
                          "IMMATERIAL low$ & tts>=48h": (g.w_usd < 1000) & (g.tts_h >= 48),
                          "politics_world": g.cat == "politics_world", "finance_econ": g.cat == "finance_econ",
                          "culture_tech": g.cat == "culture_tech", "mentions": g.cat == "mentions",
                          "sports": g.cat == "sports"})
        for lab, msk in cells.items():
            gg = g[msk]
            n, m, lo, hi, ne = _boot(gg.roi, gg.event_id)
            _, mp, lop, hip, _ = _boot(gg.pnl * 100, gg.event_id)
            day = gg.groupby("date").pnl.sum()
            rows.append({"cell": f"{dn} {lab}", "n": n, "events": ne, "roi": m, "lo": lo, "hi": hi, "pnl_c": mp,
                         "pnl_lo": lop, "pnl_hi": hip, "top_day_share": day.max() / day.sum() if day.sum() > 0 else np.nan,
                         "med_p": gg.mk_p.median(), "fill_lag_h": ((gg.mk_fill_ts - gg.decided_at) / 3600).median()})
    allj = j[j.kind == "jump"].groupby("def").size()
    print("posted jumps:", allj.to_dict(), " fills:", J[J.kind == "jump"].groupby("def").size().to_dict())
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    t.to_csv(OUT / f"maker_{split}.csv", index=False)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"enum": cmd_enum, "tape": cmd_tape, "jumps": cmd_jumps, "describe": cmd_describe, "maker": cmd_maker, "frame": cmd_frame}[cmd](*a)
