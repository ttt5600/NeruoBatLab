"""Online Elo predictions for every KXCS2GAME event at T = start - 60 min, using only results settled before T.
Out: preds.parquet (one row per event: team markets A/B, start, T, per-model pA, rated-result counts).
Usage: elo.py [--placebo SEED]   (placebo: permute team ids of the RATING table at prediction time)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
KS = [16, 24, 32, 48]


def team(s):
    try:
        return json.loads(s).get("esports_competitor")
    except Exception:
        return None


def load():
    g = pd.read_parquet(OUT / "markets_KXCS2GAME.parquet")
    m = pd.read_parquet(OUT / "markets_KXCS2MAP.parquet")
    for d in (g, m):
        d["team"] = d.custom_strike.map(team)
        d["settle"] = pd.to_datetime(d.settlement_ts, utc=True, format="ISO8601")
    ev = g.event_ticker.str.split("-").str[1]
    st = pd.to_datetime(ev.str[:11], format="%y%b%d%H%M", errors="coerce").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    exp = pd.to_datetime(g.expected_expiration_time, utc=True, format="ISO8601") - pd.Timedelta(hours=4)
    g["timed"] = st.notna()
    g["start"] = st.fillna(exp)
    m["gev"] = m.event_ticker.str.rsplit("-", n=1).str[0].str.replace("KXCS2MAP", "KXCS2GAME")
    return g, m


def results(d):
    """event -> (settle time, winner uuid, loser uuid); binary yes/no only."""
    out = []
    for e, x in d.groupby("event_ticker"):
        if len(x) != 2 or x.team.isna().any() or not set(x.result) == {"yes", "no"}:
            continue
        w = x[x.result == "yes"].iloc[0]; l = x[x.result == "no"].iloc[0]
        out.append((x.settle.max(), w.team, l.team, e))
    return pd.DataFrame(out, columns=["t", "w", "l", "ev"])


def main():
    placebo = int(sys.argv[sys.argv.index("--placebo") + 1]) if "--placebo" in sys.argv else None
    g, m = load()
    gr, mr = results(g), results(m)
    mr["gev"] = mr.ev.str.rsplit("-", n=1).str[0].str.replace("KXCS2MAP", "KXCS2GAME")
    gr_nomap = gr[~gr.ev.isin(set(mr.gev))]
    streams = {"series": gr.sort_values("t"), "map": pd.concat([mr, gr_nomap]).sort_values("t")}
    # queries
    q = []
    for e, x in g.groupby("event_ticker"):
        if len(x) != 2 or x.team.isna().any():
            continue
        x = x.sort_values("ticker")
        a, b = x.iloc[0], x.iloc[1]
        q.append(dict(ev=e, ta=a.ticker, tb=b.ticker, A=a.team, B=b.team, start=a.start, timed=bool(a.timed),
                      res_a=a.result, sv_a=float(a.settlement_value_dollars or "nan"),
                      res_b=b.result, sv_b=float(b.settlement_value_dollars or "nan"),
                      vol=float(pd.to_numeric(x.volume_fp, errors="coerce").sum())))
    q = pd.DataFrame(q)
    q["T"] = q.start - pd.Timedelta(minutes=60)
    q = q.sort_values("T").reset_index(drop=True)
    rng = np.random.default_rng(placebo) if placebo is not None else None
    for name, s in streams.items():
        for K in KS:
            R, N = {}, {}
            ti = 0
            sv = s[["t", "w", "l"]].to_numpy()
            pa, na, nb = np.empty(len(q)), np.empty(len(q)), np.empty(len(q))
            for i, r in enumerate(q.itertuples()):
                while ti < len(sv) and sv[ti][0] < r.T:
                    _, w, l = sv[ti]
                    rw, rl = R.get(w, 1500.0), R.get(l, 1500.0)
                    ew = 1 / (1 + 10 ** ((rl - rw) / 400))
                    kw = K * (1.5 if N.get(w, 0) < 10 else 1); kl = K * (1.5 if N.get(l, 0) < 10 else 1)
                    R[w] = rw + kw * (1 - ew); R[l] = rl - kl * (1 - ew)
                    N[w] = N.get(w, 0) + 1; N[l] = N.get(l, 0) + 1
                    ti += 1
                A, B = r.A, r.B
                if rng is not None and R:
                    keys = list(R.keys())
                    A, B = keys[rng.integers(len(keys))], keys[rng.integers(len(keys))]
                ra, rb = R.get(A, 1500.0), R.get(B, 1500.0)
                p = 1 / (1 + 10 ** ((rb - ra) / 400))
                if name == "map":
                    p = p * p * (3 - 2 * p)
                pa[i] = p; na[i] = N.get(r.A, 0); nb[i] = N.get(r.B, 0)
            q[f"p_{name}_{K}"] = pa
            q[f"n_{name}"] = np.minimum(na, nb)
    tag = "" if placebo is None else f"_placebo{placebo}"
    q.to_parquet(OUT / f"preds{tag}.parquet", index=False)
    print("events", len(q), "results series", len(streams["series"]), "map", len(streams["map"]))


if __name__ == "__main__":
    main()
