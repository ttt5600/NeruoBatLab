"""U1 DEV scoring (prereg notes/ufc_tsa.md). DEV only; asserts no sealed card is touched."""
import sys
from math import ceil
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from quotes import ufc_quotes
SEED = 20261119; PNC = 0.0175; THR = 0.04
def fee(c): return ceil(round(100 * 0.07 * 10 * c * (1 - c), 9)) / 100 / 10
def build():
    m = pd.read_parquet(OUT / "ufc_markets.parquet"); q = ufc_quotes()
    d = m.merge(q, on="ticker", how="inner", validate="one_to_one")
    d = d[d.split == "dev"]; assert (d.card < pd.Timestamp("2026-08-01")).all()
    d["y"] = (d.result == "yes").astype(int)
    fights = pd.read_parquet(OUT / "ufc_fights.parquet"); fights = fights[fights.fid.notna() & (fights.split == "dev")]
    P = pd.read_parquet(OUT / "ufc_probs.parquet"); P = P[P.split == "dev"]
    # FIGHT winner quotes: map fighter code -> mid
    fq = d[d.series == "KXUFCFIGHT"].copy(); fq["code"] = fq.ticker.str.split("-").str[2]
    fq = fq[(fq.bid > 0) & (fq.ask < 1) & fq.bid.notna() & fq.ask.notna()]
    fq["mid"] = (fq.bid + fq.ask) / 2
    mv = d[d.series == "KXUFCMOV"].copy()
    sfx = mv.ticker.str.split("-").str[2]
    mv = mv[~sfx.str.startswith("DRAW")].copy(); sfx = mv.ticker.str.split("-").str[2]
    mv["meth"] = sfx.str.extract(r"(KOTKODQ|SUB|DEC)$")[0].map({"KOTKODQ": "KO", "SUB": "SUB", "DEC": "DEC"})
    mv["code"] = sfx.str.replace(r"(KOTKODQ|SUB|DEC)$", "", regex=True)
    anyy = d[d.series == "KXUFCMOV"].groupby("event_ticker").y.sum()
    mv = mv[mv.event_ticker.map(anyy) > 0]
    mv = mv[mv.event_ticker.isin(fights.event_ticker)]
    rows = []
    for ev, g in mv.groupby("event_ticker"):
        fight = g.fight.iloc[0]
        f = fq[fq.fight == fight]
        if f.code.nunique() != 2 or not set(g.code) <= set(f.code): continue
        mids = f.set_index("code").mid; s = mids.sum()
        for r in g.itertuples():
            pr = P[(P.event_ticker == ev) & (P.code == r.code)]
            if len(pr) != 1: continue
            pr = pr.iloc[0]
            rows.append(dict(event_ticker=ev, card=r.card, ticker=r.ticker, code=r.code, meth=r.meth, y=r.y, bid=r.bid, ask=r.ask,
                             pwin=mids[r.code] / s * (1 - PNC), m=pr["m_" + r.meth], b=pr["b_" + r.meth], fid=pr.name))
    return pd.DataFrame(rows)
def bets(X, col):
    X = X.copy(); X["q"] = X.pwin * X[col]
    out = []
    for ev, g in X.groupby("event_ticker"):
        best = None
        for r in g.itertuples():
            for side, cost, ok, qs, win in (("YES", r.ask, pd.notna(r.ask) and r.ask < 1 and r.ask > 0, r.q, r.y == 1),
                                            ("NO", 1 - r.bid if pd.notna(r.bid) else np.nan, pd.notna(r.bid) and r.bid > 0, 1 - r.q, r.y == 0)):
                if not ok or not (0.03 <= cost <= 0.97): continue
                fe = fee(cost); e = qs - cost - fe
                if e >= THR and (best is None or e > best["edge"]):
                    best = dict(event_ticker=ev, card=r.card, ticker=r.ticker, side=side, cost=cost, fee=fe, edge=e, q=qs, win=int(win), meth=r.meth)
        if best: out.append(best)
    B = pd.DataFrame(out)
    if len(B): B["pnl"] = B.win - B.cost - B.fee; B["stake"] = B.cost + B.fee
    return B
def roi(B): return B.pnl.sum() / B.stake.sum() if len(B) else np.nan
def card_boot(Bs, cards, nb=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    agg = [B.groupby("card")[["pnl", "stake"]].sum().reindex(cards, fill_value=0).values for B in Bs]
    out = np.empty((nb, len(Bs)))
    for i in range(nb):
        idx = rng.integers(0, len(cards), len(cards))
        for k, a in enumerate(agg):
            s = a[idx].sum(0); out[i, k] = s[0] / s[1] if s[1] > 0 else np.nan
    return out
if __name__ == "__main__":
    X = build(); X.to_parquet(OUT / "u1_dev_frame.parquet")
    print("dev rungs", len(X), "fights", X.event_ticker.nunique(), "cards", X.card.nunique())
    BU = bets(X, "m"); BP = bets(X, "b")
    cards = np.array(sorted(X.card.unique()))
    bt = card_boot([BU, BP], cards)
    for nm, B, k in (("U1", BU, 0), ("P0", BP, 1)):
        v = bt[:, k]; v = v[~np.isnan(v)]
        print(f"{nm}: bets {len(B)} cards {B.card.nunique()} wins {B.win.sum()} ROI {roi(B):+.4f} CI [{np.percentile(v,2.5):+.4f},{np.percentile(v,97.5):+.4f}] p {np.mean(v<=0):.4f}")
    dlt = bt[:, 0] - bt[:, 1]; dlt = dlt[~np.isnan(dlt)]
    print(f"U1-P0 diff point {roi(BU)-roi(BP):+.4f} CI [{np.percentile(dlt,2.5):+.4f},{np.percentile(dlt,97.5):+.4f}]")
    h = BU.sort_values("card"); cut = cards[len(cards)//2]
    print("halves U1", round(roi(h[h.card < cut]), 4), round(roi(h[h.card >= cut]), 4))
    print("U1 by side/method", BU.groupby(["side", "meth"]).agg(n=("pnl", "size"), roi=("pnl", "sum"), stake=("stake", "sum")).assign(roi=lambda z: z.roi / z.stake).round(3).to_string())
    # shuffled-model placebo: permute fighter-specific m columns across fights (keep pwin)
    rng = np.random.default_rng(SEED); fid = X[["event_ticker", "code"]].drop_duplicates()
    sh = []
    mm = X.pivot_table(index=["event_ticker", "code"], columns="meth", values="m")
    for i in range(200):
        perm = mm.sample(frac=1, random_state=int(rng.integers(1e9))).values
        mp = pd.DataFrame(perm, index=mm.index, columns=mm.columns).stack().rename("ms").reset_index()
        Xs = X.merge(mp, on=["event_ticker", "code", "meth"], how="left", validate="one_to_one")
        sh.append(roi(bets(Xs, "ms")))
    sh = np.array(sh)
    print(f"PS shuffled: mean {np.nanmean(sh):+.4f} p95 {np.nanpercentile(sh,95):+.4f}; U1 rank pct {np.mean(sh < roi(BU)):.3f}")
    # secondary: Brier of q vs mid on two-sided rungs
    V = X[(X.bid > 0) & (X.ask < 1)].copy(); V["mid"] = (V.bid + V.ask) / 2
    for col in ("m", "b"):
        V["d"] = (V.pwin * V[col] - V.y) ** 2 - (V.mid - V.y) ** 2
        g = V.groupby("card").d.agg(["sum", "size"]).reindex(cards, fill_value=0).values
        r2 = np.random.default_rng(SEED); bb = []
        for i in range(5000):
            s = g[r2.integers(0, len(cards), len(cards))].sum(0); bb.append(s[0] / s[1])
        print(f"Brier(model {col}) - Brier(mid): {V.d.mean():+.5f} CI [{np.percentile(bb,2.5):+.5f},{np.percentile(bb,97.5):+.5f}] n {len(V)}")
    BU.to_csv(OUT / "u1_bets_dev.csv", index=False); BP.to_csv(OUT / "p0_bets_dev.csv", index=False)
