"""UFCStats (via github Greco1899/scrape_ufc_stats CSVs in dl/) -> fight table + point-in-time fighter method features.
Outcome-only data; no Kalshi prices. Method classes follow Kalshi: KO = KO/TKO, doctor stoppage, DQ; SUB; DEC = any decision."""
import re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).split()
def nkey(s): return " ".join(norm(s))
def fights():
    r = pd.read_csv(OUT / "dl/ufc_fight_results.csv"); e = pd.read_csv(OUT / "dl/ufc_event_details.csv")
    r["EVENT"] = r.EVENT.str.strip(); e["EVENT"] = e.EVENT.str.strip()
    r = r.merge(e[["EVENT", "DATE"]], on="EVENT", how="inner")
    r["date"] = pd.to_datetime(r.DATE, format="%B %d, %Y")
    ab = r.BOUT.str.split(" vs. ", n=1, expand=True)
    r["f1"] = ab[0].str.strip(); r["f2"] = ab[1].str.strip()
    m = r.METHOD.str.strip()
    r["mclass"] = np.select([m.str.startswith("Decision"), m.isin(["KO/TKO", "TKO - Doctor's Stoppage", "DQ"]), m == "Submission"], ["DEC", "KO", "SUB"], "OTHER")
    o = r.OUTCOME.str.strip()
    r["winner"] = np.select([o == "W/L", o == "L/W"], [r.f1, r.f2], None)
    r["loser"] = np.select([o == "W/L", o == "L/W"], [r.f2, r.f1], None)
    r["draw"] = o == "D/D"
    r["sched5"] = r["TIME FORMAT"].str.startswith("5 Rnd")
    r["women"] = r.WEIGHTCLASS.str.contains("Women")
    wc = r.WEIGHTCLASS.str.replace("UFC ", "").str.replace(" Title", "").str.replace(" Bout", "").str.replace("Interim ", "").str.strip()
    r["wclass"] = wc
    r["round"] = pd.to_numeric(r.ROUND, errors="coerce")
    r["fid"] = r.URL
    return r.sort_values("date").reset_index(drop=True)
def pit_features(fr, a, b, date):
    """Counts for fighter a (as potential winner) and b (opponent) from fights strictly before date."""
    h = fr[fr.date < date]
    out = {}
    for tag, name in (("a", a), ("b", b)):
        k = nkey(name)
        w = h[h.winner.map(nkey) == k]; l = h[h.loser.map(nkey) == k]
        n = ((h.f1.map(nkey) == k) | (h.f2.map(nkey) == k)).sum()
        out[f"{tag}_n"] = int(n)
        for mc in ("KO", "SUB", "DEC"):
            out[f"{tag}_w{mc}"] = int((w.mclass == mc).sum()); out[f"{tag}_l{mc}"] = int((l.mclass == mc).sum())
    return out

ALPHA = 3.0
def long_table(fr):
    rows = []
    for side, opp in (("f1", "f2"), ("f2", "f1")):
        x = fr[["date", "fid", side, opp, "winner", "loser", "mclass", "draw"]].copy()
        x["name"] = x[side]; x["k"] = x[side].map(nkey)
        x["won"] = x.winner == x[side]; x["lost"] = x.loser == x[side]
        rows.append(x[["date", "fid", "k", "won", "lost", "mclass"]])
    L = pd.concat(rows).sort_values(["k", "date"]).reset_index(drop=True)
    for mc in ("KO", "SUB", "DEC"):
        L["w" + mc] = (L.won & (L.mclass == mc)).astype(int); L["l" + mc] = (L.lost & (L.mclass == mc)).astype(int)
    L["n"] = 1
    cols = ["n"] + [p + mc for p in "wl" for mc in ("KO", "SUB", "DEC")]
    L[[c + "_cum" for c in cols]] = L.groupby("k")[cols].cumsum()
    return L, cols
def asof(L, cols, name, date):
    """Inclusive cumulative counts from fights strictly before date (0 if none)."""
    x = L[(L.k == nkey(name)) & (L.date < date)]
    if not len(x): return {c: 0 for c in cols}
    return {c: int(x[c + "_cum"].iloc[-1]) for c in cols}
def base_rates(fr, before):
    h = fr[(fr.date < before) & fr.winner.notna() & fr.mclass.isin(["KO", "SUB", "DEC"])]
    return h.groupby("wclass").mclass.value_counts(normalize=True).unstack().fillna(0), h.mclass.value_counts(normalize=True)
def feats(a, b, wc, sched5, women, br, brall):
    """a = would-be winner counts, b = opponent counts. Returns shrunk shares + base rates."""
    p = br.loc[wc] if wc in br.index else brall
    aw = a["wKO"] + a["wSUB"] + a["wDEC"]
    f = {}
    for mc in ("KO", "SUB", "DEC"):
        f["a_w" + mc] = (a["w" + mc] + ALPHA * p[mc]) / (aw + ALPHA)
        f["b_l" + mc] = (b["l" + mc] + ALPHA * p[mc] * 0.5) / (b["n"] + ALPHA)   # opponent finished-by rate per fight
        f["base_" + mc] = p[mc]
    f["sched5"] = float(sched5); f["women"] = float(women)
    f["a_ln"] = np.log1p(a["n"]); f["b_ln"] = np.log1p(b["n"])
    return f
FEATS = ["a_wKO", "a_wSUB", "a_wDEC", "b_lKO", "b_lSUB", "b_lDEC", "base_KO", "base_SUB", "base_DEC", "sched5", "women", "a_ln", "b_ln"]
def design(X):
    Z = X[FEATS].copy()
    for c in ["a_wKO", "a_wSUB", "a_wDEC", "b_lKO", "b_lSUB", "b_lDEC", "base_KO", "base_SUB", "base_DEC"]:
        Z[c] = np.log(Z[c].clip(1e-3, 1))
    return Z.values
def train_rows(fr, L, cols, start, end):
    br, brall = base_rates(fr, end)
    h = fr[(fr.date >= start) & (fr.date < end) & fr.winner.notna() & fr.mclass.isin(["KO", "SUB", "DEC"])]
    rows = []
    for r in h.itertuples():
        a = asof(L, cols, r.winner, r.date); b = asof(L, cols, r.loser, r.date)
        f = feats(a, b, r.wclass, r.sched5, r.women, br, brall); f["y"] = r.mclass; f["date"] = r.date
        rows.append(f)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    fr = fights(); L, cols = long_table(fr)
    tr = train_rows(fr, L, cols, pd.Timestamp("2012-01-01"), pd.Timestamp("2026-01-01"))
    tr.to_parquet(OUT / "ufc_train.parquet")
    # internal time check: fit < 2024, evaluate 2024-2025 log loss vs weight-class base rate
    a = tr[tr.date < "2024-01-01"]; b = tr[tr.date >= "2024-01-01"]
    m = LogisticRegression(max_iter=2000, C=1.0).fit(design(a), a.y)
    P = m.predict_proba(design(b)); cls = list(m.classes_)
    yi = b.y.map({c: i for i, c in enumerate(cls)}).values
    ll = -np.mean(np.log(P[np.arange(len(b)), yi]))
    Bp = b[["base_" + c for c in cls]].values; Bp = Bp / Bp.sum(1, keepdims=True)
    llb = -np.mean(np.log(Bp[np.arange(len(b)), yi]))
    print("train rows", len(tr), "fit<2024", len(a), "test 2024-25", len(b), "classes", cls)
    print(f"method|win log loss 2024-25: model {ll:.4f} vs weight-class base {llb:.4f}")
