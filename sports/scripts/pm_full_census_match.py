"""pm_full_census: match PM US markets to global Gamma markets (event-title token Jaccard + outcome-name containment)."""
import re, collections, pandas as pd
STOP = set("the a an of in on by to will be who what which is for and or at end 2026 2027 2028 before after than".split())
def norm(s): return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower().replace("&", " and ")).split()
def toks(s): return {t for t in norm(s) if t not in STOP and len(t) > 1}
def nstr(s): return " ".join(norm(s))

def match(us: pd.DataFrame, gev: pd.DataFrame, gmk: pd.DataFrame, min_j=0.5):
    """us: slug, question, title. gev: eid, title. gmk: eid, mid, q, (+ any price cols). Returns us rows + gmk cols."""
    idx = collections.defaultdict(set); gt = {}
    for eid, t in zip(gev.eid, gev.title):
        gt[eid] = toks(t)
        for x in gt[eid]: idx[x].add(eid)
    by_e = {k: v for k, v in gmk.groupby("eid")}
    out = []
    for r in us.itertuples():
        qt = toks(r.question)
        if not qt: continue
        cand = collections.Counter(e for x in qt for e in idx.get(x, ()))
        best = []
        for e, c in cand.most_common(30):
            j = len(qt & gt[e]) / len(qt | gt[e])
            if j >= min_j: best.append((j, e))
        if not best: continue
        best.sort(reverse=True)
        oname = nstr(r.title)
        hits = []
        for j, e in best[:5]:
            g = by_e.get(e)
            if g is None: continue
            if oname in ("yes",) and len(g) == 1:
                hits.append((j, g.iloc[0])); continue
            if len(oname) < 3: continue
            m = g[g.q.map(lambda q: f" {oname} " in f" {nstr(q)} ")]
            if len(m) == 1: hits.append((j, m.iloc[0]))
        if not hits: continue
        top = max(h[0] for h in hits)
        hits = [h for h in hits if h[0] == top]
        if len(hits) != 1: continue  # ambiguous -> drop
        j, g = hits[0]
        d = r._asdict(); d.update({f"g_{k}": v for k, v in g.items()}); d["jacc"] = j
        out.append(d)
    return pd.DataFrame(out)
