"""A3 match filter (fixed before outcomes): end-date gap <= 14 d AND every non-generic PM US question token appears in
the global event title or market question."""
import sys, pandas as pd
sys.path.insert(0, "scripts")
from pm_full_census_match import toks
GENERIC = set("winner leader round third first second final champion championship election season stage elimination "
              "decision next who which top week this house nominee runner up outstanding best tournament presented".split())
def a3(m, gev):
    t = dict(zip(gev.eid, gev.title))
    keep = []
    for r in m.itertuples():
        need = {x for x in toks(r.question) if x not in GENERIC and not x.isdigit()}
        have = toks(t.get(r.g_eid, "")) | toks(r.g_q)
        keep.append(need <= have and r.end_gap_d <= 14)
    return m[pd.Series(keep, index=m.index)]
if __name__ == "__main__":
    m = pd.read_parquet("results/pm_full_census/xvus_match.parquet"); gev = pd.read_parquet("results/pm_full_census/xvus_gev.parquet")
    k = a3(m, gev); print(len(m), "->", len(k), "events", k.question.nunique(), k.category.value_counts().to_dict())
    print(k.sample(30, random_state=5)[["question", "g_q", "end_gap_d"]].to_string(max_colwidth=55))
