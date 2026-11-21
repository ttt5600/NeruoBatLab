"""Rank series on the TAKER-REACHABLE tail pool (lead 11-21) and write the top-N markdown table + top-5 events.
Inputs: pool_table.csv, top_events.json (build_table.py), tested_map.py. Output: top40.md, top40.json."""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tested_map import label
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
R = list(csv.DictReader(open(HERE / "pool_table.csv")))
TOP = json.load(open(HERE / "top_events.json"))
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
INTL = ("UEFANL", "INTLFRIENDLY", "CONCACAFNL", "AFCON", "WCQ")
rows = []
for r in R:
    t, m = fl(r["tail30"]), fl(r["mirror30"])
    phi = fl(r["phi_tail"])
    reach = fl(r["reach_tail30"])
    st, cite = label(r["series"])
    seasonal = ""
    if any(k in r["series"] for k in INTL): seasonal = "intl window only (x30/7 overstates)"
    if r["src"] == "60d" and r["series"].startswith(("KXMLB", "KXWNBA")): seasonal = "season ended"
    rows.append(dict(series=r["series"], cat=r["category"], src=r["src"], reach=reach, phi=phi, tail=t, mirror=m,
                     reach_m=fl(r["reach_mirror30"]), phim=fl(r["phi_mirror"]), net_blind=(t - m) if t is not None and m is not None else None,
                     ev=fl(r["events30"]), top1=fl(r["top1_share_tail"]), rec=r["recurring"] == "True",
                     far2s=fl(r["share_far_two_sided"]), fspr=fl(r["med_far_spread"]), nfar=int(float(r["n_far"] or 0)),
                     fee=r["fee_type"], lb=r["lower_bound"] == "True", sampled=r["sampled"] == "True", status=st, cite=cite,
                     seasonal=seasonal, title=r["title"]))
# rank key: reach where phi measured; else tail x median phi of measured series (flagged 'phi~')
phis = sorted(x["phi"] for x in rows if x["phi"] is not None)
med_phi = phis[len(phis) // 2] if phis else None
for x in rows:
    x["key"] = x["reach"] if x["reach"] is not None else (x["tail"] * med_phi if med_phi and x["tail"] is not None else 0)
    x["phi_est"] = x["phi"] is None
rows.sort(key=lambda x: -x["key"])
def M(v): return "n/a" if v is None else (f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k")
lines = [f"median measured phi_tail = {med_phi} (used, flagged '~', where a series has no tape sample)", "",
         "| # | series | cat | reach TAIL $/30d | phi | raw TAIL | MIRROR | ev/30d | top1 | recurring | far 2-sided | far spr | fee | TESTED? |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, x in enumerate(rows[:N], 1):
    flags = ("~" if x["phi_est"] else "") + (" LB" if x["lb"] else "") + (" 60d" if x["src"] == "60d" else "")
    lines.append(f"| {i} | {x['series']} | {x['cat'][:8]} | {M(x['key'])}{flags} | {x['phi'] if x['phi'] is not None else '~'} | "
                 f"{M(x['tail'])} | {M(x['mirror'])} | {x['ev']:.0f} | {x['top1']} | {'Y' if x['rec'] else 'N'}{' ('+x['seasonal']+')' if x['seasonal'] else ''} | "
                 f"{'n/a' if x['far2s'] is None else round(x['far2s'], 2)} | {'n/a' if x['fspr'] is None else x['fspr']} | "
                 f"{x['fee'].replace('quadratic', 'q').replace('_with_maker_fees', '+maker')} | {x['status']}: {x['cite']} |")
lines += ["", "Top-5 TAIL events (event, $/30d-scaled contribution, share of the series' measured TAIL):"]
for x in rows[:N]:
    ev = TOP.get(x["series"], {}).get("tail", [])
    lines.append(f"- {x['series']}: " + "; ".join(f"{e} {M(v)} ({s})" for e, v, s in ev))
open(HERE / "top40.md", "w").write("\n".join(lines) + "\n")
json.dump(rows[:max(N, 80)], open(HERE / "top40.json", "w"), indent=1)
print("\n".join(lines[:N + 4]))
