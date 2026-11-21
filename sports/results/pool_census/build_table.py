"""Aggregate candle pools into the series table (week sample + low-frequency 60-day sample).
Inputs: week_markets.jsonl + week_plan.json (x 30/7), lowfreq_markets.jsonl + lowfreq_plan.json (x 30/60), series_all.json.
Output: pool_table.csv (all measured series), top_events.json (top-5 TAIL and MIRROR events per series)."""
import csv, json, collections, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
meta = {s["ticker"]: s for s in json.load(open(HERE / "series_all.json"))}

def hmed(h):
    tot = sum(h.values()); c = 0
    for k in sorted(h):
        c += h[k]
        if c >= tot / 2: return k / 100
    return None

def load(mfile, pfile, f30, src):
    if not (HERE / mfile).exists(): return {}
    plan = json.load(open(HERE / pfile)) if (HERE / pfile).exists() else {}
    S = collections.defaultdict(lambda: dict(mirror_best=0.0, tail=0.0, mirror=0.0, full=0.0, tail_rng=0.0, mirror_rng=0.0, contracts=0.0,
                                             ev_tail=collections.Counter(), ev_mirror=collections.Counter(), spreads=[],
                                             tail_spreads=[], markets=0, missing=0, events=set(), far=0, far2=0,
                                             far_spreads=[], far_tail=0.0, far2_tail=0.0, far_c=0, far2_c=0,
                                             far_hist=collections.Counter()))
    for line in open(HERE / mfile):
        x = json.loads(line); s = x["series"]
        w = plan.get(s, {}).get("w", {}).get(x["event"], 1.0) if plan.get(s, {}).get("w") else 1.0
        a = S[s]
        for k in ("tail", "mirror", "full", "tail_rng", "mirror_rng", "contracts"): a[k] += x[k] * w * f30
        a["mirror_best"] += (x["mirror"] if x.get("per") == 60 else x["mirror_rng"]) * w * f30   # validate_hybrid.json
        a["ev_tail"][x["event"]] += x["tail"] * f30          # unweighted: the event's own 30d-scaled $ contribution
        a["ev_mirror"][x["event"]] += x["mirror"] * f30
        a["markets"] += 1; a["missing"] += int(x.get("missing", False)); a["events"].add(x["event"])
        a["far_c"] += x.get("far_c", 0); a["far2_c"] += x.get("far2_c", 0)
        a["far_hist"].update({int(k): v for k, v in (x.get("far_spr_hist") or {}).items()})
        if x.get("far0"):
            a["far"] += 1; a["far_tail"] += x["tail"] * w * f30
            if x.get("two_sided0"):
                a["far2"] += 1; a["far2_tail"] += x["tail"] * w * f30
                if x.get("spread0") is not None: a["far_spreads"].append(x["spread0"])
        if x.get("spread0") is not None:
            a["spreads"].append(x["spread0"])
            if x.get("mid0") is not None and (x["mid0"] <= 0.30 or x["mid0"] >= 0.70): a["tail_spreads"].append(x["spread0"])
    out = {}
    for s, a in S.items():
        p = plan.get(s, {})
        n_ev = p.get("n_events", len(a["events"]))
        out[s] = dict(series=s, src=src, tail30=a["tail"], mirror30=a["mirror_best"], mirror_mean30=a["mirror"], full30=a["full"], tail_rng30=a["tail_rng"],
                      mirror_rng30=a["mirror_rng"], contracts30=a["contracts"], events30=n_ev * f30, markets_measured=a["markets"],
                      missing=a["missing"], sampled=bool(p.get("w")), n_events_measured=len(a["events"]),
                      med_spread=statistics.median(a["spreads"]) if a["spreads"] else None,
                      med_spread_tail=statistics.median(a["tail_spreads"]) if a["tail_spreads"] else None,
                      lower_bound=bool(p.get("w")) and all(v == 1.0 for v in p["w"].values()) and len(p["w"]) < n_ev,
                      n_spread=len(a["spreads"]), ev_tail=a["ev_tail"], ev_mirror=a["ev_mirror"],
                      n_far0=a["far"], share_far0_two_sided=(a["far2"] / a["far"]) if a["far"] else None,
                      n_far=a["far_c"], share_far_two_sided=(a["far2_c"] / a["far_c"]) if a["far_c"] else None,
                      med_far_spread=hmed(a["far_hist"]))
    return out

PHI = {}
for fn in ("phi_week.json", "phi_week2.json", "phi_low.json"):
    if (HERE / fn).exists(): PHI.update(json.load(open(HERE / fn)))
week = load("week_markets.jsonl", "week_plan.json", 30.0 / 7.0, "week")
week2 = load("week2_markets.jsonl", "week2_plan.json", 30.0 / 7.0, "week")
for k, v in week2.items():
    if k not in week: week[k] = v
low = load("lowfreq_markets.jsonl", "lowfreq_plan.json", 30.0 / 60.0, "60d")
rows = dict(low); rows.update(week)   # a series measured in the week sample uses the week numbers
top = {}
for s, r in rows.items():
    m = meta.get(s, {})
    r["category"] = m.get("category"); r["fee_type"] = m.get("fee_type"); r["fee_mult"] = m.get("fee_multiplier")
    r["frequency"] = m.get("frequency"); r["title"] = m.get("title")
    for kind in ("tail", "mirror"):
        ev = r.pop("ev_" + kind); tot = sum(ev.values())
        lst = [(e, round(v, 2), round(v / tot, 3) if tot else None) for e, v in ev.most_common(5)]
        top.setdefault(s, {})[kind] = lst
        r["top1_share_" + kind] = lst[0][2] if lst else None
    ph = PHI.get(s, {})
    r["phi_tail"] = ph.get("phi_tail"); r["phi_mirror"] = ph.get("phi_mirror"); r["phi_pages"] = ph.get("pages")
    r["reach_tail30"] = r["tail30"] * r["phi_tail"] if r["phi_tail"] is not None else None
    r["reach_mirror30"] = r["mirror30"] * r["phi_mirror"] if r["phi_mirror"] is not None else None
    r["recurring"] = bool(r["top1_share_tail"] is not None and r["top1_share_tail"] < 0.5 and r["events30"] >= 4)
cols = ["series", "title", "category", "frequency", "fee_type", "fee_mult", "src", "reach_tail30", "phi_tail", "reach_mirror30",
        "phi_mirror", "phi_pages", "n_far", "share_far_two_sided", "med_far_spread", "n_far0", "share_far0_two_sided", "tail30", "mirror30", "full30", "tail_rng30",
        "mirror_rng30", "contracts30", "events30", "top1_share_tail", "top1_share_mirror", "recurring", "med_spread",
        "med_spread_tail", "n_spread", "markets_measured", "n_events_measured", "sampled", "lower_bound", "missing"]
with open(HERE / "pool_table.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader()
    for r in sorted(rows.values(), key=lambda r: -(r["reach_tail30"] if r["reach_tail30"] is not None else -1)):
        w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})
json.dump(top, open(HERE / "top_events.json", "w"), indent=1)
print("series", len(rows))
for r in sorted(rows.values(), key=lambda r: -r["tail30"])[:60]:
    print(r["series"], r["category"], "tail", round(r["tail30"]), "reach", None if r["reach_tail30"] is None else round(r["reach_tail30"]),
          "mirror_rng", round(r["mirror_rng30"]), "ev", round(r["events30"], 1), "top1", r["top1_share_tail"], r["recurring"],
          "farN", r["n_far"], "far2s", None if r["share_far_two_sided"] is None else round(r["share_far_two_sided"], 2),
          "farSpr", r["med_far_spread"], "spr", r["med_spread"], r["fee_type"], "samp", r["sampled"])
