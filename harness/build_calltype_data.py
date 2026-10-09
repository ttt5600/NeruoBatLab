#!/usr/bin/env python3
"""Row files for the Call Type Breakdown dashboard, from the two analysis JSONs.

  analysis/calltype_per_class.json          (zfeval/experiments/calltype_per_class.py)
  analysis/calltype_split_sensitivity.json  (zfeval/experiments/calltype_split_sensitivity.py)
-> docs/calltype_dashboard/{call_types,recall,vs_run16,confusion,splits}.json
"""
import json
from pathlib import Path

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = Path(__file__).resolve().parents[1] / "docs/calltype_dashboard"
NAMES = dict(Ag="Wsst (aggressive)", Be="Begging", DC="Distance call", Di="Distress", LT="Long tonal",
             Ne="Nest", So="Song", Te="Tet", Th="Thuk", Tu="Tuck", Wh="Whine")
GROUP = dict(ours="Ours, from scratch", dapt="AVES continued on zebra finch", aves="AVES, downloaded")

pc = json.load(open(ANA / "calltype_per_class.json"))
ss = json.load(open(ANA / "calltype_split_sensitivity.json"))
cl = pc["classes"]
order = list(pc["models"])

files = {
    "call_types": [dict(call_type=c, name=NAMES[c], n_clips=pc["class_counts"][c], n_birds=pc["class_birds"][c],
                        chick_share=round(pc["class_chick_fraction"][c], 4)) for c in cl],
    "recall": [dict(model=m, group=GROUP[v["group"]], model_order=i, layer=v["layer"], call_type=c,
                    recall=round(v["recall"][c], 4), correct=v["confusion"][j][j], n=sum(v["confusion"][j]),
                    overall=round(v["acc11"], 4))
               for i, (m, v) in enumerate(pc["models"].items()) for j, c in enumerate(cl)],
    "vs_run16": [dict(model=m, call_type=c, diff=round(x["diff"], 4), lo=round(x["lo"], 4), hi=round(x["hi"], 4),
                      resolved=x["resolved"])
                 for m, v in pc["models"].items() if "vs_ref" in v for c, x in v["vs_ref"].items()],
    "confusion": [dict(model=m, true_type=c, pred_type=cl[k], count=v["confusion"][j][k])
                  for m, v in pc["models"].items() for j, c in enumerate(cl) for k in range(len(cl))],
    "splits": [dict(model=m, group=GROUP[pc["models"][m]["group"]], model_order=order.index(m), split=s,
                    acc=round(a, 4), board_split=(s == 0), diff_vs_run16=round(v["diff_vs_run16"][s], 4))
               for m, v in ss["models"].items() for s, a in zip(ss["seeds"], v["acc"])],
}
OUT.mkdir(parents=True, exist_ok=True)
for k, rows in files.items():
    (OUT / f"{k}.json").write_text(json.dumps(rows))
    print(k, len(rows), "rows", (OUT / f"{k}.json").stat().st_size, "bytes")
