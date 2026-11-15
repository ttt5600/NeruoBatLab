#!/usr/bin/env python
"""Print the headline numbers of power_analysis.json."""
import json
from pathlib import Path
d = json.loads((Path(__file__).resolve().parent / "power_analysis.json").read_text())
t = d["typical_components_median_over_recipe_pairs"]
for k, v in t.items():
    print(k, {a: (round(b, 7) if isinstance(b, float) else b) for a, b in v.items()})
for k, v in d["statistics"].items():
    print(k, "resolved", v["n_pairs_resolved"], v["resolved_pairs"], "signagree", round(v["sign_agreement_with_acc_split0"], 2), "rho",
          round(v["spearman_with_acc_split0_diffs"], 2), "null birds-only", v["null_seed_pairs_resolved_birds_only_split0"],
          "seedsd", round(v["seed_sd_single_model"], 4), "median|z|", round(v["median_abs_z"], 3))
    print("   z_named", {a: round(b, 2) for a, b in v["z_named"].items()})
c = d["ceiling"]
print({k: v for k, v in c.items() if k not in ("all_wrong", "allwrong_clip_files")})
aw = c["all_wrong"]
print({k: v for k, v in aw.items() if k != "by_type"})
print({k: (v["n"], round(v["frac_of_type"], 3)) for k, v in aw["by_type"].items()})
for n, r in d["decomposition"].items():
    a = r["acc"]
    print(n, {k: round(a[k], 7) for k in ["diff_split0", "diff_10split_mean", "lo_birds", "hi_birds", "V_bird", "V_bird_1v1_mean",
                                           "V_clip_within_birds", "V_clip_iid", "V_seed_term", "seed_sd_pooled", "V_split",
                                           "halfwidth_tool", "halfwidth_full"]})
print({k: (v if isinstance(v, str) else {a: round(b, 7) for a, b in v.items() if a != "reps"}) for k, v in d["bird_scaling_check"].items()})
print("seed sd by group", {g: {a: round(b, 4) for a, b in v.items()} for g, v in d["seed_sd_by_group_acc"].items()})
g = d["power_acc_grid"]
for key in ["B48_n2_K1", "B48_n2_K10", "B48_n3_K10", "B48_n5_K10", "B48_n10_K10", "B96_n2_K10", "B96_n5_K10", "B144_n5_K10", "B192_n10_K10"]:
    print(key, {a: round(b, 4) for a, b in g[key].items()})
