import pandas as pd
c = pd.read_csv("results/derived_ladders/calib_cells.csv")
print("series", c.series.nunique(), "cells", len(c))
c["gap_sig"] = (c.gap_lo > 0) | (c.gap_hi < 0)
c["yes_sig"] = c.ry_lo > 0; c["no_sig"] = c.rn_lo > 0
print("cells with gap CI excluding 0:", int(c.gap_sig.sum()), "| taker YES ROI CI > 0:", int(c.yes_sig.sum()), "| taker NO ROI CI > 0:", int(c.no_sig.sum()))
pd.set_option("display.width", 250)
cols = ["series", "by", "cell", "n", "n_ev", "y", "mid", "spread", "gap", "gap_lo", "gap_hi", "roi_yes", "ry_lo", "ry_hi", "roi_no", "rn_lo", "rn_hi"]
print(c[c.gap_sig | c.yes_sig | c.no_sig][cols].round(3).sort_values("gap").to_string(index=False))
