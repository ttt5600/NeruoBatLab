"""Classify settlement-definition features in results/rules_quirks/rules_by_series.jsonl -> rules_features.csv."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "results" / "rules_quirks"
FEAT = {
    "incl_ties": r"including ties|ties (?:will )?count|tied .{0,40}(?:resolve|count)s? (?:to )?yes",
    "dead_heat_1overN": r"\$1/N|1/N\b|divided equally|dead[- ]heat",
    "ot_excluded": r"(?:does not|do not|not) include (?:overtime|extra time|extra innings)|regulation only|excluding overtime|before overtime",
    "ot_included": r"(?:overtime|extra innings|extra time).{0,40}(?:count|included)|including overtime",
    "shootout": r"shootout",
    "fair_price": r"fair (?:market )?(?:price|value)",
    "tz_EST_literal": r"\bEST\b|eastern standard time",
    "tz_UTC": r"\bUTC\b|GMT",
    "first_print": r"first (?:release|print|publish|report)|initial (?:release|estimate|print)|advance estimate",
    "revisions_ignored": r"revision|revised|subsequent(?:ly)? (?:changes|corrections)",
    "stat_correction": r"stat(?:istic)?s? correction|official (?:box score|scorer|statistic)",
    "geq": r"\bat least\b|or more\b|\+ |\d\+ ",
    "gt": r"\bmore than\b|\babove\b|\bover\b|\bgreater than\b",
    "rounding": r"round(?:ed|ing)? (?:to|down|up)",
    "vacant_no": r"vacant.{0,40}(?:resolve|no)",
    "interim_excluded": r"interim .{0,30}(?:not count|do not count|does not count|excluded)",
    "station_named": r"\b(?:K[A-Z]{3}|NWS|Climatological Report|CLI\b|Weather Company|Weather Underground)",
    "cancel_48h": r"48 hours",
    "acting_counts": r"acting|interim assignment",
    "announce_vs_effective": r"announce",
    "dnp_fair": r"(?:inactive|never (?:enters|takes)).{0,80}fair",
    "dnp_lastprice": r"last fair market price before",
}


def main() -> int:
    rows = [json.loads(l) for l in open(D / "rules_by_series.jsonl")]
    df = pd.DataFrame(rows)
    txt = (df.rules_primary.fillna("") + " || " + df.rules_secondary.fillna(""))
    for k, p in FEAT.items():
        df[k] = txt.str.contains(p, flags=re.I, regex=True)
    df["txt"] = txt.str.replace(r"\s+", " ", regex=True).str.slice(0, 1200)
    df.drop(columns=["rules_primary", "rules_secondary"]).to_csv(D / "rules_features.csv", index=False)
    print("series", len(df), "by category:\n", df.category.value_counts().head(15).to_string())
    for k in FEAT:
        s = df[df[k]].sort_values("vol24", ascending=False)
        print(f"\n== {k}: {len(s)} series; top by 24h vol:", ", ".join(f"{a}({v:.0f})" for a, v in zip(s.series_ticker.head(12), s.vol24.head(12))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
