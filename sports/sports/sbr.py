"""MLB closing totals 2010-2021 from the free SportsbookReviewsOnline archive.

Rows come in pairs (visitor, home). By SBR convention the total's price on
the first row of a pair is the OVER and on the second the UNDER; that is
checked, not assumed (``check_convention``), by requiring the two prices to
imply a ~4% overround the way a two-sided market does.

SBR team codes are mapped to MLB Stats API abbreviations empirically -- by
which code appears as the home team on the dates each MLB club was at home --
rather than from a hand-typed table.
"""
from __future__ import annotations

import io
from collections import Counter

import numpy as np
import pandas as pd

from .http import fetch
from .market import american_to_prob

URL = ("https://www.sportsbookreviewsonline.com/wp-content/uploads/"
       "sportsbookreviewsonline_com_737/mlb-odds-{y}.xlsx")


def season(year: int) -> pd.DataFrame:
    d = pd.read_excel(io.BytesIO(fetch(URL.format(y=year), binary=True)))
    d.columns = ["date", "rot", "vh", "team", "pitcher"] + [f"i{i}" for i in range(1, 10)] + \
                ["final", "ml_open", "ml_close", "rl", "rl_odds", "ou_open", "ou_open_odds",
                 "ou_close", "ou_close_odds"]
    d = d.reset_index(drop=True)
    a, b = d.iloc[0::2].reset_index(drop=True), d.iloc[1::2].reset_index(drop=True)
    md = a["date"].astype(int).astype(str).str.zfill(4)
    g = pd.DataFrame({
        "date": pd.to_datetime(str(year) + md, format="%Y%m%d", errors="coerce"),
        "away_code": a["team"].str.strip(), "home_code": b["team"].str.strip(),
        "neutral": a["vh"].eq("N"),
        "away_pitcher": a["pitcher"], "home_pitcher": b["pitcher"],
        "away_runs": pd.to_numeric(a["final"], errors="coerce"),
        "home_runs": pd.to_numeric(b["final"], errors="coerce"),
        "ou_open": pd.to_numeric(a["ou_open"], errors="coerce"),
        "ou_close": pd.to_numeric(a["ou_close"], errors="coerce"),
        "over_odds": pd.to_numeric(a["ou_close_odds"], errors="coerce"),
        "under_odds": pd.to_numeric(b["ou_close_odds"], errors="coerce"),
        "ml_close_away": pd.to_numeric(a["ml_close"], errors="coerce"),
        "ml_close_home": pd.to_numeric(b["ml_close"], errors="coerce"),
    })
    g["season"] = year
    g["game_no"] = g.groupby(["date", "home_code"]).cumcount()  # doubleheaders
    return g.dropna(subset=["date", "ou_close", "over_odds", "under_odds"])


def check_convention(g: pd.DataFrame) -> float:
    """Median two-sided overround of the close total; ~0.03-0.05 if prices pair up."""
    return float(np.nanmedian(american_to_prob(g["over_odds"]) + american_to_prob(g["under_odds"]) - 1))


def code_map(sbr: pd.DataFrame, mlb: pd.DataFrame) -> dict[str, str]:
    """SBR code -> MLB abbreviation, by co-occurrence of home teams on dates."""
    home_by_date = mlb.groupby("date")["home"].apply(set)
    votes: dict[str, Counter] = {}
    for d, code in zip(sbr["date"], sbr["home_code"]):
        for abbr in home_by_date.get(d, ()):
            votes.setdefault(code, Counter())[abbr] += 1
    out = {}
    for code, c in votes.items():
        abbr, n = c.most_common(1)[0]
        out[code] = abbr
    return out


def join_to_mlb(sbr: pd.DataFrame, mlb: pd.DataFrame) -> pd.DataFrame:
    m = mlb[mlb["game_type"] == "regular"].copy()
    m["date"] = pd.to_datetime(m["date"])
    cmap = code_map(sbr, m)
    s = sbr.assign(home=sbr["home_code"].map(cmap), away=sbr["away_code"].map(cmap))
    m = m.sort_values("start_utc")
    m["game_no"] = m.groupby(["date", "home"]).cumcount()
    j = s.merge(m, on=["date", "home", "away", "game_no"], how="inner", suffixes=("", "_mlb"))
    # Same final score in both sources is the join's own check.
    j["score_match"] = (j["home_runs"] == j["home_score"]) & (j["away_runs"] == j["away_score"])
    return j
