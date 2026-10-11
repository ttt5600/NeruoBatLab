"""Grade ESPN prop lines against ESPN box scores (same athlete ids, no name matching).

    graded = settle.grade(props, box, games)

Each market maps to box-score stats summed per athlete; ``group`` pins the
box block when a stat name is ambiguous (NFL "sacks" = taken by a QB in
"passing", made by a defender in "defensive"). Markets not in MAP keep
actual = NaN and are listed by ``unmapped`` so the table can grow on evidence.

Rules (standard book rules, simplified):
  over/under  won if actual > line (over) / < line (under); push if equal
  yes         scorer markets: won if actual >= 1
  void        player has no box-score row (did not play) -> excluded, as books do
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# market name (after normalise) -> [(box group or None for any, stat), ...] summed
_PASS, _RUSH, _REC, _DEF, _KICK = "passing", "rushing", "receiving", "defensive", "kicking"
FOOTBALL = {
    "total passing yards": [(_PASS, "passingYards")],
    "total pass completions": [(_PASS, "completions")],
    "total passing attempts": [(_PASS, "passingAttempts")],
    "total passing touchdowns": [(_PASS, "passingTouchdowns")],
    "total passing interceptions": [(_PASS, "interceptions")],
    "longest passing completion": None,  # not in the box score
    "total rushing yards": [(_RUSH, "rushingYards")],
    "total carries": [(_RUSH, "rushingAttempts")],
    "longest rush": [(_RUSH, "longRushing")],
    "total receiving yards": [(_REC, "receivingYards")],
    "total receptions": [(_REC, "receptions")],
    "longest reception": [(_REC, "longReception")],
    "total rushing plus receiving yards": [(_RUSH, "rushingYards"), (_REC, "receivingYards")],
    "total passing plus rushing yards": [(_PASS, "passingYards"), (_RUSH, "rushingYards")],
    "total tackles": [(_DEF, "soloTackles")],
    "total tackles plus assists": [(_DEF, "totalTackles")],
    "total sacks": [(_DEF, "sacks")],
    "total field goals made": [(_KICK, "fieldGoalsMade")],
    "total extra points made": [(_KICK, "extraPointsMade")],
    "total kicking points": [(_KICK, "totalKickingPoints")],
    "anytime touchdown scorer": [(_RUSH, "rushingTouchdowns"), (_REC, "receivingTouchdowns")],
}
BASKETBALL = {
    "total points": [(None, "points")],
    "total rebounds": [(None, "rebounds")],
    "total assists": [(None, "assists")],
    "total turnovers": [(None, "turnovers")],
    "total steals": [(None, "steals")],
    "total blocks": [(None, "blocks")],
    "total 3-point field goals": [(None, "threePointFieldGoalsMade")],
    "total field goals made": [(None, "fieldGoalsMade")],
    "total points and assists": [(None, "points"), (None, "assists")],
    "total points and rebounds": [(None, "points"), (None, "rebounds")],
    "total rebounds and assists": [(None, "rebounds"), (None, "assists")],
    "total points, rebounds and assists": [(None, "points"), (None, "rebounds"), (None, "assists")],
    "total points, rebounds, and assists": [(None, "points"), (None, "rebounds"), (None, "assists")],
}
HOCKEY = {
    "total goals": [(None, "goals")],
    "total assists": [(None, "assists")],
    "total points": [(None, "goals"), (None, "assists")],
    "total shots on goal": [(None, "shotsTotal")],
    "total blocked shots": [(None, "blockedShots")],
    "total saves": [("goalies", "saves")],
    "total power play points": None,  # not split out in the box score
}
BASEBALL = {
    "total strikeouts": [("pitching", "strikeouts")],
    "total pitcher strikeouts": [("pitching", "strikeouts")],
    "total outs recorded": [("pitching", "outs")],
    "total earned runs allowed": [("pitching", "earnedRuns")],
    "total hits allowed": [("pitching", "hits")],
    "total walks allowed": [("pitching", "walks")],
    "total hits": [("batting", "hits")],
    "total runs": [("batting", "runs")],
    "total rbis": [("batting", "RBIs")],
    "total home runs": [("batting", "homeRuns")],
    "total hits, runs and rbis": [("batting", "hits"), ("batting", "runs"), ("batting", "RBIs")],
}
MAP = {"nfl": FOOTBALL, "cfb": FOOTBALL, "nba": BASKETBALL, "wnba": BASKETBALL, "nhl": HOCKEY, "mlb": BASEBALL}
TEAM_MARKETS = {"team total points", "team total runs", "team total goals"}


def normalise(m: str) -> str:
    return re.sub(r"\s*\(incl\.? overtime\)\s*", "", str(m)).strip().lower()


def _actual(league: str, props: pd.DataFrame, box: pd.DataFrame) -> pd.Series:
    table = MAP[league]
    b = box.dropna(subset=["athlete_id"])
    any_grp = b.groupby(["event_id", "athlete_id", "stat"])["value"].sum()
    by_grp = b.groupby(["event_id", "athlete_id", "group", "stat"])["value"].sum()
    played = set(zip(b["event_id"], b["athlete_id"]))
    out = np.full(len(props), np.nan)
    for i, (eid, aid, m) in enumerate(zip(props["event_id"], props["athlete_id"], props["market_n"])):
        spec = table.get(m)
        if not spec or aid is None or (eid, aid) not in played:
            continue
        tot = 0.0
        for grp, stat in spec:
            key = (eid, aid, stat) if grp is None else (eid, aid, grp, stat)
            src = any_grp if grp is None else by_grp
            tot += float(src.get(key, 0.0))
        out[i] = tot
    return pd.Series(out, index=props.index)


def grade(props: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    p = props.copy()
    league = games["league"].iloc[0]
    p["market_n"] = p["market"].map(normalise)
    p["actual"] = _actual(league, p, box)
    # team totals: team_id -> that team's final score
    g = games.set_index("event_id")
    team = p["market_n"].isin(TEAM_MARKETS) & p["team_id"].notna()
    for i in p.index[team]:
        e, t = p.at[i, "event_id"], p.at[i, "team_id"]
        if e in g.index:
            r = g.loc[e]
            p.at[i, "actual"] = r["home_score"] if str(r["home_team_id"]) == str(t) else \
                r["away_score"] if str(r["away_team_id"]) == str(t) else np.nan
    st = pd.to_datetime(p["event_id"].map(g["start_utc"]), utc=True)
    p["hours_before"] = (st - pd.to_datetime(p["last_updated"], utc=True)).dt.total_seconds() / 3600
    diff = p["actual"] - p["line"]
    void = p["actual"].isna()
    push = ((p["side"] != "yes") & (diff == 0)).to_numpy()
    won = np.where(p["side"] == "over", diff > 0, np.where(p["side"] == "under", diff < 0, p["actual"] >= 1))
    # 1/0 floats, NaN = void (did not play / market not gradable)
    p["push"] = np.where(void, np.nan, push.astype(float))
    p["won"] = np.where(void, np.nan, won.astype(float))
    p["profit"] = np.where(void, np.nan, np.where(push, 0.0, np.where(won, p["dec"] - 1, -1.0)))
    return p


def unmapped(props: pd.DataFrame, league: str) -> pd.Series:
    m = props["market"].map(normalise)
    known = set(MAP[league]) | TEAM_MARKETS
    return m[~m.isin(known)].value_counts()
