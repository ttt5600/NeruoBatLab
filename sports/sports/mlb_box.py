"""MLB per-start pitcher lines and per-game team batting, from boxscores.

One boxscore per game (~2,430 a season) gives, for both teams: every pitcher's
strikeouts / batters faced / pitches / outs, which pitcher started, and the
team's batting strikeouts and plate appearances. That is enough to model a
starter's strikeout count from (his K rate) x (batters he will face) x (how
often this opponent strikes out) -- all from games strictly before the start.

Every number here is POST-game for the game it describes. Use only as lagged
history; ``strikeout_features`` in ``props.py`` does that.
"""
from __future__ import annotations

import pandas as pd

from .http import fetch_many

BOX = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"


def _ip_outs(ip) -> int | None:
    """'5.2' innings -> 17 outs."""
    if ip in (None, ""):
        return None
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + int(frac or 0)


def parse_box(pk: int, b: dict) -> tuple[list[dict], list[dict]]:
    """Boxscore -> (pitcher rows, team-batting rows). Pure; tested offline."""
    pitchers, batting = [], []
    sides = b.get("teams", {})
    for side, opp in (("home", "away"), ("away", "home")):
        t = sides.get(side, {})
        team = t.get("team", {}).get("abbreviation") or t.get("team", {}).get("name")
        opp_team = sides.get(opp, {}).get("team", {})
        opp_team = opp_team.get("abbreviation") or opp_team.get("name")
        bat = t.get("teamStats", {}).get("batting", {})
        batting.append({"game_pk": pk, "team": team, "opponent": opp_team, "is_home": side == "home",
                        "bat_k": bat.get("strikeOuts"), "bat_pa": bat.get("plateAppearances")})
        for order, pid in enumerate(t.get("pitchers", [])):
            st = t.get("players", {}).get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            if not st:
                continue
            pitchers.append({
                "game_pk": pk, "team": team, "opponent": opp_team, "is_home": side == "home",
                "pitcher_id": pid,
                "pitcher": t["players"][f"ID{pid}"]["person"]["fullName"],
                "started": bool(st.get("gamesStarted")) or order == 0,
                "k": st.get("strikeOuts"), "bf": st.get("battersFaced"),
                "pitches": st.get("numberOfPitches"), "bb": st.get("baseOnBalls"),
                "outs": st.get("outs", _ip_outs(st.get("inningsPitched"))),
            })
    return pitchers, batting


def boxscores(game_pks: list[int], cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    raws = fetch_many([BOX.format(pk=pk) for pk in game_pks], cache=cache)
    P, B = [], []
    for pk, b in zip(game_pks, raws):
        p, t = parse_box(pk, b)
        P += p
        B += t
    return pd.DataFrame(P), pd.DataFrame(B)
