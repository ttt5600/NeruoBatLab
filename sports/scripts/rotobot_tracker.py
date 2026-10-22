"""Grade a tout's posted picks (built for RotoBot / @rotobot_ai, works for any source).

Input-agnostic: picks come from a CSV/JSON you exported, a paste of tweet text, or
anything else that ends up in the PICK SCHEMA below. The tracker never fetches X.

    python scripts/rotobot_tracker.py parse  tweets.txt  -o data/rotobot/picks.csv   # free text -> schema
    python scripts/rotobot_tracker.py grade  data/rotobot/picks.csv                  # close + result + stats
    python scripts/rotobot_tracker.py power                                          # n needed for 2-3% edges

PICK SCHEMA (CSV columns / JSON keys; extra columns kept):
  posted_at   UTC timestamp the pick was PUBLISHED (tweet time, not game time)   required
  sport       nba | nfl | mlb | nhl | wnba | cfb                                     required
  game        free text naming the teams ("LAL @ BOS", "Lakers vs Celtics")      required for game markets
  market      moneyline | spread | total | <prop stat> (points, rebounds, pra, receiving yards, ...)
  side        team abbr/name (ML, spread) | over | under | yes | no (Kalshi)
  line        spread from the picked team's side (-3.5), total, or prop line; blank for ML
  price       American (-110, +150), decimal (1.91) or Kalshi cents (0.62 / 62c)
  source_url  link to the post
  optional:   player, event_id, start_utc, kalshi_ticker, close_line, close_price, result (win/loss/push)

Closing line, in priority order: manual close_line/close_price columns; Kalshi candles
(kalshi_ticker: last 1h candle mid before the game start); ESPN game lines *_close (moneyline,
spread, total); ESPN prop `line`/`dec` = ESPN's LAST stored price, which espn.py documents as only
APPROXIMATELY closing (last_updated is a median 6.4 h after kickoff) -> prop CLV is flagged
close_quality="approx" and reported separately.

Hard exclusion: a pick whose posted_at is at or after the game's start is dropped (reason
posted_after_start). A pick we cannot place on a game is dropped (no_game_match); nothing is
graded on a guess.

Metrics (1u flat stakes at the POSTED price): ROI, hit rate (pushes excluded), CLV
= posted_decimal * fair_close_prob - 1 (only where the close is at the same line; proportional
de-vig of the two closing sides), and line CLV in points where the line moved. 95% CIs are a
bootstrap over posting DAYS (picks on one card are correlated).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ESPN = ROOT / "data" / "espn"
OUT = ROOT / "data" / "rotobot"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SPORTS = ["nba", "nfl", "mlb", "nhl", "wnba", "cfb"]
COLS = ["posted_at", "sport", "game", "market", "side", "line", "price", "source_url"]
GAME_MARKETS = {"moneyline", "spread", "total"}

# pick stat word -> ESPN graded `market_n` (lower-cased ESPN market name)
STAT = {
    "points": "total points", "pts": "total points",
    "rebounds": "total rebounds", "reb": "total rebounds", "rebs": "total rebounds",
    "assists": "total assists", "ast": "total assists", "asts": "total assists",
    "threes": "total 3-point field goals", "3pm": "total 3-point field goals", "3s": "total 3-point field goals",
    "3-pointers": "total 3-point field goals",
    "pra": "total points, rebounds, and assists", "pts+reb+ast": "total points, rebounds, and assists",
    "pr": "total points and rebounds", "pts+reb": "total points and rebounds",
    "pa": "total points and assists", "pts+ast": "total points and assists",
    "steals": "total steals", "blocks": "total blocks",
    "passing yards": "total passing yards", "pass yds": "total passing yards",
    "rushing yards": "total rushing yards", "rush yds": "total rushing yards",
    "receiving yards": "total receiving yards", "rec yds": "total receiving yards",
    "receptions": "total receptions", "recs": "total receptions",
    "passing tds": "total passing touchdowns", "pass tds": "total passing touchdowns",
    "completions": "total pass completions", "attempts": "total passing attempts",
    "rush+rec yds": "total rushing plus receiving yards", "carries": "total carries",
    "strikeouts": "total strikeouts", "ks": "total strikeouts",
}
SPORT_WORDS = {"nba": "nba", "nfl": "nfl", "mlb": "mlb", "nhl": "nhl", "wnba": "wnba",
               "cfb": "cfb", "ncaaf": "cfb", "cfbplayoff": "cfb"}


# ------------------------------------------------------------------ odds
def to_decimal(price) -> float:
    """American (-110/+150), decimal (1.91), Kalshi cents (0.62, 62c) -> decimal odds (no fees)."""
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return np.nan
    s = str(price).strip().lower().replace("−", "-")
    if not s or s in ("nan", "none"):
        return np.nan
    if s.endswith("c") or s.endswith("¢"):
        return 100.0 / float(s[:-1])
    x = float(s)
    if s.startswith(("+", "-")) or abs(x) >= 100:
        return 1 + 100 / -x if x < 0 else 1 + x / 100
    if 0 < x < 1:
        return 1 / x
    if x > 1:
        return x
    return np.nan


TWITTER_EPOCH_MS = 1288834974657


def snowflake_time(url_or_id) -> pd.Timestamp | None:
    """Exact UTC post time encoded in an X status id (snowflake: ms since 2010-11-04 in bits 22+).
    So a pasted tweet URL alone fixes posted_at -- no need to trust a hand-typed time."""
    m = re.search(r"(?:status/)?(\d{15,20})", str(url_or_id))
    if not m:
        return None
    return pd.Timestamp((int(m[1]) >> 22) + TWITTER_EPOCH_MS, unit="ms", tz="UTC")


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z0-9 ]+", "", s).strip()


# ------------------------------------------------------------------ free-text parser
_AM = r"(?P<price>[+\-−]\d{3,4}|\d\.\d{2}|\d{1,2}c)"
_NUM = r"(?P<line>\d+(?:\.\d)?)"
_STATS = "|".join(sorted((re.escape(k) for k in STAT), key=len, reverse=True))
PROP_A = re.compile(  # LeBron James o24.5 pts (-115) | LeBron James over 24.5 points
    rf"(?P<player>[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){{1,3}})\s+(?P<side>over|under|o|u)\s?{_NUM}\s+"
    rf"(?P<stat>{_STATS})\b(?:\s*\(?\s*{_AM}\s*\)?)?", re.I)
PROP_B = re.compile(  # LeBron James points over 24.5 (-115)
    rf"(?P<player>[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){{1,3}})\s+(?P<stat>{_STATS})\s+(?P<side>over|under|o|u)\s?"
    rf"{_NUM}(?:\s*\(?\s*{_AM}\s*\)?)?", re.I)
TOTAL = re.compile(  # Lakers vs Celtics o224.5 (-110) | LAL/BOS under 224.5
    rf"(?P<game>[A-Za-z .]+?\s*(?:vs\.?|@|/|v)\s*[A-Za-z .]+?)\s+(?P<side>over|under|o|u)\s?(?P<line>\d{{2,3}}(?:\.\d)?)"
    rf"(?:\s*\(?\s*{_AM}\s*\)?)?", re.I)
SPREAD = re.compile(  # Celtics -4.5 (-110)
    rf"(?P<team>[A-Z][A-Za-z .]+?)\s+(?P<line>[+\-−]\d{{1,2}}(?:\.5)?)(?!\d)(?:\s*\(?\s*{_AM}\s*\)?)?")
ML = re.compile(rf"(?P<team>[A-Z][A-Za-z .]+?)\s+(?:ML|moneyline)\b(?:\s*\(?\s*{_AM}\s*\)?)?", re.I)


def _sport_of(text: str, default: str | None) -> str | None:
    for w in re.findall(r"#?([A-Za-z]+)", text):
        if w.lower() in SPORT_WORDS:
            return SPORT_WORDS[w.lower()]
    return default


def _side(s: str) -> str:
    return {"o": "over", "u": "under"}.get(s.lower(), s.lower())


def parse_tweet(text: str, posted_at, source_url: str = "", sport: str | None = None) -> list[dict]:
    """One post -> zero or more picks in the schema. Line-by-line; unparsed lines are skipped.

    Deliberately small: it handles the common pick formats (player prop o/u, game total o/u,
    team spread, team ML, each with an optional price). Anything else should be entered by hand;
    `parse` writes every post that yielded no pick to *_unparsed.txt so nothing is silently lost.
    """
    sp = _sport_of(text, sport)
    out = []
    base = {"posted_at": pd.Timestamp(posted_at, tz="UTC") if pd.Timestamp(posted_at).tzinfo is None
            else pd.Timestamp(posted_at).tz_convert("UTC"), "sport": sp, "source_url": source_url}
    for raw in text.splitlines():
        ln = re.sub(r"[^\x00-\x7f−¢]", " ", raw).strip(" -*•\t")
        ln = re.sub(r"(?i)#?\b(" + "|".join(SPORT_WORDS) + r")\b:?", " ", ln)  # sport tags are not names
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln:
            continue
        m = PROP_A.search(ln) or PROP_B.search(ln)
        if m:
            out.append({**base, "game": "", "player": m["player"].strip(), "market": STAT[m["stat"].lower()],
                        "side": _side(m["side"]), "line": float(m["line"]), "price": m["price"]})
            continue
        m = TOTAL.search(ln)
        if m:
            out.append({**base, "game": m["game"].strip(), "market": "total", "side": _side(m["side"]),
                        "line": float(m["line"]), "price": m["price"]})
            continue
        m = ML.search(ln)
        if m:
            out.append({**base, "game": m["team"].strip(), "market": "moneyline", "side": m["team"].strip(),
                        "line": np.nan, "price": m["price"]})
            continue
        m = SPREAD.search(ln)
        if m:
            out.append({**base, "game": m["team"].strip(), "market": "spread", "side": m["team"].strip(),
                        "line": float(m["line"].replace("−", "-")), "price": m["price"]})
    return out


def read_posts(p: Path) -> list[dict]:
    """Posts from .jsonl/.json ({posted_at, text, url}) or .txt blocks: first line ISO time
    [+ space + url], then the post text, blocks separated by a line of '---'."""
    if p.suffix in (".jsonl", ".json"):
        txt = p.read_text()
        recs = json.loads(txt) if p.suffix == ".json" else [json.loads(l) for l in txt.splitlines() if l.strip()]
        return [{"posted_at": r.get("posted_at") or r.get("created_at")
                 or snowflake_time(r.get("url") or r.get("id", "")), "text": r["text"],
                 "url": r.get("url") or r.get("source_url", "")} for r in recs]
    posts = []
    for block in re.split(r"^---\s*$", p.read_text(), flags=re.M):
        lines = [l for l in block.strip().splitlines()]
        if not lines:
            continue
        head = lines[0].split()
        if head and ("x.com/" in head[0] or "twitter.com/" in head[0]):  # URL only: time from the status id
            head = [str(snowflake_time(head[0])), head[0]]
        posts.append({"posted_at": head[0], "url": head[1] if len(head) > 1 else "", "text": "\n".join(lines[1:])})
    return posts


# ------------------------------------------------------------------ loading picks
def load_picks(p: Path) -> pd.DataFrame:
    df = pd.read_json(p) if p.suffix == ".json" else pd.read_csv(p, dtype={"price": str, "event_id": str})
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"{p}: missing columns {miss}")
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True)
    if "source_url" in df:  # a status URL is ground truth for the post time; disagreement is flagged
        sf = pd.to_datetime(df["source_url"].map(snowflake_time), utc=True)
        bad = sf.notna() & ((sf - df["posted_at"]).abs() > pd.Timedelta(minutes=2))
        if bad.any():
            print(f"WARNING: {bad.sum()} picks have posted_at != the time in their status URL; using the URL time")
        df.loc[sf.notna(), "posted_at"] = sf[sf.notna()]
    df["sport"] = df["sport"].str.lower()
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["market"] = df["market"].map(lambda m: STAT.get(m, m))
    df["side"] = df["side"].astype(str).str.strip()
    df["dec"] = df["price"].map(to_decimal)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ teams + game matching
def team_aliases(sport: str) -> dict[str, str]:
    """alias (normalized) -> ESPN abbreviation, from the keyless ESPN teams endpoint (cached)."""
    from sports import espn
    from sports.http import fetch
    d = fetch(espn.SITE.format(path=espn.LEAGUES[sport]) + "/teams", {"limit": 1000})
    teams = [t["team"] for t in d["sports"][0]["leagues"][0]["teams"]]
    out, loc_count = {}, {}
    for t in teams:
        loc_count[norm(t.get("location", ""))] = loc_count.get(norm(t.get("location", "")), 0) + 1
    for t in teams:
        ab = t["abbreviation"]
        for k in ("abbreviation", "displayName", "shortDisplayName", "name", "nickname"):
            if t.get(k):
                out[norm(t[k])] = ab
        if loc_count.get(norm(t.get("location", ""))) == 1:  # "Boston" ok, "Los Angeles" ambiguous
            out[norm(t["location"])] = ab
    return out


def teams_in(text: str, aliases: dict[str, str]) -> list[str]:
    t = " " + norm(text) + " "
    found = []
    for a in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(a)}\b", t):
            ab = aliases[a]
            if ab not in found:
                found.append(ab)
            t = re.sub(rf"\b{re.escape(a)}\b", " ", t)
    return found


def match_game(pick: pd.Series, games: pd.DataFrame, aliases: dict, box: pd.DataFrame | None) -> tuple:
    """-> (event_id, reason). The EARLIEST candidate game starting after posted_at - 6 h and before
    posted_at + 5 d. If that game had already started when the pick was posted, the pick is excluded
    (we never slide a late pick onto the team's next game)."""
    if isinstance(pick.get("event_id"), str) and pick["event_id"]:
        return pick["event_id"], "ok"
    t0 = pick["posted_at"]
    g = games[(games.start_utc > t0 - pd.Timedelta(hours=6)) & (games.start_utc < t0 + pd.Timedelta(days=5))]
    if pick["market"] in GAME_MARKETS:
        tm = teams_in(f"{pick['game']} {pick['side']}", aliases)
        if not tm:
            return None, "no_team_match"
        g = g[g.home.isin(tm) | g.away.isin(tm)]
        if len(tm) >= 2:
            g = g[g.home.isin(tm) & g.away.isin(tm)]
    else:
        if box is None or not isinstance(pick.get("player"), str):
            return None, "no_player"
        ev = box.loc[box.name_n == norm(pick["player"]), "event_id"].unique()
        g = g[g.event_id.isin(ev)]
    if g.empty:
        return None, "no_game_match"
    return g.sort_values("start_utc").iloc[0]["event_id"], "ok"


# ------------------------------------------------------------------ grading
def _team_side(pick, game) -> str | None:
    tm = teams_in(str(pick["side"]), ALIASES.get(pick["sport"], {})) if pick["market"] != "total" else []
    if game["home"] in tm:
        return "home"
    if game["away"] in tm:
        return "away"
    return None


ALIASES: dict[str, dict[str, str]] = {}


def settle_line(market: str, side: str, line: float, home: float, away: float, team: str | None) -> float:
    """+1 win, 0 push, -1 loss (game markets)."""
    if market == "total":
        d = (home + away) - line
        d = d if side == "over" else -d
    else:
        m = home - away if team == "home" else away - home
        d = m if market == "moneyline" else m + line
    return float(np.sign(d))


def grade_game(pick, game, lines: pd.DataFrame) -> dict:
    mk, side = pick["market"], str(pick["side"]).lower()
    team = None if mk == "total" else _team_side(pick, game)
    if mk != "total" and team is None:
        return {"reason": "side_not_in_game"}
    r = {"result": settle_line(mk, side, pick["line"], game["home_score"], game["away_score"], team)}
    L = lines[lines.event_id == game["event_id"]]
    if pick.get("book") and isinstance(pick.get("book"), str):
        L = L[L.provider.str.lower() == pick["book"].lower()] if (L.provider.str.lower() == pick["book"].lower()).any() else L
    if L.empty:
        return r
    L = L.iloc[0]
    if mk == "moneyline":
        mine, other = (f"{team}_ml_close", "away_ml_close" if team == "home" else "home_ml_close")
        cl, my_px, ot_px = np.nan, L.get(mine), L.get(other)
    elif mk == "spread":
        opp = "away" if team == "home" else "home"
        cl, my_px, ot_px = L.get(f"{team}_spread_close"), L.get(f"{team}_spread_odds_close"), L.get(f"{opp}_spread_odds_close")
    else:
        opp = "under" if side == "over" else "over"
        cl, my_px, ot_px = L.get("total_close"), L.get(f"{side}_odds_close"), L.get(f"{opp}_odds_close")
    r.update(close_line=cl, close_dec=to_decimal(my_px), close_dec_other=to_decimal(ot_px), close_quality="espn_close",
             book=L.get("provider"))
    return r


def grade_prop(pick, graded: pd.DataFrame, box: pd.DataFrame, eid: str) -> dict:
    aid = box.loc[(box.event_id == eid) & (box.name_n == norm(pick["player"])), "athlete_id"]
    if aid.empty:
        return {"reason": "no_player"}
    G = graded[(graded.event_id == eid) & (graded.athlete_id == aid.iloc[0]) & (graded.market_n == pick["market"])]
    if G.empty or G["actual"].isna().all():
        return {"reason": "no_prop_result"}
    actual = G["actual"].dropna().iloc[0]
    side = str(pick["side"]).lower()
    d = actual - pick["line"]
    r = {"result": float(np.sign(d if side == "over" else -d)), "actual": actual}
    mine, other = G[G.side == side], G[G.side == ("under" if side == "over" else "over")]
    if not mine.empty and not other.empty:
        # pair at the same line from the same book
        m = mine.merge(other, on=["provider", "line"], suffixes=("", "_o"))
        if not m.empty:  # prefer the pair at the pick's own line (books list several)
            m = m.assign(_d=(m["line"] - pick["line"]).abs()).sort_values("_d").iloc[0]
            r.update(close_line=m["line"], close_dec=m["dec"], close_dec_other=m["dec_o"], close_quality="approx",
                     book=m["provider"])
    return r


def kalshi_close(ticker: str, start: pd.Timestamp) -> dict:
    """Last hourly candle mid before `start` + the settled result, from the public Kalshi API."""
    from sports.http import fetch
    t = int(start.timestamp())
    m = None
    for url in (f"{KALSHI}/markets/{ticker}", f"{KALSHI}/historical/markets/{ticker}"):
        try:
            m = fetch(url, cache=False).get("market")
            series = m.get("series_ticker") or ticker.split("-")[0]
            break
        except Exception:
            continue
    if not m:
        return {"reason": "kalshi_not_found"}
    cs = []
    for url in (f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks", f"{KALSHI}/historical/markets/{ticker}/candlesticks"):
        try:
            cs = fetch(url, {"start_ts": t - 24 * 3600, "end_ts": t, "period_interval": 60}).get("candlesticks", [])
            if cs:
                break
        except Exception:
            continue
    c = [x for x in cs if x["end_period_ts"] <= t]
    out = {"kalshi_result": m.get("result")}
    if c:
        def g(k):
            v = c[-1].get(k) or {}
            return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")
        bid, ask = g("yes_bid"), g("yes_ask")
        bid, ask = (bid / 100, ask / 100) if (ask or 0) > 1 else (bid, ask)
        out["yes_mid_close"] = (bid + ask) / 2
    return out


def attach(picks: pd.DataFrame, games: pd.DataFrame, lines: pd.DataFrame, graded: pd.DataFrame,
           box: pd.DataFrame, aliases: dict[str, dict[str, str]], use_kalshi: bool = True) -> pd.DataFrame:
    """Match every pick to a game, enforce posted < start, attach close + result."""
    ALIASES.clear()
    ALIASES.update(aliases)
    box = box.assign(name_n=box["name"].map(norm)) if not box.empty else box.assign(name_n=[])
    rows = []
    for _, p in picks.iterrows():
        r = p.to_dict()
        r["reason"] = "ok"
        if isinstance(p.get("kalshi_ticker"), str) and p["kalshi_ticker"] and use_kalshi:
            start = pd.to_datetime(p.get("start_utc"), utc=True)
            if pd.isna(start):
                r["reason"] = "kalshi_needs_start_utc"
            elif p["posted_at"] >= start:
                r["reason"] = "posted_after_start"
            else:
                k = kalshi_close(p["kalshi_ticker"], start)
                r.update(k)
                if "yes_mid_close" in k:
                    yes = k["yes_mid_close"]
                    r.update(close_prob_fair=yes if str(p["side"]).lower() == "yes" else 1 - yes, close_quality="kalshi_mid")
                if k.get("kalshi_result") in ("yes", "no"):
                    r["result"] = 1.0 if k["kalshi_result"] == str(p["side"]).lower() else -1.0
            rows.append(r)
            continue
        G = games[games.league == p["sport"]] if "league" in games else games
        eid, why = match_game(p, G, aliases.get(p["sport"], {}), box)
        if eid is None:
            r["reason"] = why
            rows.append(r)
            continue
        g = games[games.event_id == eid].iloc[0]
        r.update(event_id=eid, start_utc=g["start_utc"], home=g["home"], away=g["away"])
        if p["posted_at"] >= g["start_utc"]:
            r["reason"] = "posted_after_start"
        elif pd.isna(g.get("home_score")) or g.get("status") not in (None, "STATUS_FINAL"):
            r["reason"] = "not_final"
        elif p["market"] in GAME_MARKETS:
            r.update(grade_game(p, g, lines))
        else:
            r.update(grade_prop(p, graded, box, eid))
        rows.append(r)
    out = pd.DataFrame(rows)
    # manual overrides win over everything
    if "close_line" in picks:
        man = pd.to_numeric(picks["close_line"], errors="coerce")
        out.loc[man.notna(), "close_line"] = man[man.notna()]
    if "close_price" in picks:
        out["close_price"] = picks["close_price"].values
        man = out["close_price"].notna()
        out.loc[man, "close_dec"] = out.loc[man, "close_price"].map(to_decimal)
        out.loc[man, "close_dec_other"] = np.nan
        out.loc[man, "close_quality"] = "manual"
    out["result"] = pd.to_numeric(out["result"], errors="coerce") if "result" in out else np.nan
    if "result" in picks:
        man = picks["result"].astype(str).str.lower().map({"win": 1.0, "loss": -1.0, "push": 0.0})
        out.loc[man.notna(), "result"] = man[man.notna()]
        # a hand-graded pick we could not match still needs a hand-entered start_utc before posted_at
        start = pd.to_datetime(picks["start_utc"], utc=True) if "start_utc" in picks else pd.Series(pd.NaT, index=picks.index)
        rescued = man.notna() & out.reason.isin(["no_game_match", "no_team_match", "no_player", "not_final"])
        out.loc[rescued & start.isna(), "reason"] = "manual_needs_start_utc"
        out.loc[rescued & start.notna() & (out.posted_at >= start), "reason"] = "posted_after_start"
        out.loc[rescued & start.notna() & (out.posted_at < start), "reason"] = "ok"
    return score(out)


def score(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("result", "close_line", "close_dec", "close_dec_other", "close_prob_fair"):
        if c not in df:
            df[c] = np.nan
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["profit"] = np.where(df.result > 0, df.dec - 1, np.where(df.result < 0, -1.0, 0.0))
    df.loc[df.result.isna(), "profit"] = np.nan
    # fair closing prob: proportional de-vig of the two closing sides; one side only -> raw implied (vig in)
    q1, q2 = 1 / df.close_dec, 1 / df.close_dec_other
    fair = np.where(q2.notna(), q1 / (q1 + q2), q1)
    df["close_prob_fair"] = df["close_prob_fair"].fillna(pd.Series(fair, index=df.index))
    same_line = df.line.isna() | df.close_line.isna() | (df.line == df.close_line)
    df["clv"] = np.where(same_line, df.dec * df.close_prob_fair - 1, np.nan)
    # line CLV in points (favourable = positive): spread +3.5 closing +2.5 -> +1; over 44 closing 45 -> +1
    side = df.side.astype(str).str.lower()
    d = df.line - df.close_line
    df["line_clv_pts"] = np.where(side.eq("over"), -d, d)
    df.loc[df.market.eq("moneyline") | df.close_line.isna(), "line_clv_pts"] = np.nan
    return df


# ------------------------------------------------------------------ stats
def boot_ci(x: np.ndarray, groups: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling posting days (clusters) with replacement."""
    x, groups = np.asarray(x, float), np.asarray(groups)
    ok = ~np.isnan(x)
    x, groups = x[ok], groups[ok]
    if len(np.unique(groups)) < 2:
        return np.nan, np.nan
    keys, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df.reason.eq("ok") & df.result.notna()]
    day = ok.posted_at.dt.strftime("%Y-%m-%d").to_numpy()
    dec = ok.result.ne(0)
    s = {"n_input": int(len(df)), "excluded": df.loc[~df.reason.eq("ok"), "reason"].value_counts().to_dict(),
         "n_graded": int(len(ok)), "n_days": int(len(set(day)))}
    if len(ok):
        s["hit_rate"] = float((ok.result[dec] > 0).mean()) if dec.any() else np.nan
        s["breakeven_hit_rate"] = float((1 / ok.dec[dec]).mean()) if dec.any() else np.nan
        s["roi"] = float(ok.profit.mean())
        s["roi_ci95"] = boot_ci(ok.profit.to_numpy(), day)
        s["units"] = float(ok.profit.sum())
    for q, lab in (("espn_close", "clv_game"), ("approx", "clv_prop_approx"), ("kalshi_mid", "clv_kalshi"),
                   ("manual", "clv_manual")):
        if "close_quality" not in ok:
            break
        sub = ok[ok.close_quality.eq(q) & ok.clv.notna()]
        if len(sub):
            s[lab] = {"n": int(len(sub)), "mean": float(sub.clv.mean()),
                      "ci95": boot_ci(sub.clv.to_numpy(), sub.posted_at.dt.strftime("%Y-%m-%d").to_numpy()),
                      "beat_close_share": float((sub.clv > 0).mean())}
    lc = ok.line_clv_pts.dropna() if "line_clv_pts" in ok else pd.Series(dtype=float)
    if len(lc):
        s["line_clv_pts"] = {"n": int(len(lc)), "mean": float(lc.mean())}
    return s


# ------------------------------------------------------------------ power
def n_needed(edge: float, dec: float, z_a: float = 1.96, z_b: float = 0.84) -> int:
    """Bets to detect ROI `edge` at decimal odds `dec` (two-sided 5%, 80% power), flat 1u stakes.
    Per-bet profit SD = dec * sqrt(p (1-p)) with p = (1 + edge) / dec."""
    p = (1 + edge) / dec
    sd = dec * np.sqrt(p * (1 - p))
    return int(np.ceil(((z_a + z_b) * sd / edge) ** 2))


def clv_sd_from_espn(league: str = "nba", seasons=(2025, 2026)) -> tuple[float, float, int]:
    """Empirical spread of CLV for a bet at the OPENING moneyline graded vs the de-vigged close:
    (SD, IQR/1.349 robust SD, n). Opening->close is the WIDEST horizon; a tout posting an hour
    before tip will have a smaller CLV spread, so these bracket the n needed from above."""
    fr = [pd.read_parquet(ESPN / league / f"lines_{s}.parquet") for s in seasons if (ESPN / league / f"lines_{s}.parquet").exists()]
    if not fr:
        return np.nan, np.nan, 0
    L = pd.concat(fr)
    L = L.dropna(subset=["home_ml_open", "home_ml_close", "away_ml_close"])
    qh, qa = 1 / L.home_ml_close.map(to_decimal), 1 / L.away_ml_close.map(to_decimal)
    clv = L.home_ml_open.map(to_decimal) * qh / (qh + qa) - 1
    clv = clv[np.isfinite(clv)]
    return float(clv.std()), float((clv.quantile(.75) - clv.quantile(.25)) / 1.349), int(len(clv))


# ------------------------------------------------------------------ CLI
def load_espn(sports_: list[str]) -> tuple[pd.DataFrame, ...]:
    def cat(kind):
        fr = []
        for sp in sports_:
            for f in sorted((ESPN / sp).glob(f"{kind}_*.parquet")):
                fr.append(pd.read_parquet(f).assign(league=sp))
        return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    games, lines, graded, box = cat("games"), cat("lines"), cat("graded"), cat("box")
    if not games.empty:
        games["start_utc"] = pd.to_datetime(games.start_utc, utc=True)
    if not box.empty:
        box = box[["event_id", "athlete_id", "name"]].drop_duplicates()
    if not graded.empty:
        graded = graded.dropna(subset=["actual"])
    return games, lines, graded, box


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parse")
    a.add_argument("posts", type=Path)
    a.add_argument("-o", "--out", type=Path, default=OUT / "picks.csv")
    a.add_argument("--sport", help="default sport when a post does not name one")
    b = sub.add_parser("grade")
    b.add_argument("picks", type=Path)
    b.add_argument("--no-kalshi", action="store_true")
    sub.add_parser("power")
    args = ap.parse_args()

    if args.cmd == "parse":
        picks, miss = [], []
        for post in read_posts(args.posts):
            got = parse_tweet(post["text"], post["posted_at"], post["url"], args.sport)
            picks.extend(got)
            if not got:
                miss.append(post)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(picks)
        for c in COLS + ["player"]:
            if c not in df:
                df[c] = None
        df[COLS + ["player"]].to_csv(args.out, index=False)
        (args.out.with_name(args.out.stem + "_unparsed.txt")).write_text(
            "\n---\n".join(f"{m['posted_at']} {m['url']}\n{m['text']}" for m in miss))
        print(f"{len(picks)} picks from {len(picks) and len(set(map(str, (p['source_url'] for p in picks))))} posts -> {args.out}; "
              f"{len(miss)} posts with no parsable pick -> {args.out.stem}_unparsed.txt (enter by hand)")
        return 0

    if args.cmd == "power":
        print("bets needed for ROI edge (two-sided 5%, 80% power, flat 1u):")
        for price, dec in (("-110", to_decimal("-110")), ("+100", 2.0), ("+150", 2.5), ("+300", 4.0)):
            print(f"  {price:>5}: 2% -> {n_needed(0.02, dec):>6,}   3% -> {n_needed(0.03, dec):>6,}")
        sd, rsd, n = clv_sd_from_espn()
        print(f"CLV spread, opening ML vs de-vigged close, ESPN NBA 2025-26, n={n:,}: SD {sd:.4f}, robust SD {rsd:.4f}")
        for e in (0.02, 0.03):
            lo, hi = (int(np.ceil(((1.96 + 0.84) * x / e) ** 2)) for x in (rsd, sd))
            print(f"  bets needed to resolve mean CLV {e:.0%}: {lo:,} (robust SD) .. {hi:,} (SD)")
        return 0

    picks = load_picks(args.picks)
    sports_ = sorted(set(picks.sport.dropna()) & set(SPORTS))
    games, lines, graded, box = load_espn(sports_)
    aliases = {sp: team_aliases(sp) for sp in sports_}
    out = attach(picks, games, lines, graded, box, aliases, use_kalshi=not args.no_kalshi)
    dst = args.picks.with_name(args.picks.stem + "_graded.csv")
    out.to_csv(dst, index=False)
    print(json.dumps(summarize(out), indent=2, default=str))
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
