"""One-off live snapshot: Kalshi soccer 1X2 vs Pinnacle power de-vig (descriptive, not a trial). Run from sports/."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, re
from scipy.optimize import brentq
from sports import live
from sports.live import PIN, PIN_HEADERS, _get, am_to_dec, kalshi_fee

LG = {"epl": (1980, "KXEPLGAME"), "laliga": (2196, "KXLALIGAGAME"), "seriea": (2436, "KXSERIEAGAME"),
      "bund": (1842, "KXBUNDESLIGAGAME"), "ligue1": (2036, "KXLIGUE1GAME"), "efl": (1977, "KXEFLCHAMPIONSHIPGAME"),
      "mls": (2663, "KXMLSGAME"), "ucl": (2627, "KXUCLGAME"), "ered": (1928, "KXEREDIVISIEGAME")}


def power(r):
    k = brentq(lambda k: (r ** k).sum() - 1, 0.3, 4.0); return r ** k


def norm(s):
    s = re.sub(r"[^a-z ]", "", s.lower().replace("&", "and"))
    for w in ("fc", "afc", "cf", "sc", "club", "de", "the", "united", "city"):
        s = re.sub(rf"\b{w}\b", "", s)
    return s.split()[0] if s.split() else s


rows = []
for lg, (lid, ser) in LG.items():
    try:
        mu = _get(f"{PIN}/leagues/{lid}/matchups", headers=PIN_HEADERS)
        mk = _get(f"{PIN}/leagues/{lid}/markets/straight", headers=PIN_HEADERS)
        k = live.LEAGUES.setdefault(lg, (lid, ser, None)) and live.kalshi(lg)
    except Exception as e:
        print(lg, "fail", e); continue
    games = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("parentId") or m.get("isLive"): continue
        side = {p["alignment"]: p["name"] for p in m.get("participants", [])}
        if "home" in side and "away" in side:
            games[m["id"]] = (pd.Timestamp(m["startTime"]), side["home"], side["away"])
    k["ev"] = k.event
    n = 0
    for x in mk:
        g = games.get(x["matchupId"])
        if g is None or x.get("type") != "moneyline" or x.get("period") != 0 or x.get("isAlternate"): continue
        pr = {p["designation"]: am_to_dec(p["price"]) for p in x["prices"]}
        if set(pr) != {"home", "draw", "away"}: continue
        inv = np.array([1 / pr["home"], 1 / pr["draw"], 1 / pr["away"]])
        fair = dict(zip(("H", "D", "A"), power(inv)))
        hn, an = norm(g[1]), norm(g[2])
        cand = k[k.event.str.contains(g[0].strftime("%y%b%d").upper())]
        evs = [e for e, gg in cand.groupby("event") if any(hn in norm(t or "") for t in gg.team) and any(an in norm(t or "") for t in gg.team)]
        if len(evs) != 1: continue
        gg = cand[cand.event == evs[0]]
        n += 1
        for _, r in gg.iterrows():
            s = "D" if r.ticker.endswith("-TIE") else ("H" if hn in norm(r.team or "") else "A")
            ask = r.yes_ask; nask = 1 - r.yes_bid
            ev_yes = fair[s] - ask - kalshi_fee(ask)
            ev_no = (1 - fair[s]) - nask - kalshi_fee(nask)
            rows.append({"lg": lg, "event": evs[0], "start": g[0], "side": s, "fair": fair[s], "bid": r.yes_bid, "ask": ask,
                         "mid_gap": (r.yes_bid + ask) / 2 - fair[s], "ev_yes": ev_yes, "ev_no": ev_no, "pin_hold": inv.sum() - 1})
    print(lg, "matched", n)
d = pd.DataFrame(rows)
d = d[(d.ask - d.bid) <= 0.05]
print(len(d), "quotes; |mid - pinnacle fair| median", d.mid_gap.abs().median().round(4), "mean gap", d.mid_gap.mean().round(4))
print("by side mean gap:", d.groupby("side").mid_gap.mean().round(4).to_dict())
d["best"] = d[["ev_yes", "ev_no"]].max(1)
print("share with best EV after fee >= 1%:", (d.best >= .01).mean().round(3), ">= 2%:", (d.best >= .02).mean().round(3))
print(d.sort_values("best", ascending=False).head(12)[["lg", "event", "side", "fair", "bid", "ask", "ev_yes", "ev_no"]].round(3).to_string())
