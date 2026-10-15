# Live scanning + execution

## Scanner (paper only)
`python scripts/live_edges.py scan|loop --every 10|report` — sports/live.py.
- Fair price: Pinnacle guest API (guest.api.arcadia.pinnacle.com/0.1/leagues/{889 NFL, 880 NCAAF, 487 NBA,
  1456 NHL, 246 MLB, 578 WNBA}/matchups + /markets/straight), public web-app key in code.
- Compared: Kalshi game series KX{NFL,NCAAF,NBA,NHL,MLB,WNBA}GAME asks after fee 0.07*p*(1-p).
- Matching: Kalshi ticker date (ET) + city-prefix names. Flag = first time EV >= 2% (conservative de-vig).
- Logs: data/live/kalshi/<ts>.parquet, data/live/flags.jsonl. `report` = CLV vs last pre-game Pinnacle + settlement.
- Durability: loop dies with the session. launchd cannot read ~/Desktop -> copy code outside Desktop to schedule.

## Execution plan (not built)
- Kalshi official API only. User creates key, stores in sports/.env (never shown to Claude).
- Script defaults to paper; live needs `--live` + per-bet and daily caps set by user.
- Gate to live: a few hundred flags with mean CLV > 0 (t > 2), then small stakes.
- Sportsbook bets (DK/FD etc.) = manual placement from flags; no automation.
- Existing: scripts/kalshi_paper.py (temperature strategy paper trader).
