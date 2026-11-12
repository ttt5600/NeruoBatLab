# gamer_athletes -- do KNOWN-GAMER athletes slump after mega game launches? (user idea 11-11, player level)

Agent: gamer_athletes subagent (protocol notes/AGENT_PROTOCOL.md). Dev only, no orders, no git, no sealed scoring.
Sibling: game_release_slump (league totals; DEAD dev, D +0.020 SD [-0.032,+0.077]) -- launch list reused read-only.
Status: PRE-REGISTERED 2026-11-11 (written before any player outcome was computed).

## Question
Difference-in-differences: in the days after a mega game launch, do players publicly documented as heavy gamers
under-perform their own trailing baseline MORE than non-gamer controls playing in the same window?

## Events (fixed = sibling's list, notes/game_release_slump.md, scripts/game_release_slump.py read-only)
Dev = launches before 2022: GTA IV 2008-04-29, MW2 2009-11-10, BO 2010-11-09, MW3 2011-11-08, BO2 2012-11-13,
GTA V 2013-09-17, Ghosts 2013-11-05, Destiny 2014-09-09, BO3 2015-11-06 (+FO4 11-10 merged), WWII 2017-11-03,
BO4 2018-10-12, RDR2 2018-10-26, MW 2019-10-25. MW II 2022-10-28 = held out (2022+ era), size reported only.
Events with no regular/post-season games in a league simply contribute nothing there.

## Gamer list (frozen BEFORE any stats; see section "Frozen gamer list" + results/gamer_athletes/gamers_frozen.csv + sha256)
Built from web sources only (two researcher subagents, NBA + NFL), each player with source URL, evidence type and
EARLIEST evidence date. A player is "gamer" for an event only if evidence_date < event date (no hindsight). Before
that date he is excluded from both arms for that event (not used as a control). No adds/removes after outcomes.

## Data
- NBA: stats.nba.com LeagueGameLog (player, regular season + playoffs) seasons 2007-08..2021-22 via nba_api
  -> results/gamer_athletes/nba_playerlogs.parquet. Held out: 2022-23+ (size only).
- NFL: nflverse player weekly stats (data/nfl_player_week.parquet = 2015+; older seasons from nflverse releases if
  reachable). Dev = seasons <= 2021.

## Outcome and design (fixed)
- Player-game residual: stat - player's own trailing baseline = mean of his previous 20 games played (min 8; NFL:
  previous 8 games, min 4), strictly BEFORE the game date (no lookahead), crossing season boundaries, EXCLUDING any
  game inside any event's days 0..13 (so the baseline is pre-treatment).
- NBA stats: PTS (primary), MIN, FG% (FGM/FGA; residual on player-game FG%, weighted by FGA), PLUS_MINUS.
  NFL stats: PPR fantasy points (primary), skill positions QB/RB/WR/TE, plus snaps-proxy = touches+targets.
- Windows: W7 = days 0..6, W14 = days 0..13 after release (NFL: games in those days = 1-2 weeks).
- Treated = gamer-player-games in window. Controls = all non-gamer player-games in the SAME window and SAME league
  (same games/weeks), with matching: same position group (NBA: none available in logs -> match on baseline minutes
  bucket [<15, 15-25, 25-32, 32+]; NFL: position) . Also a pre-period comparison: same players days -21..-8.
- DiD per event = [mean resid gamers(window) - mean resid controls(window)] - [same in pre-period days -21..-8].
  Pooled over events weighted by gamer-game count. 95% CI = bootstrap over EVENT clusters (resample events),
  and as a secondary, over players.
- Minutes (NBA) / touches (NFL) reported to separate availability/fatigue from efficiency.

## Pre-registered success rule
PASS (lead only) = pooled W7 DiD on PTS (NBA) and on PPR (NFL) both NEGATIVE, at least one with event-cluster 95% CI
excluding 0, the other same sign; and the sign is negative in a majority of events with >= 5 gamer games.
KILL = pooled DiD CI crosses 0 in both leagues (-> "not distinguishable"), or sign positive.
If < 5 event clusters have gamer games in a league, the cluster CI is not trusted; then player-cluster CI is reported and
the verdict is at most "underpowered / not distinguishable".
If PASS: write (not run) a sealed 2022+ prop-unders test spec in edgelab + GTA VI paper-log list. Lead decides.

## Frozen gamer list (2026-11-11, before any stats were loaded for outcomes)
results/gamer_athletes/gamers_frozen.csv -- 50 players (NBA 25, NFL 25), sha256
9759d90d513bc78a92d62efbbcdd53c3cf652e69f76a986fcfafa2a191149b52 (results/gamer_athletes/gamers_frozen.sha256).
Raw researcher output with duplicate rows: gamers_nba_raw.csv, gamers_nfl_raw.csv. Rule for duplicates: keep the
EARLIEST evidence row per player (fixed before outcomes). YYYY-MM dates -> day 28 (conservative).
Evidence types: a stream, c esports ownership/investment, d interview/article on heavy play.
Key structural fact: almost all evidence is dated 2016-2020 (one Bleacher Report Fortnite piece 2018-03-20 alone
covers 8 NBA players). Only Arenas (2006) and Hayward (2011) qualify for the 2008-2015 launches. Dev launches after
2018-03 are RDR2 2018-10-26, BO4 2018-10-12 (NFL only) and MW 2019-10-25 -> very few event clusters.
Dropped by researchers (no qualifying source): Embiid, Oubre, Kuzma, Herro, Tatum, Dinwiddie, Lamar Jackson,
Kamara, etc. Retired-before-evidence: Shaq, KG, Ochocinco (listed, contributes 0 games).
Data note: stats.nba.com timed out on 11-11 (all requests) -> NBA box scores from ESPN site API summary endpoint,
only the date spans needed (scripts/gamer_athletes.py NBA_SPANS; fetch_nba_espn). NFL: nflverse player_stats
2009-2014 (github release) + data/nfl_player_week.parquet 2015+, dates from nflverse games.csv.

## Power (filled from counts only, before outcomes)
Amendment A1 (before outcomes): BO4 (10-12-2018) days 0..13 overlapped RDR2's pre period (-21..-8); add_offsets now
lets a game inside any event's 0..13 never serve as a pre/control game of another event.
NFL (scripts/gamer_athletes.py prep('NFL'), counts only): dev gamer player-games W7 = 25 (WWII 2, BO4 7, RDR2 8,
MW19 8), pre = 41, W14 = 46; 9 distinct players; 4 event clusters (WWII only 2 games). PPR residual SD (all dev
skill players) 6.78 -> MDE (80%, iid) = 4.8 PPR points/game, i.e. ~70% of a typical WR's output. Held-out MWII 2022:
7 W7 games. => NFL cannot detect anything plausible. Pre-registered rule: < 5 event clusters -> verdict at most
"underpowered / not distinguishable".
Amendment A2 (decided and CODED before the first score run, from counts only; this note text was written a few
minutes after the score run because an edit failed -- the code change, PRE=False in scripts/gamer_athletes.py, preceded
scoring). Reason: ESPN scoreboard includes PRESEASON -> dropped (openers = first date after the 3-day gap: 2009-10-27
... 2019-10-22). The -21..-8 pre period then has ZERO NBA gamer games for 6 of 9 events (it falls in preseason for
early-Nov launches). PRIMARY estimator = window-only DiD: (gamer residual vs own 20-game baseline) - (mean residual
of non-gamers in the same event window and same baseline-minutes bucket [NFL: same position]); the baseline is the
"before" difference. Pre-period-subtracted version = secondary. Success/kill rules unchanged.
NBA (counts only): dev gamer player-games W7 = 140 (RDR2 54, MW19 41, BO4 15, WWII 10, BO3+FO4 7, Ghosts 4, BO2 4,
BO 3, MW2 2), W14 = 311; 21 distinct players; 5 events with >= 5 W7 games. PTS residual SD (all players) 6.10
-> MDE (80%, iid) = 1.44 pts W7, 0.97 pts W14; player clustering (21 players) widens this. ~80% of gamer games
come from Fall 2018-2019 -- the event-cluster CI rests on ~5 effective clusters.

## RESULT 2026-11-11 -- DEAD (dev); the NBA sign is the WRONG way
Source: results/gamer_athletes/score.txt (`scripts/gamer_athletes.py score`), dev only (< 2022), primary = A2
window-only DiD vs matched non-gamers; event-cluster bootstrap 4000, player-cluster bootstrap 1000.
- NBA PTS W7: **+1.02 pts [+0.25, +2.12] event-CI** (9 events), player-CI [-0.39, +2.38] (21 players), n 140;
  0 of 5 events (n>=5) negative. W14 +0.75 [+0.07, +1.66], player-CI [-0.50, +2.12], n 311.
  -> gamers scored MORE than matched controls after launches. Kill rule (sign positive) met.
- NBA MIN W7 +1.12 [-1.17, +2.36] (no fewer minutes); FG% W7 -0.003 [-0.016, +0.019]; W14 FG% -0.010
  [-0.019, -0.003] event-CI but player-CI [-0.028, +0.006] -> not distinguishable (and 1 of ~8 secondary stats).
  +/- W7 -0.10 [-2.94, +1.60]; W14 -1.02 [-2.77, -0.05] event-CI, player-CI [-2.73, +0.61] -> not distinguishable.
- NFL PPR W7 +1.54 [-0.86, +5.13] (4 events, 25 games, 10 players) -> not distinguishable, positive point.
  Touches W7 -1.23 [-2.66, +0.88], W14 -0.92 [-1.70, +0.25]; 3/3 events negative but n tiny -> not distinguishable.
- Rows spot-checked (RDR2 2018-10-26: Curry 29 vs baseline 25.1, Durant 41 vs 26.7, Towns 16 vs 22.9) -- aligned.
Likely reason for the positive NBA sign (not tested; a guess): the list is dominated by young players whose 2018-19
evidence comes from 2018 articles (Fox +6.3/game, Hart, Drummond, Beal), and early-season games vs a trailing
20-game baseline from the PREVIOUS season reward improving young players; the minutes-bucket match does not control
age/trajectory. Either way there is no slump to bet.
Caveats: the event-cluster CI is narrow partly because tiny events (Ghosts 4 games +9.8, MW2 2 games) carry little
weight; the player-cluster CI is the honest one and it crosses 0 for every stat. ~80% of gamer games are Fall 2018-19.
No betting trial was run, so nothing registered in edgelab (performance DiD only; no prop prices touched).
Held out: 2022+ (MWII) -- NFL 7 W7 gamer games; NBA 2022+ not fetched (dev answer already kills it).
NOT DONE (by rule, since it did not pass): sealed 2022+ prop-unders spec and GTA VI paper list.
GTA VI week: no action recommended. If the lead wants a $0 look anyway, the list in gamers_frozen.csv (evidence before
2026-11-19 = all 50) could be logged vs closing props 11-19..11-25 as descriptive only -- one event can't pass a gate.

## Log
- 2026-11-11 pre-reg written; gamer-list researchers launched; data fetch started.
