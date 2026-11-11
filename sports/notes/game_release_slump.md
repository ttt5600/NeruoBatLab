# game_release_slump -- do athletes play worse after mega video-game releases? (user idea 11-11)

Agent: game_release_slump subagent (protocol: notes/AGENT_PROTOCOL.md). Dev data only, no orders, no git.
Status: PRE-REGISTERED 2026-11-11 (this section written before any outcome was aggregated).

## Question
After a huge video-game launch, do players across a league lose sleep and play worse? Both teams are exposed, so
sides/spreads should cancel. The testable market consequence is game TOTALS going UNDER the closing total in the days
after release, beyond what the close prices.

GTA VI: November 19, 2026 (PS5 / Xbox Series X|S; Rockstar moved it from fall 2025 -> 2026-05-26 -> 2026-11-19; the
latest confirmation I found is Take-Two's Feb 2026 call. I found no newer delay. Sources: engadget.com "delayed to
November 19, 2026"; gtaboom.com Feb 2026 earnings article).

## Event inclusion rule (fixed before outcomes; based on publisher-reported launch revenue only)
Include a title if it is a PAID console/PC game whose publisher (or widely cited launch PR) reported >= $500M revenue
(sell-through OR sell-in) within its first 7 days. Excluded by rule: free-to-play (Fortnite chapters, Warzone),
handheld-first (Pokemon), annual sports titles (Madden/2K: confounded, and launch with the season), titles with no
launch revenue figure (CoD Advanced Warfare 2014, Infinite Warfare 2016, Cold War 2020, Vanguard 2021; Skyrim, Halo 3/
Reach below $500M; Cyberpunk 2077 units only, not within 7 days; Elden Ring 12M in ~17 days).
Launch figures are from memory of publisher PR except where verified by search (GTA V ~$1B 3 days; RDR2 $725M 3 days;
BO3 $550M 3 days; BO4 $500M 3 days -- siliconera / activision blog / dexerto). The rule does not depend on outcomes,
so an error in a figure changes which events are in, not the direction of the test.

| Event | Release (US) | Basis |
|---|---|---|
| GTA IV | 2008-04-29 | $500M first week |
| CoD Modern Warfare 2 | 2009-11-10 | $550M 5 days |
| CoD Black Ops | 2010-11-09 | $650M 5 days |
| CoD Modern Warfare 3 | 2011-11-08 | $775M 5 days |
| CoD Black Ops II | 2012-11-13 | $500M day one |
| GTA V | 2013-09-17 | $800M day one, $1B 3 days |
| CoD Ghosts | 2013-11-05 | $1B sell-in day one |
| Destiny | 2014-09-09 | $500M sell-in day one |
| CoD Black Ops III | 2015-11-06 | $550M 3 days |
| Fallout 4 | 2015-11-10 | $750M sell-in first day |
| CoD WWII | 2017-11-03 | $500M 3 days |
| CoD Black Ops 4 | 2018-10-12 | $500M 3 days |
| Red Dead Redemption 2 | 2018-10-26 | $725M 3 days |
| CoD Modern Warfare (2019) | 2019-10-25 | $600M 3 days |
| CoD Modern Warfare II (2022) | 2022-10-28 | $800M 3 days |

Overlapping windows in one season (BO3 + Fallout 4, 4 days apart) are MERGED into one event cluster starting at the
first release. Rockstar subset (GTA IV, GTA V, RDR2) = the closest GTA VI analog: descriptive only (n = 3).

## Leagues / data (DEV ONLY; no sealed season is opened)
| League | Source | Seasons used | Price |
|---|---|---|---|
| NFL | lab/datasets.py nfl_total (data/lab/nfl_total.parquet), real close over/under odds | 2006..2019 (dev) | real |
| NBA | data/nba_odds/OddsData.sqlite (SBR close OU) | 2007-08..2018-19 (dev, same as nba_angles) | -110 |
| NHL | data/sbr_nhl (scripts/nhl_luck.py parse_sbr), close total + price | 2007-08..2022-23 (sealed = 2025-26 only) | real |
| CFB | data/sbr_cfb (scripts/cfb_simple_model.py parse_season), close total | 2007..2022 (2025+ sealed) | -110 |
MLB not used: mlb_total dev = 2015-2020; only RDR2 2018 hits it and that is the World Series (1 game/day). Player
props not tested: ESPN props are 2022+ = the sealed era for NBA/NFL (needs the user's yes); noted as a gap.

## Windows and control (fixed)
- Event day 0 = US release date. WINDOW W7 = days 0..6; W14 = days 0..13.
- CONTROL = same league-season, days -21..-8 (before) and +21..+34 (after) -- games outside any other event's
  [-7, +34] span are kept; games in another event's window are dropped from control.
- Residual r = (final total) - (closing total). Effect D = mean r(window) - mean r(control), computed per event x league,
  then averaged over events weighted by games; 95% CI = bootstrap over EVENT CLUSTERS (all leagues of one release move
  together). Also reported in league-SD units (r / league SD of r) for the pooled number.

## Trials (registered in edgelab, new family `game_release`; 2 trials)
- GR1: bet UNDER on every in-season game (NFL/NBA/NHL/CFB) in W7 at the closing price (real where available, else -110).
- GR2: same, W14.
SUCCESS (both needed): edgelab dev ROI 95% CI (clustered by event) lower bound > 0 and p < family bar, AND the
control-adjusted D < 0 with its CI excluding 0. KILL: ROI CI crosses 0 or D CI crosses 0 -> "not distinguishable",
idea dead for betting. A pass is still only a lead: GTA VI is one event, sealed-era data unopened.
Descriptive (no trials): per-league D, Rockstar subset, NBA pace/FG%/TOV if box data exist.

## Power (computed from game counts only, before outcomes) -- see section below
Source: `scripts/game_release_slump.py build` + `power` (results/game_release_slump/all_games.parquet, frame.parquet).
Games: W7 NFL 175 / NBA 388 / NHL 476 / CFB 727 (1,766); W14 352 / 802 / 962 / 1,444 (3,560); control 619 / 1,097 /
1,793 / 2,035. Gaps are structural: NBA has no MW3 (2011 lockout), no 2019/2022 (sealed); NHL no BO2 (2012 lockout),
GTA V/Destiny fall in NHL preseason (0 window games); BO4 2018 hits only NBA preseason-to-opening week (16 games).
Residual SD (final - close, all dev games): NFL 13.43, NBA 17.80, NHL 2.24, CFB 16.49.
MDE (80% power, two-sided .05, iid): W7 NFL 3.2 pts, NBA 2.9, NHL 0.32 goals, CFB 2.0; pooled 0.077 SD;
under win-rate MDE 3.3 pp (~6% ROI). W14 pooled 0.060 SD, 2.3 pp (~4% ROI). Event clustering (13-14 clusters) widens these.
READING: a plausible sleep effect (say 0.5-1 NBA point, ~0.03-0.05 SD) is BELOW what this design can detect. Only a large
league-wide slump (>~1.4 NBA pts / ~1 NFL pt pooled-equivalent) is detectable. A null here therefore = "no big effect",
not "no effect".

## Edgelab
Family `game_release` (new). Specs edgelab/specs/gr7_release_unders.yaml, gr14_release_unders.yaml; rule
edgelab/rules/game_release.py; loader game_release_totals; vault dev 2007-01-01..2023-06-30, sealed null.

## RESULT 2026-11-11 -- DEAD (dev), and the direction is wrong
edgelab (lab/edgelab_registry.jsonl; `python -m edgelab status`), cluster = release event, bar p < .025 (2 trials):
- gr7_release_unders  (days 0..6):  n 1,766, ROI **-4.97% [-8.63%, -0.95%]**, p .994 -> dead
- gr14_release_unders (days 0..13): n 3,560, ROI **-3.43% [-6.24%, -0.69%]**, p .990 -> dead
Control-adjusted residual (window minus days -21..-8 / +21..+34, same league-season; bootstrap over 14 event clusters;
results/game_release_slump/score.txt, `scripts/game_release_slump.py score`):
- W7 pooled D = **+0.020 SD [-0.032, +0.077]** (positive = MORE scoring after release) -> not distinguishable from 0.
  NBA +0.58 pts (+0.033 SD [-0.056,+0.110]); NFL +0.08 pts; NHL +0.12 goals (+0.053 SD [-0.013,+0.131]); CFB +0.14 pts.
- W14 pooled D = +0.014 SD [-0.023, +0.053]; every league CI crosses 0.
- Rockstar subset (GTA IV, GTA V, RDR2; n 268 W7 games): D +0.057 SD, under ROI -10.1% -- wrong sign, n = 3 events.
- Per-event W7 under ROI: 4 of 14 positive (BO3+FO4 +1.5%, BO4 +12.8%, WWII +1.4%, ...), 10 negative.
What it rules out: the lower 95% bound of the pooled W7 effect is -0.032 SD, i.e. a slump bigger than ~0.6 NBA points,
~0.4 NFL points, ~0.07 NHL goals per game is excluded. Smaller effects (sleep loss on a few players) are not testable
here and would not beat -110 anyway (break-even needs ~+2.4 pp under win rate = ~0.06 SD).
Window under ROI (-5.0%) vs control (-2.7%) ~ the vig: unders at the close win 49.6% in the window.
Rows spot-checked (RDR2 day 0: NBA/NHL/CFB 10-26-2018, NFL 10-28 Sunday, prices 1.87-1.97) -- aligned.
Not done: player props (ESPN props 2022+ = sealed era for NBA/NFL; needs the user's yes, and the totals null makes a
prop effect unlikely); pace/FG%/TOV (no historical NBA box data in data/ before 2015; nba_games.parquet has scores only).

## Forward-paper PROPOSAL (lead decides; $0, no orders)
GTA VI ships 2026-11-19 (Thu). If the lead wants a forward check anyway: log closing totals for every NBA/NHL/NFL/CFB
game 11-19..11-25 (W7) and 11-26..12-02 (W14 tail) plus control 10-29..11-11 and 12-10..12-23, from the ESPN core API
(existing loaders), paper UNDER at the close at $0. No kill needed -- one event cannot pass a gate (MDE for one event
~0.2 SD). Descriptive only. RECOMMENDATION: do not run as a betting test; at most a one-line residual check on 12-03
from data the existing loggers already collect.
