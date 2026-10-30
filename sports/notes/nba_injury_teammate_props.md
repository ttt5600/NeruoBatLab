# Q: (A) do NBA OPENING teammate props under-adjust when a rotation player is OUT? (B) do hot/cold streaks predict actual - open?

Brief 2026-10-30 (coordinator). Ledger datasets: `nba_injury_teammate_props` (A), `nba_hot_streaks` (B).
Code: scripts/nba_injury_props_reports.py (injury reports 2022-25, news side), scripts/nba_injury_teammate_props.py (signals,
join, trials). Output: results/nba_injury_teammate_props/.

Prior evidence (why a pass is unlikely): nba_sim's own check found the ESPN BET game line moves only ~half the sim's
absence delta open->close and the sim overstates single-player deltas (absence coefficient ~4); Q011 found half the
market move precedes the official report; FUND_BOOK: NBA shooting luck priced at team level; nba_player_opp_venue: the
book shades overs (over hit 48.3% vs no-vig 49.7%, hold 6.9%), so OVER trials start ~ -5% ROI.

## PRE-REGISTRATION (written 2026-10-30 BEFORE any signal was joined to a prop line or outcome)

What I had looked at before writing this (structure only, no signal, no outcome):
- ESPN props (`graded_{2023,2024,2025}.parquet`) have NO opening timestamp: only `line_open/dec_open` and a
  `last_updated` that is post-tip (hours_before ~ -3.7). So "was the OUT known before the open?" cannot be read
  per quote. It is answered two ways below: (i) WHEN the official report first listed him OUT, bucketed, and
  (ii) a measured incorporation slope (how much of the sim delta is already in the opening line), by bucket.
- Official injury report PDFs exist for 2022-23..2024-25 at the Q011 URL (hourly labels; probed 2023-01-15 05PM:
  header 05:30 PM, Last-Modified 22:30:04 GMT = 17:30 ET, so label HH publishes at HH:30). The 2022-25 PDFs use a
  different column layout than 2025-26; the parser locates columns from each page's header row (checked on the
  2023-01-15 and 2025-01-15 05PM reports: 129 and 161 rows, both include the next day's games).
  Fetched labels: 07,09,11 AM, 01,03,05,07,09,11 PM ET on every game day and the day before (6,255 labels, 2 req/s).
  Publication time pub = max(Last-Modified, PDF header) (Q011 rule).
- nba_sim.py redistribution (project_minutes / side_inputs) and State/walk (point-in-time box state).

Data / books (as nba_player_opp_venue): DEV = 2022-23 DraftKings, 2023-24 DraftKings, 2024-25 ESPN BET, OPENING
line and price only. One obs per player-game-market: the over/under pair at the open line with no-vig p closest to .5.
Pushes leave the sample. SEALED = 2025-26 Oct-Nov (ESPN BET, ~299 games; Q011 already holds those reports from
2025-10-21) -- usable, NOT loaded, scored once only on the user's explicit yes and only if a DEV trial passes.

### A. Injury -> teammate props
Definitions (all state from games dated strictly before the game; nba_sim conventions):
- Rotation player j of team T: in T's full set (players in T's last 10 games, still on T) with mean minutes over
  his last 10 games played >= 25.0.
- NEWS absence: j played T's PREVIOUS game, did not play this game, and was listed Out or Doubtful in the last
  official report published before tip that lists this game. (Carryover absences -- he also missed the previous
  game -- are already in the teammates' recent form and are not events. Unlisted DNPs are late scratches nobody
  could see pre-tip: counted and reported, NOT in trials.)
- Report -> ESPN join: report "Last,First" -> normalized first+last == normalized ESPN box name of a player on that
  team in the season (Q011 rule); report game -> ESPN event by (ET game date, team code via nba_sim.ALIAS).
  Unmatched/ambiguous counted.
- Timing class of a NEWS absence by t_out = pub of the FIRST report listing him Out/Doubtful for this game
  (he must stay Out/Doubtful through the last pre-tip report):
  EARLY  t_out before 10:00 ET on game day (incl. the day-before reports),
  MID    10:00 <= t_out < 17:00 ET game day,
  LATE   t_out >= 17:00 ET game day.
  Hypothesis for the split: DraftKings/ESPN BET NBA props are believed to open on game-day morning -> EARLY is
  (almost surely) known at the open, LATE almost surely not. The incorporation slope below tests this belief.
- Signal (nba_sim): for each teammate i on T, delta_i(stat) = proj_i(avail - NEWS) - proj_i(avail), where avail =
  full set minus every player listed Out/Doubtful in the last pre-tip report who did not play, proj = minutes from
  side_inputs (team participation weighting, known_play=False) x role-shrunk per-minute rate. NOTE the sim only moves
  MINUTES; per-minute usage is held fixed (that is the given model, not tuned here).
- Top beneficiary per (game, team, stat) = teammate with the largest delta among those who have an opening quote
  in that stat.

Trials (A, all on NEWS absences pooled over timing classes unless stated; OVER at the open price):
- A1 points: top beneficiary, delta_pts >= X = 2.0.
- A2 rebounds + assists pooled: top rebound beneficiary with delta_reb >= 1.0 and top assist beneficiary with
  delta_ast >= 1.0.
- A3 all teammates pooled, all three stats: every teammate quote with delta >= 1.0 pts / 0.5 reb / 0.5 ast.
- A4 regression: (actual - line_open) ~ delta_pts over ALL teammate points quotes in NEWS team-games
  (OLS slope; one-sided slope > 0). Slope 0 = open fully priced the sim delta.
- A5 = A1 restricted to EARLY (the only class where betting the open with this signal is surely tradeable).
LOOKAHEAD CAVEAT (decided now): in MID/LATE the news may postdate the open. A bet "at the open" there uses
information that did not exist when that price was posted; it is tradeable only if the line was still up and
unmoved when the news hit, which we cannot see. So a pass that lives in MID/LATE is NOT a tradeable edge at the
open; it is evidence of staleness and would need a timestamped-line forward test. Reported per class.

Descriptive (not trials): share of NEWS absences per timing class (+ carryover, unlisted counts) = "how often news
preceded the open"; incorporation slope per class = OLS of (line_open_i this game - line_open_i in his previous
quoted game) on delta_pts (1 = the open fully moved by the sim delta, 0 = not at all); the same for 2023-24
"current" (the one season where current looks pre-game); actual - projected-without-news vs delta (did the sim
delta come true: realized slope).

### B. Hot / cold streaks
- Baseline b_t = player's mean points over games t-25..t-6 (20 games, needs >= 10), i.e. EXCLUDING the last 5.
- hot_t = sum over the last 5 games (t-5..t-1) of (pts - b_t) / (5 + K), K = 5 (shrink to 0 with a 5-game prior);
  needs all 5 games. Descriptive alternate: 3P% residual over the last 5 (3PM - 3PA x baseline 3P%), same shrink.
- Cutoffs: p10 / p90 of hot among all box player-games (>= 10 baseline games) of the PREVIOUS season (walk-forward).
- B1 points OVER at the open when hot >= p90.
- B2 points UNDER at the open when hot <= p10.
- B3 OLS (actual - line_open) ~ z(hot) over all points quotes (z by previous-season SD); one-sided slope > 0
  (momentum). (If the true sign is negative = market over-reacts; that is reported, but it is not the registered
  direction, and the over-shading means B2 is the natural test of it.)
- Reliability (box only, descriptive): corr(hot over games t-5..t-1, mean residual over t..t+4) for non-overlapping
  windows (one window pair per player per 10 games), and split-half (odd vs even games within the 10).
  Expected near zero.

### Bar (both studies, k = 8 trials A1-A5, B1-B3)
Bonferroni one-sided p < 0.05 / 8 = **0.00625**. p = share of 4000 GAME-DATE-clustered bootstrap resamples with
units <= 0 (slope <= 0 for A4/B3). PASS = p < bar AND observed ROI (slope) > 0 in both DEV halves (split at the
median bet date). Reported: bets, observed ROI with date-cluster 2.5/97.5 percentiles, hit rate minus no-vig p.
Kill: 0/5 A -> A dead; 0/3 B -> B dead; one line each to FUND_BOOK 3d (by the lead).
Too-good guard: any ROI > +10% on > 200 bets is a bug until rows are inspected (injury pub vs tip, join, side).
Power note: points SD ~7; ~5 pp of hit rate is needed to beat the ~7% hold on overs.

### Amendments (2026-10-30, after building the NEWS side, BEFORE any outcome or line was joined)
1. "Last pre-tip report": the 2-hourly grid's last pre-tip report is often 17:30 for a 19:00 tip, and 27% of
   rotation DNPs looked "unlisted" because they went Q -> Out in the 18:30 report I did not fetch. Spot check of 12
   such cases: 11 were listed Out in the first post-tip report (e.g. KCP 2025-02-01, Brunson 2024-01-13, Towns
   2025-02-26). Rule now: reports published up to tip + 90 min count (they show the pre-tip status of a game that
   has started); a first Out listing seen only after tip is class LATE. Unlisted rotation DNPs fell 750 -> 190.
2. Name aliases (report -> ESPN): Jimmy Butler -> Jimmy Butler III, Carlton -> Bub Carrington, Alexandre -> Alex
   Sarr, Brandon Boston -> Jr., Reggie Bullock -> Jr., Cui Yongxi order. Remaining unmatched report rows are
   two-way / never-played names (1.7% of rows overall).
3. Team-game timing class with several NEWS players = the LATEST of their classes.
4. Implementation of delta: baseline avail = full set minus listed-Out non-players, WITH the news players; tonight =
   without them (first build had them removed from both -> all deltas 0; caught before any join).
News-side counts after the amendments (results/nba_injury_teammate_props/signals_summary.json): 7,920 team-games,
7,776 with report coverage; 2,596 NEWS absences in 2,040 team-games: EARLY 542, MID 471, LATE 1,027 team-games;
4,451 listed carryover rotation absences; 190 unlisted rotation DNPs. Top-beneficiary sim delta_pts per NEWS
team-game: median 2.75, max 15.9 (BKN rest game with 42-min caps binding).
Hot-streak reliability (box only, run before any line join): first-5 vs next-5 residual r = +0.40 [0.39, 0.42],
odd/even +0.50 over 16,949 blocks -- NOT near zero, because the baseline excludes the last 5 games, so "hot" also
measures genuine role/minutes drift that a stale 20-game mean misses. Next-game correlation vs a 25-game baseline
that includes the streak: +0.156; for the 3P makes-above-expected version: +0.105.

### Provenance of the scoring (2026-10-30)
- The amendments above (tip + 90 min rule, name aliases; code edit ~01:17 PT) were made BEFORE I had seen any trial
  outcome: my first `run` happened after they were written here. The delta fix (amendment 4) was made earlier, on a
  partial-report signals build, also before any join.
- Separately, the lead session (after a /clear, not knowing this study was still running) ran `table`, `signals`
  and `run` at ~01:10 PT on the PRE-amendment code (NEWS 2,036, unlisted 750; A1 -2.2% on 1,133 bets, A4 +0.080
  p .081) and wrote a RESULTS section from it. I never saw that output before my amendments, and it has been replaced
  by the section below. Its conclusions are the same (0/8).
- Ledger: the lead's `run --ledger` at 08:18:12Z already used the amended code (sha 618ff4a5686e). I then appended a
  second, identical set at 08:18:47Z (same code, same numbers) before I got the lead's message. **The 08:18:47 rows
  (lab/ledger.jsonl lines 429-436) are a duplicate and should be ignored or removed; 08:18:12 is canonical.** This is
  8 trials, not 16.

## RESULTS (run 2026-10-30; DEV 2022-23..2024-25 at the ESPN OPEN; 0/8 pass, bar p < 0.00625) -- A DEAD, B DEAD
Source: results/nba_injury_teammate_props/results.json (+ bets_A1/A3/B1/B2.csv, team_games.parquet,
teammate_deltas.parquet, signals_summary.json). Ledger: lab/ledger.jsonl, 8 rows (ts 2026-10-30T08:18:47, sha =
sha1 of the script file 618ff4a5686e; scripts uncommitted at run time). SEALED 2025-26 not loaded.
Join: 148,059 open quotes; 36,292 teammate quotes in NEWS team-games (pts 11,395 / reb 11,564 / ast 9,539).
Base rate, all open quotes, OVER ROI: points -9.4%, rebounds -11.7%, assists -9.3% (over hit - no-vig p: -1.4,
-2.5, -1.0 pp). Every OVER trial starts from there.

| trial | n | ROI or slope [date-cluster 95%] | hit - no-vig p | p | halves | pass |
|---|---|---|---|---|---|---|
| A1 pts top beneficiary over, delta >= 2 | 1,438 | -2.3% [-6.7, +2.3] | +2.4 pp | .83 | +0.3% / -4.7% | no |
| A2 reb/ast top beneficiary over | 1,467 | -15.3% [-20.6, -9.8] | -4.6 pp | 1.00 | -19.2% / -11.5% | no |
| A3 all teammates over (pts/reb/ast) | 18,494 | -7.6% [-9.2, -6.0] | -0.4 pp | 1.00 | -9.2% / -6.0% | no |
| A4 slope (actual - open) ~ delta_pts | 11,395 | +0.085 [-0.006, +0.179] | | .035 | +0.119 / +0.052 | no (p) |
| A5 = A1, EARLY news only | 354 | -2.1% [-10.9, +6.7] | +2.4 pp | .68 | +0.7% / -4.7% | no |
| B1 hot (>= p90) over | 6,064 | -12.4% [-14.7, -10.1] | -2.9 pp | 1.00 | -11.8% / -13.0% | no |
| B2 cold (<= p10) under | 4,712 | -7.2% [-10.0, -4.4] | -0.3 pp | 1.00 | -6.9% / -7.6% | no |
| B3 slope (actual - open) ~ z(hot) | 46,316 | -0.156 [-0.216, -0.098] pts/SD | | 1.00 | -0.131 / -0.187 | no (wrong sign) |

### A: how often did the news precede the open, and did the open include it?
- NEWS team-games by when the official report first listed the player Out/Doubtful: EARLY (before 10:00 ET game
  day, incl. day-before reports) 542 (27%), MID 471 (23%), LATE (>= 17:00 ET, incl. 6% of final Out listings first
  seen only in the first post-tip report) 1,027 (50%). 190 rotation DNPs were never listed (late scratches).
  The prop open has no timestamp, so the measured answer is the incorporation slope below.
- Incorporation (teammate points open minus his previous open, <= 10 days apart, on delta_pts): EARLY +0.53
  [0.40, 0.64], MID +0.49 [0.37, 0.61], LATE +0.39 [0.32, 0.46]. Realized (actual - no-news projection on delta):
  ALL +0.63 [0.51, 0.74]. So the sim delta is ~1.6x too big, the open moves ~0.4-0.5 per sim point, and the gap to
  the realized 0.63 is what A4 measures: +0.085 (p .035), below the bar and ~0.2 pts for a 2-pt delta, far inside
  the ~9% over-hold.
- 2023-24 open -> "current" (pre-game-looking) move on delta: EARLY +0.017 [-0.09, +0.13], MID +0.024, LATE +0.108
  [+0.035, +0.194]. So EARLY/MID news is fully in the open (nothing left to move); LATE news is partly AFTER the
  open (the line keeps moving ~0.1 per sim point) -- consistent with ~half of NEWS cases postdating the open, and
  with the books catching up before tip. A1 at "current" 2023-24: -11.6% (494) vs -5.7% at the open.
- By class (descriptive): A1 EARLY -2.1% (354), MID +2.5% [-7.9, +13.0] (333), LATE -4.5% (751). The LATE cases
  (the "line could not include it" ones) are NOT better than EARLY; and betting them at the open would be lookahead
  anyway (the price was posted before the news).
- A1 by season: 2022-23 +2.0% (523), 2023-24 -5.7% (494), 2024-25 -3.5% (421).
- What A1 does show: the top beneficiary's over beats the no-vig p by +2.4 pp and his ROI (-2.3%) is ~7 pp better
  than the base points over (-9.4%) -- a real but small under-adjustment that does not pay the over-shading.
  Rebounds/assists beneficiaries are the opposite (A2 -4.6 pp): books over-adjust those.

### B: hot streaks
- The market OVER-reacts: the open sits +1.73 pts per SD of hotness above the 20-game baseline [1.69, 1.77],
  actual comes in +1.58 [1.51, 1.64]; hence B3 slope -0.156 pts/SD (registered direction was momentum > 0: wrong
  sign, not a pass). The fade (B1 hot -> UNDER, descriptive) is -1.3% [-3.6, +1.0] (6,064): still under water.
  3P-luck version: slope -0.092 [-0.151, -0.031] per SD.
- Reliability of "hotness" is NOT near zero (+0.40 first-5 vs next-5, +0.50 odd/even, vs a baseline that excludes
  the streak) because it carries role/minutes drift; vs a baseline that includes the streak, next-game r = +0.156
  (box only). The persistent part is real and the books price it -- slightly too much.

### Verdict and lessons
A DEAD (0/5), B DEAD (0/3). Mechanism: the opening line already moves ~0.4-0.5 pts per sim point of redistributed
minutes for news known by morning; the residual under-adjustment (~0.09 per point) is ~0.2 pts per bet against a
~9% over-hold. "Late" news is not exploitable at the open by construction and the books close it before tip.
Hot streaks are over-priced by ~0.16 pts/SD, but fading them still pays the vig.
Oddities: (1) sim deltas hit 42-min caps in rest games (max delta_pts 15.9); (2) ESPN open has no timestamp, so the
timing classes are by report time, not by open time; (3) the NEWS definition uses "did not play" -- only 8 of
29,477 final Out/Doubtful listings played, so this is ~no lookahead; (4) one-process PDF parsing was ~0.4/s, the
fetch was re-run as 5 shards paced to <= 2 req/s total.
