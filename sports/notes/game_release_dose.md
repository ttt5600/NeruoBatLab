# game_release_dose -- does the post-launch totals effect grow with launch HYPE? (user pushback 11-11)

Agent: game_release_dose subagent (protocol notes/AGENT_PROTOCOL.md). Dev data only, no orders, no git, no sealed scoring.
Builds on notes/game_release_slump.md (prior agent: 14 launch clusters >= $500M, W7 pooled D +0.020 SD [-0.032,+0.077],
dead). User: "GTA VI will be far bigger than GTA V; look at extremely hyped games."

## PRE-REGISTRATION (written 2026-11-11 BEFORE any new outcome was computed)
Contamination, stated up front: the prior agent already published per-event W7 D for its 14 clusters (positive pooled,
Rockstar subset +0.057 SD). So the slope is not fully blind: about half of the clusters' outcomes are known. The NEW
launches added here (and every 2023+ launch) are unseen.

### Hype measure (primary, ONE): launch-week revenue H
H = the publisher/tracker-reported sales of the title within its FIRST 7 DAYS (the figure covering the longest window
<= 7 days), worldwide where given. Revenue in nominal USD if reported; if only units are reported, H = units x US
standard-edition launch price ($60 before 2020-11, $70 after; $40 Pokemon on DS/3DS; Switch titles $60, $70 from 2023).
Sell-in and sell-through both accepted (flagged). No scaling across different day-windows (1 vs 3 vs 7 days) --
that measurement error attenuates the slope toward 0 (a known bias against success, accepted).
Sources: researcher subagent, web search, one cited URL per title (table below). Figures recalled-but-unverified are
flagged; a title with no <= 7-day figure is EXCLUDED (not estimated).
Why revenue, not Google Trends: no Trends client here and Trends is not comparable across 18 years; revenue is what GTA
and CoD publishers report and what the GTA VI forecasts are stated in.

### Inclusion
Paid software (any platform), H >= $250M. Excluded by rule: free-to-play (Fortnite chapters, Warzone, Apex), annual
sports titles (Madden/FIFA/2K: launch with the season, confounded), hardware launches (a $400 console x 1M units is
not 1M x engagement; Switch 2 HARDWARE therefore out, Mario Kart World in only if it has its own <= 7-day figure),
expansions/DLC. Fortnite: excluded (F2P, no revenue figure) -- decided now, not after.
Clusters: launches whose release dates are within 6 days of each other merge into one cluster that starts at the first
release; cluster H = sum of member H.

### Eras
- DEV (this analysis): games dated <= 2022-12-31 in the prior agent's dev seasons: NFL 2006-2019 (lab nfl_total, real
  close odds), NBA 2007-08..2018-19 (SBR, -110), NHL 2007-08..2022-23 (SBR, close price) cut at 2022-12-31, CFB
  2007..2022 (SBR, -110). Elden Ring (2022-02) and Pokemon Scarlet/Violet (2022-11) fall in dev (NHL/CFB rows only).
- SEALED (prep only, NOT scored): games dated 2023-01-01..2026-10-31, all four leagues from ESPN lines (data/espn/*/
  lines_*.parquet `total`, real over/under price) -- launches 2023+ (Hogwarts, TotK, Diablo IV, BG3, Starfield, ...).

### Model (primary)
z = (final - closing total) / league SD (SD over all dev games; NFL 13.43, NBA 17.80, NHL 2.24, CFB 16.49 from prior).
Rows: window games (days 0..6 after cluster start, W=1) and matched control games (same league-season, days -21..-8
and +21..+34, W=0; a game inside any cluster's days 0..13 is never control; a game in two windows is kept once, in the
earlier cluster).
OLS:  z = alpha[cluster x league] + W * (b0 + b1 * (ln H - ln $1B)) + e
  b1 = the dose slope per log-unit of hype; b0 = window effect at GTA-V scale ($1B in 3 days).
CI: cluster bootstrap over launch clusters (4,000 reps, percentile); CR1 cluster-robust t(G-1) CI also reported.
GTA VI extrapolation ASSUMPTION (fixed now): H_VI = $3B in its first <= 7 days (3x GTA V's $1B 3-day record; range
$2B-$5B reported as sensitivity). THIS IS AN EXTRAPOLATION BEYOND THE DATA (max dev H ~ $1B): it assumes the effect is
linear in log hype past the largest observed launch.
  effect_VI = b0 + b1 * ln(3).  "Upper bound" = the most negative end (2.5th pct) of its bootstrap CI.
Secondary (report only): same window term on ALL dev games with league x season FE + league x ISO-week FE.

### Decision rule
SUCCESS (both): b1 < 0 with 95% CI excluding 0, AND effect_VI point estimate <= -0.06 SD (the ~+2.4 pp under win-rate
needed to beat -110). KILL: b1 CI crosses 0 (-> "not distinguishable") or effect_VI > -0.06 SD -> no GTA VI bet.

### Sealed trial (prep; registered, NOT scored)
grh7_hyped_unders: UNDER every in-season NFL/NBA/NHL/CFB game at the close on days 0..6 after any launch cluster with
H >= $600M (fixed now, before the hype table is in). Dataset game_release_hype: dev = rows <= 2022-12-31, sealed =
2023-01-01..2026-10-31, forward 2026-11-19+. edgelab only unseals sealed after the SAME trial passes dev.

## Log
- 11-11: pre-registration above written; researcher subagent sourcing launch-week figures (no outcomes touched).
- 11-11: HYPE TABLE FROZEN (before outcomes) in scripts/game_release_dose_hype.py: H per title + basis. Researcher
  sources (one URL each; most verified via search summaries only, original PRs not opened): GTA V $1B/3d gamespot 6414762;
  GTA IV gamespot 6190520; MW2 phys.org 2009-11-duty_1; BO gamedeveloper "650m first five days"; MW3 cbsnews 775m;
  BO2 engadget 2012-11-16; Ghosts nbcnews 8c11545197 (SI); Destiny theworkprint (weak); BO3/FO4 ibtimes 2183997 /
  mcvuk; WWII thurrott 143405; BO4 gamespot 1100-6462642; RDR2 gamesradar 725-million; MW2019 gamesbeat 600-million;
  MW II gamespot 1100-6508775; Halo 3 nbcnews wbna20998966; Fallout 3 gamedeveloper 4-7-million; BF3 gamespot
  1100-6343166; Diablo III engadget 2012-05-23; Division businesswire 20160315006723; Sun/Moon hardcoregamer 236362;
  SwSh (no URL); Pokemon SV businesswire 20221123005476; TotK videogameschronicle 10-million-3-days; Diablo IV
  activision newsroom; Wukong bloomberg 2024-08-24; MH Wilds businesswire 20250304186922; BF6 VGC 7-million-3-days;
  Z-A VGC 5-8-million. No <= 7-day figure: Elden Ring, Hogwarts, Cyberpunk, BG3, Starfield, CoD 2020+ (BO6/BO7),
  Mario Kart World -> excluded by rule (the lead's list named some of them; the rule, not the list, decides).
  GTA VI forecasts (researcher): Newzoo 2026-07-16 week 1 37-51M units / $3.3-4.5B; Piper Sandler >45M day 1;
  Bloomberg-cited >25M day 1; Ampere 30-35M by end 2026. Pre-registered H_VI = $3B is at/below Newzoo's low case;
  $5B sensitivity covers Newzoo's high case.
- 11-11 BUILD (labels only; `scripts/game_release_dose.py build` -> results/game_release_dose/dev_frame.parquet,
  clusters.csv): 22 dev clusters with H >= $250M (BO3+FO4 merged, H $1.3B); 9,880 rows, 2,565 window games
  (CFB 1,000 / NBA 553 / NFL 246 / NHL 766). Pokemon S/V has NHL window games but 0 control (SBR ends 2022-11-27) ->
  absorbed by its fixed effect, contributes nothing.
- 11-11 POWER (before scoring; `... power`): ln H spans only 19.52..20.99 (= $300M..$1.3B, a 4.4x range); GTA VI at $3B
  is ln 21.82, i.e. 0.83 log-units beyond the top dev cluster and 1.10 beyond GTA V. iid SE of b1 = 0.055 SD per log-unit
  (MDE 0.154); SE of effect_VI = 0.095 SD (MDE 0.267). READING: the design cannot detect a GTA-VI-scale effect smaller
  than ~0.27 SD even before cluster heterogeneity -- the extrapolation is the binding constraint, not n.

## RESULT 2026-11-11 -- KILL (dev): no dose-response; the slope has the WRONG sign
Source: `scripts/game_release_dose.py score` -> results/game_release_dose/score.txt (boot.npy = 4,000 cluster resamples).
22 clusters, 9,880 games (2,565 window). z = (final - close)/league SD; negative = unders.
- b1 (per ln-unit of hype) = **+0.042 SD**, boot [-0.046, +0.188], CR1 t(21) [-0.055, +0.140]; P(b1<0) 0.17.
  -> not distinguishable from 0; the point estimate says MORE scoring after bigger launches.
- b0 (window effect at $1B, GTA-V scale) = +0.024 SD, boot [-0.039, +0.146], CR1 [-0.058, +0.106].
- effect at GTA-VI scale ($3B, EXTRAPOLATION 1.1 log-units past GTA V, 0.83 past the top dev cluster):
  **+0.070 SD**, boot [-0.073, +0.356], CR1 [-0.109, +0.250]. $2B: +0.053 [-0.058,+0.280]; $5B: +0.092 [-0.096,+0.454].
- Most negative plausible GTA-VI effect (pre-registered: 2.5th pct boot) = **-0.073 SD** (CR1: -0.109). Even that
  edge of the CI is only ~1.2-1.8x the 0.06 SD break-even, and it rests on a linear-in-log extrapolation.
- Secondary (all 50,493 dev games, league x season + league x ISO-week FE): b0 +0.035, b1 +0.049, effect_VI +0.089.
- Per-cluster D (window - control): the four biggest (GTA V +0.068, Ghosts -0.084, BO3+FO4 -0.021, MW II -0.064)
  average ~-0.03; the $500-800M tier has the largest positives (MW2019 +0.224, BO +0.195). No monotone pattern.
- Pre-reg verdict: KILL (b1 CI crosses 0 and its sign is +; effect_VI +0.070 > -0.06).
- The bootstrap CI is right-skewed vs CR1 (few high-H clusters carry the slope; resamples that drop them swing it), so
  quote both. Rows spot-checked: window k 0..6, control k in [-21,-8]u[21,34], no duplicate game_ids, GTA V CFB
  rows from 2013-09-19 aligned with the prior frame.

## SEALED PREP -- grh7_hyped_unders (registered; NOT scored; cannot be under edgelab rules)
- Trial id **grh7_hyped_unders-9d169a91** (family game_release now 3 trials, bar p < .0167). Spec
  edgelab/specs/grh7_hyped_unders.yaml, rule edgelab/rules/game_release_hype.py, loader game_release_hype (loaders.py),
  vault dataset game_release_hype (dev <= 2022-12-31, sealed 2023-01-01..2026-10-31, forward 2026-11-19+).
- DEV evaluated (allowed): n 1,129 under bets at H >= $600M, ROI **-7.90% [-11.37, -4.01]**, p 1.0, halves -6.5% /
  -9.3% -> verdict fail, state **dead**. So edgelab's vault will REFUSE the sealed split (it requires dev_pass).
- Sealed coverage (counts only, `... sealed`): 4 clusters with H >= $600M -- BF6 + Pokemon Z-A (2025-10-10, H $838M,
  99 games), Zelda TotK (12), Diablo IV (5), Black Myth Wukong (4) = **120 games** (CFB 38, NBA 9, NFL 15, NHL 58).
  MH Wilds ($560M) is below the bar; Hogwarts/Elden Ring/BG3/Starfield/BO6/BO7/Switch 2 have no <= 7-day sales figure.
- Power: MDE under win-rate 12.8 pp (~24% ROI) at n 120; power to detect the +2.4 pp break-even = **0.08**, +4 pp = 0.14.
- Command (it will be refused while the trial is dead -- do not force):
  `cd sports && ../.venv_quant/bin/python -m edgelab evaluate grh7_hyped_unders --split sealed`
  RECOMMENDATION: do not ask for it. Dev failed, the sealed set is 120 games at 8% power, and 99 of them are one cluster.

## What this means for GTA VI
The data cannot rule out a GTA-VI-scale slump of up to ~0.07-0.11 SD (~1.3-2 NBA points, ~0.2 NHL goals) because the
extrapolation is long, but nothing in 22 launches points that way: bigger launches, if anything, saw slightly MORE
scoring. Any forward check stays $0 paper (see the notes/game_release_slump.md proposal).
