# kalshi_soccer_sharp -- S12 logic with Kalshi as the "soft book" (agent started 2026-11-13)

## Question
E3/S12 (soft books vs power-de-vigged Pinnacle, edge >= 3%) passed sealed but is untradeable from CA. Does the same
rule work with Kalshi soccer 1X2 contracts as the soft book, TAKER at the ask + fee?

## Data (what exists, checked BEFORE any outcome was loaded)
- `~/.claude/quant-kalshi/data/gap_log.jsonl`: one row per Kalshi contract side whose EV at the ask after the taker fee
  was >= 1% vs Pinnacle POWER de-vig (soccer: fair_lo = fair_hi = power fair), per 2-min tick since 2026-10-19.
  CENSORED: only flagged sides are stored and raw Pinnacle odds are NOT stored -> Shin cannot be recomputed from
  history, and Pinnacle close is not stored -> CLV vs Pinnacle close is impossible from history.
  For thresholds >= 1% the flagged set is complete (every quote with power EV >= 1% was written), so a 3% rule on
  power is fully reconstructable on history.
- `gap_ticks.jsonl`: per-tick quote counts per league (denominator).
- Kalshi API (read-only GET): market result + candlesticks -> settlement and Kalshi close (CLV proxy).

## Pre-registration (written 2026-11-13 before any settlement / candle was fetched)
Trial KS12-P (dev = gap_log history 2026-10-19 .. 2026-11-13, matches already settled):
- Universe: gap_logger soccer leagues (epl laliga seriea bund ligue1 efl mls ucl ered), pre-game, lead >= 10 min,
  asks in [0.02, 0.98] (logger filters).
- Signal: EV = (fair_power - ask - fee(ask)) / (ask + fee(ask)) >= 0.03 on YES or NO of H/D/A.
- Entry: FIRST tick on which a given (ticker, side) qualifies; taker at the logged ask; 1 contract.
- Outcome: Kalshi settlement (90 min + stoppage). Fee = live.kalshi_fee (0.07*p*(1-p)).
- Stats: ROI = sum pnl / sum cost; 95% CI + one-sided p by bootstrap over MATCHES (event), 10k resamples.
- CLV proxy: Kalshi close = last 1-min candle YES mid (bid/ask) before kickoff (Pinnacle start time), for the side
  bought; CLV = close_value - ask (prob points). Pinnacle-close CLV = forward only.
- SUCCESS (-> forward paper, not money): ROI CI lower > 0 AND p < cross_venue family bar after registering this trial
  AND mean Kalshi-close CLV > 0.
- KILL: ROI point <= 0 OR mean Kalshi-close CLV <= 0 (with n >= 30 bets) -> dead on dev.
- If n < 30 bets or < 20 matches: verdict = "too thin", write the forward paper rule instead.
- Descriptive only (not trials): thresholds 1/2/5%, by outcome (H/D/A), by side (YES/NO).
Shin variant KS12-S: forward only (raw odds not logged). Diagnostic now: live snapshot of Pinnacle soccer odds,
power vs Shin fair per outcome, and how many current power flags survive Shin.

## Log
### 2026-11-13 census (no outcomes) -- `scripts/kalshi_soccer_sharp.py census`
Soccer flagged rows 15,830 (10-19 16:19Z .. 11-13 08:03Z). First-tick bets: EV>=1% 121 / 80 matches; >=2% 71 / 54;
>=3% 40 / 32 matches (tie share 0.12; 30 YES, 10 NO); >=5% 21 / 15. Matched soccer quote-sides per tick: median 207
(~288 in Oct, 102-156 on 11-09..11-13). The old F3 "mostly draws" headline counted ROWS (draw gaps persist across
many ticks); by first-tick bet only 12% are draws.

### 2026-11-13 KS12-P result (pre-registered) -- `build` + `score`; rows results/kalshi_soccer_sharp/ks12p_bets.csv
- 37/40 settled bets, 30 matches: ROI -26.19%, 95% CI [-63.35%, +12.69%] (match-clustered bootstrap), p 0.907,
  units -3.19 on 1-contract stakes. Hit 0.243 vs mean fair 0.355.
- Kalshi-close CLV (last 1-min candle mid before kickoff minus ask) mean -0.0115, CI [-0.0226, +0.0006], >0 share 0.27.
- KILL met (ROI <= 0 and CLV <= 0, n >= 30) -> DEAD on dev. Registered + retired in edgelab:
  ks12p_kalshi_soccer_power-55e7078e (cross_venue now 10 trials, bar p < 5.0e-03).
- Descriptive: >=1% ROI -8.5% [-25.1, +6.4] CLV -0.0068 (n112); >=2% -10.1% [-36.0, +15.7] CLV -0.0101 (n67);
  >=5% -77.9% [-100, -17.6] CLV -0.0144 (n18). Bigger "edge" = worse: the gap is Pinnacle stale / mis-matched.
- Row audit: WHU-WBA (EFL, 10-31) logged EV +146% = TEAM-MATCH BUG in gap_logger._first(): "West Ham" and
  "West Bromwich" both reduce to "west". That bet happened to win; excluding it ROI is -33%.
- Kalshi moved toward its own price, away from Pinnacle's, into kickoff (73% of bets negative CLV): Kalshi soccer
  is not a lagging soft book on these signals.

### 2026-11-13 Shin vs power (live snapshot, `shin`; results/kalshi_soccer_sharp/shin_snapshot_20261113T0806.parquet)
122 Pinnacle 1X2 matches, 52 matched to Kalshi (international break: none kick off within 7 days).
Mean Shin - power fair: draw +0.24pp (range +0.03..+1.03), home -0.21pp, away -0.02pp. Shin makes draws MORE likely
than power, so the suspected "power over-states draws" artefact runs the other way; power was the conservative one
for draw YES. Flags now: power 1 side >= 3% of 310, Shin 3 (Barca-Villarreal draw, Real Madrid-Celta draw).
Kalshi mids sit 0.1-0.5pp below both fair sets on average (no systematic gap).

### Forward / proposals (not deployed; lead decides)
- No Shin forward trial proposed: the power version has negative Kalshi-close CLV, and Shin only adds big-favourite
  draws (history: draw YES 0/3 at >= 3%). Not worth a cross_venue trial slot.
- gap_logger fixes (proposal): (1) team matcher -- full normalized name or exact Kalshi yes_sub_title map; the "west"
  collision mis-sides WHU/WBA; (2) log raw Pinnacle decimal odds + age for every matched match every ~15th tick, so
  Shin and Pinnacle-close CLV become computable; (3) mlb/wnba KeyError noise in gap_ticks.
- Volume if ever forward: ~11 bets/week at >= 3% (40 in 3.6 weeks); ~30-50 Kalshi+Pinnacle matched soccer matches
  listed at a time in club weeks (207 quote-sides/tick / 6).

## Learnings
### Kalshi soccer is not a soft book vs Pinnacle
S12 rule on Kalshi (power de-vig, EV >= 3% at ask + fee): 37 bets/30 matches ROI -26% [-63, +13], Kalshi-close CLV
-0.0115 [-0.0226, +0.0006]; bigger flagged edges did worse (>= 5%: -78%). Source: this file,
results/kalshi_soccer_sharp/ks12p_bets.csv. The book-vs-Pinnacle soccer edge (E3) does not transfer to Kalshi.
### Shin raises soccer draw probability relative to power
On 122 live Pinnacle 1X2 lines, Shin - power = +0.24pp on the draw, -0.21pp home. "Fix Shin" cannot remove draw
flags; it adds them. Source: results/kalshi_soccer_sharp/shin_snapshot_20261113T0806.parquet.
### Count gap flags by first tick, not by row
F3 "mostly draws" came from row counts; persistent gaps repeat every 2 min. By first-tick bet, draws are 12%.
### gap_logger team matcher collides on first word
"West Ham"/"West Bromwich" -> "west" produced a fake +146% EV row. Any first-token matcher needs a uniqueness check.
