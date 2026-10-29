# FUND BOOK -- master index of the prediction-market research fund (written 2026-10-29)

Every edge we have ever considered, how the fund is run, what runs live, how research is done, and what waits on the PM.
Numbers are copied from the file cited next to them; nothing here is new analysis. REPO IS PUBLIC: no brokerage
balances or positions appear here (they live only in gitignored private notes, which this book never quotes).

Citation keys: **[S]** STATE.md · **[H5]** notes/HANDOFF_2026-10-29.md · **[H4]** notes/HANDOFF_2026-10-25.md ·
**[EA]** notes/EDGES_AND_AGENTS.md · **[R]** notes/results.md · **[L:x]** lab/ledger.jsonl dataset x · **[LB]**
lab/lockbox.jsonl · **[Q]** QUESTIONS.md · **[FS]** `scripts/fund_stats.py report` run 2026-10-29 · **[FE]**
notes/fund_economics.md · **[FA]** notes/FUND_ARCHITECTURE.md · **[PL]** notes/promising_leads_2026-10-26.md.

---------------------------------------------------------------------------------------------------------------------
## 0. How to use this book

Start here after any /clear. Section 1 = the rules the PM set; 2 = what is running with money and how to stop it;
3 = the edge registry (every idea, one row each, grouped by status -- check section 3d before proposing anything,
it is the do-not-redo list); 4 = how a new idea becomes a trade; 5 = commands; 6 = lessons; 7 = what the user owes us.
Update this file (and EDGES_AND_AGENTS.md, which keeps the long-form detail) whenever a trial is scored or a status moves.

| What | Where |
|---|---|
| Session resume | `sports/STATE.md` (top blocks), latest `notes/HANDOFF_*.md` |
| Long-form registry (numbers, agent IDs, briefs) | `notes/EDGES_AND_AGENTS.md` |
| Every trial (371 rows) / sealed scores (15 rows) | `lab/ledger.jsonl` / `lab/lockbox.jsonl`; stocks: `lab/stocks_ledger.jsonl` (15) |
| Older trial ledgers | `lab/reports/prop_loop_ledger.jsonl`, `parlay_ledger.jsonl`, `pattern_confirm.jsonl`, `pattern_survivors.json` |
| Stats gate | `scripts/fund_stats.py` (report / verdict / families / flags / power) |
| Pre-registered questions | `QUESTIONS.md` (Q001-Q008), hub project `fund`, notes log `notes/FUND_NOTES.md` |
| Fund blueprint / economics / cycle | `notes/FUND_ARCHITECTURE.md`, `notes/fund_economics.md`, `notes/FUND_CYCLE.md` |
| Model internals | `notes/MODELS_GUIDE.md` |
| Sizing | `notes/risk.md`, `notes/kalshi_sizing.md` |
| **Edge math standard (binding, protocol rule 8)** | `notes/EDGE_MATH.md` (12-step table + worked math per edge) via `scripts/edge_math.py`; playbook `notes/TRADING_PRINCIPLES.md`; situational factors tested/untested `notes/SITUATIONAL_FACTORS.md` |
| Venues | `notes/venues.md`, `notes/venue_expansion.md`, `notes/soft_venues.md` |
| Deployed code (launchd can't read ~/Desktop) | `~/.claude/quant-kalshi/` (own venv; redeploy with `scripts/install_*.sh`) |
| Daily brief output | `notes/briefs/brief_YYYY-MM-DD.md` |
| Private (gitignored, never copy numbers out) | notes/private_account.md, robinhood_snapshot.md, robinhood_strategies_private.md, thesis_v2_private.md |

---------------------------------------------------------------------------------------------------------------------
## 1. Fund mandate & management

**LONG-RUN GOAL 10-29 (user):** ~$400k over time. Reality check given (session 10-29): today's edges are capacity-capped (~$150-300/month even at $5k; profit = edge x absorbable volume, not % of capital); compounding only works with edges that absorb capital. Path: prove live -> add income-funded capital as edges prove capacity -> find capacity-rich edges (LIP market making, more venues, options/volatility). Never via over-Kelly risk or the leveraged brokerage account.

**CAPITAL DECISION 10-29 (user):** no new deposits yet; current ~$50 Kalshi + ~$60 IBKR = small-stakes live test of E1/E2 + E5; scale only after forward results pass the gates below.


| Item | Rule (source) |
|---|---|
| Goal | Double a $1-2k UNLEVERED bankroll in 12 months, proven live; scale capital only as verified qualified daily volume grows (~$140-280/day of 2%-edge bets) [S], [Q] |
| Base bankroll | $2,000 on Kalshi, user to deposit (NOT yet deposited as of 10-29) [S], [H5] |
| Unit | 1u = $50 (2.5% of $2k). Units = dollars at risk incl. fee (scripts/bet.py docstring) |
| Tier A (sealed pass: weather E1/E2/E4) | <= 1u per night total [S] |
| Tier B (paper lead with forward evidence) | <= 0.5u per bet [S]. E5 ForecastEx runs as Tier B [S] |
| Tier C (judgment) | <= 0.25u per bet, monthly C loss cap [S] |
| Loss stops | daily -3u, weekly -6u [S]; executor stops in .env (section 2) |
| Capital gates | $2k -> **$5k** when tier-A live ROI stays inside the sealed CI ~60 days AND verified qualified volume > ~$250/day; -> **$10k** when a 2nd edge (LIP / Polymarket-US maker / ForecastEx weather) passes sealed + forward [S] |
| Funding | Never fund betting from the leveraged brokerage account [S], [H5] |
| Router | Every bet (API or phone) through `scripts/bet.py`; phone bets via `bet.py log-external` [S] |

**PM (user) decides:** sealed tests ("yes <name>"), real-money size and caps, .env changes, deposits/allocations,
going live on a new edge, spend on data/APIs. **Agents may:** read-only status/report commands, dev-only backtests
(pre-registered, ledgered), build paper loggers, self-approve cheap reversible questions (no money, no orders, no
sealed data) [Q], [notes/FUND_CYCLE.md hard rules]. Agents never place orders, touch .env, score sealed data, or
print account values [FUND_CYCLE.md rules 1-3].

**Kelly guidance.**
- General: fractional Kelly on the CONSERVATIVE edge, never the point estimate; f* = (p - c)/(1 - c); quarter Kelly
  default; daily stop 5% of B, weekly 10%; turn a strategy off if live CLV/ROI after 200 bets is below the sealed CI
  [notes/risk.md].
- Weather E1/E2: confidence-based sizing NOT adopted -- price already carries the confidence; YES hits ~0.69x the
  NO-ask-implied rate (sealed 0.665); quarter Kelly at the 80% LCB = ~4.4% of B per pick, flat; the lever is unit size +
  maker [notes/kalshi_sizing.md].
- Weather maker capacity: above ~$75/pick adverse selection eats the gain; maker ceiling ~$144/month at plan ROI [FE].
- E5 ForecastEx: half-Kelly on the LCB ~5% per pick; liquidity (~29 contracts/pick) binds before Kelly does [H5]. Derivation (session 10-29): mean cost 0.898 + $0.01 fee -> b = 0.092/0.908 = 0.101; sealed win rate 0.954 (10/219 losses) -> full Kelly ~0.50; at the Wilson 95% LCB p = 0.918 -> f* = (0.101*0.918 - 0.082)/0.101 = ~0.11 -> half ~5%. Per-night total cap still needed (same-night weather picks are correlated). **Correction 10-29 (notes/EDGE_MATH.md):** the lockbox mean_cost .8977 already includes the $0.01 fee, so c = .8977, f*(LCB) = .198, half 9.9%, quarter 5.0%; the 5% pick cap and ~$26 liquidity bind either way.

---------------------------------------------------------------------------------------------------------------------
## 2. Live operations

### 2.1 Venues / accounts
| Venue | Access | Status |
|---|---|---|
| Kalshi | API (Ed25519 key in ~/.claude/quant-kalshi/keys; user-owned) | LIVE: weather executor E1/E2 taker. Account ~$49.7 on 10-29 [H5]. NBA markets sit on a separate shard ("insufficient_shard_balance") -> `bet.py shards`, `bet.py allocate 0=70 3=30 --confirm` after the $2k deposit [S], [H5] |
| IBKR ForecastEx | IB Gateway 10.50 + IBC 3.24.2 in ~/ibkr, launchd `com.jonathanwang.ibc-gateway`; `FX_BACKEND=tws` (ib_async); password from Keychain item `fund-ibkr` | LIVE E5 since 10-28/29. Logged in 10-29 09:54; gateway must stay running (minimized); auto-restarts nightly 11:45 PM without 2FA; **weekly re-auth (Sunday, IBKR Mobile)**. Old Client Portal Gateway (port 5050) retired [S], notes/ibkr_ibc_setup.md |
| Polymarket US | `scripts/polymarket_us.py` / `scripts/venues/` client: read-only works; trading gated by `--live` + user API key | read-only; F4 paper logger [EA], notes/venue_expansion.md |
| Robinhood | robinhood-trading MCP lives in peer session `connect-robinhood-mcp`; agent trades ONLY the empty agentic account; main + Roth read-only (even review_* forbidden) | NO event-contract API (its event contracts route to Kalshi/ForecastEx; manual only) [S], [EA s6], notes/venues.md |
| Sportsbooks | manual only (no API; automating breaks ToS); CA has no legal books; Bovada refused (offshore) | E3 soccer not tradeable from CA [S] |

### 2.2 launchd agents (all from ~/Library/LaunchAgents; deployed copy ~/.claude/quant-kalshi; loaded = `launchctl list`, 10-29)
| Label (com.jonathanwang.*) | What | Cadence | Report / check |
|---|---|---|---|
| quant-kalshi-paper | `kalshi_paper.py tick` (E1/E2 picks 22:00-23:59 city-local, 24 cities) then `kalshi_exec.py run --live` (REAL $) | hourly :03 | `kalshi_exec.py status`; `kalshi_paper.py report` |
| quant-forecastex-exec | `forecastex_exec.py run --live` (E5 LIVE, maker at best bid) | 5 min | `forecastex_exec.py status` |
| quant-forecastex-paper | FX1-fwd paper on public bid/ask + Q008 econ | 5 min | `forecastex_paper.py report` |
| quant-forecastex-live | old per-pick live script (other session) | 5 min | **plist present, NOT loaded** (stopped 10-29) [H5] |
| ibc-gateway | `~/ibkr/gatewaystartmacos.sh -inline` (IB Gateway via IBC) | hourly :07 | port 4001 up / ForecastEx status |
| quant-fund-brief | `fund_brief.py` daily PM brief -> notes/briefs/ | 07:30 daily | read the brief |
| quant-lip-paper | L40 LIP paper quotes (no orders) | 60 s | `lip_paper.py report` |
| quant-nba-open-totals | L32 forward + Q004 pick_tight | 10 min | `nba_open_totals.py report` |
| quant-open-lines | NHL/NFL open-line loggers + Q004 | 15 min | `open_lines_logger.py report --league nhl\|nfl` |
| quant-kalshi-wx-monthly | L39 monthly rain/snow paper (decides 22:00 local, last 7 days of month) | hourly :05 | `kalshi_wx_monthly_logger.py report` |
| quant-pmus-weather-paper | F4 Polymarket US weather paper + same-pick Kalshi price | 5 min | `pmus_weather_paper.py report` |
| quant-live-board | NBA/NFL/NHL live board recorder (no orders) | 30 s | `live_board.py show\|report`; `lb <team>` |
| quant-nba-midgame | L37 mid-game recorder (sealed FAIL; calibration only) | 90 s | `nba_midgame_scan.py report` |
| quant-nba-bigleads | L28 logger | 2 min | **plist present, NOT loaded** (L28 dead) [S] |
| quant-live-signals | F2 NFL pre-kickoff rain -> totals paper | 15 min | `live_signals.py report` |
| quant-gap-logger | F3 Kalshi vs Pinnacle game gaps | 2 min | `gap_logger.py report` |
| quant-thesis-tracker | F6/F7 thesis v1/v2 baskets | weekdays 17:30 (+ 08:00 catch-up [H4]) | `thesis_tracker.py report` |
| quant-paper-step | separate 60/40 paper-journal project (`~/.claude/quant-paper/launch.py`) | weekdays 14:30 | its own journal |

All report commands run as `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/<x>.py ...`.

### 2.3 Executors + caps (.env key NAMES only; values are the user's)
- **Kalshi** (`~/.claude/quant-kalshi/.env`): `EXEC_ENV`, `EXEC_MODE`, `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`,
  `KALSHI_BANKROLL_USD`, `KALSHI_MAX_CONTRACTS`, `KALSHI_MAX_ORDER_USD`, `KALSHI_MAX_DAILY_USD`, `KALSHI_MAX_OPEN_USD`;
  not yet set: `KALSHI_EXEC_STYLE=maker` (E4) or `KALSHI_AB=1` (A/B). Current sizing per [H5]: bankroll setting 400,
  3 contracts, $30/day, $40 open; loss stops 24h -$20 / 7d -$40 [S 10-27]. Taker.
- **ForecastEx**: `FX_BACKEND` (tws), `FX_MAX_ORDER_USD`, `FX_MAX_CONTRACTS`, `FX_MAX_DAILY_USD`,
  `FX_EXEC_LOSS_STOP_USD`, `FX_ALLOW_LIVE_ACCOUNT`, `FX_AUTO_CONFIRM`, `IBKR_MODE`, `IBKR_API_BASE`, `IBKR_ACCOUNT_ID`.
  Caps raised by user 10-29: $15/order, 20 contracts, $100/day, loss stop -$30 (backup .env.bak-2026-10-29) [S].
  TWS contract date = EVENT day (fixed, commit 7e31e02).
- Backups of every .env change sit next to it as `.env.bak-<date>[-tag]` [S].

### 2.4 Maintenance checklist
- **Daily (morning):** read `notes/briefs/` brief; `kalshi_exec.py status` (fills, KILL, errors); `forecastex_exec.py
  status`; confirm IB Gateway alive; `bet.py pnl`; glance at `lip_paper.py report` (until 11-08).
- **Game nights:** `lb <team>` / `live_board.py show` (decision support only); `nba_tonight_sheet.py` for the card.
- **Tuesdays:** `nfl_ingame_fade.py interim` (F1, descriptive only, until 11-30).
- **Weekly:** Sunday IBKR Mobile approval for IBC; `fund_stats.py report`; `launchctl list | grep quant` (16 expected
  loaded incl. ibc-gateway); check disk space.
- **After any code edit:** redeploy with the matching `scripts/install_*_agent.sh`, then a status run.

### 2.5 Emergency stop
1. `touch ~/.claude/quant-kalshi/KILL` -- halts the Kalshi executor and the ForecastEx executor (shared KILL) [S].
2. Unload: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.jonathanwang.quant-kalshi-paper.plist` and
   `.../com.jonathanwang.quant-forecastex-exec.plist` (re-enable with `launchctl bootstrap gui/$(id -u) <plist>`).
3. Cancel resting orders: `bet.py cancel --all` (Kalshi); ForecastEx orders are DAY / cancelled at cutoff by the executor.

---------------------------------------------------------------------------------------------------------------------
## 3. EDGE REGISTRY

Status words: PASSED SEALED = dev pass then one-shot sealed pass on the user's yes (the only real edges). PAPER = running
forward, no money. LEAD = dev/descriptive signal not past the bar, or awaiting a PM decision. DEAD = failed; do not
redo without a new mechanism. "dev" = development-period backtest; "sealed" = lockbox, scored once.

### 3a. PASSED SEALED -- live or ready (5)
| ID | Edge | Mechanism | Evidence | Status | Files |
|---|---|---|---|---|---|
| E1 | Kalshi daily-high temp consensus-reject, orig 7 cities: buy NO on brackets all 5 forecast models miss by >3F, ~22:00 local night before | stale far tails in a retail weather book; forecasts beat the price on brackets nobody models | sealed 2025Q3-2026Q4 +1.45% ROI, 1,844 bets, p<.0001 [LB], [EA]; fwd n 38 [FS] | **LIVE taker** since 10-22. E1+E2 live 59 settled 58-1, P&L -$0.67 (first loss 10-29, expected) [H5] | scripts/kalshi_paper.py, kalshi_exec.py, notes/go_live.md |
| E2 | X1 = same frozen rule on 17 more daily-high cities | same | sealed 2026-03..10 +0.78%, 2,464 bets, p .0047 [LB]; fwd n 91, -0.6%, p .66 [FS] | **LIVE** with E1 | scripts/wx_expand.py |
| E4 | E1/E2 executed as MAKER (post-only NO bid at mid, <= ask-1c, rest to close, no taker fallback) | weather charges takers only; saves spread + fee | sealed 4,308 picks: ROI +1.07% -> +2.22%, +30.6u [+26.2,+34.9] p<.001, fill 80% [LB], [R] | **code deployed, OFF** until user says "yes maker" (`KALSHI_EXEC_STYLE=maker`) [H5] | scripts/kalshi_temp_maker.py, notes/kalshi_temp_maker.md |
| E3 | Soccer S12: best of B365/WH/bwin vs power-de-vigged Pinnacle pre-close, edge >=3%, price <10 | soft books lag the sharp book | sealed 2019-26 CLV +3.63%, ROI +8.71%, 2,004 bets, p .010 [LB] | **not tradeable from CA**; US-book version (L9) untested | scripts/soccer_fd.py |
| E5 | ForecastEx (IBKR) weather consensus-reject (WU settlement, different stations) | same tail mispricing on a thinner, independent venue | sealed PASS-UNDERPOWERED 10-28: 219 bets, 10 losses, +15.05u, ROI +6.9%, p .0004; last-trade prices, median 29/contract, below 500 floor [EA], [LB] | **LIVE Tier B** (user overrode the >=100 paper-pick gate) [S] | scripts/forecastex_exec.py, forecastex_wx.py, notes/forecastex_wx.md, forecastex_setup.md |

### 3b. PAPER / FORWARD TESTING (13) -- judge rule in the last column
| ID | Edge | Mechanism | Evidence | Judge rule | Files |
|---|---|---|---|---|---|
| FX1-fwd | E5 rule on live ForecastEx bid/ask | E5 at real quotes | 93 picks by 10-29 [H5] | originally >= 100 settled before live (overridden) [S] | scripts/forecastex_paper.py |
| Q008 | ForecastEx econ: buy the Kalshi-favoured side when FX is >= 2c cheaper | FX = compressed copy of Kalshi + FLB at >= .85 | dev exploratory 64 bets/57 events, 0 losses, ROI +16% at last trade [Q] | judge at 60 settled: KILL if ROI <= 0 or qualifying-ask rate < 25%; early kill >= 3 losses in first 30 [Q] | forecastex_paper.py (q008), notes/q007_forecastex_breadth.md |
| L40/Q002 | Kalshi Liquidity Incentive Program on weather brackets (paper quotes) | LIP pays resting size; 75% of temp brackets paid nobody (YES side < 1,000) | $20/bracket/period; 1,000 YES @1c could claim up to ~$10/period [EA], notes/kalshi_lip.md | pre-registered 10-day test 10-29..11-08; program ends 2027-01-01 | scripts/lip_paper.py |
| L32 | NBA totals vs OPENER (model >= 3 pts off first-posted total) | stale opener | dev +8.7% no-vig p .001; **sealed FAIL** +6.4% p .054 (~25% power) [LB]; forward 12 picks CLV +1.42 p .005 [FS]; 8-4, CLV +1.4 [H5] | logger rule at 100 settled: CLV <= 0 or ROI CI < 0 kills [Q004]; never re-score sealed | scripts/nba_open_totals.py, nba_simple_model.py |
| Q004 | Totals models taken at the first TIGHT Kalshi quote (NBA/NHL/NFL) | timing edge needs early low-fee entry | deployed 10-28 (L35 basis: Kalshi early NFL CLV +3.3c p<.0002) [Q] | >= 100 settled pick_tight: CLV > 0 AND ROI at ask+fee > 0 -> sealed ask [Q] | nba_open_totals.py, open_lines_logger.py, kalshi_tight.py |
| L34 | NHL ML/totals vs opener (open-lines logger) | stale opener | dev CLV passes (p<1/4000), money fails (totals no-vig +7.1% p .0125) [EA]; fwd n 20 [FS] | gate PAPER; >= 100 settled [FS] | open_lines_logger.py, nhl_simple_model.py |
| L33 | NFL spread/total vs opener | stale opener | dev CLV +0.76/+0.81 pts; totals C +6.6% at -110 p .014 [EA]; fwd n 0 [FS]; live coefficients refit 2020-25 -> sealed forfeited, forward only | gate PAPER; >= 100 settled | open_lines_logger.py, nfl_simple_model.py |
| L39/Q006 | Monthly rain/snow "unreachable" NO (maker) | thresholds already unreachable still trade a few cents | dev W1m +3.6%/146, 1 loss, market-null p .016 (bar .0083) [L:kalshi_wx_winter] | kill: any 2 losses in first 100; one sealed shot ~Feb 2027; capacity ~$7.5/month [Q], [FE] | scripts/kalshi_wx_monthly_logger.py, kalshi_wx_winter.py |
| F4 | Polymarket US weather consensus-reject (6 cities matching Kalshi) | second book for the same bet | dev -0.14%/283 p .53 [L:pmus_temp]; forward "= Kalshi price" [H5] | pre-registered 10-28 (trial F4_pmus_weather_paper) | scripts/pmus_weather_paper.py |
| F1 | NFL in-game maker fade of the post-big-play overshoot | Kalshi overshoots ~1.8 pts then reverts (post-hoc) [R] | forward games 2026-10-22..11-30 | score ONCE after 2026-12-02; `interim` Tuesdays (descriptive) [EA] | scripts/nfl_ingame_fade.py |
| F2 | NFL pre-kickoff rain -> totals under | market under-reacts to weather | dev +8.0%/219 p .11 (after join-bug fix); old sealed INVALID [R]; gate DEAD [FS] | recorder; fires ~15-20x/season | scripts/live_signals.py |
| F3 | Kalshi vs Pinnacle gap logger (EV >= 1% after fee) | Kalshi lags the sharp book? | 2,244 ticks/168 episodes, almost all soccer draws (3-way de-vig artefact suspected), 0 US flags [H4] | N4: fix Shin de-vig then re-judge | scripts/gap_logger.py |
| F6/F7 | Thesis baskets: v1 inference vs training; v2 memory / compute-inference / energy-gen / energy-equip (rules 7e822f587214) | user thesis | hindsight backtest only; forward record since 10-23 [H4] | forward record, no pass bar | scripts/thesis_tracker.py, notes/inference_thesis.md, thesis_v2.md |

Recorders (no edge claimed, kept for calibration): L37 mid-game scanner, live_board (NBA/NFL/NHL), F5 Sleeper pick'em
logger (built, NOT installed; PK1 pass = >= 200 legs, ROI > 0, p < .05) [EA].

### 3c. LEADS -- awaiting a sealed test, a logger, or a PM decision (34)
| ID | Edge | Mechanism | Evidence | Status / next | Files |
|---|---|---|---|---|---|
| Q001 | E4 maker live + per-pick cap ladder toward ~$75 | maker > taker per $ | sealed E4 (above); ceiling ~$144/mo at plan [FE] | **PM: "yes maker"**; success = live maker ROI inside sealed CI [+1.57,+2.84]% after >= 486 filled [Q] | QUESTIONS.md Q001 |
| Q003/L41 | Kalshi Mentions maker NO (IN1_d6) | retail YES over-buying; maker side | dev IN1_d6 +7.08%/859 fills p<.0002, MID_d6 +5.8%; d24 dead; decaying (halves +11.5/+2.9%) [EA], [L:q003_retail_maker] | **SEALED ASK "yes Q003"** (events >= 2026-06-25) | scripts/q003_retail_maker.py, notes/q003_retail_maker.md |
| L36 | CFB totals vs opener | stale opener | dev CLV +1.02 pts, no-vig +7.4% p .0004, -110 +2.6% p .10 [L:cfb_simple_model] | gate LEAD, no logger [FS]; build CFB logger | scripts/cfb_simple_model.py |
| L35 | Kalshi early NFL game prices stale at listing | first listing before sharp money | CLV +3.3c p<.0002 (109 games); maker at first bid adverse +16% p .09; taker first ask -4.5% [L:kalshi_early_nfl] | folded into Q004 forward (don't re-mine the 109) | scripts/kalshi_early_nfl.py |
| L30 | P-PAINT: high paint-pts-allowed teams | style the line misses | -0.54 ATS/SD p 2.5e-5, 10/10 seasons [EA] | feed into L2 game model | scripts/nba_comebacks.py |
| L31 | P-TOT3: season 3P% -> unders; big spreads -> overs | 3P% regression | walk-forward 1,417 bets 53.6%, +2.3% (post hoc, not sig) [EA] | forward paper; sealed 2025-26 only on yes | results/nba_comebacks.json |
| L2 | Healthy-roster + preseason ratings for single games | roster info vs price | DEN@OKC 10-22: no bet by pre-committed rule [EA] | freeze model + threshold, log ~200 games | scripts/nba_matchup.py, nba_fourfactors.py, team_rating.py |
| L1 | Public models (538/ESPN) vs the OPENER | stale opener | lose to close; never tested vs open [EA] | untested | scripts/nba_public_models.py |
| L29 | Miami thin-depth: MIA under closing spread next 20 games | user thesis | 0-2 at 10-23 [H4] | track forward | -- |
| L9 | S12 soccer logic at US books in Colorado | E3 at legal books | untested [EA] | needs user's state + CO accounts | scripts/soccer_fd.py |
| L5 | Shrink P2 prop model toward the market | P2 overconfident (+17 claimed, ~0 delivered) | untested [EA] | new prereg | scripts/nba_props_dist.py |
| L7 | Situational prop model (arena, b2b, travel, altitude, nightlife, refs), unders first | books shade overs | untested as combined model [EA] | new prereg | notes/research_situational_vars.md, research_prop_models.md |
| L11 | Earnings straddles (enter ~3-5 d before) / straddle selling | VRP at earnings | literature only (Gao-Xing-Zhang +3.34%; ORATS vendor 85%) [EA], notes/social_edges_2026-10-22.md | untested; needs bid/ask history | notes/earnings.md |
| L12 | Daily earnings-consensus logging | price each contract's threshold | not started [EA] | -- | -- |
| L14 | Company graph as a data product (12,917 dated customer filings) | screens, not alpha | data built [EA] | no alpha (see dead) | data/company_graph/, notes/company_graph.md |
| L16 | X post counts vs Kalshi rumor markets | X may lead (GDELT too late to tell) | one-off test ~$13 [EA] | needs X_BEARER_TOKEN + "yes, spend" | scripts/rumor_signals.py |
| L17 | RotoBot pick record (social) | AI picks app | no public record; tracker built [EA] | needs posts (X API or paste to data/rotobot/posts.txt) | scripts/rotobot_tracker.py |
| L18 | Midterms: LA Senate 2-venue cost .9752; TX Gov Abbott .70-.72 vs poll model ~.75 | cross-venue gap / poll model | Kalshi rule text says "Kentucky"; resolution mismatch [S] | judgment <= 1% B; PM's call before 11-03 | scripts/xvenue_elections.py |
| L21 | PredictIt midterm NO-bundle sums > 110 | NBER abstract | untested; verify fees/caps [EA] | time-boxed to 11-03 | -- |
| L22 | Kalshi politics underconfidence | arXiv; contradicted by Burgi et al. | weak prior [EA] | untested | notes/prior_research_2026-10-22.md |
| -- | Kalshi crypto vs Deribit implied probs | options-implied fair | untested [EA] | -- | notes/prior_research_2026-10-22.md |
| -- | Kalshi combos/parlays above leg product, as maker | combo mispricing | untested [EA], social sweep #7 | -- | notes/social_edges_2026-10-22.md |
| H1 | Polymarket mention + culture/award NO at real ask | worst-calibrated categories (YES +4-7pp) | exploratory [EA] | ~4.5k legs/90 days forward; tens of $/day | scripts/smallmarket_*.py, notes/smallmarket_calibration.md |
| H2 | Polymarket longshot-NO as MAKER on untouched 2026-05..10 holdout | FLB via maker | exploratory [EA] | pre-register first | same |
| DKU | NBA under on high lines at DraftKings | pattern-mine survivor +1.9%/54k dev | 0 sealed bets (untestable on ESPN data) [R] | only via Odds API ($119/month, PM's call) [S] | scripts/pattern_mine.py |
| G1 | Kalshi sports favourite >= 85% 1 h pre-start | FLB | dev +2.0%/383 p .062 (bar .0083) [L:kalshi_sports] | never-scored sealed candidate, not recommended [EA] | scripts/kalshi_longshot.py |
| G4 | Kalshi sports dog 20-50c (post hoc) | FLB | dev +1.4%/4,429 p .21 [L:kalshi_sports] | same | same |
| MKdog | Kalshi maker on dogs 20-50c | spread capture | dev +9.8%/898 fills p .009 (bar .0063); gate LEAD q .066 [FS]; ladder test says spread capture doesn't survive [R] | never-scored sealed candidate | scripts/kalshi_maker.py |
| WCN | NFL west_coast_night | body clock | dev +21.5%/107 p .007; gate LEAD q .053 [FS]; circadian mirror lost -10.6% [S] | NFL lockbox pending, not recommended | lab/strategies |
| DUL | NFL div_unders_late (wk >= 10, post hoc) | familiarity | dev +9.8%/623 p .0098; gate LEAD q .070 [FS] | NFL lockbox pending | lab/strategies |
| QBC | NFL QB-change weeks | stale opener on QB news | spread CLV +1.15 pts/210 [L:nfl_simple_model T14] | tier 3 [PL] | nfl_simple_model.py |
| RYU | NFL rush-yard unders in low-total games / big-dog RB rush unders | game script | C7 +7.7pp p .039 n 155; C3 ROI +12% p .078 n 99 (2024 only) [L:nfl_props_script] | tier 3; L4 as a whole dead [PL] | scripts/nfl_props_script.py |
| ALT | NBA altitude home teams | altitude | +1.0 pt ATS p .055 [PL] | tier 3 | -- |
| W99 | Weather maker NO at 99c (tape subset) | descriptive | 275 fills, 0 losses (~1c/contract) [PL] | descriptive only | notes/wx_lockin.md |

### 3d. DEAD -- the do-not-redo list (one line each: why it died)

**In-game / live**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| N0 H1-H3 | NFL in-game: model gap hold, lag after big plays, late favourite | -5.6%/295, -7.2%/340, 6 bets [L:nfl_ingame] | Kalshi reprices big plays within ~1 min; beats nflfastR (first run had an epoch bug: fake +62%) | scripts/nfl_ingame.py |
| M4 H1-H3 | NBA in-game same three | -4.9%/1,016, -7.8%/8,397, -7.6%/27 [L:nba_ingame] | Kalshi Brier .171 < ESPN .180 | scripts/nba_ingame.py |
| T1/T2 | NBA late-game comeback calibration ("Pacers down 3 at 23%") | -1.5%/311 p .55, -5.4%/331 [L:nba_late_wp] | Kalshi calibrated 7/7 bins | notes/late_game_calibration.md |
| B1-B3 | NBA big leads ("teams up 20 win more than priced") | -8.9%/37, 15 bets exact p .71, -4.3%/23 [L:nba_big_lead] | leaders win ~what Kalshi charges | notes/nba_big_lead.md |
| L28 | NBA big-lead buy (>=25 Q3 / >=20 Q4, ask <= .96) | **sealed FAIL** 15 bets 13-2, -8.6% [LB] | adverse selection: cheap leaders = market knows something | scripts/nba_bigleads_logger.py |
| L37 | NBA mid-game comeback rule | **sealed FAIL** 477 bets -3.9c/contract p .97 [LB], [S] | in-game Kalshi sharper than model (Brier .1284 vs .1319) | scripts/nba_midgame_scan.py |
| L38 | Fade in-game trailing longshots | 0/8 trials, 3,051 snapshots [L:ingame_longshot] | sealed-period 1.5c bias did not replicate | notes/ingame_longshot.md |
| -- | NBA live totals model vs KXNBATOTAL | Brier .1196 < model .1240; rule -2.1%/336 [L:nba_live_totals] | pace adds ~nothing beyond pregame total/time/margin | notes/nba_live_totals.md |
| CB1 | Fade trailing NFL dogs in-game | Kalshi +1.8pp over vegas_wp, ~1% after fees [EA] | too small to test | notes/nfl_comebacks_matchups.md |
| -- | Live Pinnacle-vs-Kalshi scanner flags | 0 flags >= 2% EV [R] | Kalshi within 1-2c | scripts/live_edges.py |

**Sports models / angles vs the close**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| O1 | NBA simple model vs opener, gap >= .06 | **sealed FAIL** 554 bets EV -2.55%, ROI -10.2%; CLV +0.68 p .005 [LB], [R] | CLV half the dev size, below ~4% overround | scripts/nba_o1_sealed.py |
| L8 | Player-level simulation (RotoBot style) | sim open ROI -5.8%/1,784, close -9.0% [L:nba_player_sim] | too much variance; simple model better | notes/player_sim.md |
| -- | Public NBA models (538 Elo/CARM/RAPTOR, ESPN) vs close + stacking | 0/7; log loss +.008..+.019 [L:nba_public_models] | all worse than the close | notes/nba_public_models.md |
| -- | NBA angles: rest ATS, big-spread dog, heavy fav ML, b2b unders, home dog ML | -1.5%..-3.9% (+0.6% dog) [L:nba_sbr] | priced; unsigned-spread bug caught | scripts/nba_angles.py |
| -- | NBA Elo disagree >= .10, walk-forward logit edge | +0.6%/3,916 p .40, -4.2%/744 [L:nba_sbr] | market Brier .199 < Elo .207 | scripts/nba_model.py |
| -- | NBA comebacks / matchups / coaching / timeouts (152 tests) | sides efficient; 0/69 trait cells [EA] | comebacks = deficit x time x spread only | notes/nba_comebacks_matchups.md |
| -- | NBA referees: crew totals, crew home, series extension, L2M home/fav, bet-by-crew | B1 -1.9%/2,222 p .83 [L:nba_referees] | crew effect priced (L2M trailing-team bias real, no betting use) | notes/nba_referees.md |
| -- | NBA shooting luck (14 trials) | quintile bet -1.1%/4,719; slope real but tiny [L:nba_luck] | inside the model; fails ledger bar | notes/nba_luck.md |
| -- | NFL comebacks / PROE / pace / 4th-down / coach / head ref (70 tests) | 0 survive [EA] | vegas_wp calibrated ~1pp | notes/nfl_comebacks_matchups.md |
| -- | NFL bad-D x bad-D unders vs opener | 214 bets CLV -0.16, -7.7% [L:nfl_leads_open] | market raises these totals | notes/nfl_leads_open.md |
| -- | NFL offence-EPA ATS | slope +0.98/SD p .0003 but rule 51.5%, -1.6% [L:nfl_leads_open] | already inside model B | same |
| -- | NFL recency/overreaction (blowouts, upsets, cover, luck; 12 trials) | fades 48.7-51.7% ATS [L:nfl_recency] | line drift is sharp (EPA-driven) | notes/nfl_recency.md |
| -- | NFL moneyline: home control, Elo disagree, heavy fav | -3.3%, -0.9%, -1.1% [L:nfl_moneyline] | vig | lab/strategies |
| -- | NFL spreads: late home dog, fade hot form, bye fade, west_early_east, den altitude, big fav fade, dome team cold | -7.8%..+4.7%, none past bar [L:nfl_spread] | priced (west_early_east -10.6% kills circadian story) | lab/strategies |
| -- | NFL totals: wind, TNF, div (all-season), primetime, pre-rain+wind | +7.3% p .11, +0.8%, +6.0% p .036, +1.4%, -10.2% (corrected) [L:nfl_total] | below bar / wrong sign after join fix | lab/strategies |
| -- | NFL weather oracle (observed in-game rain+wind) | +21.4%/70 p .033 corrected [R] | not causal (observed after kickoff) | scripts/nfl_game_weather.py |
| -- | MLB ML fav >= 60% at open vs close; follow/fade moves | -2.7%, -2.6%, -3.0%, -2.3% [L:mlb_ml_sbr] | no open-line FLB; close efficient | scripts/mlb_ml_open.py, mlb_ml_move.py |
| -- | MLB totals: weather GLM, Wrigley wind, temp tail, umpire, precip/cloud, day-night temp, line-move momentum | 7 trials, best +3.2% p .32 [L:mlb_total] | priced | lab/strategies |
| -- | MLB bullpen / starter rest / day-after-night | 0/21 BH, 12,808 games [R] | priced | -- |
| L34-bet | NHL simple model as a bet at ESPN BET | close-implied EV at open -1.8% [EA] | 4% overround (kept as L34 forward CLV) | notes/nhl_simple_model.md |
| -- | NHL PDO luck + blind unders by opening total (17 trials) | 6.5 under -4.3%/1,160 [L:nhl_luck] | luck priced; L34 under lean doesn't replicate | notes/nhl_luck.md |
| -- | CFB spreads model | -7.8% at -110, CLV negative [L:cfb_simple_model] | miscalibrated (v2 needs new prereg) | notes/cfb_simple_model.md |
| -- | Multi-book consensus deviation (NBA 2023) | +7.4%/23k dev [R] | stale-quote illusion | -- |
| -- | Edge report shrinkage (NFL candidates) | shrink to -3..+4% [R] | unproven | reports/ |

**Props / pick'em / parlays**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| R_REB | NBA rebound unders (P2 rule) | **sealed FAIL** -1.56%/1,214 p .72 [LB] | dev +2.7% didn't replicate | scripts/nba_props_dist.py |
| P1 | NBA props rolling-form model | -3.29%/130,000 [L:nba_props_model] | books shade overs; form priced | scripts/nba_props_model.py |
| P2 | NBA props empirical-distribution unders | +0.27%/60,826 p .26 [L:nba_props_dist] | selection +4.3 pts, not past vig; skew mechanism refuted | same |
| PM10 | Backwards pattern mine (2,280 cells) | sealed confirm 0/10 [R] | survivors were artefacts (MLB hits over reversed) | scripts/pattern_mine.py |
| -- | NFL 252-trial prop model loop | 0 at p<.05, median -7.5% [R] | no edge | reports/prop_loop_ledger.jsonl |
| -- | NFL FG props (defense style + script); zone/blitz -> FGs | -8.2u/53; r = 0.00 [R] | market wins | -- |
| L4 | NFL props by category x game script (8 cells) | 0/8, best q .31 [L:nfl_props_script] | one book/season, underpowered | notes/nfl_props_script.md |
| L6b | Ref crew FT tendency -> player points props | overs -8.1%/2,174 [L:nba_ref_ft_props] | lines move toward the crew | notes/nba_ref_ft_props.md |
| -- | MLB strikeout props (workload, short rest; 2025 real lines) | -4.1%/486; +1.1u/1,970 break-even [L:mlb_k_props], [R] | market beats model | -- |
| -- | Kalshi vs DraftKings player props (10-28 scan) | median fair gap 0c [S] | no edge | notes/props_scan_2026-10-28.md |
| -- | 2-leg parlays of high-prob legs | props -9.3%/105k; NFL ML -1.35%/885 [R] | parlays square the legs' edge | scripts/parlay_test.py |
| -- | Sleeper pick'em vs Pinnacle | 0/1,047 legs +EV, hold 12.4% [R] | margin too big | scripts/pickem.py |

**Social-media / practitioner systems**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| KICK | NFL kicker over 3.5 "90%" system (@linemakersports) | 84.7% hit (n 354) vs 83.0% base [L:nfl_kicker_system] | base rate; needs ~-555 odds | notes/nfl_kicker_system.md |
| -- | GDELT news/rumor spikes -> Kalshi rumor markets | 4/4 rules lose (-7%..-13%) [L:rumor_gdelt_kalshi] | price moves BEFORE news (median 4 h) | notes/rumor_signals.md |
| -- | Practitioner/vendor claims (X/Reddit/Substack) | mostly unverifiable vendor P&L [EA] | the verifiable ones became L24-L27 (E4, L40) | notes/social_edges_2026-10-22.md |
| -- | "Harden worse in strip-club cities" style nightlife effects | untested as such [S] | folded into L7 (section 3c) | notes/research_situational_vars.md |

**Weather variants**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| -- | Kalshi temp: forecast-error, ensemble, spread regime, fade longshots, longshot thin volume, revision lag, lead-2 check | -23.7%..+33% (revision lag n 69 p .13) [L:kalshi_temp] | only consensus-reject survived (= E1) | scripts/kalshi_temp*.py |
| X2 | Consensus-reject on daily LOWS | -6.38%/224 [L:kalshi_temp_expand] | evening lows outside forecast window (unverified) | scripts/wx_expand.py |
| R1 | Daily rain all-models-dry -> NO | dev +3.42%/361; **sealed FAIL** +0.49%/274 p .28 [LB] | summer dev didn't carry into autumn | scripts/kalshi_rain.py |
| R2 | Daily rain all-wet -> YES | -0.9%/47 [L:kalshi_rain] | -- | same |
| N1 | Day-of temp consensus-reject (06:00) | D1 +0.19%/1,279 p .34; D2 -1.31%/364 [L:kalshi_temp_dayof] | Kalshi prices overnight runs by 06:00 | scripts/wx_dayof.py |
| N3 | Hourly temperature markets | median 1-71 contracts [R] | too thin | -- |
| L26 | Late-day lock-in from observed station max | ABOVE_2 -8.3%/460; below-max NO already $1.00 [L:wx_lockin] | traders see 1-5 min obs | notes/wx_lockin.md |
| -- | Decide E1/E2 earlier on D-1 (11/14/18:00) | 0/6 taker pass, all deltas negative [L:wx_timing] | cheaper NO but loss rate 7.6% vs 4.1% | notes/wx_timing.md |
| -- | New Kalshi weather cities/series | census: none to add [L:kalshi_wx_new] | all 24 KXHIGH covered | notes/kalshi_wx_new.md |
| -- | ForecastEx lows (UL), intl (SH), legacy highs (DH) | UL +3.5%/77 p .35; others ~0 bets [L:forecastex_wx2] | D-1 trades rare; event-day market | notes/forecastex_wx2.md |
| -- | Winter rain/snow taker + locked-YES (W1, W2, S1, S2) | 0/6 pass [L:kalshi_wx_winter] | locked-YES never trades; maker kept as L39 | notes/kalshi_wx_winter.md |
| -- | Confidence-based Kelly sizing for E1/E2 | Kelly 1.25x vs flat 1.32x dev [EA] | price already carries confidence | notes/kalshi_sizing.md |

**Kalshi structural (non-weather)**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| L19 K-FAV | Favourite side 88-97c in non-sports daily markets | taker -0.32%/4,419; maker -2.19%/2,829 [L:kalshi_fav_maker] | adverse selection: filled win 90.5% vs 99.3% unfilled | notes/kalshi_fav_maker.md |
| M3 | Mention markets buy NO (YES <= 15c), taker | **sealed FAIL** d24 +0.45%/2,674 p .22; d6 +0.72% p .061 [LB] | edge shrank as volume grew ~10x | scripts/kalshi_mentions.py |
| M1/M2 | Mention NO at wider YES bands | +2.2..+5.5%, none past bar [L:kalshi_mentions] | -- | same |
| G2/G3/L1/L2 | Kalshi sports fav 65-85, dog <= 15c, ladder tails | -5.6%, -45%, +0.5%, +0.4% [L:kalshi_sports] | longshots overpriced but no rule past bar | scripts/kalshi_longshot.py |
| -- | Kalshi maker fav 50-80c; symmetric ladder maker YES/NO | -5.4%; -2.6% / +2.7% [L:kalshi_sports] | spread capture doesn't survive adverse selection | scripts/kalshi_maker_ladder.py |
| S1-S3 | Kalshi soccer 1X2: draws YES, fav >= 60 YES, tail NO | -6.5%, -1.9%, +1.9% p .087 [L:kalshi_soccer] | mids sum 1.005; made off Pinnacle | scripts/kalshi_soccer.py |
| N2 | Kalshi CPI ladders vs Cleveland Fed nowcast | 0 bets [L:kalshi_cpi] | ladders only quote rungs inside ~3 sd | scripts/kalshi_cpi.py |
| N6 | Kalshi daily gas vs AAA random walk | 0 bets [L:kalshi_gas] | same ladder structure | scripts/kalshi_gas.py |
| N7 | Kalshi TSA throughput | deprioritized [S] | same ladder structure | -- |
| -- | Econ consensus-reject (claims, U3, payrolls, GDP, core PCE) | T1 33 bets 0 losses, market-null p .08; T2 p .030 [L:econ_reject] | quoted tails fairly priced; revised-FRED leakage | notes/econ_reject.md |
| -- | Kalshi early NFL TOTALS ladders | CLV +0.33 p .11/47; taker +1.7%/-1.1% [L:kalshi_early_nfl_totals] | not stale (first tight ladder near close) | notes/kalshi_early_nfl_totals.md |
| -- | "Small markets are mispriced" (by lifetime volume) | look-ahead artefact [EA] | ex-ante volume shows no gradient | notes/smallmarket_calibration.md |
| -- | Polymarket negRisk "overround legs" | +10-14% is display-price artefact [EA] | live median spread 63c | same |

**Cross-venue / soft venues**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| M1 | Kalshi vs Polymarket midterms | median gap 0.5c; flagged arbs = 3-candidate races [R] | not arbs (LA Senate kept as L18) | scripts/xvenue_elections.py |
| M2 | Kalshi vs Polymarket sports game winners | median 0.5-1c, cheapest round trip $1.021 [R] | market-made against each other | scripts/xvenue_sports_snapshot.py |
| -- | Kalshi vs Polymarket weather | 0/48 brackets under $1 after fees [EA] | no arb | notes/venues.md |
| -- | Polymarket US game winners vs Pinnacle | 0/116 +EV, within .72c [R] | made off sharp line | scripts/pmus_vs_pin.py |
| -- | Kalshi soccer vs Pinnacle (live) | 0/296 quotes >= 1% EV [R] | E3 edge doesn't exist on Kalshi | scripts/kalshi_soccer_vs_pin_snapshot.py |
| Q007 | ForecastEx other families (crypto/fin dailies, sports, elections, climate, symmetric cross-venue arb) | do-not-pursue list [Q] | stale last-trade / delisted / no gap | notes/q007_forecastex_breadth.md |
| -- | Venue declines: Bovada (offshore), PrizePicks (captcha), tennis-data (403), sign-up promos (not legal in CA) | [S], notes/soft_venues.md | legal / ToS / never bypass | -- |

**Stocks / thesis / earnings**
| ID | Idea | Evidence | Why dead | File |
|---|---|---|---|---|
| -- | 14 ETF rules: dip/rip x6, tech gate x2, learned lead-lag, 2% loss cap x2, sector momentum, sector reversal, vol-target | 0/14; vol-target alpha +2.7%/yr p .024 (bar .0036) = risk control [lab/stocks_ledger.jsonl] | simple timing doesn't beat holding | scripts/stocks_bt.py |
| -- | Classic pairs KO/PEP + 6 | +1.89%/yr p .052, ~0 after 2009 [R] | decayed as published | scripts/pairs_bt.py |
| -- | Twin securities (dual-class, ETF twins, dual-listed, GGR screen; 47 trials) | 0 pass; dual-class +3.8% gross -> -3.5% net [L:stocks_twins_yf] | costs | notes/pairs_twins.md |
| A1-B3 | Customer momentum, Fama-MacBeth, earnings spillovers both ways, SIC4 | 0/5 [L:company_graph_edgar_yf] | turnover 315%/mo eats it | notes/company_graph.md |
| C1 | Graph-smoothed momentum minus plain | **sealed FAIL** -0.69%/mo t -1.96 [LB] | graph neighbours add noise | same |
| E1p/E3p | Polymarket earnings beats: all YES, rich NO | +0.07%/971; -18.6%/368 [L:poly_earnings_beat] | calibrated (.732 vs 74.3%) | notes/earnings.md |
| E2p | Polymarket earnings cheap YES <= .60 | **sealed FAIL** -6.5%/97 [LB] | dev +11.4% was noise | same |
| -- | BE earnings straddle 10-22 | implied +-9.8% vs mean move 10.4-10.8% [S] | fair, not +EV | notes/earnings.md |
| -- | Retail anomalies (overnight, pre-FOMC, index adds, PEAD large-cap, 0DTE) | literature [EA] | reported dead for retail | notes/prior_research_2026-10-22.md |

Count: 3a 5 · 3b 13 (+3 recorders) · 3c 34 · 3d 91 rows (many rows bundle several ledger trials; the ledger has 351
scored trials in 56 families [FS]).

---------------------------------------------------------------------------------------------------------------------
## 4. Research system

**Pipeline** [FA s2]: Scout (researcher agent + scanners) -> Designer (`prereg` skill / experiment-designer: baseline,
ONE change, metric, success/kill, power, trial budget in QUESTIONS.md) -> Builder (script with pre-registration in the
docstring; `shasum -a 256` recorded BEFORE any market join; after the hash the scorer is frozen, a fix = `void_bug` row +
new hash) -> Auditor (data-auditor: joins, timestamps/units, lookahead, duplicates, sealed-file access) -> run dev
trials, every one appended to lab/ledger.jsonl -> Critic + Replicator (independent recompute, no shared code) ->
deterministic gate `fund_stats.py verdict` (DEAD / INCOMPLETE / LEAD / PAPER / SEALED-ASK; agents can't override) ->
`finding` skill -> forward paper logger (launchd) -> PM sealed ask ("yes <name>") -> PM sizes -> deterministic executor
with caps, KILL, loss stops, post-only. Every arrow is a file in the repo.

**Protocol** (notes/protocol.md): freeze first (data/frozen/2026-10-14 MANIFEST); every trial counts; sealed = one shot,
human yes, never re-scored; cluster duplicates by event; evaluate vs the de-vigged market, not 50%; distrust EV > ~5%,
one-book/one-side profit, line-move or "current" timestamps; shrink then 1/4 Kelly.

**Bars** [FS]: 351 scored trials in 56 families; ledger-wide Bonferroni p < 1.42e-4; dataset bar = .05/trials in that
dataset; LEAD needs BH q < .10 + both halves agree; forward needs >= 100 settled; 11 trials flagged with q < .1 and no
halves check (`fund_stats.py flags`). Power: n = ((1.645 + 0.842) sigma / mu)^2 (`fund_stats.py power MU SIGMA`) [FE].

**Hub project `fund`** [notes/FUND_NOTES.md]: STATE + NOTES (`hub note fund "..."`) + QUESTIONS.md (Q001 proposed,
Q002 running, Q003 sealed ask, Q004 running, Q005 done, Q006 running, Q007 done, Q008 running) + findings/. One-agent
lease. Headless cycle: `hub launch fund "Run ONE research cycle exactly as written in notes/FUND_CYCLE.md (section 'The
cycle'). Obey its hard rules."` (not scheduled) [notes/FUND_CYCLE.md].

**Agent types**: researcher (prior art), experiment-designer (prereg), general-purpose (builder), data-auditor, critic,
replicator, Explore (search); skills: prereg, finding, fanout, hub-start / hub-handoff. Past agent IDs are unreachable
after a /clear -- re-launch with the brief at the top of each report [EA s6].

**Credit-light /loop prompt** (verbatim, as launched 2026-10-28 10:20Z; stopped for /clear 10-29 [H5]):

```
/loop Fund research loop (quant-trader worktree /Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader, sports/). CREDIT BUDGET: keep ticks cheap. Each tick: (1) run ONE combined status check (cd ~/.claude/quant-kalshi; .venv/bin/python scripts/kalshi_exec.py status | head; scripts/bet.py pnl | head; scripts/lip_paper.py report | head; scripts/nba_open_totals.py report | head -4) and note only NEW things (settlements, errors, kill-rule hits, alerts); (2) only if no research agent is running AND >= 3 h since the last one was launched (track in sports/notes/FUND_NOTES.md), launch ONE background agent on the top [proposed] question in sports/QUESTIONS.md or the next build item in sports/notes/FUND_ARCHITECTURE.md milestones (prereg first, dev-only, ledger, fund_stats verdict; builds = platform work like IBKR/ForecastEx connector prep, LIP report tooling, brief improvements); (3) when an agent finishes: record verdict in EDGES_AND_AGENTS.md/STATE.md/QUESTIONS.md, commit + push (git -c credential.helper='!gh auth git-credential' push -q https://github.com/ttt5600/NeruoBatLab.git worktree-quant-trader). NEVER place orders, change .env/caps, run sealed tests, or touch accounts without the user's explicit yes -- queue asks in STATE.md. Replies: 1-3 lines unless something material happened.
```

**Launching research by hand**: write the question in QUESTIONS.md (template at its top) -> `/prereg` -> background
general-purpose agent with a written brief (dev only, sealed never read, ledger every trial, evaluate vs market, CIs,
bar) -> critic/replicator -> `fund_stats.py verdict <strategy>` -> update EDGES_AND_AGENTS.md + this book + STATE.

---------------------------------------------------------------------------------------------------------------------
## 5. Tools / commands cheat sheet
Deployed tools: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/<x>.py`. Research tools: `cd sports &&
../.venv_quant/bin/python scripts/<x>.py`.

| Command | Does |
|---|---|
| `lb <team> [--fresh]` (/usr/local/bin) | live game check: state, pregame line, model, Kalshi, gap (cached, decision support only) |
| `rating <team>` (/usr/local/bin) | Bayesian team rating (prior + season blend; scripts/team_rating.py) |
| `bet.py place TICKER yes\|no --units U [--price P] [--tier A\|B\|C] [--taker] [--dry]` | the one router for API bets (post-only by default) |
| `bet.py log-external --venue ... --market ... --side ... --price ... --tier C --strategy judgment` | log phone/Robinhood bets |
| `bet.py settle \| pnl \| orders \| cancel ID\|--all \| shards \| allocate 0=70 3=30 [--confirm]` | journal / shard balances |
| `kalshi_exec.py status \| preflight \| kill` | weather executor |
| `kalshi_paper.py report` | weather paper |
| `forecastex_exec.py status` / `run` (dry) / `run --live` | E5 executor |
| `forecastex_paper.py report \| q008-validate` | FX1-fwd + Q008 |
| `lip_paper.py report \| programs` | L40 LIP paper |
| `nba_open_totals.py report \| show` ; `open_lines_logger.py report --league nhl\|nfl` | L32 / L33 / L34 / Q004 |
| `kalshi_wx_monthly_logger.py report` ; `pmus_weather_paper.py report` | L39 ; F4 |
| `live_board.py show \| report` ; `nba_midgame_scan.py show\|report` | live recorders |
| `gap_logger.py report` ; `live_signals.py report` ; `thesis_tracker.py report` | F3 ; F2 ; F6/F7 |
| `fund_brief.py` (07:30 auto) | PM brief -> notes/briefs/ |
| `nba_tonight_sheet.py` | tonight's NBA card (sports/scripts) |
| `nfl_ingame_fade.py interim` (Tuesdays) / `score` (once, after 12-02) | F1 |
| `fund_stats.py report \| verdict S \| families \| flags \| power MU SIGMA` | stats gate |
| `ev_check.py` | EV after fees + fractional-Kelly stake for any single bet |
| `nfl_simple_model.py week --json` ; `nba_simple_model.py today` | model numbers (NFL live output not to be bet before refit check) |
| `xvenue_elections.py` | midterm cross-venue scan (until 11-03) |
| `manual_bets.py` | legacy manual post-only orders (superseded by bet.py) |

---------------------------------------------------------------------------------------------------------------------
## 6. Lessons learned
1. **Markets are efficient where they are watched.** Every NFL/NBA/MLB closing-line angle loses ~the vig; in-game
   Kalshi reprices within ~1 min and beats our models (Brier .1284 vs .1319) [S], [R].
2. **Our edges are structural, not predictive**: stale far tails in thin retail books (weather), maker vs taker, and
   thin/second venues (ForecastEx). Literature agrees: makers earn what takers lose [FA s1].
3. **Models lead the OPENER, never the close** (NBA/NFL/NHL/CFB CLV p < 1/4000) -- but positive CLV is necessary, not
   sufficient: it must clear the opener's overround (O1 sealed fail) [R]. The money question is low-vig early execution.
4. **Cheap prices are adverse selection.** A cheap leader means the market knows something (L28: 253/268 triggers never
   offered <= .96); filled maker bids lose more than unfilled (K-FAV 90.5% vs 99.3%) [S], [EA].
5. **Base rate vs price.** An 85% hit rate is not an edge at -200 when fair is ~-555 (kicker system); "75% of companies
   beat" is already in the price [notes/nfl_kicker_system.md], [notes/earnings.md].
6. **Leakage/bug list** (check every new dataset): datetime64[us] // 1e9 epoch bug froze quotes -> fake +62%; NFL weather
   joined by row position (53% wrong); NBA sqlite spread unsigned (fake +7.2%); Kalshi expected_expiration = start+6h;
   soccer occurrence_datetime = kickoff+3h; ESPN prop last_updated is post-tip; proportional de-vig fakes longshot CLV;
   lifetime volume is look-ahead; display prices aren't quotes; revised FRED vintages leak [S], [R], [EA].
7. **Dev edges shrink.** M3 +4.3% -> +0.45% sealed at 10x volume; R_REB +2.7% -> -1.56%; ~300 trials give ~15 false
   p < .05 by chance [PL]. Sealed once, never re-scored.
8. **Capacity caps everything.** Weather maker tops out ~$144/month; size above ~$75/pick loses to adverse selection;
   P(losing month) does not depend on stake [FE].
9. **Learning time depends on bet count, not bankroll** (maker weather ~1,050 filled picks ~3 months; L32 +3% ~12
   seasons) [FE].
10. **Ops**: never infer which job to run from the wall clock (launchd runs late after sleep); verify a fill before a
   convention KILL; Kalshi taker fee is charged exactly, not rounded to 1c [S], [H4].

---------------------------------------------------------------------------------------------------------------------
## 7. Pending user decisions / next steps

| # | Ask / event | Exact wording or action | Source |
|---|---|---|---|
| 1 | Deposit $2k on Kalshi, then fund the NBA shard | `bet.py allocate 0=70 3=30 --confirm` + raise weather caps | [H5] |
| 2 | Turn on maker (E4 / Q001) | say "yes maker" (`KALSHI_EXEC_STYLE=maker`) | [H5] |
| 3 | Sealed test Q003 Mentions maker NO (decaying edge) | "yes Q003" or no | [H5] |
| 4 | Optional ~$500 to ForecastEx (unlevered); optional Polymarket US API key | user | [H5], [S] |
| 5 | Weekly IBKR Mobile approval (Sunday) for IBC | user | [S] |
| 6 | Revoke + replace the Kalshi PROD key that was pasted in chat (file named kalshi-demo.pem) | user | [S], [EA s8] |
| 7 | Agent authority / fund shard 3 asks | user | [S] |
| 8 | X post-count test (~$13) yes/no; install F5 pick'em logger; free disk; confirm home state (CA vs CO) | user | [EA s8] |
| 9 | LA Senate / TX Gov judgment (election 11-03) | PM's call | [S] |
| 10 | Fix `espn.py::parse_prop` (ESPN dropped the over/under marker; first of each pair = over), re-run `collect.py props` | agent | [S] |
| 11 | L40 LIP paper ends 11-08 -> score per pre-registration | agent | lip_paper.py |
| 12 | F1 NFL in-game fade: score once after 2026-12-02 | agent | [EA] |
| 13 | L32/Q004/L34 at 100 settled picks -> decision rule | agent | [Q] |
| 14 | L39 sealed shot ~Feb 2027 | PM yes later | [Q] |
| 15 | Build a CFB totals logger for L36; fix Shin de-vig in gap logger (N4) | agent | [FS], [H4] |
| 16 | Restart the research loop (prompt in section 4) only if wanted -- loops are off by default | user | [H5] |
