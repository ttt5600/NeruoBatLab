# fund: NOTES (append-only, one dated line per session; `hub note fund "..."`)

- 2026-10-27 M1 built (notes/FUND_ARCHITECTURE.md): scripts/fund_stats.py statistician gate (+ tests/test_fund_stats.py),
  hub project `fund` registered (STATE.md, this file, QUESTIONS.md Q001-Q006 [proposed], findings/),
  scripts/fund_brief.py daily PM brief (+ scripts/install_fund_brief_agent.sh, NOT installed), notes/FUND_CYCLE.md
  nightly research-cycle task (NOT launched). Goal: double a $1-2k unlevered bankroll in 12 months, proven live.
- 2026-10-28T09:18Z loop: launched agent Q003 (maker NO in thin retail categories). Q002 LIP paper running, Q005 done (smallmarket scan), Q006 L39 logger running. NBA open-totals logger 4-1 CLV +1.2.
- 2026-10-28T09:29Z loop: Q003 done -> LEAD (IN1_d6 +7.1%/859 fills, decaying); sealed ask queued.
- 2026-10-28T12:22Z loop: launched build agent: ForecastEx/IBKR connector + public-quote forward paper logger (E5).
- 2026-10-28T12:30Z loop: ForecastEx connector + FX1-fwd paper logger (public live bid/ask) deployed; 17 tests.
- 2026-10-28T15:25Z loop: launched Q004 build (first-tight-quote + depth capture in open-line loggers).
- 2026-10-28T15:32Z loop: Q004 deployed (first-tight-quote + depth fields, pick_tight paper picks in nba_open_totals + open_lines_logger; censors pre-existing listings).
- 2026-10-28T18:28Z loop: launched build: 'rating' team command (Bayesian prior+season blend) + brief adds E5/ForecastEx + Q003/Q004 sections.
- 2026-10-28T18:31Z loop: 'rating' command + brief sections (E5, Q003, Q004, picks) deployed.
- 2026-10-28T21:29Z loop: launched Q007 ForecastEx breadth scan.
- 2026-10-28T21:38Z loop: Q007 done -- ForecastEx favourites underpriced across families (+1.5-3.5% at last trade, no bid/ask); econ vs Kalshi: FX favourites 1.4-2.2c cheaper; Q008 [proposed] Kalshi-favoured side on FX when >=2c cheaper (64 bets 0 losses, tiny capacity) -> fold into FX paper logger.
- 2026-10-29T00:33Z loop: launched Q008 fold-in (FX favourite cheaper than Kalshi) into forecastex_paper.py.
- 2026-10-29T00:37Z loop: Q008 forward paper deployed in forecastex_paper.py (first snapshot 10-29 15:00 CT; econ events >30d out not yet checked).
- 2026-10-29T02:54Z user-directed: launched 3 edge searches (ForecastEx other weather; more Kalshi weather cities/series; econ-data nowcast consensus-reject).
- 2026-10-29T02:56Z Kalshi weather census: no new cities/series (all 24 KXHIGH covered; intl/others never listed; no daily snow). Volume limit = our caps, not the market.
- 2026-10-29T02:59Z forecastex_wx2: UL/SH/DH FAIL (D-1 trades rare; UL contested prints lose 18%); no daily rain on FX. Unverified: FX file trade date may be UTC (E5 prices staler, no lookahead).
- 2026-10-29T03:05Z econ_reject DEAD: Kalshi econ ladders only list ~+-2.3sd; quoted tails fairly priced (~6/yr); T1 ROI driven by revised-FRED leakage.
- 2026-10-28 20:51 IBKR gateway: SSO login works (Keychain + iMessage 2FA automated) but ssodh/init 'invalid challenge' on every try; Java ruled out; waiting on account funding/paper status
- 2026-10-28 21:04 IBKR: same ssodh error after funding; live-switch diagnostic inconclusive (gateway reused paper SSO); stopped burning SMS; options: IBKR support or IB Gateway/TWS API
- 2026-10-28 21:22 IBKR CONNECTED (live-mode session, read-only): cause = no paper account; ForecastEx books with sizes readable; trading still gated (paper mode, no account id)
- 2026-10-28 21:30 E5 ForecastEx LIVE: 26 orders night 1 (cap 5/day), launchd forecastex-live every 5 min
- 2026-10-29T05:38Z IBKR gateway session DOWN -> user must re-login at https://localhost:5050 before next 22:00-local decisions.
- 2026-10-31 15:59 2026-10-31: ForecastEx executor never filled: (1) Pacific/Mountain contracts are ET-dated event+1 -> patched forecastex_tws_client.py resolve() to look up by public conId first (backup .bak-2026-10-31; offline /tmp/fxdiag/t.py passes LAS+CLT; needs one read-only resolve on real gateway); (2) IBKR returns Inactive 'verify with the token we emailed you' -> user must verify in Client Portal; (3) spent_today counts rejected intents (not fixed). Side-income roadmap board: https://claude.ai/artifact/45XM3EdgKMCMap59B6r7Q6
- 2026-11-01 17:37 2026-11-01: conId patch VERIFIED read-only on real gateway (LAS/BKF lastTrade=event+1, CLT=event, all resolve). spent_today now excludes intents whose IBKR response is Inactive/Rejected (backup forecastex_client.py.bak-2026-11-01); 10-30..11-01 counter 99.x -> 0, 10-29 CP-path rows still count 24.27. Orders still blocked on the IBKR emailed token.
- 2026-11-18 20:47 2026-11-19 handoff: 11-15..18 waves (~25 tests) found no new money edge; E4 strengthened (price bands, X1 restart, NBM rule C forward), edgelab SCORING v2 + BAR v2 amendment adopted on user approval, health monitor + robustness fixes after 3 silent failures; Fed + mortgage agents running at /clear (merge their inbox rows).
