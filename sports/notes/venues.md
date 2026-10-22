# Venues -- where else the edges can be traded (2026-10-22)

User ask 2026-10-22: "Don't just do Kalshi, do some other ones too, diversify." + priority: programmatic connection to
Polymarket US. Code: `scripts/polymarket_us.py` (read-only client), `scripts/venues_weather.py` (weather-rule port,
cross-venue arb, paper logger). Tags: [F] = fetched by us 2026-10-22; [S] = search-result / third-party summary,
not confirmed against the venue; UNVERIFIED = no source.

## Polymarket US connection (priority)

- Two hosts [F docs.polymarket.us/api-reference/introduction.md]:
  - PUBLIC market data, NO key: `https://gateway.polymarket.us/v1/...` -- markets, events, series, search, order book
    (`/markets/{slug}/book`), BBO, settlement, price history (`/price-history?symbol=<slug>`; custom ranges <= 24 h give
    ~1-minute book-derived quotes: longPrice = YES ask, shortPrice = 1 - YES bid; history present back to May 2026).
  - TRADING, key required: `https://api.polymarket.us/v1/...` -- orders (create/modify/cancel/batch of 20/preview/close
    position), portfolio, balances; WebSockets `wss://api.polymarket.us/v1/ws/private` and `/v1/ws/markets`.
- Auth [F .../authentication.md]: API Key ID + Secret Key (Ed25519). Each request signs `timestamp + METHOD + path` ->
  headers `X-PM-Access-Key`, `X-PM-Timestamp` (ms, within 30 s of server time), `X-PM-Signature`. No wallet / no
  on-chain signing.
- Automated retail trading IS supported: create-order has `manualOrderIndicator` = MANUAL | AUTOMATIC, and the rate-limit
  page has a section "For automated systems" (email support@polymarket.us for higher limits). Order types LIMIT/MARKET;
  TIF DAY/GTC/GTD/IOC/FOK; intent BUY_LONG (YES) / BUY_SHORT (NO) / SELL_* [F .../orders/create-order.md].
- Rate limits [F .../rate-limits.md]: 25 req/s per IP shared across endpoints (price-history page says 20 req/s);
  429 -> wait >= 1 s, exponential backoff. "Global Rate Limit Exceeded" on orders = a 5-second latency stopgap reject,
  not a real rate limit.
- SDK [F .../sdks/python/quickstart.md]: `pip install polymarket-us` (Python >= 3.10; github.com/Polymarket/polymarket-us-python);
  also TypeScript `polymarket-us`. Public calls work with `PolymarketUS()` and no keys. We did NOT install it -- our
  read-only module uses plain `requests` against the gateway.
- Fees [F docs.polymarket.us/fees.md, effective 2026-10-19 1 pm ET]: taker = 0.0695 x C x p x (1-p) (table tennis 0.10);
  maker rebates only from $10M/month maker volume; taker rebates only from $250k/month. Essentially Kalshi's 0.07 formula.
- KYC / funding [F .../learn/get-started/signup.md, fund-account.md]: sign up with Google/Apple ID in the app; DOB, phone,
  legal name, address, SSN; sometimes ID + selfie. KYC before deposit. Deposits: debit card, ACH, Apple Pay, PayPal,
  Venmo, USDC (iPhone), wire ($5k min). Daily limit $50k card/ACH.
- State availability: Polymarket US publishes NO state list in its docs (searched llms.txt; "Trading Restrictions" page
  is about insiders only). California: listed available incl. sports by defirate.com/prediction-markets/california/
  (updated 2026-10-04) [S-fetched third party]. Third-party exclusion list (pm.wiki): AZ, IL, MA, MD, MI, MT, NJ, NV, OH [S].
  The app's KYC is the authoritative check.
- STEPS FOR THE USER (we never see the keys):
  1. Install the Polymarket US app (App Store id6648798962), create the account (Google/Apple/email), pass KYC.
  2. Go to https://polymarket.us/developer, sign in with the SAME method as the app (switching methods can break key access).
  3. Create an API key -> Key ID + Secret Key; the secret is shown ONCE.
  4. Put them in `sports/.env` yourself as `POLYMARKET_KEY_ID=...` and `POLYMARKET_SECRET_KEY=...` (gitignored); revoke at
     the same page if exposed. Nothing in this repo trades on Polymarket yet; an executor would mirror kalshi_exec.py
     (caps, KILL file, dry-run first) and needs the user's go.
- Tested 2026-10-22: `../.venv_quant/bin/python scripts/polymarket_us.py test` -> PASS (markets, event, book, BBO,
  price history, settlement, search; 1.4 s). Also `book <slug>` and `temps [date]`.

## Venue inventory (2026-10-22)

The question that matters: does the venue run its OWN order book? If it routes to Kalshi, it adds no diversification.
CA = California (probably the user's state). Most state facts come from defirate.com/prediction-markets/california/
(updated 2026-10-04, fetched by us) -- a third party, so the venue's own KYC is the final word.

| venue | own book? | weather? | other products | fees | public data API / trading API | CA | notes |
|---|---|---|---|---|---|---|---|
| Kalshi | own DCM | YES: daily highs+lows ~48 US series; CLI/TWC | econ, politics, sports, etc. | 0.07*C*p(1-p) (our KALSHI_FEE; kalshi.com 429'd today, so not re-checked) | yes / yes (we use both) | yes incl. sports [defirate] | only venue with a sealed-passed edge |
| Polymarket US | own DCM (QCEX) [F docs] | YES: highs NYC/CHI(MDW)/MIA/LAX/SFO since 2026-04-29, PHL + lows since 2026-10-16; NWS CLI, same stations as Kalshi [F] | sports (most), politics, crypto, macro, culture, climate [F gateway counts: 69k sports / 79.6k open mkts] | taker 0.0695*C*p(1-p) [F] | gateway, no key / api.polymarket.us, key, automated orders allowed [F] | yes incl. sports [defirate]; no official list | see section above |
| ForecastEx (IBKR) | own exchange + clearing [S agent, CFTC filings F] | YES: "Daily Temperature" exceed-X thresholds, 1F steps, ~24 cities highs + lows + hourly; settles on Weather Underground daily obs at AIRPORT codes (NYC = LGA, not Central Park) [F CFTC ptc01232637912.pdf, ptc08062614351.pdf; F price file] | econ, rates, elections, crypto; sports reportedly dropped [S] | UNVERIFIED (launched commission-free + coupon) [S] | daily OHLC/VWAP files only, forecastex.com/api/download?type=prices&date=YYYYMMDD [F]; trading via IBKR API (UNVERIFIED) | "Live", no sports [defirate] | 10-20 file: 320,307 pairs traded across UH (high) contracts [F] -- liquid |
| Robinhood | broker: routes to Kalshi, ForecastEx, Rothera (ex-MIAXdx), or CDNA [F Robinhood page, per agent] | yes, but its NYC high = KLGA/WU thresholds = ForecastEx's product (inference) | sports, econ | $0.01 + exchange fee [S] | no event-contract API (Agentic Trading = stocks only, STATE.md) | yes incl. sports [defirate] | no new book for weather |
| Coinbase | routes to Kalshi [S The Block] | Kalshi's | Kalshi's | -- | -- | yes [defirate] | zero diversification |
| Webull | routes to Kalshi [S] | Kalshi's | Kalshi's | -- | -- | not listed | zero diversification |
| PrizePicks | routes to Kalshi (Polymarket planned) [S] | no | sports | -- | -- | yes [defirate] | zero diversification |
| Crypto.com / CDNA (ex-Nadex) | own DCM [S] | none found (UNVERIFIED) | sports, crypto | up to $0.10 + $0.10 per contract [S] | UNVERIFIED | via OG app [defirate] | also powers Fanatics Markets |
| Fanatics Markets | CDNA [S] | no | sports | -- | -- | yes [defirate] | |
| DraftKings Predictions | CME + CDNA; Railbird own exchange planned [S] | no | sports | UNVERIFIED | no | yes [defirate] | |
| FanDuel Predicts | CME book [S] | no | sports, financial | UNVERIFIED | no | yes [defirate] | DK/FD share the CME book [S] |
| Underdog | own (Aristotle exchange bought 2026-03) [S] | no | sports | UNVERIFIED | no | yes [defirate] | |
| Gemini Predictions | UNVERIFIED | no | sports | UNVERIFIED | UNVERIFIED | yes [defirate] | |
| Novig | own (Ludlow Exchange, 2026-08) [S] | no | sports | UNVERIFIED | UNVERIFIED | not listed | |
| ProphetX | own exchange, approved 2026-06 [S] | no | sports | UNVERIFIED | UNVERIFIED | not listed | |
| PredictIt | Aristotle; politics only [S] | no | politics | UNVERIFIED | UNVERIFIED | UNVERIFIED | |

KYC everywhere = name, DOB, address, SSN (+ ID/selfie sometimes). Polymarket US deposits: debit, ACH, Apple Pay, PayPal,
Venmo, USDC, wire [F]. Other venues' deposit methods were not verified.

## Weather rule ported (scripts/venues_weather.py; pre-registered in its docstring)

Only Polymarket US is an exact port: same NWS CLI stations as our Kalshi series, same 2F brackets (every bracket of
NYC/CHI/MIA/LAX/PHL on 10-22 matched a Kalshi bracket 1:1), public 1-minute quote history. ForecastEx is a different
station (LGA etc.) and source (Weather Underground hourly obs) and thresholds instead of brackets = a NEW trial, and its
only public data is daily OHLC, so the 22:00 decision price does not exist -> not built (needs an IBKR account).

Data (build 2026-10-22): 5,069 settled Polymarket brackets, 5 cities x 168 days (2026-04-29..10-20) + PHL 6 days;
5-model forecasts present (GEM 98.6%, others 100%). Settlement cross-check vs Kalshi's official value: 2 brackets on
1 day disagree (MIA 2026-08-29, sealed side) -> excluded. Kalshi's candlestick service answered 503 all morning, so
the historical Kalshi-vs-Polymarket quote comparison is EMPTY; re-run `build` when it is back (cache resumes).

TRAP FOUND (amended into the pre-registration before scoring): Polymarket's price-history "display" quote is not the
book. With no YES bid resting, it showed yes_bid 0.01, 0.02 (above a 0.01 ask) and 0.09. 140/3,083 dev quotes are
locked/crossed -> dropped. An empty bid shown as a plausible price cannot be detected, and it flatters the backtest
(buying NO against a bid that wasn't there). So P1 is indicative only; the live-book paper logger is the clean test.

P1 DEV (2026-04-29..08-14, ledger `venues_weather_P1_pmus`, dataset pmus_temp, 1 trial, bar p < .05):
  283 bets (~80/mo), 8 losses, units -0.39, ROI -0.14%, 95% CI on units [-6.97, +5.51], p .53, months + 3/5, avg NO .972
  by city: LAX +0.33/27, CHI +1.07/52, MIA -0.30/73, NYC -0.52/74, SFO -0.97/57 -> FAIL.
  The power warning was stated before scoring: Kalshi's sealed +1.45% would need thousands of bets to resolve, so this
  is "no edge visible", not "proven no edge". Sealed P1 (2026-08-15..) is NOT worth a yes on this dev result.
Cross-venue arb: live books 2026-10-22 ~07-08Z, two snapshots: 30 and 18 identical brackets matched; after-fee
  YES-here + NO-there cost < $1 on 0; median best cost $1.044 / $1.054. Mids within ~1-2c. No arb.
Polymarket book depth at the top YES bid (validation 10-22): 1 to 606 contracts (Kalshi median 617 within 1c).

Forward paper (no orders, no launchd installed): `scripts/venues_weather.py paper tick` (22-23h local window, live book,
logs data/venues_weather_paper.jsonl, settles via /v1/market/slug); `paper validate` exercises it now (4 picks today).
To install hourly (user's call), copy the kalshi-paper pattern (scripts/install_kalshi_paper_agent.sh; launchd cannot
read ~/Desktop): a launchd job at Minute=7 running
  `cd <code copy outside Desktop>/sports && ../.venv_quant/bin/python scripts/venues_weather.py paper tick`

## What to open (ranked by usable edge, not hype)

1. Kalshi -- keep; the only sealed-passed edge (temp orig7 + X1).
2. Polymarket US -- open it (CA listed as available, automated API allowed, same fee). Its weather markets are the SAME
   bet as our Kalshi picks (same stations and days), so it adds capacity and a second price, not independent edge;
   dev port shows no edge on display quotes and no arb. Expected $/month today ~ $0 (see below); option value for
   politics/sports cross-venue gaps (xvenue_elections.py) and if Kalshi depth ever binds.
3. Interactive Brokers (ForecastEx) -- the only liquid weather venue with a DIFFERENT station/source; worth an account
   only to run a new pre-registered trial (needs its live quotes). $0/month until that trial passes.
4. Skip for diversification: Robinhood, Coinbase, Webull, PrizePicks (Kalshi/ForecastEx routed), and the sports-only
   books (DK, FanDuel, Underdog, Fanatics, Crypto.com, Novig, ProphetX) -- no tested edge there.

$/month at risk.md stakes (B = $1,000, ~1% of B = $10 per pick): Polymarket weather ~80 bets/mo x $10 = $800/mo turnover.
At the dev point estimate (-0.14%) that is about -$1/mo; even at Kalshi's sealed +1.45% it would be about +$12/mo, and
those bets are correlated with picks we already make on Kalshi. Nothing in this pass adds evidence-backed $/month.
