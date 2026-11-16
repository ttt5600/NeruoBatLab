# crypto_barrier_v2 -- CB2 (Deribit-smile barrier) + CB3 (MAX-rung NO structural rule), FORWARD ONLY

Agent: crypto_barrier_v2 (research/ops subagent, 2026-11-16, started 10:23Z). No orders, no git, no .env/kill changes.
Parent study + CB1: notes/crypto_barrier.md. Scratch: results/crypto_barrier_v2/.

## Why (from notes/crypto_barrier.md, DEV = 12 months / 21 events, ALREADY LOOKED AT -> nothing is scored on it)
(a) Both CB1 models UNDER-predict MIN touches (spread<=5c MIN rows: M1 .080 / M2 .096 / mid .124 / realised .123) --
consistent with put skew that a flat ATM DVOL + symmetric t ignore. (b) MAX rows: Kalshi YES mid .093 vs realised .045
(post hoc, one 12-month path). Both are hypotheses generated on DEV; the only honest test is forward.

## Pre-registration (written 2026-11-16 ~10:30Z, BEFORE any forward CB2/CB3 data exists; first decision 16:00Z today)
Shared with CB1 (unchanged): universe KXBTCMAXMON, KXBTCMINMON, KXETHMAXMON, KXETHMINMON (+ legacy KXBTCMAXM if listed);
same daily snapshot (first logger run >= 16:00 UTC; a bet only if the Kalshi GET starts in [16:00, 16:30) UTC, else
`off_time`); all model inputs read BEFORE the Kalshi GET; tau to the window end from rules_primary (never close_time);
spot = Coinbase ticker; skip if spot already through the strike; taker at the ask, 1 contract, fee 0.07 a(1-a) exact;
at most ONE bet per market per rule (first firing day); outcome = Kalshi `result`; cluster = ET month of window end.
Judge (both): edgelab forward split, family kalshi_data_ladders bar (alpha .05 / #trials); no verdict before n >= 60
settled bets AND >= 12 month-clusters; kill roi_ci_below_zero after 60; ROI > +15% on >= 60 = bug until rows inspected.
CB1, CB2, CB3 share snapshots and outcomes (one BTC/ETH path per month) -> they are NOT independent evidence of each other.

### CB2 = smile barrier model, same decision rule as CB1 with M1 replaced by MS
Code: scripts/crypto_barrier_v2_model.py (frozen at registration; sha256 recorded below).
- Smile input: Deribit public GET /public/get_book_summary_by_currency?currency=BTC|ETH&kind=option, read before the
  Kalshi GET. Per expiry e (08:00 UTC on the instrument date): F_e = underlying_price; OTM options only (puts K < F_e,
  calls K >= F_e) with mark_iv > 0 AND bid_price > 0; smile point (k = ln(K/F_e), w = (mark_iv/100)^2 T_e).
  Expiries with < 5 points dropped.
- Total variance at log-moneyness k and horizon tau: linear interpolation of w in k within each expiry (flat in sigma
  beyond the outermost quoted strike: w scaled by T), then linear in T between the two expiries bracketing tau
  (tau below the first expiry: sigma of the first expiry; above the last: sigma of the last). Monthly window ends
  (last day 23:59 ET) fall between Deribit expiries -> both neighbours used.
- Driftless mapping (as CB1, r = 0, Coinbase spot S): the barrier B is read at k = ln(B/S); prices are Black with F = S.
  (Deribit forward basis is ignored, same as CB1.)
- Barrier probability = smile-consistent static-replication formula, exact for GBM with r = 0 (reflection with drift
  -sigma^2/2 in log space) and the standard skew-aware approximation otherwise:
    MAX:  P = 2 D_call(B) + C(B)/B,   D_call(K) = -dC/dK  (skew-adjusted digital, finite difference h = 1e-3 K)
    MIN:  P = 2 D_put(B)  - Put(B)/B, D_put(K)  = +dPut/dK
  C, Put = Black(F = S, K, sqrt(w(ln K/S, tau))). Per-minute monitoring: BGK shift B' = B exp(+/- 0.5826 sigma_B sqrt(1 min))
  (outward), sigma_B = smile vol at the barrier. P clipped to [0, 1]. Sanity: with a flat smile MS == CB1's M0.
- Rule: scripts/crypto_barrier_model.decide (CB1's frozen function) with p1 := MS, p2 := M2 (same hourly bootstrap):
  P_cons = min(MS_X, M2_X); fire iff P_cons - ask - fee >= .05, ask in [.05, .95], spread <= 5c, tau >= 2 d.
- Diagnostic only (not a trial): MS for SOL is not computed (no Deribit SOL options used); yearly series logged.

### CB3 = structural NO on MAX rungs (M1 only)
- Fire (side NO, taker at NO ask = 1 - yes_bid) iff barrier is MAX, yes_bid/yes_ask valid with yes_ask - yes_bid <= .05,
  mid = (yes_bid + yes_ask)/2 in [0.05, 0.25], CB1's M1 P(touch) < mid / 2, NO ask in [.05, .95], tau >= 2 d.
- Choice of [x, y] = [0.05, 0.25], fixed now without any outcome beyond the recorded fact (b): the DEV MAX gap was a mean
  over low-priced rungs (mid mean .093), i.e. the longshot zone; .05 is the floor below which the NO ask is >= .95 (the
  shared ask cap) once spread is counted; .25 caps the rule at longshots -- above it the YES side is no longer a
  longshot and there is no stated reason for overpricing. mid/2 (M1 must say the rung is at most half as likely as the
  book) is a coarse, round margin; with NO ask ~ .80-.95 the fee is <= .011 so no extra edge threshold is added.
- No outcome-dependent tuning was done; no DEV ROI of CB3 is computed (a would-fire COUNT on DEV, without outcomes, may
  be reported for power only).

### Logging (no new launchd job)
scripts/crypto_barrier_forward.py, same tick: adds Deribit book summaries to the inputs (before the Kalshi GETs), adds
quote columns ps (MS), sigma_b, fires_cb2, fires_cb3, and writes CB2/CB3 picks (field `rule`) to the SEPARATE stream
~/.claude/quant-sports/data/kx_forward/crypto_barrier_v2.jsonl (so CB1's file/adapter are untouched); settle covers
both streams. A CB2 failure (Deribit down) is logged as `cb2_error` on the day row and never blocks CB1/CB3.
Adapter edgelab/adapters/crypto_barrier_v2_fwd.py (params rule CB2 | CB3); specs cb2_crypto_barrier_smile_taker,
cb3_crypto_max_no_taker; scorecard cards CB2, CB3.

## Log
- 10:23Z started; inbox row. Prereg above written ~10:25Z (x, y = .05, .25 fixed before anything below).
- Model code scripts/crypto_barrier_v2_model.py sha256 0db239e9d1bac91d5a94284f2ae85fa78ee8dffca9ccd466517b0d79a3ad4d57
  (frozen). CB1's crypto_barrier_model.py untouched (sha256 6093f148...99ba, same as deployed).
- Sanity (results/crypto_barrier_v2/sanity.py): flat smile -> MS == CB1 M0 to 8.9e-6 (finite-difference error) on
  MAX and MIN. Live smile 10:2xZ: 11 Deribit expiries BTC and ETH (20-57 OTM points each); window end 14.8 d sits between
  27NOV26 (10.9 d) and 4DEC26 (17.9 d). Put skew visible: BTC sigma at B/S .8 = .626 vs ATM .415, so MS MIN 0.8 = .053 vs
  M1(ATM) .020; MAX 1.2 MS .021 vs M1 .037 (MS lower on far MAX) -- the direction the DEV fact (a) points to. Model
  values only, no outcomes.
- Logger patched (results/crypto_barrier_v2/patch_logger.py; pre-patch copy crypto_barrier_forward.pre_v2.py):
  scripts/crypto_barrier_forward.py sha256 ee4f4dd1f7eb709954da3525ff1fe5a6228f48d4eb9fd869bce6053b2c8511cf. CB1 logic,
  CB1 file and CB1 adapter (sha256 ab820521... unchanged) untouched; CB1 day row gains a `v2` summary key only; settle
  now loops over both streams (same per-ticker semantics).
- Deployed (copied, no plist change): ~/.claude/quant-sports/scripts/{crypto_barrier_forward.py, crypto_barrier_v2_model.py}
  (hashes match the repo). install_crypto_barrier_agent.sh now also copies crypto_barrier_v2_model.py.
- Verified 10:29Z under the DEPLOYED venv: `tick --force-now` -> 71 quotes, 57 scored CB1, 46 scored CB2 (BTC/ETH trial +
  yearly; SOL has no smile), cb2_error none, fires CB1 0 / CB2 0 / CB3 0; quote rows carry ps, sigma_b, fires_cb2,
  fires_cb3 (results/crypto_barrier_v2/inspect_manual.py); v2 day row written to crypto_barrier_v2_manual.jsonl. Idle
  `tick` rc 0, `settle` rc 0, launchd kickstart -> heartbeat 02:30 PST rc 0, err log empty. First real decision 16:00Z.
- edgelab (main registry, verify first_break null): cb2_crypto_barrier_smile_taker-8d73f36a (family kalshi_data_ladders
  57 trials -> bar 8.77e-4), cb3_crypto_max_no_taker-c008cbd0 (58 -> bar 8.62e-4). vault.yaml kalshi_crypto_barrier gains
  both forward_adapters (edgelab/adapters/crypto_barrier_v2_fwd.py, params rule CB2|CB3, sha256 cb37c452...). Sandbox
  register + evaluate --split forward (results/crypto_barrier_v2/registry_sandbox.jsonl): runs, 0 bets. Synthetic
  adapter check (results/crypto_barrier_v2/adapter_check.py): rules separated, off_time/unsettled excluded, PIT ok.
- Scorecards: card_cb_v2("CB2"/"CB3") in scripts/scorecards.py; test render results/crypto_barrier_v2/scorecards_test.md
  -> both too-early.

## Power (counts only, no outcome read)
- CB3 would-fire COUNT on the DEV frame (results/crypto_barrier_v2/dev_wouldfire_count.py; reads no `result` column):
  18 market-days, 11 markets (first firing), 6 events in 12 months -> ~1 bet/month. 60 bets is ~5 years away; CB3 is a
  long-run log, not something that will produce a verdict soon. Not changed after counting.
- CB2 has no DEV analogue (no historical Deribit surface snapshots at 16:00Z) -> its fire rate is unknown; today's
  snapshot fired 0 (the book sat between MS and M2 on every rung).

## Verdict
Nothing scored; both trials are forward-only and registered before their first decision (16:00Z 2026-11-16). CB1, CB2,
CB3 share snapshots and one BTC/ETH path per month, so a pass in one is not independent support for another.

## Learnings
### A smile-consistent one-touch probability is 2 x digital + call/B (MAX) or 2 x digital - put/B (MIN)
With r = 0 the reflection result for GBM equals P_up = 2 D_call(B) + C(B)/B and P_down = 2 D_put(B) - P(B)/B; plugging
skew-adjusted digitals (-dC/dK from the smile) gives a skew-aware barrier price in a few lines. Flat-smile check equals
CB1 M0 to 9e-6 (results/crypto_barrier_v2/sanity.py).

### Deribit put skew roughly doubles far MIN touch probabilities vs a flat ATM vol
11-16 BTC, 14.8 d: barrier at 0.8 x spot has smile vol .626 vs ATM .415 -> MS .053 vs flat-t4 .020; far MAX goes the
other way (1.2 x: .021 vs .037). That is the size of the gap CB1's flat DVOL leaves on MIN rungs.

### Keep a new rule's picks in a separate stream when an old trial's adapter is hash-pinned
CB2/CB3 picks go to crypto_barrier_v2.jsonl so CB1's file and adapter sha stay unchanged; one logger, one snapshot,
one settle loop over both files.

### A structural longshot rule on monthly ladders fires about once a month
CB3 on DEV: 11 markets in 12 months (count only). Monthly one-path clustering makes any such rule a multi-year trial.
