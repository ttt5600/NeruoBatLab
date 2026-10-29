# IBKR for ForecastEx via IB Gateway + IBC (one approval a week instead of a login every day)

Built 2026-10-29. This replaces the Client Portal Gateway (`~/.claude/quant-kalshi/ibkr/`, port 5050). That gateway
needs a browser login with 2FA every day. In the new setup, IB Gateway is run by IBC. It logs itself in, restarts
every night without asking for 2FA, and needs **one IBKR Mobile push approval a week**. The executor talks to it
through `ib_async`.

## What is installed (nothing is running yet)

| What | Where |
|---|---|
| IB Gateway 10.50 (stable, standalone, arm64, ships its own Java) | `~/ibkr/Applications/IB Gateway 10.50/` |
| IBC 3.24.2 (macOS) | `~/ibkr/ibc-3.24.2/` |
| IBC config: **placeholders for you to replace**, mode 600 | `~/ibkr/ibc/config.ini` |
| Start script (IBC's own, with our paths filled in) | `~/ibkr/gatewaystartmacos.sh` |
| launchd agent, written but **not loaded** | `~/ibkr/launchd/com.jonathanwang.ibc-gateway.plist` |
| Load / unload / status helper | `~/ibkr/install_ibc_agent.sh [install\|remove\|status]` |
| Gateway settings and IBC logs | `~/ibkr/Jts/`, `~/ibkr/ibc/logs/` |
| TWS client (same venue interface as the Web API client) | `scripts/venues/forecastex_tws_client.py` |
| Backend switch in the executor | `FX_BACKEND=webapi` (default) or `tws` in `~/.claude/quant-kalshi/.env` |

Everything lives under `~/ibkr`, not `~/Desktop`, because launchd cannot read `~/Desktop` (macOS TCC).

Settings changed from IBC's sample config: `TradingMode=live`, `ReloginAfterSecondFactorAuthenticationTimeout=yes`,
`SecondFactorAuthenticationExitInterval=90`, `AutoRestartTime=11:45 PM` (nightly restart, no 2FA),
`ColdRestartTime=10:00` (**Sunday 10:00 PT** full login, which is the weekly push),
`ExistingSessionDetectedAction=primary`, `AcceptIncomingConnectionAction=accept`, `ReadOnlyApi=no`,
`OverrideTwsApiPort=4001`, `AcceptNonBrokerageAccountWarning=yes`. The start script uses `TWOFA_TIMEOUT_ACTION=restart`.

**Password storage:** IBC cannot read the macOS Keychain. It takes credentials only from `config.ini` or the command
line, and the command line would expose them in `ps`. So the password goes in `config.ini`, which only you can read
(mode 600, folder 700, protected by FileVault). The `fund-ibkr` Keychain item stays where it is for the old `login.py`.

## Your steps (one time, about 15 minutes)

1. **Enrol IBKR Mobile (IB Key)** so logins approve with a push notification instead of an SMS code.
   - Install **IBKR Mobile** on your phone and open it. Choose *Authenticate* (or *Register Two-Factor*), then
     follow the IB Key activation using your username, password and the SMS code.
   - Check it worked: in Client Portal (web), *Settings > Security > Secure Login System* should show **IB Key**.
     A test web login should now send a push.
   - Note: the old `ibkr/login.py` reads an SMS code from Messages. It stops working once IB Key replaces SMS.
     That is expected, because this setup replaces it.
2. **Fill in your credentials.** Run `open -e ~/ibkr/ibc/config.ini` (or use nano). Replace `YOUR_IBKR_USERNAME` and
   `YOUR_IBKR_PASSWORD` on the `IbLoginId=` and `IbPassword=` lines, then save. Keep it private with
   `chmod 600 ~/ibkr/ibc/config.ini`. Do not paste the password into a chat or a commit.
3. **Stop the Client Portal Gateway.** One username can hold only one brokerage session, so the two gateways would
   fight over it: `pkill -f clientportal.gw`. No launchd agent restarts it; it was started by hand with
   `ibkr/start_gateway.sh`.
4. **Start IB Gateway (first approval):** `~/ibkr/install_ibc_agent.sh`. The helper refuses to run while the
   placeholders or the Client Portal Gateway are still there. The Gateway window opens, IBC types the login, and your
   phone gets a push: **approve it within about 3 minutes**. If you miss it, IBC tries again. The first time, macOS may
   ask to allow Java or a background item; allow it.
5. **Check it:** `~/ibkr/install_ibc_agent.sh status` should report port 4001 listening. Then:
   ```
   cd ~/.claude/quant-kalshi
   .venv/bin/python scripts/venues/forecastex_tws_client.py preflight   # expect CONNECTED, account U2***
   .venv/bin/python scripts/venues/forecastex_tws_client.py resolve UHLGA_<MMDDYY>_<strike>   # use a listed ticker
   .venv/bin/python scripts/venues/forecastex_tws_client.py book    UHLGA_<MMDDYY>_<strike>
   ```
   `resolve` must print a conId for both YES (C) and NO (P). It fails if the Gateway's conId differs from the public
   feed's.
6. **Switch the executor to TWS.** Do it when `scripts/forecastex_exec.py status` shows `open 0`, because orders
   placed through the Web API are tracked by Web API order ids, which the TWS backend cannot look up.
   ```
   cp .env .env.bak-$(date +%F)-fxtws && echo 'FX_BACKEND=tws' >> .env
   .venv/bin/python scripts/forecastex_exec.py status    # first line must read "backend tws; gateway session AUTHENTICATED"
   ```
   The launchd executor (every 5 minutes) uses the new backend on its next run. Caps, KILL, the loss stop and tier-B
   limits are unchanged.

## Weekly maintenance

- **Sunday about 10:00 PT:** approve the IBKR Mobile push. IBKR ends every session once a week. IBC's cold restart
  starts the new login at that time, and if you miss the push it keeps asking again.
- **Nightly 23:45 PT:** the Gateway restarts by itself with no 2FA. The executor may miss one 5-minute run.
- **The Mac has to be awake and logged in** (the Gateway is a GUI app). Sleep pauses everything until it wakes.
- **If it is down:** run `~/ibkr/install_ibc_agent.sh status` and check the newest `~/ibkr/ibc/logs/ibc-*.txt`. The
  agent retries at :07 every hour, and `launchctl kickstart -k gui/$(id -u)/com.jonathanwang.ibc-gateway` forces a
  start. While it is down the executor prints `NOT TRADING: no IB Gateway API session` and places nothing.
- **Logging into the IBKR website or app with the same username** can conflict with the Gateway session (`primary`
  means the Gateway keeps it). If that gets in the way, create a second IBKR username for the API in Client Portal
  (*Settings > Users & Access Rights*) and put that one in `config.ini`.
- **Gateway updates:** this is the offline "stable" build, so it never updates itself, which IBC requires. To upgrade,
  install the new version under `~/ibkr/Applications/IB Gateway <ver>` and change `TWS_MAJOR_VRSN` in
  `~/ibkr/gatewaystartmacos.sh`.

## Rollback (back to the Client Portal Gateway)

1. In `~/.claude/quant-kalshi/.env`, delete the `FX_BACKEND=tws` line (or set it to `webapi`).
2. `~/ibkr/install_ibc_agent.sh remove`, then quit the Gateway window (or `pkill -f ibcalpha.ibc.IbcGateway`).
3. `~/.claude/quant-kalshi/ibkr/start_gateway.sh`, then log in at https://localhost:5050 as before.
   (If IB Key is now your 2FA, the login is approved by push rather than SMS.)

## Technical notes

- Contract (IBKR TWS API docs): secType `OPT`, exchange `FORECASTX`, currency USD, symbol `UH<code>`. YES = right
  `C`, NO = right `P`. `lastTradeDateOrContractMonth` is the IB **expiration**, which for UH* stations is the **event
  date + 1** (checked against the public feed on 2026-10-29 for LGA, DFW, LAX and PHX). The client takes it from the
  feed's `expiration` field and falls back to event date + 1.
- Never sell: the client has one order factory, it is BUY / LMT / whole contracts, and it has no action parameter.
  `_place()` checks the action again before the socket. To exit a position, BUY the opposite side.
- Order id = IB `permId` (global). Client id is fixed at 17 (`FX_TWS_CLIENT_ID`) so a later run can still cancel.
  Other settings: `FX_TWS_HOST` and `FX_TWS_PORT` (default 4001), `FX_TWS_QUOTE_WAIT` (4 s), `FX_TWS_ACK_WAIT` (10 s),
  `FX_TWS_MKT_DATA_TYPE` (1 = live).
- If the TWS snapshot returns no quote, the book falls back to the public ForecastTrader top of book, which has no
  sizes.
- Not yet tested against a real session: the IBC login, `reqContractDetails` on a real UH* contract, and whether
  ForecastEx snapshot data arrives without a subscription. Step 5 is that test.
