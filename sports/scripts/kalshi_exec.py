"""Kalshi order executor for the sealed-passed temperature consensus-reject rule (orig7 + X1). DRY BY DEFAULT.

    python scripts/kalshi_exec.py preflight      # env, key-file perms, API reachability, balance, launchd, last tick
    python scripts/kalshi_exec.py status         # spend vs caps, open exposure, loss stops, P&L vs paper, positions
    python scripts/kalshi_exec.py run            # dry run: size fresh paper picks against the live book, log only
    python scripts/kalshi_exec.py run --live     # orders -- ONLY if .env also has EXEC_MODE=live (the user sets it)
    python scripts/kalshi_exec.py smoke          # DEMO ONLY: rest 1 unfillable contract, re-send it (dedupe), cancel
    python scripts/kalshi_exec.py kill [reason]  # write the KILL file and cancel this executor's resting orders
    python scripts/kalshi_exec.py check          # read-only: does the key sign requests (GET balance)
    python scripts/kalshi_exec.py report         # alias of status

Modes (resolved every run): dry  = nothing sent (default; what the launchd agent does today)
                            demo = EXEC_ENV=demo + EXEC_MODE=live + --live: real orders on Kalshi's DEMO exchange
                            live = EXEC_ENV=prod (default) + EXEC_MODE=live + --live: real money
EXEC_MODE and EXEC_ENV are read only from the .env file next to scripts/ (deployed: ~/.claude/quant-kalshi/.env),
never from the process environment, so nothing but the user's edit of that file can turn orders on.

Inputs: picks written by kalshi_paper.py (data/kalshi_paper.jsonl). The rule itself is NOT changed here; every pick
is re-checked against kalshi_paper.rejected() on its own logged forecasts before an order is considered.

Safety, all enforced here (not by the caller):
  * kill switch: KILL in this root or ~/.claude/quant-kalshi/ stops everything (one notification per KILL file)
  * ONE decision per (mode, ticker), ever: the ticker is written to data/kalshi_exec.jsonl BEFORE the POST
    (write-ahead), the client_order_id is deterministic (uuid5 of strategy|mode|ticker) so a retry or a re-run
    re-sends the same id and Kalshi rejects the duplicate (409), and before every order the exchange itself is
    asked for existing orders/positions on that ticker
  * caps (.env): KALSHI_MAX_ORDER_USD=25, KALSHI_MAX_DAILY_USD=100, KALSHI_MAX_OPEN_USD=300, KALSHI_MAX_CONTRACTS=1
    (per demo/live order). With KALSHI_BANKROLL_USD=B (required for demo/live) notes/risk.md also binds:
    <= 1% of B per orig7 pick, <= 0.5% per X1 pick, <= 10% of B per day and open; daily loss stop 5% of B
    (trailing 24 h of settlements -> no new orders); weekly 10% of B (7 d) -> writes KILL (live only)
  * stale-signal guard: pick older than KALSHI_MAX_PICK_AGE_MIN (120) or market not active / closing within
    KALSHI_MIN_MINUTES_TO_CLOSE (60) -> skip
  * price sanity: NO ask now > pick + KALSHI_MAX_SLIPPAGE (0.01) or < pick - KALSHI_MAX_PRICE_DROP (0.03) -> skip
  * size <= KALSHI_DEPTH_FRAC (0.5) x resting depth within 1c; limit price = the ask; immediate_or_cancel, so
    nothing rests; any of OUR orders found resting > KALSHI_STALE_ORDER_MIN (10) is cancelled
  * fill-price convention check: a fill whose NO cost differs from the ask by > 2c writes KILL (wrong convention)
  * KALSHI_EXEC_STYLE=maker (default taker; KALSHI_AB=1 -> maker iff sha1(ticker) even): sealed-passed M3_N10
    (lab/lockbox.jsonl kalshi_temp_maker_M3_N10, notes/kalshi_temp_maker.md). Resting post-only GTC NO bid at
    q = min(NO mid floored to the cent, NO ask - 1c), expiration_time = market close, no taker fallback; skip if
    NO ask - q > 5c. Fee 0 only when GET /series fee_type == quadratic (else taker fee in sizing). While resting it
    commits n*q to the caps; the stale sweep leaves it alone until its market is no longer active; post_only
    reject -> rejected_post_only (no retry); the convention check compares fills with q, not the ask.
  * every fill / partial fill / error / kill / loss stop -> macOS notification (KALSHI_NOTIFY=0 turns it off)
  * per-cohort / per-price contract caps (all default OFF; added 2026-11-15, notes/ops_e4_prep.md), applied on top of
    KALSHI_MAX_CONTRACTS in demo/live: KALSHI_X1_FRAC (bankroll share per X1 pick, default 0.005),
    KALSHI_X1_MAX_CONTRACTS (X1-only contract cap), KALSHI_X1_DAY_USD (X1 sub-cap on BOTH today's X1 spend and open
    X1 exposure, inside the shared day/open caps, so X1 cannot use up orig7's room), KALSHI_PRICE_BAND_CAPS="0.97:24,0:19" (contracts by NO price at
    the order: the highest floor <= price applies). X1 kill (only while x1 is in KALSHI_COHORTS): settled X1 fills
    since KALSHI_X1_KILL_SINCE (ISO; unset = all) -> >= 2 losses in the first 30, or ROI < 0 once >= 60 -> a sticky
    cohort_kill row is journaled and X1 picks become skip_cohort_killed (orig7 keeps trading).
Credentials: KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH in .env, put there by the user. The key file is read only to
sign requests; the key, its path contents and request headers are never printed or logged.

API checked against docs.kalshi.com 2026-10-21 (OpenAPI 3.34.0): hosts prod external-api.kalshi.com, demo
external-api.demo.kalshi.co; signature = base64(sign(ts_ms + METHOD + /trade-api/v2/path-without-query)), Ed25519
or RSA-PSS(SHA256, MGF1 SHA256, salt=digest); POST /portfolio/events/orders {ticker, side bid|ask (YES leg),
count, price (YES dollars), time_in_force, self_trade_prevention_type, client_order_id} -> 201 {order_id,
fill_count, remaining_count, average_fill_price, average_fee_paid}; "sell YES == buy NO at 1 - price";
duplicate client_order_id -> 409; GET /portfolio/orders (ticker, status=resting|canceled|executed, cursor);
GET /portfolio/positions (position_fp < 0 = NO); DELETE /portfolio/events/orders/{id}?market_ticker=;
429 = {"error": "too many requests"}, no Retry-After -> exponential backoff; GET /portfolio/balance 1/s.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
HOSTS = {"prod": {"trade": "https://external-api.kalshi.com", "public": "https://api.elections.kalshi.com"},
         "demo": {"trade": "https://external-api.demo.kalshi.co", "public": "https://external-api.demo.kalshi.co"}}
BASE = "/trade-api/v2"
FEE = 0.07
STRATEGY = "consensus-reject-v1"
AGENT_LABEL = "com.jonathanwang.quant-kalshi-paper"
TIER_FRAC = {"orig7": 0.01, "x1": 0.005}            # notes/risk.md: share of bankroll per pick
NIGHT_FRAC, DAY_LOSS_FRAC, WEEK_LOSS_FRAC = 0.10, 0.05, 0.10
RETRY_CODES = {429, 500, 502, 503, 504}
ZERO_COST = ("no_fill", "rejected", "rejected_post_only", "not_placed")
WEATHER_SHARD = 0  # exchange_index of every KXHIGH* market (GET /markets/{t}, verified 2026-11-15)
MAKER_MAX_IMPROVE = 0.05                        # notes/kalshi_temp_maker.md: skip if NO ask - q > 5c
MAKER_FREE_FEE_TYPES = ("quadratic",)           # General Trading Fees Table: no maker fee
X1_KILL_FIRST, X1_KILL_LOSSES, X1_KILL_ROI_N = 30, 2, 60  # X1 restart kill (notes/ops_e4_prep.md)


def set_root(root: Path, deploy: Path = DEPLOY) -> None:
    """Point every file path at one root (tests and the deployed copy use their own)."""
    global ROOT, DATA, PICKS, LOG, STATE, ENV, KILLS, RUNLOG
    ROOT = Path(root)
    DATA = ROOT / "data"
    PICKS = DATA / "kalshi_paper.jsonl"
    LOG = DATA / "kalshi_exec.jsonl"
    STATE = DATA / "kalshi_exec_state.json"
    ENV = ROOT / ".env"
    KILLS = [ROOT / "KILL", Path(deploy) / "KILL"]
    RUNLOG = Path(deploy) / "run.log"


# KALSHI_EXEC_HOME only relocates files (e.g. a scratch copy of data/); the mode still comes from THAT root's .env
set_root(Path(os.environ.get("KALSHI_EXEC_HOME") or Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------------------------------- config / files
def env() -> dict:
    out = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def config(e: dict) -> dict:
    f = lambda k, d: float(e.get(k, d))  # noqa: E731
    ex_env = e.get("EXEC_ENV", "prod").lower()
    if ex_env not in HOSTS:
        raise ValueError(f"EXEC_ENV must be prod or demo, got {ex_env!r}")
    b = e.get("KALSHI_BANKROLL_USD")
    return {"env": ex_env, "bankroll": float(b) if b else None,
            "max_order": f("KALSHI_MAX_ORDER_USD", 25), "max_day": f("KALSHI_MAX_DAILY_USD", 100),
            "max_open": f("KALSHI_MAX_OPEN_USD", 300), "max_contracts": int(f("KALSHI_MAX_CONTRACTS", 1)),
            "max_up": f("KALSHI_MAX_SLIPPAGE", 0.01), "max_down": f("KALSHI_MAX_PRICE_DROP", 0.03),
            "max_age_min": f("KALSHI_MAX_PICK_AGE_MIN", 120), "depth_frac": f("KALSHI_DEPTH_FRAC", 0.5),
            "min_close_min": f("KALSHI_MIN_MINUTES_TO_CLOSE", 60), "stale_order_min": f("KALSHI_STALE_ORDER_MIN", 10),
            "notify": e.get("KALSHI_NOTIFY", "1") != "0",
            # 2026-11-03 (user): KALSHI_COHORTS=orig7 pauses X1 (edgelab: E4-X1 dead); KALSHI_ORIG7_FRAC sizes orig7
            "cohorts": {c.strip() for c in e.get("KALSHI_COHORTS", "orig7,x1").split(",") if c.strip()},
            "orig7_frac": f("KALSHI_ORIG7_FRAC", TIER_FRAC["orig7"]),
            # 2026-11-05 (user, edgelab early 2x step): day/open share of bankroll; loss stops unchanged
            "night_frac": f("KALSHI_NIGHT_FRAC", NIGHT_FRAC),
            "exec_style": _style(e.get("KALSHI_EXEC_STYLE", "taker")), "ab": e.get("KALSHI_AB", "0") == "1",
            # 2026-11-15 (ops_e4_prep): X1 restart at minimum size + E4 price-band cap; all off unless set
            "x1_frac": f("KALSHI_X1_FRAC", TIER_FRAC["x1"]),
            "x1_max_contracts": int(e["KALSHI_X1_MAX_CONTRACTS"]) if e.get("KALSHI_X1_MAX_CONTRACTS") else None,
            "price_band_caps": parse_band_caps(e.get("KALSHI_PRICE_BAND_CAPS", "")),
            "x1_kill_since": e.get("KALSHI_X1_KILL_SINCE") or None,
            "x1_day": float(e["KALSHI_X1_DAY_USD"]) if e.get("KALSHI_X1_DAY_USD") else None}


def parse_band_caps(v: str) -> list[tuple[float, int]]:
    """"0.97:24,0:19" -> [(0.97, 24), (0.0, 19)] sorted by floor, highest first. Empty -> [] (off)."""
    out = []
    for part in (v or "").split(","):
        if not part.strip():
            continue
        lo, n = part.split(":")
        lo, n = float(lo), int(n)
        if not 0 <= lo < 1 or n < 0:
            raise ValueError(f"KALSHI_PRICE_BAND_CAPS entry {part!r}: need 0 <= floor < 1 and contracts >= 0")
        out.append((lo, n))
    return sorted(out, reverse=True)


def contract_cap(cfg: dict, cohort: str, price: float, mode: str) -> int:
    """Max contracts for one order: KALSHI_MAX_CONTRACTS, the X1 cap, and the price band. dry = uncapped (as before).
    A price below every band floor is uncapped by the bands."""
    if mode == "dry":
        return 10 ** 9
    n = cfg["max_contracts"]
    if cohort == "x1" and cfg.get("x1_max_contracts") is not None:
        n = min(n, cfg["x1_max_contracts"])
    for lo, cap in cfg.get("price_band_caps") or []:
        if price >= lo - 1e-9:
            n = min(n, cap)
            break
    return n


def _utc(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def x1_kill_verdict(state: dict, since: str | None) -> str | None:
    """Pre-set X1 restart kill on this mode's settled X1 fills (orders placed at/after `since`): >= X1_KILL_LOSSES
    losses within the first X1_KILL_FIRST, or ROI < 0 once there are >= X1_KILL_ROI_N. None = keep going."""
    lo = _utc(since) if since else None
    rows = sorted((o for o in state.values() if o.get("cohort") == "x1" and o.get("settled") and held(o) > 0
                   and (lo is None or _utc(o["ts"]) >= lo)), key=lambda o: _utc(o["ts"]))
    losses = sum(o.get("result") == "yes" for o in rows[:X1_KILL_FIRST])
    if losses >= X1_KILL_LOSSES:
        return f"X1: {losses} losses within the first {min(len(rows), X1_KILL_FIRST)} settled fills"
    if len(rows) >= X1_KILL_ROI_N:
        pnl, cost = sum(o["pnl"] for o in rows), sum(committed(o) for o in rows)
        if cost and pnl / cost < 0:
            return f"X1: ROI {pnl / cost:+.2%} after {len(rows)} settled fills"
    return None


def x1_killed(hist: list[dict], mode: str, since: str | None) -> bool:
    """Sticky: a cohort_kill row for this mode and this KALSHI_X1_KILL_SINCE keeps X1 off (a new SINCE = a new restart)."""
    return any(r.get("event") == "cohort_kill" and r.get("cohort") == "x1" and r.get("mode") == mode
               and r.get("since") == since for r in hist)


def _style(v: str) -> str:
    v = (v or "taker").strip().lower()
    if v not in ("taker", "maker"):
        raise ValueError(f"KALSHI_EXEC_STYLE must be taker or maker, got {v!r}")
    return v


def exec_style(cfg: dict, ticker: str) -> str:
    """taker | maker for one ticker. KALSHI_AB=1 overrides KALSHI_EXEC_STYLE: maker iff sha1(ticker) is even."""
    if cfg.get("ab"):
        return "maker" if int(hashlib.sha1(ticker.encode()).hexdigest()[:8], 16) % 2 == 0 else "taker"
    return cfg.get("exec_style", "taker")


def tag_style(cfg: dict) -> bool:
    """Write exec_style onto order rows only when maker is in play, so the default taker journal is unchanged."""
    return cfg.get("ab", False) or cfg.get("exec_style", "taker") != "taker"


def cent_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def maker_fee_unit(q: float, fee_type: str | None) -> float:
    """Per-contract fee assumed for a resting NO bid at q: 0 on a 'quadratic' series (no maker fee), else the taker
    formula (conservative: unknown or maker-fee schedules are sized as if they charged the full taker fee)."""
    return 0.0 if fee_type in MAKER_FREE_FEE_TYPES else FEE * q * (1 - q)


def resolve_mode(e: dict, live_flag: bool) -> str:
    if not (live_flag and e.get("EXEC_MODE") == "live"):
        return "dry"
    return "demo" if e.get("EXEC_ENV", "prod").lower() == "demo" else "live"


def caps(cfg: dict, cohort: str | None = None) -> dict:
    b = cfg["bankroll"]
    frac = (cfg.get("orig7_frac", TIER_FRAC["orig7"]) if cohort == "orig7" else
            cfg.get("x1_frac", TIER_FRAC["x1"]) if cohort in (None, "x1") else TIER_FRAC.get(cohort, 0.005))
    order = cfg["max_order"] if b is None else min(cfg["max_order"], frac * b)
    nf = cfg.get("night_frac", NIGHT_FRAC)
    day = cfg["max_day"] if b is None else min(cfg["max_day"], nf * b)
    opn = cfg["max_open"] if b is None else min(cfg["max_open"], nf * b)
    return {"order": order, "day": day, "open": opn}


def journal() -> list[dict]:
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def append(rec: dict) -> dict:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_picks() -> list[dict]:
    return [json.loads(l) for l in PICKS.read_text().splitlines() if l.strip()] if PICKS.exists() else []


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s))


def killed() -> Path | None:
    return next((k for k in KILLS if k.exists()), None)


def notify(title: str, msg: str, cfg: dict | None = None) -> None:
    """macOS notification; never raises. Message text is ours (tickers, counts), never credentials."""
    if cfg is not None and not cfg.get("notify", True):
        return
    safe = lambda s: str(s).replace("\\", "").replace('"', "'")[:200]  # noqa: E731
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe(msg)}" with title "{safe(title)}"'],
                       timeout=5, capture_output=True)
    except Exception:
        pass


def notify_once(key: str, title: str, msg: str, cfg: dict) -> None:
    st = load_state()
    seen = st.setdefault("notified", {})
    if key in seen:
        return
    seen[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(st)
    notify(title, msg, cfg)


def client_order_id(mode: str, ticker: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kalshi_exec/{STRATEGY}/{mode}/{ticker}"))


def fee_total(n: float, p: float) -> float:
    """Kalshi taker fee for one order, rounded UP to the cent (conservative for sizing)."""
    return math.ceil(round(FEE * n * p * (1 - p) * 100, 6)) / 100 if n > 0 else 0.0


def order_cost(n: int, p: float) -> float:
    return round(n * p + fee_total(n, p), 4)


def allowed_series() -> set[str]:
    """The sealed-passed universe = kalshi_paper.PAPER_CITIES (orig7 + X1)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kalshi_paper
    return set(kalshi_paper.PAPER_CITIES)


def rule_holds(p: dict) -> bool:
    """Re-check the frozen rule on the pick's own logged forecasts (integrity, not a new rule)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kalshi_paper
    vals = [v for v in (p.get("forecasts") or {}).values() if v is not None]
    if len(vals) < 3:
        return False
    fl = p.get("floor")
    ca = p.get("cap")
    fl = float(fl) if fl is not None else math.nan
    ca = float(ca) if ca is not None else math.nan
    return bool(kalshi_paper.rejected(p.get("strike_type"), fl, ca, vals))


# ----------------------------------------------------------------------------------------------- HTTP client
class TransportError(RuntimeError):
    pass


def _requests_send(method, url, headers, params, data, timeout):
    r = requests.request(method, url, headers=headers, params=params, data=data, timeout=timeout)
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {"raw": r.text[:300]}
    return r.status_code, body


class Kalshi:
    """Thin client. transport(method, url, headers, params, data, timeout) -> (status, json) is injectable."""

    def __init__(self, e: dict, env_name: str = "prod", transport=None, sleep=time.sleep, retries: int = 4):
        self.e, self.host = e, HOSTS[env_name]
        self.send, self.sleep, self.retries = transport or _requests_send, sleep, retries
        self._key = None

    def has_key(self) -> bool:
        return bool(self.e.get("KALSHI_KEY_ID") and self.e.get("KALSHI_PRIVATE_KEY_PATH"))

    def _sign(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
        if self._key is None:
            self._key = serialization.load_pem_private_key(
                Path(self.e["KALSHI_PRIVATE_KEY_PATH"]).expanduser().read_bytes(), password=None)
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path.split("?")[0]).encode()
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            sig = self._key.sign(msg)
        elif isinstance(self._key, rsa.RSAPrivateKey):
            sig = self._key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                  salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        else:
            raise ValueError(f"unsupported key type {type(self._key).__name__}")
        return {"KALSHI-ACCESS-KEY": self.e["KALSHI_KEY_ID"], "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}

    def request(self, method: str, path: str, params=None, body=None, auth: bool = True):
        """Retries 429/5xx/connection errors with exponential backoff, re-signing each attempt. POSTs carry a
        deterministic client_order_id, so re-sending one is safe (a duplicate comes back 409, never a 2nd order)."""
        url = (self.host["trade"] if auth else self.host["public"]) + BASE + path
        data = json.dumps(body) if body is not None else None
        last = None
        for i in range(self.retries):
            try:
                headers = self._sign(method, BASE + path) if auth else {"User-Agent": "Mozilla/5.0"}
                code, out = self.send(method, url, headers, params, data, 20)
                if code not in RETRY_CODES:
                    return code, out
                last = f"HTTP {code}"
            except (requests.ConnectionError, requests.Timeout, TransportError) as ex:
                last = type(ex).__name__
            if i < self.retries - 1:
                self.sleep(0.5 * 2 ** i)
        if last and last.startswith("HTTP"):
            return int(last[5:]), {"error": f"gave up after {self.retries} tries: {last}"}
        raise TransportError(f"{method} {path}: gave up after {self.retries} tries ({last})")

    def _ok(self, code, out, what):
        if code != 200:
            raise TransportError(f"{what}: HTTP {code} {str(out)[:200]}")
        return out

    def _pages(self, path, key, params):
        rows, cursor = [], None
        for _ in range(20):
            q = dict(params, limit=200, **({"cursor": cursor} if cursor else {}))
            out = self._ok(*self.request("GET", path, params=q), path)
            rows += out.get(key, [])
            cursor = out.get("cursor")
            if not cursor:
                break
        return rows

    # public
    def exchange_status(self) -> dict:
        return self._ok(*self.request("GET", "/exchange/status", auth=False), "exchange status")

    def market(self, ticker: str) -> dict | None:
        code, out = self.request("GET", f"/markets/{ticker}", auth=False)
        return out.get("market") if code == 200 else None

    def book(self, ticker: str) -> tuple[float, float, float]:
        """(best NO ask, contracts within 1c of it, best NO bid). NO ask = 1 - best YES bid; the NO bid is the best
        price on the book's NO side (= 1 - best YES ask), 0.0 when that side is empty."""
        code, d = self.request("GET", f"/markets/{ticker}/orderbook", auth=False)
        if code != 200:
            return math.nan, 0.0, 0.0
        ob = d.get("orderbook_fp") or d.get("orderbook") or {}
        lvl = lambda rows: sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in rows),  # noqa: E731
                                  reverse=True)
        lv = lvl(ob.get("yes_dollars") or ob.get("yes") or [])
        nb = lvl(ob.get("no_dollars") or ob.get("no") or [])
        no_bid = round(nb[0][0], 4) if nb else 0.0
        if not lv:
            return math.nan, 0.0, no_bid
        best = lv[0][0]
        return round(1 - best, 2), sum(q for p, q in lv if p >= best - 0.0101), no_bid

    def book_no(self, ticker: str) -> tuple[float, float]:
        """(best NO ask, contracts within 1c of it). NO ask = 1 - best YES bid."""
        ask, depth, _ = self.book(ticker)
        return ask, depth

    def series_fee_type(self, series: str) -> str | None:
        """GET /series/{s} -> series.fee_type ('quadratic' = no maker fee); None if unreadable."""
        code, out = self.request("GET", f"/series/{series}", auth=False)
        return (out.get("series") or {}).get("fee_type") if code == 200 else None

    # authenticated
    def balance(self, shard: int | None = None) -> float:
        """Cash an order can use. Kalshi keeps cash PER exchange shard (market.exchange_index; weather = 0) and an
        order only sees its own shard's cash. 2026-11-15: balance_dollars $604 but shard 0 $4.81 -> both orig7 24ct
        maker orders rejected insufficient_balance (notes/e4_ops_diag.md). With `shard`, return that shard's cash
        from balance_breakdown (0.0 if the shard is missing from a non-empty breakdown); else the account total."""
        out = self._ok(*self.request("GET", "/portfolio/balance"), "balance")
        bd = out.get("balance_breakdown") or []
        if shard is not None and bd:
            return next((float(x["balance"]) for x in bd if int(x.get("exchange_index", -1)) == shard), 0.0)
        return float(out.get("balance_dollars", 0))

    def orders(self, ticker: str | None = None, status: str | None = None) -> list[dict]:
        p = {k: v for k, v in (("ticker", ticker), ("status", status)) if v}
        return self._pages("/portfolio/orders", "orders", p)

    def positions(self, ticker: str | None = None) -> list[dict]:
        p = {"ticker": ticker} if ticker else {"count_filter": "position"}
        return self._pages("/portfolio/positions", "market_positions", p)

    def create_order(self, body: dict):
        return self.request("POST", "/portfolio/events/orders", body=body)

    def cancel(self, order_id: str, ticker: str):
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}", params={"market_ticker": ticker})


# ----------------------------------------------------------------------------------------------- journal folding
def fold(hist: list[dict], mode: str) -> dict:
    """ticker -> current state of the ONE decision for that ticker in `mode` (order + updates + settle)."""
    out = {}
    for r in hist:
        if r.get("mode") != mode or "ticker" not in r:
            continue
        ev = r.get("event", "order")
        t = r["ticker"]
        if ev == "order" and t not in out:
            o = dict(r)
            if o.get("status") == "dry":  # pre-2026-10-21 rows: count == held, cost had no fee
                o.setdefault("filled", o.get("count", 0))
                o.setdefault("avg_no", o.get("no_ask_now"))
            out[t] = o
        elif ev == "update" and t in out:
            out[t].update({k: v for k, v in r.items() if k not in ("event", "ts", "mode", "ticker")})
        elif ev == "settle" and t in out:
            out[t].update(settled=True, result=r["result"], pnl=r["pnl"], settled_ts=r["ts"])
    return out


def committed(o: dict) -> float:
    s = o.get("status", "")
    return 0.0 if s.startswith("skip") or s in ZERO_COST else float(o.get("cost", 0) or 0)


def held(o: dict) -> float:
    return float(o.get("filled", 0) or 0)


def spend_and_open(state: dict, now: datetime) -> tuple[float, float]:
    today = now.date().isoformat()
    spent = sum(committed(o) for o in state.values() if o["ts"][:10] == today)
    opn = sum(committed(o) for o in state.values() if not o.get("settled"))
    return spent, opn


def net_pnl(state: dict, now: datetime, hours: float) -> float:
    lo = now - timedelta(hours=hours)
    return sum(o["pnl"] for o in state.values() if o.get("settled") and datetime.fromisoformat(o["settled_ts"]) >= lo)


# ----------------------------------------------------------------------------------------------- settle / reconcile
def settle(hist: list[dict], picks: list[dict], now: datetime, client: Kalshi | None = None) -> int:
    """Append a settle event for every held position whose market resolved (paper results, else the market)."""
    res = {p["ticker"]: p.get("result") for p in picks}
    n = 0
    for mode in sorted({r.get("mode") for r in hist if r.get("mode")}):
        for t, o in fold(hist, mode).items():
            if o.get("settled") or held(o) <= 0 or committed(o) <= 0 or o.get("status") == "resting":
                continue  # a resting (maker) row settles after reconcile has read its final fills
            r = res.get(t)
            if r not in ("yes", "no") and client is not None and now - datetime.fromisoformat(o["ts"]) > timedelta(hours=36):
                r = (client.market(t) or {}).get("result")
            if r not in ("yes", "no"):
                continue
            cost = committed(o)
            pnl = round(held(o) - cost if r == "no" else -cost, 4)
            append({"event": "settle", "ts": now.isoformat(timespec="seconds"), "mode": mode, "ticker": t,
                    "result": r, "held": held(o), "pnl": pnl})
            n += 1
    return n


def fills_from_order(o: dict) -> dict:
    """Exchange Order object -> our fields. Costs are the user's own side in dollars per the Order schema."""
    filled = float(o.get("fill_count_fp") or 0)
    cost = float(o.get("taker_fill_cost_dollars") or 0) + float(o.get("maker_fill_cost_dollars") or 0)
    fee = float(o.get("taker_fees_dollars") or 0) + float(o.get("maker_fees_dollars") or 0)
    initial = float(o.get("initial_count_fp") or filled)
    status = ("resting" if o.get("status") == "resting" else
              "filled" if filled and filled >= initial else "partial" if filled else "no_fill")
    return {"filled": filled, "avg_no": round(cost / filled, 4) if filled else None, "fee": round(fee, 4),
            "cost": round(cost + fee, 4), "order_id": o.get("order_id"), "outcome_side": o.get("outcome_side"),
            "status": status}


def write_kill(reason: str, cfg: dict) -> None:
    k = KILLS[-1] if KILLS[-1].parent.exists() else KILLS[0]
    k.write_text(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {reason}\n")
    append({"event": "kill", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason, "file": str(k)})
    notify("Kalshi exec KILLED", reason, cfg)


def reconcile(client: Kalshi, mode: str, cfg: dict, now: datetime) -> None:
    """(1) resolve submitting/unknown orders from the exchange, (2) cancel OUR stale resting orders,
    (3) compare journal holdings with exchange positions and alert on any mismatch."""
    state = fold(journal(), mode)
    ours = {o.get("client_order_id"): t for t, o in state.items() if o.get("client_order_id")}
    for t, o in state.items():
        if o.get("status") not in ("submitting", "unknown", "resting"):
            continue
        match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == o.get("client_order_id")]
        if match:
            upd = fills_from_order(match[0])
            maker = o.get("exec_style") == "maker" and o.get("maker_q") is not None
            if maker:
                upd = maker_reserve(o, upd)
            if maker and upd["filled"] and upd["avg_no"] is not None and abs(upd["avg_no"] - float(o["maker_q"])) > 0.02:
                upd = {**upd, "cost": round(upd["filled"] * 1.0 + upd["fee"], 4)}  # worst case until a human looks
                append({"event": "update", "ts": now.isoformat(timespec="seconds"), "mode": mode, "ticker": t,
                        "source": "reconcile", **upd})
                write_kill(f"fill price convention check failed on {t}: NO cost {upd['avg_no']} vs maker bid "
                           f"{o['maker_q']}", cfg)
                return
            prev = float(o.get("filled") or 0)
            append({"event": "update", "ts": now.isoformat(timespec="seconds"), "mode": mode, "ticker": t,
                    "source": "reconcile", **upd})
            if upd["filled"] and (not maker or upd["filled"] > prev):
                notify(f"Kalshi {mode} fill (reconciled)", f"{t}: {upd['filled']:g} NO @ {upd['avg_no']}", cfg)
        elif now - datetime.fromisoformat(o["ts"]) > timedelta(minutes=30):
            append({"event": "update", "ts": now.isoformat(timespec="seconds"), "mode": mode, "ticker": t,
                    "status": "not_placed", "filled": 0, "cost": 0.0, "source": "reconcile"})
    for x in client.orders(status="resting"):
        if x.get("client_order_id") not in ours:
            continue
        if state.get(ours[x["client_order_id"]], {}).get("exec_style") == "maker":
            # a maker bid is MEANT to rest until close: cancel only once its market is no longer active / past close
            m = client.market(x.get("ticker") or ours[x["client_order_id"]]) or {}
            close = m.get("close_time")
            open_now = m.get("status") in ("active", "open") and (
                close is None or datetime.fromisoformat(close.replace("Z", "+00:00")) > now)
            if open_now:
                continue
            code, out = client.cancel(x["order_id"], x.get("ticker"))
            append({"event": "cancel", "ts": now.isoformat(timespec="seconds"), "mode": mode,
                    "ticker_cancelled": x.get("ticker"), "order_id": x["order_id"], "http": code,
                    "reduced_by": out.get("reduced_by"), "why": "maker_market_closed"})
            continue
        created = x.get("created_time")
        age = (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).total_seconds() / 60 if created else 1e9
        if age >= cfg["stale_order_min"]:
            code, out = client.cancel(x["order_id"], x.get("ticker"))
            append({"event": "cancel", "ts": now.isoformat(timespec="seconds"), "mode": mode,
                    "ticker_cancelled": x.get("ticker"), "order_id": x["order_id"], "http": code,
                    "reduced_by": out.get("reduced_by")})
    state = fold(journal(), mode)
    pos = {p["ticker"]: -float(p.get("position_fp") or 0) for p in client.positions()}
    try:
        series = allowed_series()
    except Exception:
        series = set()
    for t in sorted(set(pos) | {t for t, o in state.items() if held(o) > 0 and not o.get("settled")}):
        mine = held(state[t]) if t in state and not state[t].get("settled") else 0.0
        if abs(pos.get(t, 0.0) - mine) > 1e-6 and (t in state or t.split("-")[0] in series):
            if pos.get(t, 0.0) == 0 and (client.market(t) or {}).get("status") not in ("active", "open"):
                continue  # exchange already settled it; our settle event follows once the result is published
            if t in state and state[t].get("settled") and abs(pos.get(t, 0.0) - held(state[t])) < 1e-6:
                continue  # mirror case (11-26/27): we booked the settle from the published result; Kalshi still shows
                #           the same position until it clears it. Any OTHER exchange count after a settle still alerts.
            append({"event": "reconcile_mismatch", "ts": now.isoformat(timespec="seconds"), "mode": mode,
                    "market": t, "journal_no": mine, "exchange_no": pos.get(t, 0.0)})
            notify_once(f"mismatch:{mode}:{t}:{pos.get(t, 0.0)}", f"Kalshi {mode} position mismatch",
                        f"{t}: journal {mine:g} NO vs exchange {pos.get(t, 0.0):g}", cfg)


# ----------------------------------------------------------------------------------------------- run
def decide(p: dict, client: Kalshi, mode: str, cfg: dict, now: datetime, spent: float, opn: float,
           bal: float | None, x1_used: tuple[float, float] = (0.0, 0.0)) -> dict:
    t = p["ticker"]
    style = exec_style(cfg, t)
    if style == "maker":
        ask, depth, no_bid = client.book(t)
    else:
        ask, depth = client.book_no(t)
    rec = {"event": "order", "ts": now.isoformat(timespec="seconds"), "mode": mode, "env": cfg["env"], "ticker": t,
           "cohort": p.get("cohort", "orig7"), "pick_logged_at": p["logged_at"], "pick_no_price": p["no_price"],
           "no_ask_now": ask, "depth_1c": depth, "count": 0, "filled": 0, "cost": 0.0}
    if tag_style(cfg):
        rec["exec_style"] = style
    if rec["cohort"] not in cfg.get("cohorts", {"orig7", "x1"}):
        return {**rec, "status": "skip_cohort_paused"}
    if rec["cohort"] in cfg.get("killed_cohorts", ()):
        return {**rec, "status": "skip_cohort_killed"}
    m = client.market(t)
    cap = caps(cfg, rec["cohort"])
    close = (m or {}).get("close_time")
    mins_left = (datetime.fromisoformat(close.replace("Z", "+00:00")) - now).total_seconds() / 60 if close else -1
    budget = min(cap["order"], cap["day"] - spent, cap["open"] - opn, bal if bal is not None else math.inf)
    x1_room = math.inf
    if rec["cohort"] == "x1" and cfg.get("x1_day") is not None:
        x1_room = cfg["x1_day"] - max(x1_used)
        budget = min(budget, x1_room)
    if not rule_holds(p):
        return {**rec, "status": "skip_rule_check"}
    if m is None:
        return {**rec, "status": "skip_no_market"}
    if int(m.get("exchange_index") or WEATHER_SHARD) != WEATHER_SHARD:  # bal below is shard-0 cash only
        return {**rec, "status": "skip_shard", "exchange_index": m.get("exchange_index")}
    if m.get("status") not in ("active", "open") or mins_left < cfg["min_close_min"]:
        return {**rec, "status": "skip_market_not_active", "market_status": m.get("status")}
    if not (ask == ask) or ask >= 1:
        return {**rec, "status": "skip_no_book"}
    if ask > p["no_price"] + cfg["max_up"] + 1e-9:
        return {**rec, "status": "skip_price_moved"}
    if ask < p["no_price"] - cfg["max_down"] - 1e-9:
        return {**rec, "status": "skip_price_dropped"}
    if x1_room < ask - 0.05:  # not even one contract at the lowest maker price (ask - 5c) -> X1 sub-cap is full
        return {**rec, "status": "skip_x1_cap"}
    if style == "maker":
        return _decide_maker(rec, client, mode, cfg, ask, no_bid, budget, bal, close)
    n = int(max(budget, 0) // (ask + FEE * ask * (1 - ask)) if budget > 0 else 0)
    ct = contract_cap(cfg, rec["cohort"], ask, mode)
    n = min(n, int(depth * cfg["depth_frac"]), ct)
    if mode != "dry" and ct != cfg["max_contracts"]:
        rec = {**rec, "ct_cap": ct}
    while n > 0 and order_cost(n, ask) > budget + 1e-9:
        n -= 1
    if n < 1:
        why = ("skip_balance" if bal is not None and bal < ask else
               "skip_cap" if budget < order_cost(1, ask) else "skip_depth")
        return {**rec, "status": why}
    return {**rec, "count": n, "limit_yes_price": f"{1 - ask:.4f}", "cost": order_cost(n, ask),
            "fee": fee_total(n, ask), "avg_no": ask, "status": "go"}


def _decide_maker(rec: dict, client: Kalshi, mode: str, cfg: dict, ask: float, no_bid: float, budget: float,
                  bal: float | None, close: str) -> dict:
    """M3 (sealed-passed kalshi_temp_maker_M3_N10): resting post-only NO bid at the NO mid floored to the cent, capped
    at NO ask - 1c. No depth cap (a resting bid takes no liquidity). Fee 0 only on a 'quadratic' series."""
    q = round(min(cent_floor((ask + no_bid) / 2), ask - 0.01), 4)
    fee_type = client.series_fee_type(rec["ticker"].split("-")[0])
    unit_fee = maker_fee_unit(q, fee_type)
    rec = {**rec, "maker_q": q, "no_bid_now": no_bid, "fee_type": fee_type}
    if q <= 0 or ask - q > MAKER_MAX_IMPROVE + 1e-9:
        return {**rec, "status": "skip_maker_spread"}
    cost = lambda n: round(n * q + (fee_total(n, q) if unit_fee else 0.0), 4)  # noqa: E731
    n = int(max(budget, 0) // (q + unit_fee) if budget > 0 else 0)
    ct = contract_cap(cfg, rec["cohort"], q, mode)
    n = min(n, ct)
    if mode != "dry" and ct != cfg["max_contracts"]:
        rec = {**rec, "ct_cap": ct}
    while n > 0 and cost(n) > budget + 1e-9:
        n -= 1
    if n < 1:
        return {**rec, "status": "skip_balance" if bal is not None and bal < q else "skip_cap"}
    exp = int(datetime.fromisoformat(close.replace("Z", "+00:00")).timestamp())
    return {**rec, "count": n, "limit_yes_price": f"{1 - q:.4f}", "cost": cost(n),
            "fee": round(cost(n) - n * q, 4), "avg_no": q, "expiration_time": exp, "status": "go"}


def _is_post_only_reject(out) -> bool:
    """Kalshi does not document the post_only-would-cross error; match on its wording."""
    txt = json.dumps(out).lower() if out else ""
    return any(w in txt for w in ("post_only", "post only", "postonly", "post-only", "cross"))


def maker_reserve(rec: dict, u: dict) -> dict:
    """While a maker order rests, commit filled cost + the unfilled remainder at q (+ assumed fee) to the caps."""
    if u.get("status") != "resting":
        return u
    q, n = float(rec["maker_q"]), float(rec.get("count") or 0)
    rest = max(n - float(u.get("filled") or 0), 0.0)
    return {**u, "cost": round(float(u.get("cost") or 0) + rest * (q + maker_fee_unit(q, rec.get("fee_type"))), 4)}


def place(client: Kalshi, rec: dict, cfg: dict, now: datetime) -> dict:
    """Write-ahead, then POST once (the client retries with the SAME client_order_id), then record the outcome."""
    mode, t = rec["mode"], rec["ticker"]
    coid = client_order_id(mode, t)
    existing = client.orders(ticker=t)
    pos = [p for p in client.positions(ticker=t) if float(p.get("position_fp") or 0) != 0]
    if existing or pos:
        return append({**rec, "status": "skip_exchange_has_order", "count": 0, "cost": 0.0, "filled": 0,
                       "client_order_id": coid if any(x.get("client_order_id") == coid for x in existing) else None})
    append({**rec, "status": "submitting", "client_order_id": coid, "filled": 0})
    maker = rec.get("exec_style") == "maker"
    if maker:  # sell YES at 1-q == buy NO at q; rests until the market's close (GTC + expiration_time = GTT)
        body = {"ticker": t, "side": "ask", "count": str(rec["count"]), "price": rec["limit_yes_price"],
                "time_in_force": "good_till_canceled", "post_only": True, "expiration_time": rec["expiration_time"],
                "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}
    else:
        body = {"ticker": t, "side": "ask", "count": str(rec["count"]), "price": rec["limit_yes_price"],
                "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": coid}
    ref = rec["maker_q"] if maker else rec["no_ask_now"]  # the price a correct fill must be near
    upd = {"event": "update", "ts": now.isoformat(timespec="seconds"), "mode": mode, "ticker": t}
    import order_guard as og               # lazy: order_guard imports this module
    og.kx = sys.modules[__name__]          # same module object when this file runs as __main__
    try:
        code, out = og.guarded_create(client, body, strategy="weather", cap=float(cfg["max_order"]),
                                      on_trip=lambda why: write_kill(why, cfg), now=now)
    except TransportError as ex:
        append({**upd, "status": "unknown", "error": str(ex)[:300]})
        notify(f"Kalshi {mode} order UNKNOWN", f"{t}: {ex} -- will reconcile next run", cfg)
        return {**rec, "status": "unknown"}
    if code == 409:  # same client_order_id already accepted: read the truth back
        match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == coid]
        u = fills_from_order(match[0]) if match else {"status": "unknown"}
        if maker and match:
            u = maker_reserve(rec, u)
        elif maker and _is_post_only_reject(out):
            u = {"status": "rejected_post_only", "cost": 0.0, "filled": 0}
        append({**upd, "http": 409, "source": "409", **u})
        return {**rec, **u}
    if maker and code not in (200, 201) and _is_post_only_reject(out):
        append({**upd, "status": "rejected_post_only", "http": code, "response": out, "cost": 0.0})
        notify(f"Kalshi {mode} maker bid not placed", f"{t}: post_only would cross (HTTP {code}); no retry", cfg)
        return {**rec, "status": "rejected_post_only", "cost": 0.0}
    if code not in (200, 201):
        append({**upd, "status": "rejected", "http": code, "response": out, "cost": 0.0})
        notify(f"Kalshi {mode} order REJECTED", f"{t}: HTTP {code} {str(out)[:120]}", cfg)
        return {**rec, "status": "rejected"}
    filled = float(out.get("fill_count") or 0)
    u = {"http": code, "order_id": out.get("order_id"), "filled": filled, "response": out}
    if filled > 0:
        avg_yes = float(out.get("average_fill_price") or 0)
        avg_no = round(1 - avg_yes, 4)
        fee = round(float(out.get("average_fee_paid") or 0) * filled, 4)
        u.update(status="filled" if filled >= rec["count"] else "partial", avg_no=avg_no, fee=fee,
                 cost=round(filled * avg_no + fee, 4))
        if abs(avg_no - ref) > 0.02:
            # Cross-check with the Order object (GET fields verified on prod 2026-10-22) before killing: the create
            # response's average_fill_price convention is the one field not yet seen on a real fill.
            try:
                match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == coid]
            except Exception:  # noqa: BLE001 -- any failure -> keep the conservative path below
                match = []
            g = fills_from_order(match[0]) if match else {}
            if g.get("filled") and g.get("avg_no") is not None and abs(g["avg_no"] - ref) <= 0.02:
                u.update(status="filled" if g["filled"] >= rec["count"] else "partial", filled=g["filled"],
                         avg_no=g["avg_no"], fee=g["fee"], cost=g["cost"], source="get_order",
                         create_avg_no=avg_no)
                avg_no = g["avg_no"]
        if abs(avg_no - ref) > 0.02:  # convention / price check: stop before a second order
            u["cost"] = round(filled * 1.0 + fee, 4)  # worst case until a human looks
            append({**upd, **u})
            what = f"maker bid {ref}" if maker else f"ask {rec['no_ask_now']}"
            write_kill(f"fill price convention check failed on {t}: NO cost {avg_no} vs {what}", cfg)
            return {**rec, **u}
        if maker and filled < rec["count"] and float(out.get("remaining_count") or 0) > 0:
            u = maker_reserve(rec, {**u, "status": "resting"})
        notify(f"Kalshi {mode} {'FILL' if u['status'] == 'filled' else 'PARTIAL fill'}",
               f"{t}: {filled:g}/{rec['count']} NO @ {avg_no} (${u['cost']:.2f})", cfg)
    elif maker and float(out.get("remaining_count") or 0) > 0:
        u.update(status="resting", cost=rec["cost"])  # committed n*q (+fee) for the caps while it rests
        notify(f"Kalshi {mode} maker bid resting", f"{t}: {rec['count']} NO @ {rec['maker_q']} until close", cfg)
    elif maker:  # accepted, nothing filled, nothing left resting: the post_only bid was cancelled at entry
        u.update(status="rejected_post_only", cost=0.0)
    else:
        u.update(status="no_fill", cost=0.0)
    append({**upd, **u})
    return {**rec, **u}


def cmd_run(argv: list[str] | None = None, client: Kalshi | None = None, now: datetime | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
    now = now or datetime.now(timezone.utc)
    e = env()
    cfg = config(e)
    mode = resolve_mode(e, "--live" in argv)
    k = killed()
    if k:
        notify_once(f"kill:{k}:{k.stat().st_mtime}", "Kalshi exec KILLED", f"{k} present -> nothing done", cfg)
        print(f"KILL file present ({k}) -> nothing done")
        return 0
    if "--live" in argv and mode == "dry":
        print(f"--live ignored: {ENV} does not have EXEC_MODE=live -> dry run")
    client = client or Kalshi(e, cfg["env"])
    picks = load_picks()
    settle(journal(), picks, now, client)
    bal = None
    if mode != "dry":
        if cfg["bankroll"] is None:
            append({"event": "error", "ts": now.isoformat(timespec="seconds"), "mode": mode,
                    "error": "KALSHI_BANKROLL_USD not set; demo/live refuse to trade without it"})
            notify(f"Kalshi {mode} refused", "set KALSHI_BANKROLL_USD in .env", cfg)
            print("KALSHI_BANKROLL_USD not set -> refusing to trade")
            return 2
        if not client.has_key():
            print("no KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH in .env -> refusing to trade")
            return 2
        reconcile(client, mode, cfg, now)
        if killed():
            print("KILL written during reconcile -> stop")
            return 0
    state = fold(journal(), mode)
    if "x1" in cfg["cohorts"]:
        since = cfg.get("x1_kill_since")
        if x1_killed(journal(), mode, since):
            cfg["killed_cohorts"] = {"x1"}
        else:
            why = x1_kill_verdict(state, since)
            if why:
                append({"event": "cohort_kill", "ts": now.isoformat(timespec="seconds"), "mode": mode, "cohort": "x1",
                        "since": since, "reason": why})
                notify(f"Kalshi {mode} X1 cohort KILLED", f"{why}; orig7 continues", cfg)
                cfg["killed_cohorts"] = {"x1"}
    b = cfg["bankroll"]
    if b is not None and mode != "dry":
        week, day = net_pnl(state, now, 24 * 7), net_pnl(state, now, 24)
        if week <= -WEEK_LOSS_FRAC * b and mode == "live":
            write_kill(f"weekly loss stop: 7-day net ${week:.2f} <= -{WEEK_LOSS_FRAC:.0%} of ${b:,.0f}", cfg)
            return 0
        if day <= -DAY_LOSS_FRAC * b or week <= -WEEK_LOSS_FRAC * b:
            append({"event": "loss_stop", "ts": now.isoformat(timespec="seconds"), "mode": mode, "day": day, "week": week})
            notify_once(f"lossstop:{mode}:{now.date()}", f"Kalshi {mode} loss stop",
                        f"24h net ${day:.2f} / 7d ${week:.2f}: no new orders", cfg)
            print(f"loss stop: 24h net ${day:.2f}, 7d net ${week:.2f} -> no new orders")
            return 0
    try:
        series = allowed_series()
    except Exception as ex:
        print(f"cannot load strategy universe ({ex}) -> nothing done")
        return 1
    fresh = [p for p in picks if p["ticker"] not in state and p.get("result") is None
             and p.get("city") in series and p["ticker"].startswith(p["city"] + "-")
             and now - datetime.fromisoformat(p["logged_at"]) <= timedelta(minutes=cfg["max_age_min"])]
    if not fresh:
        print(f"[{mode}] no fresh unexecuted picks")
        return 0
    if mode != "dry":
        st = client.exchange_status()
        if not st.get("trading_active", False):
            print("exchange trading not active -> retry next run")
            return 0
        bal = client.balance(shard=WEATHER_SHARD)
    spent, opn = spend_and_open(state, now)
    x1s, x1o = spend_and_open({t: o for t, o in state.items() if o.get("cohort") == "x1"}, now)
    for p in fresh:
        rec = decide(p, client, mode, cfg, now, spent, opn, bal, (x1s, x1o))
        if rec["status"] != "go":
            append(rec)
            if rec["status"] == "skip_balance":  # a pick lost to shard cash is never retried: say so once a day
                notify_once(f"skipbal:{mode}:{now.date()}", f"Kalshi {mode} pick skipped: shard-{WEATHER_SHARD} cash",
                            f"{p['ticker']}: shard {WEATHER_SHARD} cash ${bal:.2f}; fund shard {WEATHER_SHARD}", cfg)
        elif mode == "dry":
            rec = append({**rec, "status": "dry", "filled": rec["count"]})
        else:
            rec = place(client, rec, cfg, now)
            if killed():
                print("KILL written after a fill -> stop")
                break
        c = committed(rec)
        spent, opn = spent + c, opn + c
        if rec.get("cohort") == "x1":
            x1s, x1o = x1s + c, x1o + c
        if bal is not None:
            bal -= c
        print(f"{mode:4s} {p['ticker']:28s} NO ask {rec['no_ask_now']} (pick {p['no_price']}) depth "
              f"{rec['depth_1c']:.0f} -> {rec['status']} {rec.get('filled', 0):g}/{rec['count']} contracts ${c:.2f}")
    return 0


# ----------------------------------------------------------------------------------------------- ops commands
def cmd_check() -> int:
    e = env()
    cfg = config(e)
    c = Kalshi(e, cfg["env"])
    if not c.has_key():
        print(f"no key configured: add KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH to {ENV} yourself")
        return 1
    code, body = c.request("GET", "/portfolio/balance")
    print(f"[{cfg['env']}] GET /portfolio/balance -> HTTP {code}; "
          + (f"balance ${float(body.get('balance_dollars', 0)):,.2f}" if code == 200 else str(body)[:200]))
    return 0 if code == 200 else 1


def _perm_ok(p: Path) -> tuple[bool, str]:
    s = p.stat()
    ok = not (s.st_mode & 0o077) and s.st_uid == os.getuid()
    return ok, f"mode {stat.filemode(s.st_mode)}, owner {'you' if s.st_uid == os.getuid() else s.st_uid}"


def cmd_preflight(client: Kalshi | None = None) -> int:
    rows = []
    add = lambda lvl, what, msg: rows.append((lvl, what, msg))  # noqa: E731
    e = env()
    try:
        cfg = config(e)
    except ValueError as ex:
        print(f"FAIL config: {ex}")
        return 1
    mode = resolve_mode(e, True)
    add("INFO", "files", f"root {ROOT}; .env {'present' if ENV.exists() else 'MISSING'}")
    add("INFO", "mode", f"EXEC_MODE={e.get('EXEC_MODE', '(unset)')} EXEC_ENV={cfg['env']} -> the launchd run (--live) "
        f"would be: {mode.upper()}")
    if ENV.exists():
        ok, d = _perm_ok(ENV)
        add("PASS" if ok else "WARN", ".env perms", d + ("" if ok else " -> chmod 600 .env"))
    try:
        import cryptography  # noqa: F401
        add("PASS", "cryptography", "importable")
    except ImportError:
        add("FAIL", "cryptography", "missing -> pip install cryptography")
    has_id, kp = bool(e.get("KALSHI_KEY_ID")), e.get("KALSHI_PRIVATE_KEY_PATH")
    need_key = mode != "dry"
    add("PASS" if has_id else ("FAIL" if need_key else "WARN"), "KALSHI_KEY_ID", "set" if has_id else "not set")
    keyok = False
    if kp:
        p = Path(kp).expanduser()
        if not p.exists():
            add("FAIL", "private key file", "path in .env does not exist")
        else:
            ok, d = _perm_ok(p)
            keyok = ok
            add("PASS" if ok else "FAIL", "private key file", d + ("" if ok else " -> chmod 600 <key file>"))
            if str(p.resolve()).startswith(str(Path.home() / "Desktop")):
                add("FAIL", "private key location", "under ~/Desktop: launchd cannot read it (TCC); move it")
    else:
        add("FAIL" if need_key else "WARN", "KALSHI_PRIVATE_KEY_PATH", "not set")
    b = cfg["bankroll"]
    if b is None:
        add("FAIL" if need_key else "WARN", "KALSHI_BANKROLL_USD", "not set (required for demo/live)")
    else:
        c7, cx = caps(cfg, "orig7"), caps(cfg, "x1")
        add("PASS", "caps", f"B ${b:,.0f}: per order orig7 ${c7['order']:.2f} / x1 ${cx['order']:.2f}, day ${c7['day']:.2f},"
            f" open ${c7['open']:.2f}, {cfg['max_contracts']} contract(s)/order; loss stop 24h -${DAY_LOSS_FRAC * b:.2f},"
            f" 7d -${WEEK_LOSS_FRAC * b:.2f}")
        add("PASS", "cohorts / contract caps",
            f"cohorts {','.join(sorted(cfg['cohorts']))}; x1 max ct {cfg['x1_max_contracts'] or cfg['max_contracts']}"
            f" (kill since {cfg['x1_kill_since'] or 'all'}), x1 day/open ${cfg['x1_day'] or 'shared'}; price bands "
            + (", ".join(f">={lo:.2f}:{n}" for lo, n in cfg["price_band_caps"]) or "off"))
    k = killed()
    add("WARN" if k else "PASS", "kill switch", f"{k} PRESENT (nothing will trade)" if k else "no KILL file")
    client = client or Kalshi(e, cfg["env"])
    try:
        st = client.exchange_status()
        add("PASS", "API reachable", f"{client.host['public']} exchange_active={st.get('exchange_active')} "
            f"trading_active={st.get('trading_active')}")
    except Exception as ex:
        add("FAIL", "API reachable", f"{type(ex).__name__}: {str(ex)[:120]}")
    if client.has_key() and (keyok or not kp):
        try:
            add("PASS", f"auth + balance ({cfg['env']})", f"${client.balance():,.2f} available")
        except Exception as ex:
            add("FAIL", f"auth + balance ({cfg['env']})", f"{str(ex)[:160]} (demo keys only work on demo, prod on prod)")
    if cfg["env"] == "demo":
        try:
            code, out = client.request("GET", "/markets", params={"series_ticker": "KXHIGHNY", "status": "open", "limit": 5}, auth=False)
            n = len(out.get("markets", [])) if code == 200 else 0
            add("PASS" if n else "WARN", "demo weather markets", f"{n} open KXHIGHNY markets on demo"
                + ("" if n else " -> picks will log skip_no_market; use `smoke` to test order plumbing"))
        except Exception as ex:
            add("WARN", "demo weather markets", str(ex)[:120])
    try:
        r = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{AGENT_LABEL}"], capture_output=True, text=True, timeout=10)
        add("PASS" if r.returncode == 0 else "FAIL", "launchd agent", f"{AGENT_LABEL} {'loaded' if r.returncode == 0 else 'NOT loaded'}")
    except Exception as ex:
        add("WARN", "launchd agent", str(ex)[:120])
    if RUNLOG.exists():
        last = [l for l in RUNLOG.read_text().splitlines() if l.startswith("=== ")]
        if last:
            t = datetime.strptime(last[-1][4:23], "%Y-%m-%d %H:%M:%S")
            age = (datetime.now() - t).total_seconds() / 60
            add("PASS" if age <= 75 else "WARN", "last tick", f"{last[-1][4:]} ({age:.0f} min ago; hourly at :03)")
    else:
        add("WARN", "last tick", f"{RUNLOG} not found")
    here, dep = Path(__file__).resolve(), DEPLOY / "scripts" / "kalshi_exec.py"
    if here != dep.resolve() and dep.exists():
        same = here.read_bytes() == dep.read_bytes()
        add("PASS" if same else "WARN", "deployed copy", "matches this file" if same else
            "differs from this file -> re-run scripts/install_kalshi_paper_agent.sh")
    for lvl, what, msg in rows:
        print(f"{lvl:4s}  {what:26s} {msg}")
    fails = sum(r[0] == "FAIL" for r in rows)
    print(f"\npreflight: {fails} FAIL, {sum(r[0] == 'WARN' for r in rows)} WARN")
    return 1 if fails else 0


def style_lines(st: dict, picks: dict) -> list[str]:
    """Per exec style: orders placed, filled, fill rate, filled loss rate vs the paper loss rate on the same picks.
    Empty when no row carries exec_style (pure default-taker journal), so the old report is unchanged."""
    if not any("exec_style" in o for o in st.values()):
        return []
    out = []
    for style in ("taker", "maker"):
        placed = [o for o in st.values() if o.get("exec_style", "taker") == style and float(o.get("count") or 0) > 0
                  and not o.get("status", "").startswith("skip")]
        if not placed:
            continue
        filled = [o for o in placed if held(o) > 0]
        done = [o for o in filled if o.get("settled")]
        paper = [picks.get(o["ticker"], {}).get("result") for o in placed]
        paper = [r for r in paper if r in ("yes", "no")]
        resting = sum(o.get("status") == "resting" for o in placed)
        loss = f"{sum(o['result'] == 'yes' for o in done) / len(done):.1%} of {len(done)}" if done else "n/a"
        ploss = f"{sum(r == 'yes' for r in paper) / len(paper):.1%} of {len(paper)}" if paper else "n/a"
        out.append(f"{style}: {len(placed)} placed, {len(filled)} filled ({len(filled) / len(placed):.0%}), "
                   f"{resting} resting; filled loss rate {loss} vs paper loss rate {ploss}; "
                   f"P&L ${sum(o['pnl'] for o in done):+,.2f}")
    return out


def cmd_status(client: Kalshi | None = None, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    e = env()
    cfg = config(e)
    mode_now = resolve_mode(e, True)
    hist = journal()
    picks = {p["ticker"]: p for p in load_picks()}
    k = killed()
    print(f"launchd mode: {mode_now.upper()} (EXEC_ENV={cfg['env']}); KILL: {k or 'none'}; bankroll: "
          f"{'$%s' % format(cfg['bankroll'], ',.0f') if cfg['bankroll'] else 'unset'}; exec style: "
          f"{'A/B (sha1 parity: even maker, odd taker)' if cfg['ab'] else cfg['exec_style']}")
    for mode in ("dry", "demo", "live"):
        st = fold(hist, mode)
        if not st:
            continue
        c = caps(cfg, "orig7")
        spent, opn = spend_and_open(st, now)
        rows = [o for o in st.values() if held(o) > 0 and committed(o) > 0]
        done = [o for o in rows if o.get("settled")]
        pnl = sum(o["pnl"] for o in done)
        paper = 0.0  # the same contracts at the pick's own NO price + fee: what paper assumed
        for o in done:
            q = float(o.get("pick_no_price", o.get("avg_no", 0)))
            cost = held(o) * q + fee_total(held(o), q)
            paper += held(o) - cost if o["result"] == "no" else -cost
        skips = {}
        for o in st.values():
            if o.get("status", "").startswith("skip"):
                skips[o["status"]] = skips.get(o["status"], 0) + 1
        print(f"\n[{mode}] {len(st)} decisions, {len(rows)} with contracts, {len(done)} settled "
              f"({sum(o['result'] == 'no' for o in done)} wins)  P&L ${pnl:+,.2f}  vs paper-at-pick-price ${paper:+,.2f}")
        print(f"  today ${spent:,.2f} / cap ${c['day']:,.2f}   open ${opn:,.2f} / cap ${c['open']:,.2f}   "
              f"24h net ${net_pnl(st, now, 24):+,.2f}  7d net ${net_pnl(st, now, 168):+,.2f}")
        if skips:
            print("  skips: " + ", ".join(f"{s} {n}" for s, n in sorted(skips.items())))
        for line in style_lines(st, picks):
            print("  " + line)
        for o in sorted((o for o in rows if not o.get("settled")), key=lambda o: o["ts"]):
            print(f"  open {o['ticker']:28s} {held(o):g} NO @ {o.get('avg_no')} ${committed(o):.2f} "
                  f"({o.get('status')}; paper result {picks.get(o['ticker'], {}).get('result')})")
    client = client or Kalshi(e, cfg["env"])
    if client.has_key():
        try:
            pos = client.positions()
            print(f"\nexchange ({cfg['env']}): balance ${client.balance():,.2f}; {len(pos)} open positions")
            for p in pos:
                print(f"  {p['ticker']:28s} position {p.get('position_fp')} exposure ${p.get('market_exposure_dollars')}")
        except Exception as ex:
            print(f"\nexchange ({cfg['env']}): {str(ex)[:160]}")
    errs = [r for r in hist if r.get("event") in ("error", "kill", "loss_stop", "reconcile_mismatch")][-5:]
    for r in errs:
        print(f"recent {r['event']}: {r['ts']} {r.get('error') or r.get('reason') or r.get('market') or ''}")
    return 0


def cmd_kill(argv: list[str] | None = None, client: Kalshi | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
    e = env()
    cfg = config(e)
    write_kill(" ".join(argv) or "manual kill", cfg)
    client = client or Kalshi(e, cfg["env"])
    if client.has_key():
        ours = {o.get("client_order_id") for m in ("demo", "live") for o in fold(journal(), m).values()}
        for x in client.orders(status="resting"):
            if x.get("client_order_id") in ours:
                code, _ = client.cancel(x["order_id"], x.get("ticker"))
                print(f"cancelled {x.get('ticker')} -> HTTP {code}")
    print(f"KILL written: {killed()}; remove that file to resume")
    return 0


def cmd_smoke(client: Kalshi | None = None) -> int:
    """DEMO ONLY. Rests 1 contract that cannot fill (post_only buy-NO at 1c), re-sends the same client_order_id
    (expects 409), reads it back, cancels it. Exercises signing, create, dedupe, get, cancel on mock funds."""
    e = env()
    cfg = config(e)
    if cfg["env"] != "demo":
        print("smoke refuses to run unless EXEC_ENV=demo in .env")
        return 1
    client = client or Kalshi(e, "demo")
    if not client.has_key():
        print("no demo key in .env")
        return 1
    code, out = client.request("GET", "/markets", params={"status": "open", "limit": 50}, auth=False)
    ms = [m for m in out.get("markets", []) if float(m.get("yes_bid_dollars") or 0) < 0.98] if code == 200 else []
    if not ms:
        print(f"no open demo market found (HTTP {code})")
        return 1
    t = ms[0]["ticker"]
    coid = str(uuid.uuid4())
    body = {"ticker": t, "side": "ask", "count": "1", "price": "0.9900", "time_in_force": "good_till_canceled",
            "post_only": True, "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}
    steps, ok = [], True
    c1, o1 = client.create_order(body)
    steps.append(("create", c1, o1.get("order_id"), c1 == 201))
    c2, o2 = client.create_order(body)
    steps.append(("duplicate client_order_id", c2, o2.get("error") or o2.get("code"), c2 == 409))
    oid = o1.get("order_id")
    if oid:
        c3, o3 = client.request("GET", f"/portfolio/orders/{oid}")
        o = o3.get("order", {})
        steps.append(("get order", c3, f"{o.get('status')} outcome_side={o.get('outcome_side')} no_price={o.get('no_price_dollars')}",
                      c3 == 200 and o.get("outcome_side") == "no"))
        c4, o4 = client.cancel(oid, t)
        steps.append(("cancel", c4, f"reduced_by={o4.get('reduced_by')}", c4 == 200))
    for name, code, info, good in steps:
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:28s} HTTP {code}  {info}")
    append({"event": "smoke", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "demo",
            "market": t, "steps": [(s[0], s[1], s[3]) for s in steps]})
    print(f"\nsmoke on demo market {t}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    fns = {"run": cmd_run, "check": cmd_check, "preflight": cmd_preflight, "status": cmd_status,
           "report": cmd_status, "kill": cmd_kill, "smoke": cmd_smoke}
    if cmd not in fns:
        print(__doc__)
        return 1
    try:
        return fns[cmd]()
    except Exception as ex:
        msg = f"{type(ex).__name__}: {str(ex)[:300]}"
        try:
            append({"event": "error", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "where": cmd,
                    "error": msg})
        except Exception:
            pass
        try:
            cfg = config(env())
        except Exception:
            cfg = {"notify": True}
        try:  # once per day per distinct error, so a broken key does not alert every hour
            notify_once(f"err:{cmd}:{msg[:80]}:{datetime.now(timezone.utc).date()}", f"Kalshi exec ERROR ({cmd})", msg, cfg)
        except Exception:
            notify(f"Kalshi exec ERROR ({cmd})", msg, cfg)
        print(f"ERROR {msg}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
