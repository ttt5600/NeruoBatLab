"""Live scorecards: one card per strategy, judged against the kill rule written BEFORE its first bet.

    cd sports && ../.venv_quant/bin/python scripts/scorecards.py            # offline: logs + cached exchange fills
    cd sports && ../.venv_quant/bin/python scripts/scorecards.py --kalshi   # also refresh exchange fills (GET only)
    options: --out PATH (default notes/SCORECARDS.md), --no-subproc (skip the loggers' own report commands)

READ-ONLY. Reads the deployed logs (~/.claude/quant-kalshi/data, ~/.claude/quant-sports/data); never writes there, never
places or cancels. --kalshi makes authenticated GETs only (/portfolio/fills, /portfolio/positions, /markets/{t}) through
the deployed kalshi_exec client and caches them in results/scorecards/kalshi_cache.json (later offline runs reuse it).

Kill rules are read VERBATIM from their source lines at run time (ACT_LIST.md, FUND_BOOK.md, notes, edgelab registry,
logger code), so the card shows exactly what was written and where. Status, mechanical (notes/scorecards_agent.md; NOT notes/scorecards.md -- macOS is case-insensitive, that IS the output):
  KILLED-by-rule  the written kill condition is met on the counted sample
  near-kill       one more loss trips a loss-count rule, or ROI CI upper < 0 / mean CLV < 0 before the judge n
  too-early       < 10 settled (or nothing placed) and no kill condition met
  alive           otherwise
ROI = sum P&L / sum $ risked; 95% CI = bootstrap over bets (10,000 resamples, seed 2026).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                    # sports/
NOTES = ROOT / "notes"
KD = Path.home() / ".claude" / "quant-kalshi" / "data"
SD = Path.home() / ".claude" / "quant-sports" / "data"
QS = Path.home() / ".claude" / "quant-sports"
PY = ROOT.parent / ".venv_quant" / "bin" / "python"
CACHE = ROOT / "results" / "scorecards" / "kalshi_cache.json"
LEAD = KD / "lead_exec.jsonl"
STATUSES = ("KILLED-by-rule", "near-kill", "alive", "too-early")
RNG_SEED, NBOOT, MIN_N = 2026, 10_000, 10


# ------------------------------------------------------------------------------------------------ helpers
def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT.parent))
    except ValueError:
        return str(p).replace(str(Path.home()), "~")


def rule_line(path: Path, needle: str, cell: int | None = None) -> tuple[str, str]:
    """(source 'file:line', verbatim text) of the first line containing `needle`; cell = markdown table column."""
    if not path.exists():
        return f"{rel(path)} (MISSING)", "(rule source file missing)"
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if needle in line:
            txt = line.strip()
            if cell is not None:
                cells = [c.strip() for c in txt.strip("|").split("|")]
                txt = cells[cell] if len(cells) > abs(cell) else txt
            return f"{rel(path)}:{i}", txt
    return f"{rel(path)} (needle not found: {needle!r})", "(rule text not found -- source changed?)"


def boot_roi(pnl, risk) -> tuple[float, float, float]:
    pnl, risk = np.asarray(pnl, float), np.asarray(risk, float)
    if len(pnl) == 0 or risk.sum() <= 0:
        return math.nan, math.nan, math.nan
    roi = pnl.sum() / risk.sum()
    if len(pnl) < 2:
        return roi, math.nan, math.nan
    idx = np.random.default_rng(RNG_SEED).integers(0, len(pnl), (NBOOT, len(pnl)))
    b = pnl[idx].sum(1) / np.maximum(risk[idx].sum(1), 1e-12)
    return roi, float(np.quantile(b, .025)), float(np.quantile(b, .975))


def mean_ci(x) -> tuple[float, float, float, int]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan, 0
    if len(x) < 2:
        return float(x.mean()), math.nan, math.nan, 1
    se = x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se), len(x)


def f_roi(pnl, risk) -> str:
    r, lo, hi = boot_roi(pnl, risk)
    if not np.isfinite(r):
        return "ROI n/a"
    ci = f" [{lo:+.1%}, {hi:+.1%}]" if np.isfinite(lo) else " [CI n/a, n<2]"
    tag = "" if not np.isfinite(lo) else (" (not distinguishable from 0)" if lo <= 0 <= hi else "")
    return f"ROI {r:+.1%}{ci}{tag}"


def f_clv(x, unit: str = "") -> str:
    m, lo, hi, n = mean_ci(x)
    if n == 0:
        return "CLV not logged"
    ci = f" [{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo) else ""
    return f"CLV mean {m:+.4f}{unit}{ci} (n {n})"


def wl(won: list) -> str:
    w = sum(1 for v in won if v is True)
    l_ = sum(1 for v in won if v is False)
    p = sum(1 for v in won if v is None)
    return f"{w}-{l_}" + (f"-{p}" if p else "")


def kalshi_fee(p: float, n: float = 1.0) -> float:
    """Kalshi taker fee: ceil to the cent of 0.07 * n * p * (1 - p)."""
    return math.ceil(round(0.07 * n * p * (1 - p) * 100, 6)) / 100


class Card:
    def __init__(self, cid: str, title: str):
        self.cid, self.title, self.status, self.why = cid, title, "too-early", ""
        self.rules: list[tuple[str, str]] = []
        self.lines: list[str] = []
        self.warn: list[str] = []
        self.headline = ""

    def rule(self, src_txt: tuple[str, str], label: str = "Kill rule"):
        self.rules.append((label, *src_txt))

    def set(self, status: str, why: str):
        assert status in STATUSES
        self.status, self.why = status, why

    def md(self) -> str:
        out = [f"### {self.cid} -- {self.title}", f"**Status: {self.status}** -- {self.why}", ""]
        for label, src, txt in self.rules:
            out.append(f"- {label} (verbatim, `{src}`): \"{txt}\"")
        out += [f"- {x}" for x in self.lines]
        out += [f"- WARNING: {x}" for x in self.warn]
        return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ journal fold
def fold(hist: list[dict], mode: str = "live") -> dict:
    """key (or ticker) -> first order row + its updates/settle merged. Mirrors scripts/lead_exec.py fold()."""
    out = {}
    for r in hist:
        if r.get("mode") != mode or "ticker" not in r:
            continue
        ev, t = r.get("event", "order"), r.get("key") or r["ticker"]
        if ev == "order" and (t not in out or (out[t].get("strategy") == "DUL"
                                               and str(out[t].get("status", "")).startswith("skip"))):
            out[t] = dict(r)
        elif ev in ("update", "settle") and t in out:
            out[t].update({k: v for k, v in r.items() if k not in ("event", "ts", "mode", "ticker")})
            if ev == "settle":
                out[t].update(settled=True, settled_ts=r["ts"])
    return out


def placed(o: dict) -> bool:
    s = str(o.get("status", ""))
    return not s.startswith("skip") and s not in ("no_fill", "rejected", "rejected_post_only", "not_placed") \
        and float(o.get("count") or 0) > 0 or float(o.get("filled") or 0) > 0


def lead_rows(strategy: str, folded: dict) -> dict:
    rows = [o for o in folded.values() if o.get("strategy") == strategy]
    pl = [o for o in rows if placed(o)]
    fi = [o for o in pl if float(o.get("filled") or 0) > 0]
    st = sorted([o for o in fi if o.get("settled")], key=lambda o: o.get("first_fill_ts") or o["ts"])
    return {"decisions": rows, "placed": pl, "filled": fi, "settled": st,
            "skips": sum(str(o.get("status", "")).startswith("skip") for o in rows)}


def lead_summary(d: dict, risk_key: str = "cost") -> str:
    st = d["settled"]
    pnl = [float(o.get("pnl") or 0) for o in st]
    risk = [float(o.get(risk_key) or 0) for o in st]
    s = (f"live: decisions {len(d['decisions'])} (skips {d['skips']}), placed {len(d['placed'])}, filled "
         f"{len(d['filled'])}, settled {len(st)}")
    if st:
        s += (f"; W-L {wl([bool(o.get('won')) for o in st])}, P&L ${sum(pnl):+.2f} on ${sum(risk):.2f} risked, "
              f"{f_roi(pnl, risk)}; {f_clv([o.get('clv') for o in st])}")
    return s


# ------------------------------------------------------------------------------------------------ exchange fills
def kalshi_refresh(tickers: list[str]) -> dict:
    """Authenticated GETs only. Returns {ticker: {fills, market, position}}."""
    sys.path.insert(0, str(Path.home() / ".claude" / "quant-kalshi" / "scripts"))
    import kalshi_exec as kx  # deployed client
    e = kx.env()
    c = kx.Kalshi(e, kx.config(e)["env"])
    out = {}
    for tk in tickers:
        fills = c._pages("/portfolio/fills", "fills", {"ticker": tk})            # GET
        code, pos = c.request("GET", "/portfolio/positions", params={"ticker": tk})
        m = c.market(tk) or {}                                                  # GET (public)
        out[tk] = {"fills": fills,
                   "position": ((pos.get("market_positions") or [None])[0] if code == 200 else None),
                   "market": {k: m.get(k) for k in ("status", "result", "close_time", "last_price_dollars",
                                                    "yes_bid_dollars", "yes_ask_dollars")}}
    return out


def load_cache(refresh: bool, tickers: list[str]) -> tuple[dict, str]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {"fetched_at": None, "tickers": {}}
    if refresh and tickers:
        cache["tickers"].update(kalshi_refresh(tickers))
        cache["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache["tickers"], cache.get("fetched_at") or "never"


def fill_delta(f: dict) -> float:
    """Signed YES contracts. Kalshi's fills report both 'sell YES' exits and 'buy NO' entries as action=sell
    (side=no); verified against position_fp (MIN 25-350=-325, BUF 3-63=-60) and NHL SJ journal P&L (+0.122)."""
    n = float(f.get("count_fp") or f.get("count") or 0)
    return n if f["action"] == "buy" else -n


def ticker_pnl(fills: list[dict], market: dict) -> tuple[float, bool]:
    """(P&L $, settled?) of everything filled on one ticker: cash + net YES x 1{yes} - fees; unsettled -> marked
    at the market's last price."""
    cash = sum(-fill_delta(f) * float(f["yes_price_dollars"]) for f in fills)
    fees = sum(float(f.get("fee_cost") or 0) for f in fills)
    net = sum(fill_delta(f) for f in fills)
    res = (market or {}).get("result")
    if res in ("yes", "no"):
        return cash + net * (res == "yes") - fees, True
    last = float((market or {}).get("last_price_dollars") or 0)
    return cash + net * last - fees, False


# ------------------------------------------------------------------------------------------------ cards
def card_b1(folded) -> Card:
    c = Card("B1", "L39 monthly rain/snow 'unreachable' NO, maker (lead_exec live + forward paper)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "| B1 |", -1))
    c.rule(rule_line(NOTES / "FUND_BOOK.md", "| L39/Q006 |", -2))
    d = lead_rows("L39", folded)
    c.lines.append(lead_summary(d) + f" (source `{rel(LEAD)}`; L39 live window = Nov 24-30)")
    p = jl(KD / "kalshi_wx_monthly_picks.jsonl")
    p.sort(key=lambda r: r.get("ts", ""))
    s = [r for r in p if r.get("result") in ("yes", "no")]
    first = s[:100]
    losses = sum(not r.get("won") for r in first)
    mf = [r for r in s if r.get("maker_filled")]
    c.lines.append(
        f"forward paper (`{rel(KD / 'kalshi_wx_monthly_picks.jsonl')}`): picks {len(p)}, settled {len(s)}, W-L "
        f"{wl([bool(r['won']) for r in s])}; maker tape-filled {len(mf)}/{len(s)}: P&L "
        f"{100 * sum(r['pnl_maker'] for r in mf):+.1f}c/ct, "
        f"{f_roi([r['pnl_maker'] for r in mf], [r['maker_q'] for r in mf])}; taker all: "
        f"{f_roi([r['pnl_taker'] for r in s], [r['taker_cost'] for r in s])}. CLV: not logged for L39. With 0 losses "
        "the bootstrap CI only reflects the spread of entry prices; it cannot see the tail (one loss at ~95c costs "
        "~20 wins), which is why the rule counts losses.")
    n_live = len(d["settled"])
    live_losses = sum(not o.get("won") for o in d["settled"][:100])
    if losses >= 2 or live_losses >= 2:
        c.set("KILLED-by-rule", f"{max(losses, live_losses)} losses within the first 100 picks")
    elif losses == 1 or live_losses == 1:
        c.set("near-kill", f"1 loss in the first {len(first)} paper picks; one more loss kills")
    elif n_live == 0 and len(s) < MIN_N:
        c.set("too-early", f"no live fills yet; paper {len(s)} settled")
    else:
        c.set("alive" if len(s) >= MIN_N else "too-early",
              f"0 losses in {len(first)} settled paper picks (judge: 100 picks); live {n_live} settled")
    return c


def card_b2(folded) -> Card:
    c = Card("B2", "DUL NFL division-game UNDERS, KXNFLTOTAL (lead_exec)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "| B2 |", -1))
    d = lead_rows("DUL", folded)
    c.lines.append(lead_summary(d) + f" (source `{rel(LEAD)}`; first DUL window Thu 11-12)")
    st = d["settled"]
    pnl = [float(o.get("pnl") or 0) for o in st]
    risk = [float(o.get("cost") or 0) for o in st]
    clv = [o["clv"] for o in st if o.get("clv") is not None]
    roi = sum(pnl) / sum(risk) if sum(risk) else math.nan
    if len(st) >= 40 and (roi < 0 or (clv and np.mean(clv) < 0)):
        c.set("KILLED-by-rule", f"after {len(st)} bets ROI {roi:+.1%}, mean CLV {np.mean(clv) if clv else math.nan:+.4f}")
    elif len(st) < MIN_N:
        c.set("too-early", f"{len(st)} settled of the 40 the rule waits for")
    elif roi < 0 or (clv and np.mean(clv) < 0):
        c.set("near-kill", f"{len(st)}/40 settled with ROI {roi:+.1%} or CLV < 0")
    else:
        c.set("alive", f"{len(st)}/40 settled, ROI and CLV >= 0 so far")
    return c


def f1_paper(run: bool) -> tuple[str, dict]:
    """The logger's own report (it fetches ESPN closing lines read-only); returns (line, {n, clv_hi})."""
    if not run:
        return "forward paper: not run (--no-subproc)", {}
    try:
        out = subprocess.run([str(PY), "scripts/nba_early_form_logger.py", "report", "--no-refresh"], cwd=QS,
                             capture_output=True, text=True, timeout=240).stdout
    except Exception as ex:  # noqa: BLE001
        return f"forward paper: report failed ({type(ex).__name__})", {}
    keep = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(("F1 primary", "in-window", "clv_", "rows "))]
    info = {}
    for ln in keep:
        if ln.startswith("F1 primary"):
            info["n"] = int(ln.split("bets ")[1].split(",")[0])
        if ln.startswith("clv_ask"):
            info["clv_hi"] = float(ln.split("[")[1].split(",")[1].strip(" ]").split("]")[0])
            info["clv"] = ln
    line = ("forward paper (`~/.claude/quant-sports/scripts/nba_early_form_logger.py report --no-refresh`, taker at "
            "the Kalshi ask + fee): " + " | ".join(keep)) if keep else "forward paper: report printed no summary"
    return line, info


def card_b3(run_sub: bool) -> Card:
    c = Card("B3", "F1 NBA early-form sides (user-placed on Robinhood; bet.py journal)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "| B3 |", -1))
    rows = jl(KD / "bets.jsonl")
    bets = {r["id"]: dict(r) for r in rows if r.get("event") == "bet" and r.get("strategy") == "early_form_F1"}
    for r in rows:
        if r.get("id") in bets and r.get("event") in ("settle", "update"):
            bets[r["id"]].update({k: v for k, v in r.items() if k not in ("event", "ts")})
    st = [b for b in bets.values() if b.get("result") in ("win", "loss", "push")]
    pnl = [float(b.get("pnl_usd") or 0) for b in st]
    risk = [float(b.get("stake_usd") or 0) for b in st]
    c.lines.append(f"user journal (`{rel(KD / 'bets.jsonl')}`, strategy early_form_F1): bets {len(bets)}, settled "
                   f"{len(st)}, W-L {wl([{'win': True, 'loss': False}.get(b['result']) for b in st])}, P&L "
                   f"${sum(pnl):+.2f} on ${sum(risk):.2f}, {f_roi(pnl, risk)}; CLV not in the journal")
    line, info = f1_paper(run_sub)
    c.lines.append(line)
    if info.get("clv"):
        c.lines.append("CLV note: the rule does not say vs ask or vs mid. vs the ask (what a taker pays) it is the "
                       "number used for status; vs the mid it is in the report line above -- part of the ask gap is the "
                       "spread itself")
    r, lo, hi = boot_roi(pnl, risk)
    pn, clv_hi = info.get("n", 0), info.get("clv_hi", math.nan)
    if len(st) >= 100 and np.isfinite(hi) and hi < 0:
        c.set("KILLED-by-rule", f"100+ bets and ROI CI upper {hi:+.1%} < 0")
    elif pn >= 100 and np.isfinite(clv_hi) and clv_hi < 0:
        c.set("KILLED-by-rule", f"paper n {pn} >= 100 and CLV vs the ask CI upper {clv_hi:+.4f} < 0")
    elif np.isfinite(clv_hi) and clv_hi < 0:
        c.set("near-kill", f"paper CLV vs the ask is already below 0 at 95% (CI upper {clv_hi:+.4f}, n {pn}); the "
                           f"rule judges at 100 bets (user journal {len(st)}/100)")
    elif len(st) < MIN_N:
        c.set("too-early", f"{len(st)} settled user bets of the 100 the rule judges at")
    elif np.isfinite(hi) and hi < 0:
        c.set("near-kill", f"ROI CI upper {hi:+.1%} < 0 before the 100-bet judge")
    else:
        c.set("alive", f"{len(st)}/100 settled, CI upper >= 0")
    return c


def card_b4(folded) -> Card:
    c = Card("B4", "TSA weekly maker (KXTSAW, lead_exec mirrors the tsa_t2_maker paper posts)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "| B4 |", -1))
    c.rule(rule_line(ROOT / "scripts" / "e1x_tsa_logger.py", "if sum(losses[:30]) >= 3"))
    d = lead_rows("TSA", folded)
    resting = sum(1 for o in d["placed"] if o.get("status") == "resting" and not o.get("settled"))
    c.lines.append(lead_summary(d) + f"; resting now {resting} (source `{rel(LEAD)}`; the 11-08 week's 8 picks were "
                   "all skip_stale_signal, so live starts with KXTSAW-26NOV15)")
    rs = jl(SD / "e1x" / "tsa_paper.jsonl")
    posts = {r["ticker"]: r for r in rs if r.get("kind") == "post"}
    fills = {r["ticker"]: r for r in rs if r.get("kind") == "fill"}
    res = {r["ticker"]: r["result"] for r in rs if r.get("kind") == "settle"}
    st = [(tk, posts[tk]) for tk in sorted(fills, key=lambda k: fills[k]["fill_ts"]) if tk in res and tk in posts]
    pnl = [(1.0 if res[tk] == p["side"] else 0.0) - p["q"] for tk, p in st]
    c.lines.append(f"forward paper (`{rel(SD / 'e1x' / 'tsa_paper.jsonl')}`): posted {len(posts)}, tape-filled "
                   f"{len(fills)}, settled fills {len(st)}, W-L {wl([res[tk] == p['side'] for tk, p in st])}, "
                   f"{f_roi(pnl, [p['q'] for _, p in st])}; settled posts (filled or not) {len(res)}")
    live_l = sum(not o.get("won") for o in d["settled"][:30])
    if live_l >= 2:
        c.set("KILLED-by-rule", f"{live_l} live losses within the first 30 fills")
    elif live_l == 1:
        c.set("near-kill", "1 live loss in the first 30 fills; one more kills")
    elif len(d["settled"]) < MIN_N:
        c.set("too-early", f"live settled fills {len(d['settled'])} (rule counts the first 30)")
    else:
        c.set("alive", f"0 losses in {len(d['settled'])} live settled fills")
    return c


def nhl_paper() -> tuple[list[dict], str]:
    sig = jl(SD / "live" / "nhl_ingame_rule_signals.jsonl")
    res = {r["ticker"]: r.get("result") for r in jl(SD / "live" / "nhl_ingame_results.jsonl")}
    out = []
    for s in sig:
        r = res.get(s["ticker"])
        if r not in ("yes", "no"):
            continue
        q = float(s["price"])                                   # price paid for the backed side (taker)
        cost = q + kalshi_fee(q)
        won = r == s["side"]
        out.append({"won": won, "pnl": (1 - cost) if won else -cost, "risk": cost})
    return out, (f"forward paper trial nhl_ig_t1_late_taker (`{rel(SD / 'live' / 'nhl_ingame_rule_signals.jsonl')}` "
                 f"x results): signals {len(sig)}, settled {len(out)}, W-L {wl([o['won'] for o in out])}, "
                 f"{f_roi([o['pnl'] for o in out], [o['risk'] for o in out])} (1 ct at the signal price + taker fee; "
                 f"CLV not logged by the paper trial)")


def card_b5(folded, cache) -> Card:
    c = Card("B5", "NHL last-15-min taker (ingame_exec live $1 + paper trial nhl_ig_t1_late_taker)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "- B5 NHL last-15-min taker"))
    src, txt = rule_line(ROOT / "lab" / "edgelab_registry.jsonl", '"preset_kill", "trial_id": "nhl_ig_t1_late_taker')
    if txt.startswith("{"):
        j = json.loads(txt)
        txt = f"trial {j['trial_id']} split {j['split']} rules {json.dumps(j['rules'])}"
    c.rule((src, txt), "Paper-trial preset kill (registry row, rules field)")
    d = lead_rows("NHL", folded)
    c.lines.append(lead_summary(d) + f" (source `{rel(LEAD)}`, mode live only; 1 dry row 11-07 excluded)")
    xs = [(o["ticker"], float(o.get("pnl") or 0)) for o in d["settled"]]
    chk = [(tk, p, ticker_pnl(cache[tk]["fills"], cache[tk]["market"])[0]) for tk, p in xs if tk in cache]
    bad = [(tk, p, q) for tk, p, q in chk if abs(p - q) > 0.011]
    c.lines.append(f"exchange check (cached fills): {len(chk)}/{len(xs)} tickers compared, "
                   f"{'all match the journal within 1c' if not bad else 'MISMATCH ' + str(bad)}")
    kills = [r for r in jl(LEAD) if r.get("event") == "kill"]
    for k in kills:
        c.warn.append(f"journal kill event {k['ts']}: {k.get('reason')} (kill file later renamed "
                      "LEAD_KILL.cleared-2026-11-09-false-alarm)")
    paper, line = nhl_paper()
    c.lines.append(line)
    st = d["settled"]
    l30 = sum(not o.get("won") for o in st[:30])
    if l30 >= 10:
        c.set("KILLED-by-rule", f"{l30} live losses in the first 30")
    elif len(paper) >= 100:
        r, lo, hi = boot_roi([o["pnl"] for o in paper], [o["risk"] for o in paper])
        c.set("KILLED-by-rule" if hi < 0 else "alive", f"paper n {len(paper)}: ROI CI upper {hi:+.1%}")
    elif l30 == 9:
        c.set("near-kill", "9 live losses in the first 30; one more kills")
    elif len(st) < MIN_N:
        c.set("too-early", f"live {len(st)} settled ({l30} losses of the 10-in-30 limit); paper {len(paper)} of 100")
    else:
        c.set("alive", f"live {len(st)} settled, {l30} losses (limit 10 in first 30)")
    return c


def nflf_exchange(folded, cache) -> tuple[list[str], dict]:
    """Split every NFLF ticker's exchange fills into clean (entry + the first exit order) vs incident (every later
    sell order = the runaway retries) P&L."""
    keys = {k: o for k, o in folded.items() if o.get("strategy") == "NFLF" and float(o.get("filled") or 0) > 0}
    entry_oid = {o.get("order_id"): k for k, o in keys.items() if o.get("order_id")}
    by_tk = defaultdict(list)
    for k, o in keys.items():
        by_tk[o["ticker"]].append(k)
    lines, tot = [], {"clean": 0.0, "incident": 0.0, "all": 0.0, "clean_rows": [], "unsettled": [], "missing": [],
                      "inc": {}}
    for tk, ks in sorted(by_tk.items()):
        if tk not in cache:
            tot["missing"].append(tk)
            continue
        fills, mkt = cache[tk]["fills"], cache[tk]["market"]
        orders = defaultdict(list)
        for f in fills:
            orders[f["order_id"]].append(f)
        seq = sorted(orders.values(), key=lambda fs: min(f["created_time"] for f in fs))
        all_pnl, settled = ticker_pnl(fills, mkt)
        clean_fills, cur = [], None
        n_exit_orders = defaultdict(int)
        for fs in seq:
            oid = fs[0]["order_id"]
            if oid in entry_oid:
                cur = entry_oid[oid]
                clean_fills.append((cur, fs))
            elif cur is not None:
                n_exit_orders[cur] += 1
                if n_exit_orders[cur] == 1:
                    clean_fills.append((cur, fs))
        for k in ks:
            kf = [f for kk, fs in clean_fills if kk == k for f in fs]
            p, _ = ticker_pnl(kf, mkt)
            entry = sum(fill_delta(f) for f in kf if f["order_id"] in entry_oid)
            exited = n_exit_orders[k] > 0
            tot["clean_rows"].append({"key": k, "ticker": tk, "pnl": p, "risk": float(keys[k].get("cost") or 0),
                                      "won": p > 0, "settled": settled or exited and abs(
                                          sum(fill_delta(f) for f in kf)) < 1e-9})
            lines.append(f"`{k.split('|')[-1]}` {tk.split('-', 1)[1]}: bought {entry:g}, exit orders that filled "
                         f"{n_exit_orders[k]} (clean = first only) -> clean P&L ${p:+.2f}; journal said "
                         f"${float(keys[k].get('pnl') or 0):+.2f}" + ("" if keys[k].get("settled") else " (journal: open)"))
        clean_tk = sum(r["pnl"] for r in tot["clean_rows"] if r["ticker"] == tk)
        tot["all"] += all_pnl
        tot["clean"] += clean_tk
        tot["incident"] += all_pnl - clean_tk
        if abs(all_pnl - clean_tk) > 0.005:
            tot.setdefault("inc", {})[tk] = all_pnl - clean_tk
        net = sum(fill_delta(f) for f in fills)
        tag = f"settled {mkt.get('result')}" if settled else (f"UNSETTLED ({mkt.get('status')}), marked at last "
                                                             f"{mkt.get('last_price_dollars')}")
        if not settled:
            tot["unsettled"].append(tk)
        lines.append(f"  ticker {tk}: net YES {net:+g} after all fills, {tag}: all-fills P&L ${all_pnl:+.2f}, of which "
                     f"incident ${all_pnl - clean_tk:+.2f}")
    return lines, tot


def card_b6(folded, cache, fetched) -> Card:
    c = Card("B6", "NFLF live big-play maker fade (ingame_exec; DISABLED after the 11-09/10 runaway-exit incident)")
    c.rule(rule_line(NOTES / "ACT_LIST.md", "- B6 NFLF live big-play maker fade"))
    d = lead_rows("NFLF", folded)
    c.lines.append("journal as recorded (intended trades only): " + lead_summary(d))
    lines, tot = nflf_exchange(folded, cache)
    if not cache:
        c.warn.append("no exchange-fill cache: run with --kalshi; the journal alone UNDER-COUNTS (see below)")
    else:
        cr = tot["clean_rows"]
        cs = [r for r in cr if r["settled"]]
        c.lines.append(f"exchange truth (fills cached {fetched}; clean = entry + FIRST exit order, incident = every "
                       f"later sell order): clean {len(cs)} settled/exited, W-L {wl([r['won'] for r in cs])}, P&L "
                       f"${sum(r['pnl'] for r in cs):+.2f} on ${sum(r['risk'] for r in cs):.2f}, "
                       f"{f_roi([r['pnl'] for r in cs], [r['risk'] for r in cs])}")
        c.lines.append(f"INCIDENT (runaway exit fills, separate from the strategy): ${tot['incident']:+.2f}; all fills "
                       f"together ${tot['all']:+.2f}" + (f"; includes UNSETTLED {', '.join(tot['unsettled'])} marked at "
                                                        "last price (re-run after settlement)" if tot["unsettled"] else "")
                       + f". The sign is luck: unintended NO legs on {len(tot['inc'])} tickers, landed on "
                       f"{sum(v > 0 for v in tot['inc'].values())}; the biggest: "
                       + ", ".join(f"{t.rsplit('-', 1)[-1]} ${v:+.2f}" for t, v in
                                   sorted(tot["inc"].items(), key=lambda kv: -abs(kv[1]))[:3])
                       + ". It says nothing about the strategy, and the exposure was uncapped.")
        c.lines.append("runaway exits were NOT only 11-09 BUFMIN: on 11-08 every exit attempt filled too "
                       "(journal exit_attempts = number of filled sell orders); STATE's 'NFLF 1-7 -$2.80' is the "
                       "journal view")
        c.lines += ["per entry: " + ln if not ln.startswith("  ") else ln for ln in lines]
        if tot["missing"]:
            c.warn.append(f"no cached fills for {tot['missing']}")
    cs = [r for r in tot["clean_rows"] if r["settled"]] if cache else d["settled"]
    pnl = sum(float(r["pnl"]) for r in cs)
    l20 = sum(not r["won"] for r in cs[:20])
    if pnl <= -15 or l20 >= 8:
        c.set("KILLED-by-rule", f"clean record: P&L ${pnl:+.2f}, {l20} losses in the first {min(20, len(cs))} "
                                "(already disabled by the user for the incident)")
    elif pnl <= -12 or l20 == 7:
        c.set("near-kill", f"clean record P&L ${pnl:+.2f}, {l20} losses in first 20 (strategy is disabled anyway)")
    else:
        c.set("alive" if len(cs) >= MIN_N else "too-early",
              f"clean record P&L ${pnl:+.2f}, {l20} losses in first {min(20, len(cs))}; DISABLED (LEAD_DISABLE_NFLF), "
              "do not re-arm before the find_order root cause is known")
    return c


def card_l32b() -> Card:
    c = Card("L32b", "NBA debiased totals vs opener, forward paper (l32b_logger)")
    c.rule(rule_line(NOTES / "l32b_debiased_totals.md", "- JUDGE:"))
    p = SD / "l32b_state.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    picks = [v for v in st.values() if v.get("pick")]
    s = [v for v in picks if v.get("result") in ("win", "loss", "push")]
    won = [{"win": True, "loss": False}.get(v["result"]) for v in s]
    nv = [v.get("pnl_novig", 0) for v in s]
    c.lines.append(f"`{rel(p)}`: picks {len(picks)} (live {sum(v['kind'] == 'live' for v in picks)}, backfill "
                   f"{sum(v['kind'] == 'backfill' for v in picks)}), settled {len(s)}, W-L-P {wl(won)}, ROI no-vig "
                   f"{f_roi(nv, [1.0] * len(s))[4:]}, ROI at -110 "
                   f"{f_roi([v.get('pnl_110', 0) for v in s], [1.0] * len(s))[4:]}; "
                   f"{f_clv([v.get('clv_pts') for v in s], ' pts')}")
    m, lo, hi, n = mean_ci([v.get("clv_pts") for v in s])
    if n >= 100 and m <= 0:
        c.set("KILLED-by-rule", f"mean CLV {m:+.2f} pts <= 0 after {n}")
    elif n < MIN_N:
        c.set("too-early", f"{n} settled with CLV of 100 (kill) / 200 (judge)")
    elif np.isfinite(hi) and hi < 0:
        c.set("near-kill", f"CLV CI upper {hi:+.2f} pts < 0 at n {n} (kill check at 100)")
    else:
        c.set("alive", f"n {n}/100, CLV {m:+.2f} pts")
    return c


def opener_rows(path: Path, settle_kind: str, clv_key: str):
    rows = jl(path)
    picks = {(r["key"], r.get("market")): r for r in rows if r.get("type") in ("pick",)}
    out = []
    for r in rows:
        if r.get("type") != settle_kind:
            continue
        p = picks.get((r["key"], r.get("market")))
        res = r.get("result")
        won = {"win": True, "loss": False}.get(res)
        if settle_kind == "settle_tight":
            pnl = r.get("roi")
        elif p is not None and p.get("nv_prob_open"):
            pnl = (1 / p["nv_prob_open"] - 1) if won else (-1.0 if won is False else 0.0)
        else:
            pnl = None
        out.append({"market": r.get("market") or "total", "won": won, "pnl": pnl, "clv": r.get(clv_key)})
    return picks, out


def opener_card(cid, title, path, settle_kind, clv_key, unit, rules, need_roi_pass=False) -> Card:
    c = Card(cid, title)
    for r in rules:
        c.rule(r)
    picks, s = opener_rows(path, settle_kind, clv_key)
    n_picks = len(picks) if settle_kind == "settle" else sum(1 for r in jl(path) if r.get("type") == "pick_tight") \
        or "n/a"
    for mk in sorted({x["market"] for x in s}) + (["ALL"] if len({x["market"] for x in s}) > 1 else []):
        x = [v for v in s if mk == "ALL" or v["market"] == mk]
        xr = [v for v in x if v["pnl"] is not None]
        c.lines.append(f"[{mk}] settled {len(x)}, W-L-P {wl([v['won'] for v in x])}, "
                       f"{f_roi([v['pnl'] for v in xr], [1.0] * len(xr))} per 1u "
                       f"({'at ask+fee' if settle_kind == 'settle_tight' else 'at the no-vig opener'}); "
                       f"{f_clv([v['clv'] for v in x], unit)}")
    c.lines.append(f"source `{rel(path)}` rows type={settle_kind}; picks logged {n_picks}")
    m, lo, hi, n = mean_ci([v["clv"] for v in s])
    r, rlo, rhi = boot_roi([v["pnl"] for v in s if v["pnl"] is not None],
                           [1.0] * sum(v["pnl"] is not None for v in s))
    if len(s) >= 100 and (m <= 0 or (need_roi_pass and r <= 0)):
        c.set("KILLED-by-rule", f"n {len(s)} >= 100: CLV {m:+.4f}" + (f", ROI {r:+.1%}" if need_roi_pass else ""))
    elif len(s) < MIN_N:
        c.set("too-early", f"{len(s)} settled of 100")
    elif (np.isfinite(hi) and hi < 0) or (np.isfinite(rhi) and rhi < 0) or m < 0:
        c.set("near-kill", f"n {len(s)}/100: CLV {m:+.4f}{unit} [{lo:+.4f}, {hi:+.4f}], ROI {r:+.1%} "
                           f"[{rlo:+.1%}, {rhi:+.1%}] -- below the bar at the moment (rule judges at 100)")
    else:
        c.set("alive", f"n {len(s)}/100, CLV {m:+.4f}{unit}")
    return c


def card_lock(run_sub: bool) -> Card:
    c = Card("LOCK", "lock-in 99c NBA queue-depth logger (read-only measurement for lock_t2_nba_maker99)")
    c.rule(rule_line(ROOT / "scripts" / "lock_queue_logger.py", "N_QUEUE = 5000 was a guess"),
           "No kill rule (measurement); assumption under test")
    p = SD / "live" / "lock_queue.jsonl"
    rows = jl(p)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r.get("kind")] += 1
    locks = [r for r in rows if r.get("kind") == "lock"]
    c.lines.append(f"`{rel(p)}`: rows {len(rows)} ({dict(kinds)}), locks {len(locks)} "
                   f"(late_detect {sum(bool(r.get('late_detect')) for r in locks)}), games "
                   f"{len({r.get('game_id') for r in locks})}, dates {sorted({r.get('espn_date') for r in locks})}")
    for series in ("KXNBAGAME", "KXNBATOTAL"):
        q = [r["yes_99"] for r in rows if r.get("kind") == "snapshot" and r.get("series") == series
             and r.get("delay_s") == 0 and not r.get("late_detect") and r.get("yes_99") is not None]
        if q:
            a = np.asarray(q, float)
            c.lines.append(f"{series} at lock (+0 s{' = the horn' if series == 'KXNBAGAME' else ''}): n {len(a)}, "
                           f"99c queue min {a.min():,.0f} / p10 {np.quantile(a, .1):,.0f} / median "
                           f"{np.median(a):,.0f} / p90 {np.quantile(a, .9):,.0f} / max {a.max():,.0f}; "
                           f"<= 5,000: {np.mean(a <= 5000):.0%}; median = {np.median(a) / 5000:.0f}x the frozen "
                           "N_QUEUE=5,000")
    if run_sub:
        try:
            out = subprocess.run([str(PY), "scripts/lock_queue_logger.py", "report"], cwd=QS, capture_output=True,
                                 text=True, timeout=120).stdout
            c.lines.append("logger's own report (share = volume reaching a joiner behind the queue):\n\n```\n"
                           + out.strip() + "\n```")
        except Exception as ex:  # noqa: BLE001
            c.warn.append(f"lock_queue_logger report failed: {type(ex).__name__}")
    c.set("too-early" if len(locks) < 30 else "alive",
          "measurement only (no money, no kill rule); edgelab trial lock_t2_nba_maker99 is needs_audit")
    return c


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kalshi", action="store_true", help="refresh exchange fills (authenticated GETs only)")
    ap.add_argument("--no-subproc", action="store_true", help="skip the loggers' own report commands")
    ap.add_argument("--out", default=str(NOTES / "SCORECARDS.md"))
    a = ap.parse_args(argv)
    hist = jl(LEAD)
    folded = fold(hist, "live")
    ing = sorted({o["ticker"] for o in folded.values() if o.get("strategy") in ("NFLF", "NHL")
                  and float(o.get("filled") or 0) > 0})
    cache, fetched = load_cache(a.kalshi, ing)
    run_sub = not a.no_subproc
    fb = NOTES / "FUND_BOOK.md"
    cards = [card_b1(folded), card_b2(folded), card_b3(run_sub), card_b4(folded), card_b5(folded, cache),
             card_b6(folded, cache, fetched),
             card_l32b(),
             opener_card("L34", "NHL ML/totals vs opener (open_lines_logger, forward paper)",
                         KD / "open_lines_nhl_picks.jsonl", "settle", "clv", " prob",
                         [rule_line(fb, "| L34 |", -2),
                          rule_line(ROOT / "scripts" / "open_lines_logger.py", "KILL RULE: mean CLV <= 0")]),
             opener_card("L33", "NFL spread/total vs opener (open_lines_logger, forward paper)",
                         KD / "open_lines_nfl_picks.jsonl", "settle", "clv", " pts",
                         [rule_line(fb, "| L33 |", -2),
                          rule_line(ROOT / "scripts" / "open_lines_logger.py", "KILL RULE: mean CLV <= 0")]),
             opener_card("Q004-NHL", "first TIGHT Kalshi quote, NHL", KD / "open_lines_nhl_picks.jsonl", "settle_tight",
                         "clv", " prob", [rule_line(fb, "| Q004 |", -2)], need_roi_pass=True),
             opener_card("Q004-NFL", "first TIGHT Kalshi quote, NFL", KD / "open_lines_nfl_picks.jsonl", "settle_tight",
                         "clv", " pts", [rule_line(fb, "| Q004 |", -2)], need_roi_pass=True),
             opener_card("Q004-NBA", "first TIGHT Kalshi quote, NBA totals", KD / "open_totals_picks.jsonl",
                         "settle_tight", "clv_pts", " pts", [rule_line(fb, "| Q004 |", -2)], need_roi_pass=True),
             card_lock(run_sub)]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    killed = [c for c in cards if c.status == "KILLED-by-rule"]
    near = [c for c in cards if c.status == "near-kill"]
    out = [f"# SCORECARDS -- every live / forward strategy vs its pre-set kill rule (generated {now})", "",
           f"Generated by `sports/scripts/scorecards.py` (read-only; `--kalshi` refreshes exchange fills by GET). "
           f"Exchange-fill cache: {fetched}. Rules are quoted verbatim from the cited line at run time. The lead "
           "decides; this file never touches kill files.", ""]
    out.append("## KILL RULES THAT HAVE FIRED")
    out += [f"- **{c.cid}**: {c.why}" for c in killed] or ["- none"]
    out += ["", "## Near kill"] + ([f"- **{c.cid}**: {c.why}" for c in near] or ["- none"])
    out += ["", "## Summary", "| ID | Strategy | Status | Why |", "|---|---|---|---|"]
    out += [f"| {c.cid} | {c.title} | {c.status} | {c.why} |" for c in cards]
    out += ["| E4 | Kalshi weather maker (Tier A) | see notes/e4_live_audit.md | audited by agent e4_live_audit |", ""]
    out += ["## Cards", ""] + [c.md() for c in cards]
    out += ["### E4 -- pointer only", "Audited separately: `sports/notes/e4_live_audit.md` (agent e4_live_audit).", ""]
    Path(a.out).write_text("\n".join(out))
    print(f"wrote {a.out}: " + ", ".join(f"{c.cid}={c.status}" for c in cards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
