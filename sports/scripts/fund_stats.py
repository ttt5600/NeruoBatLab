"""STATISTICIAN GATE for the research fund (notes/FUND_ARCHITECTURE.md role 6). Deterministic; no LLM judgement.

    ../.venv_quant/bin/python scripts/fund_stats.py report            # every live/paper lead + any other non-DEAD trial
    ../.venv_quant/bin/python scripts/fund_stats.py verdict STRATEGY  # one trial (latest ledger row with that name)
    ../.venv_quant/bin/python scripts/fund_stats.py families          # per-dataset family counts and bars
    ../.venv_quant/bin/python scripts/fund_stats.py flags             # q<.10 trials lacking a both-halves check
    ../.venv_quant/bin/python scripts/fund_stats.py power MU SIGMA [--alpha .05 --power .8]
    add --json to report/verdict for machine output; --ledger / --lockbox / --fund-data override the inputs.

Inputs (read-only): lab/ledger.jsonl (every dev trial), lab/lockbox.jsonl (sealed one-shot results) and the forward
paper loggers' files under ~/.claude/quant-kalshi/data (env FUND_DATA).

Rules (fixed here so no agent can argue a verdict):
- A TRIAL = a ledger row with status "ok" and a numeric p, minus rows cancelled by a LATER row of the same strategy
  with status invalidated/void_bug (matching `voids_trial_number` and sha when given). Re-runs under the same name still count (every trial counts); look-ups use the latest row.
- p used = the most conservative p recorded on the row: max(p, p_null, p_null_1per_cm, net_p, roi_p, p_exact_vs_breakeven). A p of exactly 0 from a
  finite bootstrap is replaced by the normal-approximation p from the row's own CI when one exists (units/roi/delta
  CI), else floored at P_FLOOR = 1/5000 (a 5,000-draw bootstrap cannot show p below that).
- Ledger-wide Bonferroni bar = .05 / N_trials. Dataset bar = min(.05 / n_trials_in_dataset, the row's own
  pre-registered `bonferroni_bar`). BH-FDR q-values over all N trials.
- Both-halves check = the effect has the same sign in the earlier and later half of the sample (per_half, halves,
  half1/half2, *_h1/*_h2, half1_units/half2_units, sub-periods, or per_fold split chronologically).
- verdict(trial):
    sealed FAIL in lockbox                                   -> DEAD (never re-score; a forward logger is a NEW trial)
    sealed PASS in lockbox                                   -> PASSED
    not scored, effect <= 0, or BH q >= .10                  -> DEAD
    q < .10 but no both-halves check on the row              -> INCOMPLETE (add the check; would be LEAD)
    q < .10 and halves disagree in sign                      -> DEAD
    q < .10, halves agree                                    -> LEAD
      ... and a CLV metric                                   -> stays LEAD (CLV is a leading indicator, not P&L)
      ... and a forward paper logger is running              -> PAPER
      ... and passes Bonferroni or the dataset bar AND forward evidence (>= FWD_MIN_N settled, mean > 0,
          one-sided p < .05)                                 -> SEALED-ASK
- Power: n = ((z_a + z_b) * sigma / mu)^2 (one-sided alpha, power 1-beta).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent  # sports/
LEDGER = ROOT / "lab" / "ledger.jsonl"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
FUND_DATA = Path(os.environ.get("FUND_DATA", Path.home() / ".claude" / "quant-kalshi" / "data"))

ALPHA = 0.05
LEAD_Q = 0.10
P_FLOOR = 1 / 5000
VOID_STATUSES = ("invalidated", "void_bug")
P_KEYS = ("p", "p_null", "p_null_1per_cm", "net_p", "roi_p", "p_exact_vs_breakeven")  # conservative = max of these
FWD_MIN_N = 100          # the loggers' own kill-rule checkpoint
FWD_ALPHA = 0.05
N = NormalDist()

EFFECT_KEYS = ("net_ann", "roi", "mean", "clv", "delta", "delta_units", "units", "profit_per_contract_c", "excess",
               "slope")
CLV_METRICS = {"clv", "clv_prob", "clv_pts", "slope_pts_per_sd", "slope", "resid_vs_devig_open", "prob_pts",
               "excess_err"}

# Lockbox strategy names that differ from the ledger trial they promoted.
LOCKBOX_ALIASES = {"nba_L32_tot_open": "nba_simple_T5_tot_strict_open_novig"}

# Registered live/paper/passed leads (notes/EDGES_AND_AGENTS.md). `strategy` = the decisive DEV ledger trial;
# `lockbox` = its sealed row if named differently; `forward` = (parser, arg) for the forward paper logger.
LEADS = [
    dict(id="E1", name="Weather consensus-reject, orig 7 (taker)", status="LIVE", strategy="kalshi_reject_safe_check",
         forward=("weather_paper", "orig7")),
    dict(id="E2", name="Weather consensus-reject, X1 17 cities", status="LIVE", strategy="wx_expand_X1_new_highs",
         forward=("weather_paper", "x1")),
    dict(id="E4", name="E1/E2 as MAKER", status="PASSED (exec off)", strategy="kalshi_temp_maker_M3_N10",
         forward=None),
    dict(id="E3", name="Soccer soft books vs Pinnacle", status="PASSED (no CA venue)", strategy="soccer_best3_pow3",
         forward=None),
    dict(id="L32", name="NBA totals vs opener", status="PAPER", strategy="nba_simple_T5_tot_strict_open_novig",
         forward=("open_totals", None)),
    dict(id="L34", name="NHL totals vs opener", status="PAPER", strategy="nhl_simple_T5_tot_strict_novig",
         forward=("open_lines", "nhl")),
    dict(id="L33", name="NFL totals vs opener", status="PAPER", strategy="nfl_simple_T12_total_roi110_C",
         forward=("open_lines", "nfl")),
    dict(id="L36", name="CFB totals vs opener", status="LEAD (no logger)", strategy="cfb_simple_T06_total_roi_novig",
         forward=None),
    dict(id="L39", name="Monthly rain unreachable-NO (maker)", status="PAPER",
         strategy="kalshi_wx_winter_W1m_rain_unreachable_NO_maker", forward=("wx_monthly", None)),
    dict(id="F2", name="NFL pre-kickoff rain unders", status="PAPER", strategy="pre_rain_unders", forward=None),
    dict(id="L37", name="NBA mid-game comeback", status="PAPER (recorder)", strategy=None,
         lockbox="nba_L37_midgame", forward=None),
]


# ----------------------------------------------------------------------------------------------- helpers
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _val(x):
    """A half/period entry may be a number or [value, n]."""
    if _num(x):
        return float(x)
    if isinstance(x, (list, tuple)) and x and _num(x[0]):
        return float(x[0])
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def effect(r: dict) -> float | None:
    for k in EFFECT_KEYS:
        if _num(r.get(k)):
            return float(r[k])
    return None


def is_clv(r: dict) -> bool:
    m = str(r.get("metric") or "")
    return m in CLV_METRICS or m.startswith("clv") or "_clv" in str(r.get("strategy", "")).lower()


def _ci_p(r: dict) -> float | None:
    """One-sided normal-approximation p from the row's own 95% CI (estimate / ((hi-lo)/3.92))."""
    for est, lo, hi in (("units", "ci_lo", "ci_hi"), ("roi", "roi_ci_lo", "roi_ci_hi")):
        if all(_num(r.get(k)) for k in (est, lo, hi)) and r[hi] > r[lo]:
            return 1 - N.cdf(r[est] / ((r[hi] - r[lo]) / 3.92))
    d = r.get("delta_ci")
    if _num(r.get("delta")) and isinstance(d, list) and len(d) == 2 and all(_num(v) for v in d) and d[1] > d[0]:
        return 1 - N.cdf(r["delta"] / ((d[1] - d[0]) / 3.92))
    return None


def p_used(r: dict) -> float | None:
    ps = [float(r[k]) for k in P_KEYS if _num(r.get(k))]
    if not ps:
        return None
    p = max(ps)
    if p <= 0:
        cp = _ci_p(r)
        p = cp if cp is not None and cp > 0 else P_FLOOR
        p = max(p, 1e-12)
    return min(p, 1.0)


def halves(r: dict) -> tuple[float, float] | None:
    """(earlier-half value, later-half value), or None when the row carries no time-split check."""
    for k in ("per_half", "halves", "sub"):
        d = r.get(k)
        if isinstance(d, dict) and len(d) >= 2:
            vals = [_val(v) for _, v in sorted(d.items())]
            if all(v is not None for v in vals):
                m = len(vals) // 2
                return (sum(vals[:m]) / m, sum(vals[m:]) / (len(vals) - m)) if len(vals) > 2 else (vals[0], vals[1])
    for a, b in (("half1", "half2"), ("half1_units", "half2_units"), ("units_h1", "units_h2"),
                 ("delta_h1", "delta_h2")):
        va, vb = _val(r.get(a)), _val(r.get(b))
        if va is not None and vb is not None:
            return va, vb
    pf = r.get("per_fold")
    if isinstance(pf, dict) and len(pf) >= 2:
        items = [(k, _val(v)) for k, v in sorted(pf.items())]
        if all(v is not None for _, v in items):
            m = len(items) // 2
            return sum(v for _, v in items[:m]), sum(v for _, v in items[m:])
    return None


def sign_consistent(r: dict) -> bool | None:
    h, e = halves(r), effect(r)
    if h is None or e is None:
        return None
    s = 1 if e > 0 else -1
    return h[0] * s > 0 and h[1] * s > 0


def bh_qvalues(ps: list[float]) -> list[float]:
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    q = [1.0] * n
    run = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * n / rank)
        q[i] = min(run, 1.0)
    return q


def n_required(mu: float, sigma: float, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Bets needed to detect a per-bet mean edge mu with per-bet SD sigma: ((z_a+z_b) sigma/mu)^2, one-sided."""
    if mu <= 0 or sigma <= 0:
        return math.inf
    return ((N.inv_cdf(1 - alpha) + N.inv_cdf(power)) * sigma / mu) ** 2


def power_at(n: float, mu: float, sigma: float, alpha: float = ALPHA) -> float:
    if n <= 0 or sigma <= 0:
        return 0.0
    return N.cdf(mu * math.sqrt(n) / sigma - N.inv_cdf(1 - alpha))


def mean_test(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "p": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    p = (1 - N.cdf(m / (sd / math.sqrt(n)))) if sd else None
    return {"n": n, "mean": m, "sd": sd, "p": p}


# ----------------------------------------------------------------------------------------- forward parsers
def _fw_weather(cohort: str, data: Path) -> dict:
    xs = []
    for r in read_jsonl(data / "kalshi_paper.jsonl"):
        if r.get("cohort") != cohort or r.get("result") not in ("yes", "no") or not _num(r.get("no_price")):
            continue
        c = r["no_price"] + (r.get("fee_per_contract") or 0.0)
        xs.append((1 - c) / c if r["result"] == "no" else -1.0)
    return {"metric": "ROI per $ (paper, taker)", **mean_test(xs)}


def _settles(path: Path) -> list[dict]:
    return [r for r in read_jsonl(path) if r.get("type") == "settle"]


def _fw_open_totals(_, data: Path) -> dict:
    xs = [r["clv_pts"] for r in _settles(data / "open_totals_picks.jsonl") if _num(r.get("clv_pts"))]
    return {"metric": "CLV pts vs close", **mean_test(xs)}


def _fw_open_lines(league: str, data: Path) -> dict:
    xs = [r["clv"] for r in _settles(data / f"open_lines_{league}_picks.jsonl") if _num(r.get("clv"))]
    return {"metric": "CLV prob pts vs close", **mean_test(xs)}


def _fw_wx_monthly(_, data: Path) -> dict:
    xs = []
    for r in read_jsonl(data / "kalshi_wx_monthly_picks.jsonl"):
        if r.get("result") in ("yes", "no") and _num(r.get("taker_cost")):
            c = r["taker_cost"]
            xs.append((1 - c) / c if r["result"] == "no" else -1.0)
    return {"metric": "ROI per $ (paper, taker)", **mean_test(xs)}


FORWARD = {"weather_paper": _fw_weather, "open_totals": _fw_open_totals, "open_lines": _fw_open_lines,
           "wx_monthly": _fw_wx_monthly}


def forward_stats(spec, data: Path = None) -> dict | None:
    if not spec:
        return None
    data = Path(data or FUND_DATA)
    try:
        st = FORWARD[spec[0]](spec[1], data)
    except Exception as ex:  # a logger's file format changed: report, never crash the gate
        return {"metric": "error", "n": 0, "mean": None, "sd": None, "p": None, "error": str(ex)[:120]}
    st["passes"] = bool(st["n"] >= FWD_MIN_N and st["mean"] is not None and st["mean"] > 0
                        and st["p"] is not None and st["p"] < FWD_ALPHA)
    return st


# ---------------------------------------------------------------------------------------------- the gate
@dataclass
class Gate:
    rows: list[dict]
    lockbox: list[dict] = field(default_factory=list)
    fund_data: Path = FUND_DATA

    def __post_init__(self):
        kept: list[dict] = []
        for r in self.rows:  # ledger order = time order; a void row cancels an EARLIER ok row of the same strategy
            st = str(r.get("status", ""))
            if st in VOID_STATUSES or r.get("voids_trial_number") is not None:
                tn = r.get("voids_trial_number")
                for i in range(len(kept) - 1, -1, -1):
                    k = kept[i]
                    if k.get("strategy") == r.get("strategy") and (tn is None or k.get("trial_number") == tn) \
                            and (not r.get("sha") or k.get("sha") == r.get("sha")):
                        kept.pop(i)
                        break
                continue
            if st == "ok" and p_used(r) is not None and "INVALID" not in str(r.get("note", "")):
                kept.append(r)
        self.trials = kept
        self.n = len(self.trials)
        self.bonf = ALPHA / self.n if self.n else 0.0
        self.family: dict[str, int] = {}
        for r in self.trials:
            self.family[r["dataset"]] = self.family.get(r["dataset"], 0) + 1
        qs = bh_qvalues([p_used(r) for r in self.trials])
        self._q = {id(r): q for r, q in zip(self.trials, qs)}
        self.by_name: dict[str, dict] = {}
        for r in self.trials:
            self.by_name[r["strategy"]] = r  # latest row wins
        self.sealed: dict[str, str] = {}
        for r in self.lockbox:
            name = LOCKBOX_ALIASES.get(r.get("strategy"), r.get("strategy"))
            self.sealed[r.get("strategy")] = self.sealed[name] = _sealed_outcome(r)

    @classmethod
    def load(cls, ledger=LEDGER, lockbox=LOCKBOX, fund_data=FUND_DATA) -> "Gate":
        return cls(read_jsonl(Path(ledger)), read_jsonl(Path(lockbox)), Path(fund_data))

    def q(self, r: dict) -> float | None:
        return self._q.get(id(r))

    def dataset_bar(self, r: dict) -> float:
        bar = ALPHA / self.family.get(r.get("dataset"), 1)
        if _num(r.get("bonferroni_bar")):
            bar = min(bar, float(r["bonferroni_bar"]))
        return bar

    def tier(self, r: dict) -> str:
        p = p_used(r)
        if p is None:
            return "-"
        if p < self.bonf:
            return "ledger"
        if p < self.dataset_bar(r):
            return "dataset"
        return "none"

    def evaluate(self, r: dict | None, forward=None, lockbox_name: str | None = None) -> dict:
        """Full record behind a verdict. r = ledger row (or None for a lead scored only in the lockbox)."""
        name = (r or {}).get("strategy") or lockbox_name
        out = {"strategy": name, "dataset": (r or {}).get("dataset"), "p": p_used(r) if r else None,
               "q": self.q(r) if r else None, "tier": self.tier(r) if r else "-", "effect": effect(r) if r else None,
               "halves": halves(r) if r else None, "sign_ok": sign_consistent(r) if r else None,
               "clv": is_clv(r) if r else False, "sealed": self.sealed.get(lockbox_name or name),
               "forward": forward_stats(forward, self.fund_data) if forward else None, "reasons": []}
        why = out["reasons"]
        if out["sealed"] == "FAIL":
            why.append("sealed FAIL (never re-score)")
            out["verdict"] = "DEAD"
            return out
        if out["sealed"] == "PASS":
            why.append("sealed PASS")
            out["verdict"] = "PASSED"
            return out
        if r is None or id(r) not in self._q:
            why.append("not a scored ledger trial")
            out["verdict"] = "DEAD"
            return out
        if out["effect"] is None or out["effect"] <= 0:
            why.append("effect <= 0")
            out["verdict"] = "DEAD"
            return out
        if out["q"] >= LEAD_Q:
            why.append(f"BH q {out['q']:.3f} >= {LEAD_Q}")
            out["verdict"] = "DEAD"
            return out
        if out["sign_ok"] is None:
            why.append("no both-halves check on the row")
            out["verdict"] = "INCOMPLETE"
            return out
        if out["sign_ok"] is False:
            why.append("halves disagree in sign")
            out["verdict"] = "DEAD"
            return out
        why.append(f"BH q {out['q']:.3f} < {LEAD_Q}, halves agree")
        if out["clv"]:
            why.append("CLV metric: a leading indicator, not P&L")
            out["verdict"] = "LEAD"
            return out
        fw = out["forward"]
        if out["tier"] in ("ledger", "dataset") and fw and fw.get("passes"):
            why.append(f"passes {out['tier']} bar + forward evidence")
            out["verdict"] = "SEALED-ASK"
        elif fw is not None:
            why.append(f"forward n {fw['n']} < {FWD_MIN_N} or not significant" if not fw.get("passes")
                       else "forward passes but dev misses both bars")
            out["verdict"] = "PAPER"
        else:
            out["verdict"] = "LEAD"
        return out

    def verdict(self, trial) -> str:
        r = self.by_name.get(trial) if isinstance(trial, str) else trial
        lead = next((L for L in LEADS if r is not None and L.get("strategy") == r.get("strategy")), None)
        return self.evaluate(r, forward=(lead or {}).get("forward"),
                             lockbox_name=trial if isinstance(trial, str) else None)["verdict"]

    def missing_halves(self, max_q: float = LEAD_Q) -> list[dict]:
        return [r for r in self.by_name.values() if self.q(r) < max_q and (effect(r) or 0) > 0 and halves(r) is None]


def _sealed_outcome(r: dict) -> str:
    v = str(r.get("verdict") or "").upper()
    if v.startswith("PASS"):
        return "PASS"
    if v.startswith("FAIL"):
        return "FAIL"
    if isinstance(r.get("passed"), bool):
        return "PASS" if r["passed"] else "FAIL"
    return "PASS" if _num(r.get("p")) and r["p"] < ALPHA else "FAIL"


_DEFAULT: Gate | None = None


def verdict(trial, gate: Gate | None = None) -> str:
    """DEAD / INCOMPLETE / LEAD / PAPER / SEALED-ASK / PASSED for a ledger row (dict) or a strategy name."""
    global _DEFAULT
    if gate is None:
        _DEFAULT = _DEFAULT or Gate.load()
        gate = _DEFAULT
    return gate.verdict(trial)


# ------------------------------------------------------------------------------------------------- report
def lead_rows(g: Gate) -> list[dict]:
    out = []
    for L in LEADS:
        r = g.by_name.get(L["strategy"]) if L.get("strategy") else None
        ev = g.evaluate(r, forward=L.get("forward"), lockbox_name=L.get("lockbox") or L.get("strategy"))
        out.append({"id": L["id"], "name": L["name"], "status": L["status"], **ev})
    return out


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def report(g: Gate, as_json: bool = False) -> str:
    leads = lead_rows(g)
    lead_names = {L["strategy"] for L in LEADS}
    others = []
    for name, r in g.by_name.items():
        if name in lead_names:
            continue
        ev = g.evaluate(r)
        if ev["verdict"] != "DEAD":
            others.append(ev)
    others.sort(key=lambda e: e["q"])
    if as_json:
        return json.dumps({"n_trials": g.n, "bonferroni_bar": g.bonf, "leads": leads, "other_non_dead": others},
                          default=str, indent=1)
    L = [f"STATISTICIAN GATE -- {g.n} scored trials in {len(g.family)} dataset families; ledger Bonferroni bar "
         f"p < {g.bonf:.2e}; LEAD needs BH q < {LEAD_Q} + both halves; forward needs >= {FWD_MIN_N} settled",
         "",
         f"{'id':<4} {'status':<21} {'verdict':<11} {'p':>8} {'q':>6} {'tier':<7} {'halves':<5} "
         f"{'fwd n':>5} {'fwd mean':>9} {'fwd p':>6}  name / reason"]
    for e in leads:
        fw = e["forward"] or {}
        L.append(f"{e['id']:<4} {e['status']:<21} {e['verdict']:<11} {_f(e['p'], '.2e'):>8} {_f(e['q'], '.3f'):>6} "
                 f"{e['tier']:<7} {({True: 'ok', False: 'FLIP', None: '-'})[e['sign_ok']]:<5} "
                 f"{_f(fw.get('n'), 'd'):>5} {_f(fw.get('mean'), '+.4f'):>9} {_f(fw.get('p'), '.3f'):>6}  "
                 f"{e['name']} -- {'; '.join(e['reasons'])}")
    L += ["", f"Other ledger trials not DEAD ({len(others)}): latest row per strategy, by q"]
    for e in others[:40]:
        L.append(f"  {e['verdict']:<11} q {e['q']:.3f} p {e['p']:.2e} {e['tier']:<7} {e['dataset']:<22} "
                 f"{e['strategy']}  ({'; '.join(e['reasons'])})")
    miss = g.missing_halves()
    L += ["", f"Flags: {len(miss)} trials with q < {LEAD_Q} and a positive effect have NO both-halves check "
              f"(`fund_stats.py flags`)."]
    return "\n".join(L)


def families(g: Gate) -> str:
    L = [f"{'dataset':<28} {'trials':>6} {'bar .05/n':>10} {'q<.10':>6} {'min p':>9}"]
    for d, n in sorted(g.family.items(), key=lambda kv: -kv[1]):
        rs = [r for r in g.trials if r["dataset"] == d]
        L.append(f"{d:<28} {n:>6} {ALPHA / n:>10.5f} {sum(g.q(r) < LEAD_Q for r in rs):>6} "
                 f"{min(p_used(r) for r in rs):>9.2e}")
    L.append(f"{'ALL':<28} {g.n:>6} {g.bonf:>10.2e}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["report", "verdict", "families", "flags", "power"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--lockbox", default=str(LOCKBOX))
    ap.add_argument("--fund-data", default=str(FUND_DATA))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--power", type=float, default=0.8)
    a = ap.parse_args(argv)
    if a.cmd == "power":
        mu, sigma = float(a.args[0]), float(a.args[1])
        n = n_required(mu, sigma, a.alpha, a.power)
        print(f"n = ((z_{a.alpha} + z_{1 - a.power:.2f}) * {sigma} / {mu})^2 = {math.ceil(n):,} bets")
        return 0
    g = Gate.load(a.ledger, a.lockbox, a.fund_data)
    if a.cmd == "report":
        print(report(g, a.json))
    elif a.cmd == "families":
        print(families(g))
    elif a.cmd == "flags":
        for r in sorted(g.missing_halves(), key=g.q):
            print(f"q {g.q(r):.3f} p {p_used(r):.2e} {r['dataset']:<22} {r['strategy']}")
    elif a.cmd == "verdict":
        for name in a.args:
            r = g.by_name.get(name)
            lead = next((L for L in LEADS if L.get("strategy") == name), {})
            ev = g.evaluate(r, forward=lead.get("forward"), lockbox_name=name)
            print(json.dumps(ev, default=str) if a.json else f"{name}: {ev['verdict']} ({'; '.join(ev['reasons'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
