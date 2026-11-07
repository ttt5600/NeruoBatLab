"""Family: post-election margin brackets (Kalshi KXMIDTERMMOV) vs the live AP count in NYT's results JSON.

Market: "If <party> wins the 2026 <office> election in <state> by K points or more" (event KXMIDTERMMOV-<ST><OFF><P>).
Source: candidate vote totals + eevp from NYT results-{governor,senate}.json. The NYT `nyt_model_estimates` block
froze 2026-11-04 08:08Z and is IGNORED (census 11-07). Fair model is deliberately naive and is what the paper test
judges: final margin ~ Normal(current margin, sd = remaining_share * REM_SD + CANVASS_SD).
Known blind spot (the election lesson): uncounted provisional/cure/late-mail ballots can lean one way and are NOT in
eevp; the grader measures whether markets that price them beat this count-only fair value.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .. import core

NAME = "canvass_margin"
SERIES = ["KXMIDTERMMOV"]
NYT = "https://static01.nyt.com/elections-assets/pages/data/2026-11-03/results-{}.json"
OFFICES = {"GOV": "governor", "SEN": "senate"}
REM_SD = 20.0        # pts: spread of the uncounted votes' margin around the counted margin
CANVASS_SD = 0.15    # pts: canvass corrections even at 100% reporting
FRESH_HOURS = 36     # post-election counts move daily; older than this = stale
EVENT_RE = re.compile(r"^KXMIDTERMMOV-([A-Z]{2})(GOV|SEN)([DR])$")


def observe() -> dict[tuple[str, str], core.Observation]:
    obs = {}
    for off, file in OFFICES.items():
        for r in core.get(NYT.format(file))["races"]:
            t = r.get("top_reporting_unit") or {}
            st = t.get("state_postal")
            cm = r.get("candidate_metadata") or {}
            votes = {}
            for c in t.get("candidates") or []:
                pid = ((cm.get(c.get("nyt_id")) or {}).get("party") or {}).get("nyt_id")
                v = (c.get("votes") or {}).get("total") or 0
                votes[c.get("nyt_id")] = (pid, v)
            total = sum(v for _, v in votes.values())
            if not st or not total:
                continue
            upd = r.get("updated_at")
            try:
                age_h = (core.now() - datetime.fromisoformat(upd.replace("Z", "+00:00"))).total_seconds() / 3600
            except Exception:
                age_h = None
            fresh = age_h is not None and age_h <= FRESH_HOURS and r.get("race_call_status") != "certified"
            obs[(st, off)] = core.Observation(
                key=f"{st}-{off}", source="nyt-ap-counts",
                value={"votes": {k: v for k, v in votes.items()}, "total": total, "eevp": t.get("eevp"),
                       "call": r.get("race_call_status")},
                observed_at=core.now().isoformat(timespec="seconds"), source_updated_at=upd, fresh=fresh,
                note="" if fresh else f"stale age_h={age_h}")
    return obs


def margin_pts(o: core.Observation, party: str) -> float | None:
    code = {"D": "DEM", "R": "GOP"}[party]
    mine = sum(v for p, v in o.value["votes"].values() if p == code)
    others = sorted((v for k, (p, v) in o.value["votes"].items() if p != code), reverse=True)
    if not others or not o.value["total"]:
        return None
    return 100.0 * (mine - others[0]) / o.value["total"]


def fair_yes(m_now: float, eevp: float | None, k: float) -> float:
    rem = max(0.0, 1.0 - (eevp or 0) / 100.0)
    sd = rem * REM_SD + CANVASS_SD
    return core.phi((m_now - k) / sd)


def scan(min_ev: float = 0.05):
    obs = observe()
    flags, seen = [], 0
    used = {}
    for series in SERIES:
        for m in core.kalshi_markets(series):
            g = EVENT_RE.match(m.get("event_ticker", ""))
            k = core.f(m.get("floor_strike"))
            if not g or k is None or m.get("strike_type") != "greater_or_equal":
                continue
            st, off, party = g.groups()
            o = obs.get((st, off))
            if not o:
                continue
            mn = margin_pts(o, party)
            if mn is None:
                continue
            seen += 1
            used[o.key] = o
            p = fair_yes(mn, o.value.get("eevp"), k)
            if not o.fresh:
                continue  # rule 1: no flag on a stale source (still logged as an obs)
            for fl in core.gap(NAME, m, p, o, min_ev):
                fl.extra.update({"margin_now": round(mn, 3), "k": k, "eevp": o.value.get("eevp")})
                flags.append(fl)
    core.record(NAME, list(used.values()), flags, seen)
    return list(used.values()), flags, seen
