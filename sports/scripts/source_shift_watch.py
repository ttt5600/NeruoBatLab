"""source_shift_watch -- P-1 (settlement-definition shift) watcher. READ-ONLY GETs; no orders, no state outside its own dir.

Why (notes/IDEA_PATTERNS.md P-1, notes/source_shift.md): OpenRouter switched its default token chart to BYOK-excluded mid-week
(2026-11-20/21); bots repriced in minutes and Kalshi listed new low strikes 7 minutes later. The case was found only because an
exact settlement rebuild disagreed with its own earlier capture. This script makes that check routine: every run snapshots each
rebuildable settlement source under EVERY known definition, plus the Kalshi strikes / rules / settlement-source text of the
series that settle on it, appends one JSON line to ~/.claude/quant-sports/data/source_shift/snapshots.jsonl and prints
`ALERT <kind> <source> <detail>` when
  (i)  two definitions of the same source diverge more than usual (rolling median +- max(floor, 5 x MAD) of this script's own
       history), or a completed period's value is revised;
  (ii) the source's format / default view / precision changes (signature differs from the last run);
  (iii) Kalshi lists strikes on an open event that were not there last run (FAR if outside the old ladder by > 2 steps), or a
       series' rules template / settlement source / contract URL changes.
Sources: OpenRouter token chart (default, byok=organic, byok=all) + author share, open/closed, TEXT REQUEST share; Vercel AI Gateway labs
export + chef/weightClass endpoints; AAA national averages (all grades, current/yesterday); TSA (tsa.gov 403s scripts and is NOT fetched;
Kalshi KXTSAD first prints logged); The Weather Company display that Kalshi settles weather on (weather.com/kalshi/api/climate/
primary) vs the NWS CLI product for the E4 stations; NOAA NCEI monthly global land+ocean CSV (vintage = sha256 + last 24 months).
Cadence the lead may install: hourly (OpenRouter refreshes ~hourly; TWC preliminary values move intraday). NOT installed here.
Usage: ../.venv_quant/bin/python scripts/source_shift_watch.py [--quiet]
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

OUTDIR = Path.home() / ".claude" / "quant-sports" / "data" / "source_shift"
SNAP, STATE = OUTDIR / "snapshots.jsonl", OUTDIR / "state.json"
KX = "https://api.elections.kalshi.com/trade-api/v2"
OR = "https://openrouter.ai/api/frontend/v1/rankings/"
UA = "research-script/0.1 (source_shift_watch; read-only)"  # honest UA; every source answers it (2026-11-21)
ET = ZoneInfo("America/New_York")
OR_SERIES = ["KXTOKENUSE", "KXTOKENUSEM", "KXTOKENUSED", "KXANTHSHARE", "KXGOOGSHARE", "KXDEEPSHARE", "KXOPENSHARE", "KXTOPUSAGEAI",
             "KXOPENSOURCESHAREOR", "KXZAISHARE", "KXMETASHARE", "KXMISTRALSHARE", "KXTOP3AID"]
VERCEL = {"KXOPENSOURCESHARE": ("W", "Open Weights", "tokens"), "KXOSSPEND": ("W", "Open Weights", "cost"),
          "KXOPENVREQ": ("L", "openai", "requests"), "KXOPENVSPEND": ("L", "openai", "spend"), "KXMOONVSPEND": ("L", "moonshotai", "spend"),
          "KXANTHVSPEND": ("L", "anthropic", "spend"), "KXANTHVREQ": ("L", "anthropic", "requests"), "KXGOOGVREQ": ("L", "google", "requests"),
          "KXDEEPVREQ": ("L", "deepseek", "requests"), "KXGOOGVSPEND": ("L", "google", "spend"), "KXZAIVSPEND": ("L", "zai", "spend"),
          "KXZAIVREQ": ("L", "zai", "requests")}
E4 = {"KXHIGHNY": "NYC", "KXHIGHCHI": "MDW", "KXHIGHMIA": "MIA", "KXHIGHAUS": "AUS", "KXHIGHDEN": "DEN", "KXHIGHLAX": "LAX",
      "KXHIGHPHIL": "PHL"}
OTHER_SERIES = ["KXAAAGASD", "KXAAAGASW", "KXAAAGASM", "KXTSAD", "KXTSAW", "KXHMONTHRANGE", "KXHMONTH"]
AUTHORS = ["anthropic", "google", "deepseek", "openai", "z-ai", "meta-llama", "mistralai", "tencent", "qwen", "x-ai", "moonshotai"]
MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
ALERTS: list[dict] = []
QUIET = "--quiet" in sys.argv


def alert(kind: str, source: str, detail: str) -> None:
    ALERTS.append({"kind": kind, "source": source, "detail": detail})
    print(f"ALERT {kind} {source} {detail}", flush=True)


def get(url: str, params: dict | None = None, ua: str = UA, as_json: bool = True):
    err = None
    for a in range(4):
        try:
            r = requests.get(url, params=params, timeout=40, headers={"User-Agent": ua})
            if r.status_code == 429:
                time.sleep(3 * (a + 1)); continue
            if r.status_code >= 400:
                return {"_status": r.status_code}
            return r.json() if as_json else r.text
        except Exception as e:  # noqa: BLE001 -- a missing source is logged as an error, never fatal
            err = str(e)[:200]; time.sleep(2 * (a + 1))
    return {"_error": err}


def sig(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def template(text: str | None, subs) -> str:
    t = text or ""
    for s in subs:
        if s and len(s) > 1:
            t = t.replace(s, "<S>")
    t = re.sub(MONTHS, "<M>", t)
    t = re.sub(r"[-+]?\$?\d[\d,]*(\.\d+)?%?", "#", t)
    return re.sub(r"\s+", " ", t).strip()


def diverge(state: dict, key: str, value: float | None, floor: float, source: str, label: str) -> None:
    """(i) alert when value leaves its own rolling band (median +- max(floor, 5 MAD) of the last 48 snapshots)."""
    if value is None:
        return
    h = state.setdefault("hist", {}).setdefault(key, [])
    if len(h) >= 3:
        med = statistics.median(h[-48:])
        mad = statistics.median([abs(x - med) for x in h[-48:]])
        band = max(floor, 5 * mad)
        if abs(value - med) > band:
            alert("DIVERGENCE", source, f"{label}={value:.4g} vs usual {med:.4g} (band +-{band:.3g}, n={len(h[-48:])})")
    elif h and abs(value - h[-1]) > floor:
        alert("DIVERGENCE", source, f"{label}={value:.4g} vs last {h[-1]:.4g} (floor {floor}; short history)")
    h.append(value)
    del h[:-200]


def fmt_change(state: dict, source: str, signature, what: str) -> None:
    """(ii) alert when a source's format / default / precision signature changes."""
    s = sig(signature)
    old = state.setdefault("sig", {}).get(source)
    if old and old["sig"] != s:
        alert("FORMAT", source, f"{what} changed: {old['desc']} -> {json.dumps(signature, default=str)[:300]}")
    state["sig"][source] = {"sig": s, "desc": json.dumps(signature, default=str)[:300]}


def revised(state: dict, source: str, period: str, value: float | None, tol: float) -> None:
    """(i) completed-period revision: a value already recorded for a closed period changes."""
    if value is None:
        return
    book = state.setdefault("periods", {}).setdefault(source, {})
    old = book.get(period)
    if old is not None and abs(old - value) > tol:
        alert("REVISION", source, f"{period}: {old:.6g} -> {value:.6g}")
    book[period] = value
    if len(book) > 400:
        for k in sorted(book)[:-400]:
            del book[k]


# ----------------------------------------------------------------------------------------------------------------- sources
def openrouter(state: dict) -> dict:
    out = {}
    views = {v: get(OR + "model-rankings-chart", None if v == "default" else {"byok": v}) for v in ("default", "organic", "all")}
    wk = {}
    for v, j in views.items():
        d = (j or {}).get("data") if isinstance(j, dict) else None
        if not isinstance(d, dict) or not d.get("data"):
            out[f"chart_{v}"] = {"error": j.get("_error") or j.get("_status") if isinstance(j, dict) else "bad"}; continue
        bars = d["data"]
        wk[v] = {b["x"]: sum(b["ys"].values()) / 1e12 for b in bars[-3:]}
        top = sorted(((k, x) for k, x in bars[-1]["ys"].items() if k != "Others"), key=lambda kv: -kv[1])[:3]
        out[f"chart_{v}"] = {"cachedAt": d.get("cachedAt"), "keys": sorted(d.keys()), "byok_field": d.get("significantByokSeries") is not None,
                             "n_models": len(bars[-1]["ys"]), "weeks_T": {k: round(x, 4) for k, x in wk[v].items()},
                             "top_models_T": [[k, round(x / 1e12, 3)] for k, x in top]}
    if {"default", "organic", "all"} <= wk.keys():
        cur = max(wk["all"]); prev = sorted(wk["all"])[-2]
        dflt = wk["default"].get(cur)
        match = "organic" if abs(dflt - wk["organic"].get(cur, -1)) < 1e-6 * max(dflt, 1) else (
            "all" if abs(dflt - wk["all"][cur]) < 1e-6 * max(dflt, 1) else "neither")
        out["default_matches"] = match
        fmt_change(state, "openrouter_chart", {"default_view": match, "keys": out["chart_default"]["keys"]}, "default view / keys")
        r_cur = wk["organic"][cur] / wk["all"][cur] if wk["all"][cur] else None
        r_prev = wk["organic"].get(prev, 0) / wk["all"][prev] if wk["all"].get(prev) else None
        out["organic_over_all"] = {cur: r_cur, prev: r_prev}
        # "top model / top lab by usage" markets (KXTOPUSAGEAI, KXTOP3AID) flip if the #1 differs between the two views
        t_org, t_all = (out["chart_organic"].get("top_models_T") or [[None]])[0][0], (out["chart_all"].get("top_models_T") or [[None]])[0][0]
        out["top_model"] = {"organic": t_org, "all": t_all}
        fmt_change(state, "openrouter_top_model_views", {"same_top_model": t_org == t_all}, "#1 model organic vs byok=all")
        diverge(state, "or_ratio_prev", r_prev, 0.02, "openrouter_tokens", f"organic/all week {prev}")
        diverge(state, "or_ratio_cur", r_cur, 0.08, "openrouter_tokens", f"organic/all week {cur} (partial)")
        # completed week revisions (prev week is complete once the current week exists)
        for v in ("all", "organic"):
            revised(state, f"openrouter_tokens_{v}", prev, wk[v].get(prev), tol=0.002 * wk[v].get(prev, 1))
    # market share + open/closed: are they BYOK-filterable yet? (2026-11-21: both ignore byok -> BYOK-inclusive only)
    shares = {}
    for ep in ("author-market-share", "open-closed"):
        for v in ("default", "organic", "all"):
            j = get(OR + ep, None if v == "default" else {"byok": v})
            d = (j or {}).get("data") if isinstance(j, dict) else None
            if not isinstance(d, dict) or not d.get("data"):
                shares[f"{ep}_{v}"] = {"error": str(j)[:100]}; continue
            res = {}
            for b in d["data"][-2:]:
                tot = sum(b["ys"].values()) or 1
                keys = AUTHORS if ep == "author-market-share" else list(b["ys"])
                res[b["x"]] = {"total_T": round(tot / 1e12, 4), **{k: round(100 * b["ys"].get(k, 0) / tot, 4) for k in keys}}
            shares[f"{ep}_{v}"] = res
        a, o = shares.get(f"{ep}_all", {}), shares.get(f"{ep}_organic", {})
        if a and o and "error" not in a and "error" not in o:
            cur = max(a)
            gap = abs(a[cur]["total_T"] - o.get(cur, {}).get("total_T", a[cur]["total_T"]))
            fmt_change(state, f"openrouter_{ep}", {"byok_filterable": gap > 1e-6 * a[cur]["total_T"]}, "BYOK filter on " + ep)
            prev = sorted(a)[0]
            for k, v in a[prev].items():
                if k != "total_T":
                    revised(state, f"openrouter_{ep}_{k}", prev, v, tol=0.05)
    # The Kalshi share series (KXANTHSHARE/KXGOOGSHARE/KXDEEPSHARE...) settle on "text market share by model author" = TEXT
    # REQUEST share (modality-chart?routeSegment=text -> marketShareData); verified 2026-11-21: week 11-09 anthropic 6.27 ->
    # KXANTHSHARE-26NOV16 settled 6.3. byok is ignored there today (BYOK-inclusive); if an organic view appears, shares jump
    # (tencent's 13.2% BYOK spike in week 11-16 would vanish) -> FORMAT alert on 'byok_filterable'.
    tx = {}
    for v in ("default", "organic", "all"):
        p = {"routeSegment": "text"} | ({} if v == "default" else {"byok": v})
        j = get(OR + "modality-chart", p)
        d = (j or {}).get("data") if isinstance(j, dict) else None
        msd = d.get("marketShareData") if isinstance(d, dict) else None
        if not msd:
            tx[v] = {"error": str(j)[:100]}; continue
        res = {}
        for b in msd[-2:]:
            tot = sum(b["ys"].values()) or 1
            res[b["x"]] = {"total_req": tot, **{k: round(100 * x / tot, 4) for k, x in b["ys"].items()}}
        tx[v] = res
    if all("error" not in tx.get(v, {"error": 1}) for v in ("organic", "all")):
        cur = max(tx["all"])
        filt = tx["organic"].get(cur, {}).get("total_req") != tx["all"][cur]["total_req"]
        fmt_change(state, "openrouter_text_request_share", {"byok_filterable": filt, "default_eq_all": tx["default"].get(cur) == tx["all"][cur]},
                   "BYOK filter / default on text request share (Kalshi share series source)")
        prev = sorted(tx["all"])[0]
        for k, x in tx["all"][prev].items():
            if k != "total_req":
                revised(state, f"openrouter_text_share_{k}", prev, x, tol=0.05)
    out["text_request_share"] = tx
    out["shares"] = shares
    return out


def vercel(state: dict) -> dict:
    out = {}
    today = datetime.now(timezone.utc).date()
    f, t = (today - timedelta(days=4)).isoformat(), today.isoformat()
    lab = get("https://vercel.com/api/ai/leaderboard-export", {"dataset": "labs", "from": f, "to": t, "format": "json"})
    wc = get("https://vercel.com/api/ai/v5/gateway-model-leaderboard", {"groupBy": "weightClass", "from": f, "to": t})
    L, W = {}, {}
    if isinstance(lab, dict) and lab.get("rows"):
        for r in lab["rows"]:
            L[(r["date"], r["name"], r["metric"])] = r["share_percent"]
        fmt_change(state, "vercel_labs_export", {"keys": sorted(k for k in lab if k != "rows"),
                                                  "metric_modality": sorted({(r["metric"], r["modality"]) for r in lab["rows"]})}, "export schema")
    else:
        out["labs_error"] = str(lab)[:200]
    if isinstance(wc, list):
        for o in wc:
            for k, v in o.get("chef_values", []):
                W[(o["day"][:10], k, o["metric"])] = v
        fmt_change(state, "vercel_weightclass", {"metrics": sorted({o["metric"] for o in wc}),
                                                  "classes": sorted({k for o in wc for k, _ in o.get("chef_values", [])})}, "weightClass schema")
    else:
        out["wc_error"] = str(wc)[:200]
    # second definition of the lab shares: the page's own endpoint (groupBy=chef; 'cost' there = 'spend' in the export)
    ch = get("https://vercel.com/api/ai/v5/gateway-model-leaderboard", {"groupBy": "chef", "from": f, "to": t})
    C = {}
    if isinstance(ch, list):
        for o in ch:
            for k, v in o.get("chef_values", []):
                C[(o["day"][:10], k, "spend" if o["metric"] == "cost" else o["metric"])] = v
    vals, gaps = {}, {}
    done_day = (today - timedelta(days=1)).isoformat()
    for s, (src, name, metric) in VERCEL.items():
        tab = W if src == "W" else L
        vals[s] = {d: round(v, 4) for (d, n, m), v in tab.items() if n == name and m == metric}
        # Kalshi settles daily share to 1 decimal; a past day moving after the fact is a vintage change
        for d, v in vals[s].items():
            if d <= (today - timedelta(days=2)).isoformat():
                revised(state, f"vercel_{s}", d, v, tol=0.05)
        if src == "L" and (done_day, name, metric) in L and (done_day, name, metric) in C:
            gaps[s] = round(C[(done_day, name, metric)] - L[(done_day, name, metric)], 4)
            diverge(state, f"vercel_page_minus_export_{s}", gaps[s], 0.1, "vercel", f"{s} page-export {done_day} (pp)")
    out["values"], out["page_minus_export"] = vals, gaps
    return out


def aaa(state: dict) -> dict:
    h = get("https://gasprices.aaa.com/", as_json=False)
    if not isinstance(h, str):
        return {"error": str(h)[:200]}
    m = re.search(r"Price as of\s*(\d{1,2}/\d{1,2}/\d{2})", h)
    head = re.findall(r"<th>([A-Za-z0-9\- ]+)</th>", h)[:6]
    rows = {}
    for lab in ("Current", "Yesterday", "Week Ago", "Month Ago", "Year Ago"):
        mm = re.search(lab + r" Avg\.</td>((?:\s*<td>\$[0-9.]+</td>)+)", h)
        if mm:
            rows[lab] = re.findall(r"\$([0-9.]+)", mm.group(1))
    if not m or "Current" not in rows:
        alert("FORMAT", "aaa", "could not parse 'Price as of' / Current Avg. row (page format changed?)")
        return {"error": "parse", "len": len(h)}
    asof = datetime.strptime(m.group(1), "%m/%d/%y").date()
    dec = len(rows["Current"][0].split(".")[1])
    fmt_change(state, "aaa", {"decimals": dec, "grades": head, "rows": sorted(rows)}, "precision / grades / rows")
    reg = {k: float(v[0]) for k, v in rows.items()}
    revised(state, "aaa_regular", asof.isoformat(), reg["Current"], tol=0.00005)
    if "Yesterday" in reg:  # yesterday's print as shown today vs what we logged yesterday as Current
        revised(state, "aaa_regular", (asof - timedelta(days=1)).isoformat(), reg["Yesterday"], tol=0.00005)
    return {"asof": asof.isoformat(), "decimals": dec, "grades": head, "rows": rows}


def tsa(state: dict) -> dict:
    # tsa.gov answers 403 to scripts; we do not work around that. The first print is what Kalshi records as expiration_value.
    j = get(f"{KX}/markets", {"series_ticker": "KXTSAD", "status": "settled", "limit": 200})
    firsts = {}
    for mk in (j.get("markets") or []) if isinstance(j, dict) else []:
        firsts.setdefault(mk["event_ticker"], mk.get("expiration_value"))
    for e, v in firsts.items():
        try:
            revised(state, "kalshi_tsad_first_print", e, float(str(v).replace(",", "")), tol=0.5)
        except (TypeError, ValueError):
            pass
    return {"tsa_gov": "not fetched (403 to scripts)", "kalshi_first_prints": dict(sorted(firsts.items())[-5:])}


def parse_cli(text: str, issued: str) -> dict:
    """NWS CLI product. FINAL = the summary is for a date before the local issuance date (the overnight report, no 'VALID ... AS OF'
    line); partial reports ('VALID TODAY AS OF 0400 PM', 'VALID AS OF 0600 AM') cover only part of the day and are not compared."""
    hdr = re.search(r"CLIMATE SUMMARY FOR ([A-Z]+ \d{1,2} \d{4})", text)
    issue_line = re.search(r"\n\s*\d{3,4} [AP]M [A-Z]{3} [A-Z]{3} ([A-Z]{3} \d{1,2} \d{4})", text)
    valid = re.search(r"VALID ([^\n]*)", text)
    mx = re.search(r"\n\s*MAXIMUM\s+(-?\d+|MM)", text)
    mn = re.search(r"\n\s*MINIMUM\s+(-?\d+|MM)", text)
    date = datetime.strptime(hdr.group(1), "%B %d %Y").date() if hdr else None
    idate = datetime.strptime(issue_line.group(1).title(), "%b %d %Y").date() if issue_line else None
    num = lambda g: int(g.group(1)) if g and g.group(1) != "MM" else None  # noqa: E731
    final = bool(date and idate and idate > date and not (valid and "AS OF" in valid.group(1) and "MIDNIGHT" not in valid.group(1)))
    return {"date": date.isoformat() if date else None, "valid": valid.group(1)[:40] if valid else None, "max": num(mx), "min": num(mn),
            "final": final, "issued": issued}


def weather(state: dict) -> dict:
    out = {"twc": {}, "cli": {}}
    now_et = datetime.now(ET).date()
    twc = {}
    for d in (now_et - timedelta(days=2), now_et - timedelta(days=1), now_et):  # d-2 covers the latest FINAL CLI
        j = get("https://weather.com/kalshi/api/climate/primary", {"date": d.isoformat()})
        if not isinstance(j, dict) or "results" not in j:
            out["twc"][d.isoformat()] = {"error": str(j)[:150]}; continue
        summ = {k: j.get(k) for k in ("officialReports", "revisedReports", "preliminaryReports", "noReports")}
        st = {}
        for r in j["results"]:
            cli = r["station"].get("cliId"); dd = r.get("data") or {}
            st[cli] = {"status": r.get("status"), "max": dd.get("maxTemp"), "min": dd.get("minTemp"), "official": dd.get("isOfficial")}
            if r.get("status") in ("official", "revised"):
                revised(state, f"twc_max_{cli}", d.isoformat(), dd.get("maxTemp") if isinstance(dd.get("maxTemp"), (int, float)) else None, tol=0.5)
                revised(state, f"twc_min_{cli}", d.isoformat(), dd.get("minTemp") if isinstance(dd.get("minTemp"), (int, float)) else None, tol=0.5)
        if (summ.get("revisedReports") or 0) > 0:
            alert("SOURCE_REVISED", "twc", f"{d}: {summ['revisedReports']} revised reports "
                  f"({[c for c, v in st.items() if v['status'] == 'revised'][:10]})")
        unknown = sorted({v["status"] for v in st.values() if v["status"]} - {"official", "revised", "preliminary", "no_report"})
        fmt_change(state, "twc_api", {"keys": sorted(j.keys()), "unknown_statuses": unknown, "n_stations": len(st)}, "TWC display schema")
        twc[d.isoformat()] = st
        out["twc"][d.isoformat()] = {"summary": summ, "e4": {c: st.get(c) for c in E4.values()}}
    for series, loc in E4.items():
        lst = get(f"https://api.weather.gov/products/types/CLI/locations/{loc}")
        g = lst.get("@graph") if isinstance(lst, dict) else None
        if not g:
            out["cli"][loc] = {"error": str(lst)[:100]}; continue
        reps = []
        for p in g[:4]:  # stop at the first FINAL report (overnight, for yesterday); partial reports are logged only
            txt = get(f"https://api.weather.gov/products/{p['id']}")
            if isinstance(txt, dict) and txt.get("productText"):
                c = parse_cli(txt["productText"], p["issuanceTime"]); reps.append(c)
                if c["final"]:
                    break
        out["cli"][loc] = reps
        for c in [c for c in reps if c["final"]]:
            tw = (twc.get(c["date"] or "") or {}).get(loc)
            if not tw or tw.get("max") is None or c.get("max") is None:
                continue
            dmax = tw["max"] - c["max"]
            # NY 11-13 pattern: the TWC display (what the book prices) disagrees with the final CLI -> Kalshi may hold and settle on
            # the corrected value (winners_transfer P4: 3 of 15 holds flipped a >= 85c favourite)
            if abs(dmax) >= 1:
                alert("DEFINITIONS_DISAGREE", f"weather_{loc}", f"{c['date']} TWC {tw['status']} max {tw['max']} vs NWS CLI final {c['max']} "
                      f"(Kalshi {series} settles on TWC; material error -> hold possible)")
            diverge(state, f"twc_minus_cli_{loc}", float(dmax), 1.0, f"weather_{loc}", "TWC-CLI final max")
    return out


def noaa(state: dict) -> dict:
    url = ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/"
           f"1850-{datetime.now(timezone.utc).year}/data.csv")
    t = get(url, as_json=False)
    if not isinstance(t, str):
        return {"error": str(t)[:200]}
    rows = [ln.split(",") for ln in t.splitlines() if re.match(r"^\d{6},", ln)]
    header = [ln for ln in t.splitlines() if not re.match(r"^\d{6},", ln)][:6]
    last = {r[0]: float(r[1]) for r in rows[-24:]}
    fmt_change(state, "noaa_globe", {"header": header[:4], "decimals": max(len(v.split(".")[-1]) for v in [r[1] for r in rows[-12:]])},
               "NOAA CSV header / precision")
    prev_latest = state.get("noaa_latest")
    latest = rows[-1][0] if rows else None
    if prev_latest and latest != prev_latest:
        alert("NEW_PRINT", "noaa_globe", f"first print {latest} = {rows[-1][1]} (log as the Kalshi settlement vintage)")
    state["noaa_latest"] = latest
    for k, v in last.items():
        revised(state, "noaa_globe", k, v, tol=0.0049)
    return {"sha256": hashlib.sha256(t.encode()).hexdigest()[:16], "latest": latest, "last24": last}


# ----------------------------------------------------------------------------------------------------------------- Kalshi
def kalshi(state: dict, series_list: list[str]) -> dict:
    out = {}
    ks = state.setdefault("kalshi", {})
    for s in series_list:
        se = get(f"{KX}/series/{s}")
        meta = (se or {}).get("series") if isinstance(se, dict) else None
        if meta:
            fmt_change(state, f"kalshi_series_{s}", {"sources": meta.get("settlement_sources"), "terms": meta.get("contract_terms_url"),
                                                     "contract": meta.get("contract_url")}, "settlement source / contract URL")
        j = get(f"{KX}/markets", {"series_ticker": s, "status": "open", "limit": 1000})
        mk = (j.get("markets") or []) if isinstance(j, dict) else []
        ev: dict[str, dict] = {}
        for m in mk:
            e = ev.setdefault(m["event_ticker"], {"strikes": set(), "tpl": set(), "rungs": []})
            k = m.get("floor_strike") if m.get("floor_strike") is not None else m.get("cap_strike")
            if k is not None:
                e["strikes"].add(float(k))
            subs = [m.get("yes_sub_title"), m.get("no_sub_title")]
            e["tpl"].add(sig([template(m.get("rules_primary"), subs), template(m.get("rules_secondary"), subs)]))
            e["sec"] = template(m.get("rules_secondary"), subs)
            e["rungs"].append([m["ticker"].split("-")[-1], m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("last_price_dollars"),
                               m.get("volume_fp"), (m.get("created_time") or "")[:16]])
        out[s] = {}
        for e, v in ev.items():
            old = ks.get(e)
            strikes = sorted(v["strikes"])
            if old:
                new = sorted(set(strikes) - set(old["strikes"]))
                if new and old["strikes"]:
                    o = sorted(old["strikes"]); step = statistics.median([b - a for a, b in zip(o, o[1:])]) if len(o) > 1 else 0
                    far = [x for x in new if step and (x < o[0] - 2 * step or x > o[-1] + 2 * step)]
                    alert("KALSHI_NEW_STRIKES", s, f"{e} +{new} (old ladder {o[0]}..{o[-1]}, step {step}){' FAR ' + str(far) if far else ''}")
                if set(v["tpl"]) - set(old["tpl"]):
                    alert("KALSHI_RULES_CHANGED", s, f"{e} rules template changed mid-event; secondary now: {v['sec'][:160]}")
            elif s in OR_SERIES and ks.get("_seen_series_" + s):
                alert("KALSHI_NEW_EVENT", s, f"{e} listed ({len(v['rungs'])} markets) -- check which OpenRouter view it will settle on")
            ks[e] = {"strikes": strikes, "tpl": sorted(v["tpl"]), "seen": datetime.now(timezone.utc).isoformat()[:16]}
            out[s][e] = {"strikes": strikes, "rungs": v["rungs"]}
        ks["_seen_series_" + s] = {"seen": datetime.now(timezone.utc).isoformat()[:16], "strikes": [], "tpl": []}
        # series-level template drift between events (e.g. the 2026-08-14 NWS -> Weather Company switch)
        lastt = state.setdefault("series_tpl", {}).get(s)
        cur_t = sorted({t for v in ev.values() for t in v["tpl"]})
        if lastt and cur_t and not set(cur_t) & set(lastt):
            alert("KALSHI_RULES_CHANGED", s, f"no open event shares the previous rules template (new contract text); e.g. {next(iter(ev.values()))['sec'][:160]}")
        if cur_t:
            state["series_tpl"][s] = cur_t
    # forget closed events
    for e in [e for e, v in ks.items() if v.get("seen", "") < (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()[:16]]:
        del ks[e]
    return out


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    first_run = not state
    snap = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, fn in (("openrouter", openrouter), ("vercel", vercel), ("aaa", aaa), ("tsa", tsa), ("weather", weather), ("noaa", noaa)):
        try:
            snap[name] = fn(state)
        except Exception as e:  # noqa: BLE001
            snap[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            alert("WATCH_ERROR", name, snap[name]["error"])
    try:
        snap["kalshi"] = kalshi(state, OR_SERIES + list(VERCEL) + list(E4) + OTHER_SERIES)
    except Exception as e:  # noqa: BLE001
        snap["kalshi"] = {"error": str(e)[:200]}; alert("WATCH_ERROR", "kalshi", str(e)[:200])
    snap["alerts"] = ALERTS
    snap["first_run"] = first_run
    with SNAP.open("a") as f:
        f.write(json.dumps(snap, default=str) + "\n")
    STATE.write_text(json.dumps(state, default=str))
    if not QUIET:
        o = snap.get("openrouter", {})
        print(snap["ts"], "| OR default=", o.get("default_matches"), "organic/all", {k: round(v, 3) for k, v in (o.get("organic_over_all") or {}).items() if v},
              "| AAA", (snap.get("aaa") or {}).get("asof"), (snap.get("aaa") or {}).get("rows", {}).get("Current", [None])[0],
              "| NOAA latest", (snap.get("noaa") or {}).get("latest"), "| TSA first prints", len((snap.get("tsa") or {}).get("kalshi_first_prints") or {}),
              "| kalshi series", len(snap.get("kalshi") or {}), "| alerts", len(ALERTS), "(first run: baselines only)" if first_run else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
