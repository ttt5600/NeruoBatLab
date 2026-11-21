"""KXTOKENUSE forward paper logger (agent ai_nowcast, R-TOK1 forward spec in notes/ai_nowcast.md s5). READ-ONLY GETs, no orders.

One run = one snapshot: OpenRouter weekly chart (default view AND byok=all, both are logged because the default switched to
BYOK-excluded around 2026-11-21 and which one Kalshi settles on is unknown), plus every open KXTOKENUSE rung's bid/ask/size.
Appends one JSON line per run to results/ai_nowcast/fwd/snapshots.jsonl. Scoring is offline (score_tokenuse_fwd in notes s5).
Intended cadence: hourly (the endpoint updates ~hourly). NOT installed in launchd -- the lead/user decides.
Usage: ../.venv_quant/bin/python scripts/tokenuse_logger.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "ai_nowcast" / "fwd" / "snapshots.jsonl"
OR = "https://openrouter.ai/api/frontend/v1/rankings/model-rankings-chart"
KX = "https://api.elections.kalshi.com/trade-api/v2"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url: str, params: dict | None = None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 -- log and retry; a missing source is recorded as null
            err = str(e)
            time.sleep(2 * (i + 1))
    return {"_error": err}


def weekly_totals(j: dict) -> dict:
    d = j.get("data") if isinstance(j, dict) else None
    if isinstance(d, dict):
        cached, byok, d = d.get("cachedAt"), d.get("significantByokSeries"), d.get("data")
    else:
        cached, byok = None, None
    if not d:
        return {"error": j.get("_error") if isinstance(j, dict) else "bad"}
    tot = {r["x"]: sum(r["ys"].values()) for r in d[-3:]}  # last 3 weekly bars, raw token counts
    return {"cachedAt_ms": cached, "has_byok_field": byok is not None, "weeks": tot}


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    snap = {"ts": ts, "or_default": weekly_totals(get(OR)), "or_byok_all": weekly_totals(get(OR, {"byok": "all"}))}
    mk = get(f"{KX}/markets", {"series_ticker": "KXTOKENUSE", "status": "open", "limit": 1000})
    snap["kalshi"] = [
        {k: m.get(k) for k in ("ticker", "event_ticker", "floor_strike", "strike_type", "yes_bid_dollars", "yes_ask_dollars",
                               "yes_bid_size_fp", "yes_ask_size_fp", "volume_fp", "close_time", "rules_primary")}
        for m in mk.get("markets", [])
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(snap) + "\n")
    print(ts, "rungs", len(snap["kalshi"]), "default", snap["or_default"].get("weeks"), "byok_all", snap["or_byok_all"].get("weeks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
