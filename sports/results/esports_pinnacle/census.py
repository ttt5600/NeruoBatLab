"""Read-only census of Kalshi esports series (open markets now + rules text + settled-volume stats).
Run: ../.venv_quant/bin/python results/esports_pinnacle/census.py  (from sports/). Writes census.json next to this file."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget  # noqa: E402

SERIES = ["KXCS2GAME", "KXLOLGAME", "KXVALORANTGAME", "KXDOTA2GAME", "KXCS2MAP", "KXLOLMAP", "KXVALORANTMAP",
          "KXDOTA2MAP", "KXR6GAME", "KXCODGAME", "KXROCKETLEAGUEGAME", "KXMLBBGAME", "KXDOTA2GAME3WAY"]


def markets(series, status):
    out, cur = [], None
    for _ in range(10):
        p = {"series_ticker": series, "status": status, "limit": 200}
        if cur:
            p["cursor"] = cur
        j = kget("/markets", p) or {}
        out += j.get("markets", [])
        cur = j.get("cursor")
        if not cur or status == "settled":
            break
    return out


res = {}
for s in SERIES:
    op = markets(s, "open")
    st = markets(s, "settled")
    vol = np.array([float(m.get("volume_fp") or 0) for m in st], float)
    ex = (op or st or [{}])[0]
    r = {"open_markets": len(op), "open_events": len({m["event_ticker"] for m in op}),
         "settled_n": len(st), "settled_vol_median": float(np.median(vol)) if len(vol) else None,
         "settled_vol_sum": float(vol.sum()),
         "settled_results": {k: sum(m.get("result") == k for m in st) for k in ("yes", "no", "scalar")},
         "scalar_values": [m.get("settlement_value_dollars") for m in st if m.get("result") == "scalar"][:20],
         "settled_span": [min((m.get("close_time") or "") for m in st) if st else None, max((m.get("close_time") or "") for m in st) if st else None],
         "rules_primary": ex.get("rules_primary"), "rules_secondary": ex.get("rules_secondary"),
         "example": {k: ex.get(k) for k in ("ticker", "event_ticker", "title", "yes_sub_title", "open_time", "close_time",
                                              "expected_expiration_time", "latest_expiration_time", "can_close_early",
                                              "early_close_condition", "settlement_timer_seconds")},
         "open_sample": [{k: m.get(k) for k in ("ticker", "yes_sub_title", "yes_bid_dollars", "yes_ask_dollars",
                                                 "volume_fp", "open_interest_fp", "close_time", "expected_expiration_time")}
                         for m in op[:6]]}
    res[s] = r
    print(s, "open", r["open_markets"], "/", r["open_events"], "ev | settled", r["settled_n"], "med vol",
          r["settled_vol_median"], r["settled_results"])
(HERE / "census.json").write_text(json.dumps(res, indent=1, default=str))
