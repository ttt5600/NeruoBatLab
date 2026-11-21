"""pool_census stage 1: enumerate every settled Kalshi market (non-MVE) closing in the last N days. Read-only public GETs
through scripts/kx_throttle (shared cross-process slot). Appends compact rows to OUT.jsonl page by page; resumable
(cursor kept in OUT.cursor). Usage: python enum_settled.py OUT_PREFIX DAYS [mve_filter=exclude|only]"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget

KEEP = ("ticker", "event_ticker", "open_time", "close_time", "settlement_ts", "result", "volume_fp", "last_price_dollars",
        "previous_yes_bid_dollars", "previous_yes_ask_dollars", "market_type", "strike_type", "expiration_value", "status")

def main():
    pre, days = sys.argv[1], int(sys.argv[2])
    mve = sys.argv[3] if len(sys.argv) > 3 else "exclude"
    out, curf, metaf = Path(pre + ".jsonl"), Path(pre + ".cursor"), Path(pre + ".meta.json")
    if metaf.exists():
        meta = json.load(open(metaf))
    else:
        meta = {"t_run": time.time(), "min_close_ts": int(time.time() - days * 86400), "mve_filter": mve, "pages": 0, "rows": 0}
        json.dump(meta, open(metaf, "w"))
    cur = curf.read_text().strip() if curf.exists() else None
    if cur == "DONE":
        print("already done"); return
    with open(out, "a") as f:
        while True:
            p = {"status": "settled", "min_close_ts": meta["min_close_ts"], "limit": 1000, "mve_filter": mve}
            if cur: p["cursor"] = cur
            d = kget("/markets", p)
            if d is None:
                print("4xx; stopping", flush=True); break
            ms = d.get("markets", [])
            for m in ms:
                f.write(json.dumps({k: m.get(k) for k in KEEP}) + "\n")
            f.flush()
            meta["pages"] += 1; meta["rows"] += len(ms)
            cur = d.get("cursor")
            curf.write_text(cur or "DONE"); json.dump(meta, open(metaf, "w"))
            if meta["pages"] % 25 == 0:
                print(meta["pages"], meta["rows"], ms[-1]["close_time"] if ms else None, flush=True)
            if not cur or not ms:
                curf.write_text("DONE"); break
            stop = os.environ.get("STOP_BEFORE")
            if stop and ms[-1]["close_time"] < stop:
                print("reached STOP_BEFORE", stop, flush=True); break
    print("done", meta, flush=True)

if __name__ == "__main__":
    main()
