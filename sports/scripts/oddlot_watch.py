#!/usr/bin/env python3
"""oddlot_watch: daily EDGAR full-text scan for tender / exchange offers that give odd-lot (<100 share) priority.

Why: an issuer tender with odd-lot priority buys ALL shares from holders of <100 shares without proration. When the
offer price (Dutch auction: the FLOOR of the range) is above the market price, 99 shares per brokerage account is a
near-riskless small profit (notes/money_now.md, notes/trading_ways_survey.md #3).

What it does (stdlib only, read-only GETs to sec.gov, no orders):
  1. efts.sec.gov/LATEST/search-index full-text search for several odd-lot phrases in Schedule TO filings
     (SC TO-I, SC TO-T incl. their /A amendments, SC TO-C pre-commencement communications) filed in the last N days.
  2. Dedupes by accession number against the output file, so each filing is appended once.
  3. For new ORIGINAL offers (SC TO-I / SC TO-T / SC TO-C) it fetches the matched document and extracts, by regex:
     offer type (dutch / fixed / exchange), price range, expiration sentence, the odd-lot sentence, and a
     "less than 100 shares may not participate" flag.  Extraction is best-effort: ALWAYS read the filing before acting.
  4. Appends one JSON line per new filing to ~/.claude/quant-sports/data/oddlot/filings.jsonl and prints new ones.

Usage:
  python3 sports/scripts/oddlot_watch.py                 # daily run: look back 4 days (covers weekends)
  python3 sports/scripts/oddlot_watch.py --lookback 60   # backfill
  python3 sports/scripts/oddlot_watch.py --dry-run       # print, do not write
Exit: 0 ok (new or not), 2 on EDGAR/network failure (nothing written for the failed query).
Not installed in launchd by design: the lead installs it (daily, e.g. 07:15 PT).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "fund-research bot (research use)"  # descriptive UA; deliberately no personal e-mail
EFTS = "https://efts.sec.gov/LATEST/search-index"
ARCH = "https://www.sec.gov/Archives/edgar/data"
OUT = Path.home() / ".claude" / "quant-sports" / "data" / "oddlot" / "filings.jsonl"
PHRASES = ['"odd lot"', '"odd-lot"', '"odd lots"', '"fewer than 100 shares"']
# Root forms only: efts matches amendments (/A) through root_forms. Listing "SC TO-I/A" explicitly was tested on
# 2026-11-20 and made efts return ONLY amendments (originals such as the MYPS offer vanished).
FORMS = ["SC TO-I", "SC TO-T", "SC TO-C"]
ORIGINAL_FORMS = {"SC TO-I", "SC TO-T", "SC TO-C"}
SLEEP = 0.25  # SEC fair-access limit is 10 req/s; stay well under


def get(url: str, retries: int = 3) -> bytes:
    last = None
    for k in range(retries):
        time.sleep(SLEEP * (1 + 2 * k))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def search(phrase: str, start: str, end: str) -> list[dict]:
    """All hits (paginated, 100/page) for one phrase over the Schedule TO forms."""
    hits, frm = [], 0
    while True:
        q = urllib.parse.urlencode({"q": phrase, "forms": ",".join(FORMS), "dateRange": "custom",
                                    "startdt": start, "enddt": end, "from": frm})
        d = json.loads(get(f"{EFTS}?{q}"))
        page = d.get("hits", {}).get("hits", [])
        hits.extend(page)
        total = d.get("hits", {}).get("total", {}).get("value", 0)
        frm += len(page)
        if not page or frm >= total or frm >= 1000:
            return hits


def to_text(b: bytes) -> str:
    t = b.decode("utf-8", "ignore")
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t))


def first(pattern: str, text: str, width: int = 260) -> str:
    m = re.search(pattern, text, re.I)
    if not m:
        return ""
    return text[max(0, m.start() - 40): m.end() + width].strip()


def extract(text: str) -> dict:
    """Best-effort terms from an offer document. Every field may be empty; read the filing before acting."""
    head = text[:6000]
    money = r"\$\s?\d{1,4}(?:,\d{3})*(?:\.\d{1,4})?"
    # "not less than $4.30 per share and not more than $5.50" / "not greater than $3.40 per share nor less than $2.85"
    rng = re.search(rf"not (?:less|greater|more) than ({money})[^$]{{0,60}}?(?:nor|and|or)\s+(?:not\s+)?"
                    rf"(?:less|greater|more) than ({money})", head, re.I | re.S)
    lo = hi = None
    if rng:
        a, b = (float(x.replace("$", "").replace(",", "").strip()) for x in rng.groups())
        lo, hi = min(a, b), max(a, b)
    fixed = None
    if lo is None:
        m = re.search(rf"(?:purchase price of|at a price of|at|for)\s+({money})\s+(?:net\s+)?per\s+(?:share|ordinary share"
                      rf"|american depositary share)", head, re.I)
        if m:
            fixed = float(m.group(1).replace("$", "").replace(",", "").strip())
    low = head.lower()
    if "exchange" in low[:1500] and ("offer to exchange" in low or "exchange offer" in low):
        kind = "exchange"
    elif lo is not None or "dutch auction" in low:
        kind = "dutch"
    elif fixed is not None:
        kind = "fixed"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "price_lo": lo, "price_hi": hi, "price_fixed": fixed,
        "expiration": first(r"(?:will )?expire[sd]?\b.{0,40}?(?:p\.m\.|a\.m\.|midnight)", head, 160),
        "expires_date": (lambda m: m.group(1) if m else "")(re.search(
            r"expire.{0,220}?((?:January|February|March|April|May|June|July|August|September|October|November|"
            r"December) \d{1,2}, \d{4})", head, re.I | re.S)),
        "oddlot_sentence": first(r"odd[ -]lots?", text, 300),
        "under100_excluded": bool(re.search(r"fewer than 100 shares[^.]{0,80}(?:may not|cannot|will not be able to) "
                                            r"participate", text, re.I)),
        "financing_condition": "not subject to any financing condition" not in low
                               and "not conditioned upon any financing" not in low
                               and "financing condition" in low,
    }


def load_seen(path: Path) -> set[str]:
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                seen.add(json.loads(line)["adsh"])
            except Exception:  # noqa: BLE001 - tolerate a damaged line
                continue
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lookback", type=int, default=4, help="days back from today to search (default 4)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-docs", action="store_true", help="skip fetching documents (metadata only)")
    a = ap.parse_args()

    today = dt.date.today()
    start, end = (today - dt.timedelta(days=a.lookback)).isoformat(), today.isoformat()
    seen = load_seen(a.out)
    by_adsh: dict[str, dict] = {}
    failed = []
    for ph in PHRASES:
        try:
            hits = search(ph, start, end)
        except RuntimeError as e:
            failed.append(str(e))
            continue
        for h in hits:
            s = h["_source"]
            adsh = s["adsh"]
            doc = h["_id"].split(":", 1)[1] if ":" in h["_id"] else ""
            rec = by_adsh.setdefault(adsh, {
                "adsh": adsh, "file_date": s.get("file_date"), "form": s.get("form"),
                "company": (s.get("display_names") or [""])[0], "ciks": s.get("ciks", []),
                "tickers": re.findall(r"\(([A-Z][A-Z0-9.\-]*(?:, [A-Z][A-Z0-9.\-]*)*)\)\s+\(CIK",
                                      (s.get("display_names") or [""])[0]),
                "docs": [], "phrases": []})
            if doc and doc not in rec["docs"]:
                rec["docs"].append(doc)
            if ph not in rec["phrases"]:
                rec["phrases"].append(ph)

    new = [r for adsh, r in sorted(by_adsh.items(), key=lambda kv: kv[1]["file_date"] or "") if adsh not in seen]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in new:
        cik = str(int(r["ciks"][0])) if r["ciks"] else ""
        acc = r["adsh"].replace("-", "")
        r["seen_utc"] = now
        r["window"] = [start, end]
        r["has_ticker"] = bool(r["tickers"])
        r["index_url"] = f"{ARCH}/{cik}/{acc}/" if cik else ""
        r["doc_urls"] = [f"{ARCH}/{cik}/{acc}/{d}" for d in r["docs"]] if cik else []
        if not a.no_docs and r["form"] in ORIGINAL_FORMS and r["has_ticker"] and r["doc_urls"]:
            # prefer the offer-to-purchase / offer-to-exchange exhibit if it is among the matched docs
            pick = sorted(r["doc_urls"], key=lambda u: (not re.search(r"a1a|a1i|a\)\(1\)|ex99a1|exa1a", u, re.I), u))[0]
            try:
                r["terms"] = extract(to_text(get(pick)))
                r["terms"]["doc"] = pick
            except RuntimeError as e:
                r["terms"] = {"error": str(e)[:200]}
        r.pop("docs", None)

    for r in new:
        t = r.get("terms", {})
        price = (f"${t['price_lo']}-{t['price_hi']}" if t.get("price_lo") is not None
                 else (f"${t['price_fixed']}" if t.get("price_fixed") is not None else ""))
        print(f"NEW {r['file_date']} {r['form']:<10} {','.join(r['tickers']) or '-':<12} {r['company'][:50]:<50} "
              f"{t.get('kind', '')} {price} exp={t.get('expires_date', '')}"
              f"{' UNDER100-EXCLUDED' if t.get('under100_excluded') else ''}")
    print(f"oddlot_watch {start}..{end}: {len(by_adsh)} filings matched, {len(new)} new, "
          f"{sum(r['has_ticker'] for r in new)} new with a ticker; failed queries: {len(failed)}")

    if new and not a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    if failed:
        print("FAILED:", *failed, sep="\n  ", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
