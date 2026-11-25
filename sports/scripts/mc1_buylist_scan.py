#!/usr/bin/env python3
"""mc1_buylist_scan: daily READ-ONLY scan for MTG cards whose Card Kingdom (CK) NM cash buylist price beats the cheapest
US marketplace retail (TCGplayer AND Manapool) after tax and shipping -- lead MC1 / pre-registered test T-MTG
(sports/notes/small_capacity_census.md, sections S6, T-MTG and "MC1 scanner").

Signal (fixed by T-MTG; reproduces results/small_capacity_census/scripts/mtg_name_strict.py EXACTLY, then applies the
pre-registered exclusions):
  price   = max(TCGplayer retail, Manapool retail)            (MTGJSON AllPricesToday, latest date per finish)
  cost    = price x (1 + 9.25% CA sales tax) + $1 shipping share per card
  net     = CK cash buy (MTGJSON CK buylist) - cost;          FLAG iff net >= $3, 0 < price < $50
  naming  = CK pricelist rows with the same (buy, retail, foil) tier -> Scryfall /cards/collection -> keep the card
            iff EXACTLY ONE row's Scryfall TCGplayer price (usd / usd_foil) equals the MTGJSON TCGplayer price to the cent
  exclude = sets released before 1995-01-01 and Reserved List cards (vintage condition artefacts, census cross-check 1);
            CK qty_buying < 1.  Unnamed (unresolved) signal rows are counted but are NOT flags (nobody can spot-check them).
  Not in net: outbound postage to CK (~$5-8 per batch [U]) and CK grading downgrades (foil EX pays 75% of NM).

Data sources (all public, read-only, no keys; checked 2026-11-24):
  - MTGJSON https://mtgjson.com/api/v5/AllPricesToday.json.xz (~4.4 MB). Licence MIT (mtgjson.com/license).
    robots.txt disallows /api/v5/*.json (the raw JSON) but NOT the compressed archives -> we fetch only the .xz.
  - Card Kingdom https://api.cardkingdom.com/api/v2/pricelist with Accept-Encoding: gzip (~10 MB vs 69 MB raw).
    api.cardkingdom.com/robots.txt = "User-agent: * / Disallow:" (nothing disallowed). No published API terms found
    (www.cardkingdom.com sits behind a Cloudflare challenge); one GET per day.
  - Scryfall https://api.scryfall.com/cards/collection (POST, <= 75 ids per call). scryfall.com/docs/api: accurate
    User-Agent + Accept header required; /cards/collection hard limit 2 req/s (500 ms) -- stricter than the general
    10 req/s; a 429 locks the client out for 30 s. We space calls >= 0.6 s and back off 35 s on a 429.
  Downloads go to a fresh temp dir that is deleted on exit; nothing raw is kept.

Outputs (default --out ~/.claude/quant-sports/data/mc1):
  <date>.json   every flag (name, set, number, foil, TCGplayer price + listing URL, Manapool price, CK NM cash buy,
                CK qty_buying, cost and net per copy and at full qty) + unresolved / excluded signal rows + data stamps
  <date>.md     click-through table for the user's spot-check, plus a seeded random sample of 10 flags
  daily.jsonl   one appended summary line per run (date, n_flags, net/cost at 1 copy and at full qty, data stamps).
                A manual re-run for the same date appends another line: judge on the LAST line per date.

Usage:
  python3 -I sports/scripts/mc1_buylist_scan.py                         # live run, PT date, default --out
  python3 -I sports/scripts/mc1_buylist_scan.py --out /tmp/mc1test      # live run into a scratch dir
  python3 -I sports/scripts/mc1_buylist_scan.py --mtgjson-file AllPricesToday.json[.xz] --ck-file ck.json[.gz]
                                                                         # cached inputs (Scryfall is still live)
Exit: 0 ok (including 0 flags), 2 network / download failure (nothing written), 1 anything else (a bug).
Stdlib only. No orders, no accounts, no credentials. Installed by scripts/install_mc1_agent.sh (qs_job.sh mc1).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import lzma
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---- fixed by the census / T-MTG (do not loosen) -------------------------------------------------------------------
TAX = 0.0925           # CA sales tax assumption (7.25-10.25%; census midpoint)
SHIP = 1.00            # per-card share of seller shipping
MARGIN = 3.00          # minimum net per copy
PRICE_CAP = 50.0       # max(TCG, Manapool) must be < this
PRE_CUTOFF = "1995-01-01"   # exclude prints released before this (vintage condition artefacts)
SAMPLE_N = 10          # spot-check sample size (T-MTG: "10 random flagged cards")

MTGJSON_URL = "https://mtgjson.com/api/v5/AllPricesToday.json.xz"
CK_URL = "https://api.cardkingdom.com/api/v2/pricelist"
SCRY_URL = "https://api.scryfall.com/cards/collection"
CK_BASE = "https://www.cardkingdom.com/"
UA = "quant-sports-mc1-scanner/1.0 (read-only daily MTG price research; python-urllib)"
SCRY_SPACING = 0.6     # s between /cards/collection calls (limit 2/s)
SCRY_BATCH = 75        # Scryfall maximum identifiers per call
MIN_CK_ROWS = 10000     # a full CK pricelist had 155,033 rows (2026-11-24); fewer = partial / error response
OUT_DEFAULT = Path.home() / ".claude" / "quant-sports" / "data" / "mc1"


class NetError(RuntimeError):
    """Network / download failure -> exit 2 (the launchd wrapper retries on a later run)."""


# ---- pure logic ------------------------------------------------------------------------------------------------------
def latest(series: dict):
    """MTGJSON price series {date: value} -> value at the latest date (None if empty)."""
    return series[max(series)] if series else None


def signal_candidates(data: dict) -> list[dict]:
    """Strict T-MTG signal over MTGJSON AllPricesToday['data'] (same arithmetic as mtg_name_strict.py)."""
    out = []
    for uuid, rec in data.items():
        p = rec.get("paper") or {}
        k = p.get("cardkingdom") or {}
        t = p.get("tcgplayer") or {}
        m = p.get("manapool") or {}
        for fin in ("normal", "foil"):
            b = (k.get("buylist") or {}).get(fin) or {}
            r = (t.get("retail") or {}).get(fin) or {}
            mm = (m.get("retail") or {}).get(fin) or {}
            cr = (k.get("retail") or {}).get(fin) or {}
            if not b or not r or not mm:
                continue
            tv, mv = latest(r) or 0, latest(mm) or 0
            px = max(tv, mv)
            if px <= 0 or px >= PRICE_CAP:
                continue
            bv = latest(b)
            if bv is None:
                continue
            cost = px * (1 + TAX) + SHIP
            net = bv - cost
            if net >= MARGIN:
                out.append(dict(uuid=uuid, foil=fin == "foil", ck_buy=bv, tcg=tv, mp=mv,
                                ck_retail=latest(cr), price=px, cost=round(cost, 2), net=round(net, 2)))
    return out


def tier_key(buy, retail, foil: bool):
    return (round(float(buy), 2), round(float(retail), 2), bool(foil))


def ck_index(rows: list[dict], keys: set | None = None) -> dict:
    """CK v2 pricelist rows -> {(price_buy, price_retail, is_foil): [rows]} (only `keys` if given)."""
    idx = defaultdict(list)
    for r in rows:
        try:
            key = tier_key(r["price_buy"], r["price_retail"], r.get("is_foil") == "true")
        except (KeyError, TypeError, ValueError):
            continue
        if keys is None or key in keys:
            idx[key].append(r)
    return idx


def cand_key(c: dict):
    return None if c["ck_retail"] is None else tier_key(c["ck_buy"], c["ck_retail"], c["foil"])


def needed_ids(cands: list[dict], idx: dict) -> list[str]:
    ids = set()
    for c in cands:
        for r in idx.get(cand_key(c), []) if cand_key(c) else []:
            if r.get("scryfall_id"):
                ids.add(r["scryfall_id"])
    return sorted(ids)


def scry_summary(card: dict) -> dict:
    pr = card.get("prices") or {}
    return dict(name=card.get("name"), set=card.get("set"), set_name=card.get("set_name"),
                number=card.get("collector_number"), released=card.get("released_at"), reserved=card.get("reserved"),
                usd=pr.get("usd"), usd_foil=pr.get("usd_foil"), tcgplayer_id=card.get("tcgplayer_id"),
                scryfall_uri=(card.get("scryfall_uri") or "").split("?")[0] or None)


def tcg_url(s: dict, foil: bool) -> str:
    q = urllib.parse.urlencode({"Language": "English", "Printing": "Foil" if foil else "Normal",
                                "Condition": "Near Mint"})
    if s.get("tcgplayer_id"):
        return f"https://www.tcgplayer.com/product/{s['tcgplayer_id']}?{q}"
    return "https://www.tcgplayer.com/search/magic/product?" + urllib.parse.urlencode(
        {"q": s.get("name") or "", "productLineName": "magic"})


def max_ok_price(ck_buy: float) -> float:
    """Highest listing price (with the $1 ship assumption) that still nets >= MARGIN."""
    return round((ck_buy - SHIP - MARGIN) / (1 + TAX), 2)


def resolve(cands: list[dict], idx: dict, scry: dict) -> dict:
    """Name candidates; split into flags / unresolved / excluded. Naming rule = mtg_name_strict.py."""
    flags, unresolved, excluded = [], [], []
    for c in cands:
        hits = []
        key = cand_key(c)
        for r in idx.get(key, []) if key else []:
            s = scry.get(r.get("scryfall_id"))
            if not s:
                continue
            pz = s["usd_foil"] if c["foil"] else s["usd"]
            if pz is not None and abs(float(pz) - c["tcg"]) < 0.005:
                hits.append((r, s))
        if len(hits) != 1:
            unresolved.append(dict(c, n_hits=len(hits),
                                   reason="no_ck_retail" if key is None else ("no_match" if not hits else "ambiguous")))
            continue
        r, s = hits[0]
        qb = int(r.get("qty_buying") or 0)
        row = dict(name=r["name"], set=s["set"], set_name=s["set_name"], number=s["number"], foil=c["foil"],
                   ck_edition=r.get("edition"), variation=r.get("variation") or "", released=s["released"],
                   reserved=bool(s["reserved"]),
                   tcg_price=c["tcg"], tcg_url=tcg_url(s, c["foil"]), manapool_price=c["mp"], buy_price=c["price"],
                   max_ok_price=max_ok_price(c["ck_buy"]),
                   ck_buy_nm_cash=c["ck_buy"], ck_qty_buying=qb, ck_retail=c["ck_retail"],
                   ck_qty_retail=r.get("qty_retail"), ck_sku=r.get("sku"),
                   ck_url=CK_BASE + str(r.get("url") or "").replace("\\/", "/").lstrip("/"),
                   scryfall_uri=s["scryfall_uri"], scryfall_id=r.get("scryfall_id"), mtgjson_uuid=c["uuid"],
                   cost=c["cost"], net=c["net"],
                   cost_qty=round(c["cost"] * qb, 2), net_qty=round(c["net"] * qb, 2))
        why = []
        if (s["released"] or "9999") < PRE_CUTOFF:
            why.append("pre1995")
        if s["reserved"]:
            why.append("reserved_list")
        if qb < 1:
            why.append("qty_buying_0")
        if why:
            excluded.append(dict(row, excluded=why))
        else:
            flags.append(row)
    flags.sort(key=lambda x: (-x["net"], x["name"]))
    return dict(flags=flags, unresolved=unresolved, excluded=excluded)


def summarize(date: str, res: dict, n_signal: int, stamps: dict) -> dict:
    """Census convention (notes S6 RESULT): net totals sum the per-card nets rounded to cents; cost totals sum the
    UNROUNDED per-card costs (price x 1.0925 + $1), so 2026-11-24 reproduces $101.76 / $346.92 and $423.56 / $1,515.79."""
    f = res["flags"]
    exc = res["excluded"]
    raw_cost = lambda x: x["buy_price"] * (1 + TAX) + SHIP  # noqa: E731
    return dict(date=date, n_flags=len(f), n_signal=n_signal, n_unresolved=len(res["unresolved"]),
                n_excluded_pre1995=sum("pre1995" in e["excluded"] for e in exc),
                n_excluded_reserved=sum("reserved_list" in e["excluded"] for e in exc),
                n_excluded_qty0=sum("qty_buying_0" in e["excluded"] for e in exc),
                net_1copy=round(sum(x["net"] for x in f), 2), cost_1copy=round(sum(raw_cost(x) for x in f), 2),
                net_fullqty=round(sum(x["net_qty"] for x in f), 2),
                cost_fullqty=round(sum(raw_cost(x) * x["ck_qty_buying"] for x in f), 2),
                units_fullqty=sum(x["ck_qty_buying"] for x in f), **stamps)


def spot_sample(date: str, flags: list[dict]) -> list[int]:
    """Indices (0-based, into the net-sorted flag list) of a reproducible random sample (seed = date)."""
    n = len(flags)
    return sorted(random.Random(f"mc1-{date}").sample(range(n), min(SAMPLE_N, n)))


def _money(x) -> str:
    return "" if x is None else f"${x:,.2f}"


def render_md(report: dict) -> str:
    s, flags = report["summary"], report["flags"]
    L = [f"# MC1 Card Kingdom buylist scan -- {s['date']} (PT)", "",
         f"Data: MTGJSON AllPricesToday {s.get('mtgjson_date')} ({s.get('mtgjson_version')}); CK pricelist created "
         f"{s.get('ck_created_at')}; Scryfall names fetched {s.get('run_utc')}.", "",
         f"Signal (T-MTG, fixed): CK NM cash buy >= max(TCGplayer, Manapool) x {1 + TAX:.4f} + ${SHIP:.2f} + "
         f"${MARGIN:.2f}; max(TCG, Manapool) < ${PRICE_CAP:.0f}; CK qty_buying >= 1; named uniquely; no pre-1995 "
         "prints, no Reserved List.", "",
         f"**{s['n_flags']} flags.** One copy each: net **{_money(s['net_1copy'])}** on cost {_money(s['cost_1copy'])}. "
         f"At full CK qty_buying ({s['units_fullqty']} units): net **{_money(s['net_fullqty'])}** on cost "
         f"{_money(s['cost_fullqty'])}.", "",
         f"Signal rows before naming {s['n_signal']}; unresolved (no unique name) {s['n_unresolved']}; excluded: "
         f"pre-1995 {s['n_excluded_pre1995']}, Reserved List {s['n_excluded_reserved']}, CK qty_buying 0 "
         f"{s['n_excluded_qty0']}.", "",
         "Not in net: outbound postage to CK (~$5-8 per batch) and CK grading (foil EX pays 75%, VG 50%, G 30% of NM).",
         ""]
    if s.get("warnings"):
        L += ["**Warnings:** " + "; ".join(s["warnings"]), ""]
    L += ["| # | card | set (number) | finish | TCGplayer | Manapool | price used | max OK price | CK NM cash buy | "
          "CK qty | net/copy | net @ qty |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(flags, 1):
        var = f" ({x['variation']})" if x["variation"] else ""
        L.append(f"| {i} | [{x['name']}]({x['scryfall_uri']}){var} | {x['set_name']} ({x['set'].upper()} "
                 f"#{x['number']}) | {'foil' if x['foil'] else 'nonfoil'} | [{_money(x['tcg_price'])}]({x['tcg_url']}) | "
                 f"{_money(x['manapool_price'])} | {_money(x['buy_price'])} | {_money(x['max_ok_price'])} | "
                 f"[{_money(x['ck_buy_nm_cash'])}]({x['ck_url']}) | {x['ck_qty_buying']} | {_money(x['net'])} | "
                 f"{_money(x['net_qty'])} |")
    if not flags:
        L.append("| - | no flags today | | | | | | | | | | |")
    L += ["", "Columns: *price used* = max(TCGplayer, Manapool), what the net assumes you pay; *max OK price* = highest "
          "listing price (with $1 shipping) that still nets >= $3. TCGplayer links open the product filtered to "
          "English / Near Mint / the right printing; card names link to Scryfall (it also lists other shops).", ""]
    samp = report.get("spot_sample") or []
    L += [f"## Spot-check sample ({len(samp)} random flags, seed = date)", "",
          "For each card: open the TCGplayer link (or Manapool), find the cheapest NEAR MINT English listing in the "
          "right printing, add the seller's shipping. Confirmed iff listing price <= *price used* AND "
          "CK buy - (price x 1.0925 + seller shipping) >= $3. Record price, shipping, seller and a yes/no.", ""]
    for i in samp:
        x = flags[i]
        L.append(f"- [ ] #{i + 1} {x['name']} -- {x['set_name']} #{x['number']} {'foil' if x['foil'] else 'nonfoil'}: "
                 f"price used {_money(x['buy_price'])}, CK buy {_money(x['ck_buy_nm_cash'])} -> found NM $____ + ship "
                 f"$____ -> net $____  ({x['tcg_url']})")
    L.append("")
    return "\n".join(L)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_outputs(report: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    d = report["summary"]["date"]
    _atomic_write(out / f"{d}.json", json.dumps(report, indent=1))
    _atomic_write(out / f"{d}.md", render_md(report))
    line = {k: report["summary"][k] for k in ("date", "n_flags", "net_1copy", "net_fullqty", "cost_1copy",
                                              "cost_fullqty", "units_fullqty", "n_signal", "n_unresolved",
                                              "mtgjson_date", "ck_created_at", "run_utc")}
    with open(out / "daily.jsonl", "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


# ---- network ---------------------------------------------------------------------------------------------------------
def download(url: str, dest: Path, headers: dict | None = None, retries: int = 3, timeout: int = 120) -> dict:
    """Stream url -> dest. Returns response headers. Raises NetError after `retries` failures."""
    last = None
    for k in range(retries):
        if k:
            time.sleep(10 * k)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 16)
                return dict(r.headers)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
    raise NetError(f"download failed after {retries} tries: {url}: {last!r}")


def scry_post(ids: list[str], retries: int = 4) -> dict:
    body = json.dumps({"identifiers": [{"id": x} for x in ids]}).encode()
    hdr = {"User-Agent": UA, "Accept": "application/json", "Content-Type": "application/json"}
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(SCRY_URL, data=body, headers=hdr, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(35 if e.code == 429 else 5 * (k + 1))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            last = e
            time.sleep(5 * (k + 1))
    raise NetError(f"Scryfall /cards/collection failed after {retries} tries: {last!r}")


def fetch_scryfall(ids: list[str], post=scry_post, spacing: float = SCRY_SPACING) -> dict:
    out = {}
    for i in range(0, len(ids), SCRY_BATCH):
        if i:
            time.sleep(spacing)
        js = post(ids[i:i + SCRY_BATCH])
        for c in js.get("data", []):
            out[c["id"]] = scry_summary(c)
    return out


def load_json(path: Path):
    """.xz / .gz / plain JSON by magic bytes (CK may or may not honour Accept-Encoding)."""
    with open(path, "rb") as fh:
        head = fh.read(6)
    opener = lzma.open if head.startswith(b"\xfd7zXZ") else gzip.open if head[:2] == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:
        return json.load(fh)


# ---- main ------------------------------------------------------------------------------------------------------------
def pt_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:  # noqa: BLE001 - no tz database: fall back to local time (launchd wrapper sets TZ=PT)
        return dt.date.today().isoformat()


def main(argv=None, *, dl=download, post=scry_post) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output dir (default %(default)s)")
    ap.add_argument("--date", default=None, help="PT date stamp for the output files (default: today PT)")
    ap.add_argument("--mtgjson-file", default=None, help="use a cached AllPricesToday .json/.xz instead of downloading")
    ap.add_argument("--ck-file", default=None, help="use a cached CK pricelist .json/.gz instead of downloading")
    a = ap.parse_args(argv)
    date = a.date or pt_today()
    run_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = Path(tempfile.mkdtemp(prefix="mc1_dl_"))
    try:
        try:
            if a.mtgjson_file:
                mj_path = Path(a.mtgjson_file)
            else:
                mj_path = tmp / "AllPricesToday.json.xz"
                dl(MTGJSON_URL, mj_path)
            try:
                mj = load_json(mj_path)
            except (lzma.LZMAError, EOFError, OSError, ValueError) as e:
                raise NetError(f"MTGJSON file unreadable (truncated download?): {e!r}") from e
            if not a.mtgjson_file:
                mj_path.unlink()
            meta = mj.get("meta") or {}
            cands = signal_candidates(mj["data"])
            del mj
            if a.ck_file:
                ck_path = Path(a.ck_file)
            else:
                ck_path = tmp / "ck_pricelist_v2.bin"
                dl(CK_URL, ck_path, headers={"Accept": "application/json", "Accept-Encoding": "gzip"})
            try:
                ck = load_json(ck_path)
            except (EOFError, OSError, ValueError) as e:
                raise NetError(f"CK pricelist unreadable (truncated download?): {e!r}") from e
            if not a.ck_file:
                ck_path.unlink()
            ck_meta = ck.get("meta") or {}
            keys = {cand_key(c) for c in cands if cand_key(c)}
            idx = ck_index(ck.get("data") or [], keys)
            n_ck_rows = len(ck.get("data") or [])
            del ck
            if n_ck_rows < MIN_CK_ROWS:
                raise NetError(f"CK pricelist has only {n_ck_rows} rows (partial or error response)")
            scry = fetch_scryfall(needed_ids(cands, idx), post=post)
        except NetError as e:
            print(f"mc1: NETWORK FAILURE, nothing written: {e}", file=sys.stderr)
            return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = resolve(cands, idx, scry)
    warnings = []
    mjd = meta.get("date")
    try:
        lag = (dt.date.fromisoformat(date) - dt.date.fromisoformat(mjd)).days
        if lag > 1:
            warnings.append(f"MTGJSON data is {lag} days older than the run date")
    except (TypeError, ValueError):
        warnings.append("MTGJSON meta date missing")
    stamps = dict(mtgjson_date=mjd, mtgjson_version=meta.get("version"), ck_created_at=ck_meta.get("created_at"),
                  ck_rows=n_ck_rows, scryfall_cards=len(scry), run_utc=run_utc, warnings=warnings,
                  assumptions=dict(tax=TAX, ship_per_card=SHIP, margin=MARGIN, price_cap=PRICE_CAP,
                                   pre_cutoff=PRE_CUTOFF))
    summary = summarize(date, res, len(cands), stamps)
    report = dict(summary=summary, flags=res["flags"], spot_sample=spot_sample(date, res["flags"]),
                  unresolved=res["unresolved"], excluded=res["excluded"])
    line = write_outputs(report, Path(a.out).expanduser())
    print("mc1:", json.dumps(line))
    for x in res["flags"]:
        print(f"  {x['name'][:30]:30s} | {x['set_name'][:24]:24s} #{x['number']:>5s} | {'F' if x['foil'] else 'N'} | "
              f"tcg {x['tcg_price']:6.2f} mp {x['manapool_price']:6.2f} ck {x['ck_buy_nm_cash']:6.2f} "
              f"q{x['ck_qty_buying']:>2d} net {x['net']:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
