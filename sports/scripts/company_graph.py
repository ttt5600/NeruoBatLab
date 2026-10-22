"""Company graph from free SEC data + customer-momentum / earnings-propagation / graph-momentum tests.

User 2026-10-22: "Model market twin stocks with a graph model on companies, translating relations into better options.
Model relationships and build a thesis we can market a model off of."

    python scripts/company_graph.py edges      # EDGAR full-text-search crawl -> customer edges (no prices, no P&L)
    python scripts/company_graph.py prices     # yfinance daily prices for every node (survivors only)
    python scripts/company_graph.py earn       # EDGAR submissions -> 8-K item 2.02 dates + SIC per node
    python scripts/company_graph.py data       # coverage report; NO P&L
    python scripts/company_graph.py export     # data/company_graph/edges.parquet (customer + industry + comove edges)
    python scripts/company_graph.py dev [--ledger]   # score the 6 pre-registered trials on DEV only

PRE-REGISTERED 2026-10-22, written before any price for the graph universe was downloaded and before any P&L.
(The edge crawl design below was fixed after probing the EDGAR full-text-search syntax on 2008 Wal-Mart queries only.)
DATA-CHECK AMENDMENTS (after the crawl, before any price download or P&L, 2026-10-22 ~01:00 PT):
  - FTS phrase search is case-insensitive. Auto-generated names "Key" (KeyCorp minus "Corp") and "Lab" (Labcorp) matched
    "sales to key customers" etc. -> both dropped (added to SKIP_AUTO).
  - Dedupe kept the undated auto name over the dated alias, so "AT&T" (old AT&T Corp before 2005-11-18, a different
    price lineage from T) was undated -> aliases now win the dedupe. "DuPont" -> DD only from 2019-06-03.
  - 68 queries returned HTTP 500 on the first pass and were retried (1 still failing: "FedEx Freight comprised").

PRIOR. Cohen & Frazzini (2008) report customer momentum ~1.5%/month long-short (1981-2004, Compustat segment file).
Published anomalies decay ~1/3-1/2 after publication (McLean & Pontiff 2016). Our earlier stock-side work: ~62 trials,
0 passes (stocks_bt.py, pairs_bt.py, pairs_twins.py). Expect a small, cost-sensitive result.

WINDOWS
  DEV    = signals and returns 2002-01-01 .. 2022-12-30. `dev` drops every price row and every edge with
           file_date > 2022-12-30 right after load, before any signal is built.
  SEALED = 2023-01-03 .. 2026-10-21. Scored ONCE, only on the user's explicit "yes" (not implemented until then).

GRAPH (all edges carry first_known = the EDGAR filing date; an edge is never used before that date)
  Customer edges (supplier -> customer). Customers = S&P 500 members as of 2026-10-22 (Wikipedia) + 21 large
  non-S&P customers (EXTRA below), with dated aliases (e.g. Raytheon -> RTX only from 2020-04-03, because the RTX price
  series before that is United Technologies). Source: EDGAR full-text search (efts.sec.gov, covers 2001+), forms
  10-K/10-K405/10-KSB, exact phrases (punctuation is ignored by the engine):
     "{N} accounted for" "{N} Inc accounted for" "{N} Corporation accounted for" "{N} represented"
     "{N} Inc represented" "sales to {N}" "net sales to {N}" "revenue from {N}" "revenues from {N}"
     "{N} our largest customer" "{N} is our largest customer" "{N} comprised"
  Each hit = (filer CIK = supplier, customer, file_date, accession). Self-hits (filer = customer) are dropped.
  ASC 280 makes firms name-or-describe customers > 10% of revenue; these phrases catch the sentence form, not tables
  (recall < 100%; precision is measured on a random sample of hits in `data`, and reported).
  Active window: from the first trading day AFTER file_date for 456 calendar days (15 months), extended by any
  later hit on the same pair.
  Industry edges: same SIC 4-digit (SEC submissions, CURRENT SIC -> mild look-ahead; disclosed; used only in B3).
  Co-movement edges (export only, not traded): each calendar year, each node's 5 highest-correlation partners on
  that year's daily returns; first_known = last trading day of that year.

UNIVERSE / FILTERS
  Price data = yfinance daily (auto_adjust) for tickers in SEC company_tickers.json today -> SURVIVORSHIP BIAS:
  delisted suppliers/customers (bankrupt, acquired) are missing. Direction of the bias for customer momentum is
  unclear (dead suppliers after customer collapses would mostly have been in the SHORT leg -> bias is against us).
  Eligible on a signal date: raw-ish close >= $2 (adjusted close used; disclosed) and trailing 60-day median dollar
  volume >= $1M (shifted 1 day).
  COSTS per side, tiered by that $ADV (same tiers as pairs_twins.py): >=$1B 0.5bp | >=$100M 1.5 | >=$20M 4 |
  >=$5M 10 | >=$1M 25 bp; short borrow 0.3%/yr if $ADV >= $20M else 2%/yr; SPY hedge 0.5bp/side, no borrow cost.
  These tiers are estimates, not measured quotes.

TRIALS (6; Bonferroni bar one-sided p < 0.05/6 = 0.00833; all appended to lab/ledger.jsonl)
  A1 customer momentum, long-short quintiles (Cohen-Frazzini). Month end t: for each eligible supplier with >= 1
     active customer edge (file_date < t), CR = equal-weight mean of its active customers' month-t return.
     Need >= 25 eligible suppliers. Long top quintile, short bottom quintile, equal weight, hold month t+1.
     Net of turnover costs + borrow. Metric: mean monthly net LS, Newey-West(6) t, one-sided p.
  A2 Fama-MacBeth: r_{i,t+1} on [CR rank (0..1), own r_t, own 12-1 momentum, SIC2 industry mean r_t (excl. self,
     all universe firms)] each month (same suppliers as A1); test slope on CR > 0 with NW(6). Gross (diagnostic:
     does CR survive industry momentum / own reversal?).
  B1 earnings propagation customer -> supplier. Events = first 8-K item 2.02 per firm per 30 days (EDGAR
     acceptanceDateTime; AMENDED before scoring: it is not ET -- see accepted_et(), converted to ET (matches filingDate 92-98%/yr vs 66-75% raw); some filings look true-UTC (11-12h ET bump),
     so CONSERVATIVELY: converted hour < 9 -> day0 = that day, else day0 = next trading day (a late day0 cannot leak)).
     s = customer r(day0) - SPY r(day0). If |s| >= 2%: for every eligible supplier linked at day0 whose own next
     2.02 day0_s satisfies day0+1 < day0_s <= day0+1+60 trading days, open a sleeve at the CLOSE of day0+1:
     sign(s) x (supplier - SPY), exit at the close of day0_s + 1 (holds through the supplier's own report).
     Cost: supplier tier + SPY 0.5bp, entry and exit; borrow on short sleeves. Daily portfolio = mean net return of
     live sleeves (0 when none). Metric: daily series over all DEV trading days, NW(10) t, one-sided p.
  B2 same, reversed: supplier reports -> trade its not-yet-reported customers.
  B3 same, intra-industry: event firm -> not-yet-reported peers with the same current SIC 4-digit (all universe firms).
  C1 graph-smoothed momentum vs plain momentum. Universe = eligible firms with >= 1 active customer/supplier edge
     (either direction). MOM = 12-1 month return; GMOM = mean MOM of active neighbours.
     Baseline score = pct-rank(MOM); graph score = 0.5 pct-rank(MOM) + 0.5 pct-rank(GMOM). Quintile LS for each,
     net of costs. Metric: monthly (graph LS - baseline LS), NW(6), one-sided p. Also report both levels.
  Description only (not trials): sub-periods 2002-08 / 2009-15 / 2016-22, no-filter (incl. sub-$1M ADV) A1 gross,
  number of names per leg, turnover, edge precision sample.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "company_graph"
LEDGER = ROOT / "lab" / "ledger.jsonl"
UA = {"User-Agent": "quant-research-bot admin@quantresearch.local"}
DEV_START, DEV_END, SEALED_END = "2002-01-01", "2022-12-30", "2026-10-21"
TIERS = [(1e9, 0.5), (1e8, 1.5), (2e7, 4.0), (5e6, 10.0), (1e6, 25.0), (0.0, 60.0)]
PATTERNS = ["{n} accounted for", "{n} Inc accounted for", "{n} Corporation accounted for", "{n} represented",
            "{n} Inc represented", "sales to {n}", "net sales to {n}", "revenue from {n}", "revenues from {n}",
            "{n} our largest customer", "{n} is our largest customer", "{n} comprised"]
FORMS = "10-K,10-K405,10-KSB"

# Large customers outside today's S&P 500 (survivors with US listings). (ticker, names)
EXTRA = [("KSS", ["Kohl's", "Kohls"]), ("M", ["Macy's", "Federated Department Stores"]), ("AAP", ["Advance Auto Parts"]),
         ("WBA", ["Walgreens", "Walgreen"]), ("ARW", ["Arrow Electronics"]), ("AVT", ["Avnet"]),
         ("SNX", ["Synnex", "TD Synnex"]), ("SANM", ["Sanmina"]), ("USFD", ["US Foods"]),
         ("PFGC", ["Performance Food Group"]), ("DKS", ["Dick's Sporting Goods"]), ("BURL", ["Burlington Stores"]),
         ("ACI", ["Albertsons"]), ("TM", ["Toyota"]), ("HMC", ["Honda"]), ("STLA", ["Stellantis", "FCA US"]),
         ("TSM", ["Taiwan Semiconductor", "TSMC"]), ("SONY", ["Sony"]), ("NOK", ["Nokia"]), ("ERIC", ["Ericsson"]),
         ("OMI", ["Owens & Minor", "Owens and Minor"])]
# Dated / extra aliases for S&P members: ticker -> [(name, from, to)]
ALIASES = {"WMT": [("Wal-Mart", None, None), ("Walmart", None, None), ("Sam's Club", None, None)],
           "HPQ": [("Hewlett-Packard", None, "2015-10-31"), ("HP", None, None)],
           "HPE": [("Hewlett Packard Enterprise", "2015-11-01", None)],
           "RTX": [("United Technologies", None, "2020-04-02"), ("Raytheon", "2020-04-03", None),
                   ("Pratt & Whitney", None, None)],
           "COR": [("AmerisourceBergen", None, None), ("Cencora", None, None)],
           "META": [("Facebook", None, None), ("Meta", "2021-10-28", None)],
           "GOOGL": [("Google", None, None), ("Alphabet", None, None)],
           "ELV": [("Anthem", None, None), ("WellPoint", None, None), ("Elevance", None, None)],
           "T": [("AT&T", "2005-11-18", None)],
           "MSI": [("Motorola", None, None)],
           "FLEX": [("Flextronics", None, None), ("Flex", "2016-09-01", None)],
           "GE": [("General Electric", None, None)],
           "IBM": [("International Business Machines", None, None), ("IBM", None, None)],
           "HD": [("Home Depot", None, None)], "LOW": [("Lowe's", None, None), ("Lowes", None, None)],
           "TGT": [("Target", None, None)], "COST": [("Costco", None, None)], "KR": [("Kroger", None, None)],
           "SO": [("Southern Company", None, None)], "CHTR": [("Charter Communications", None, None)],
           "AMZN": [("Amazon", None, None)], "DELL": [("Dell", "2016-09-07", None)],
           "VZ": [("Verizon", None, None)], "CVS": [("CVS", None, None)], "TJX": [("TJX", None, None)],
           "XOM": [("ExxonMobil", None, None), ("Exxon Mobil", None, None)], "LLY": [("Eli Lilly", None, None)],
           "PG": [("Procter & Gamble", None, None), ("Procter and Gamble", None, None)],
           "JNJ": [("Johnson & Johnson", None, None)], "MCK": [("McKesson", None, None)],
           "CAH": [("Cardinal Health", None, None)], "DIS": [("Disney", None, None)],
           "NOC": [("Northrop Grumman", None, None), ("Northrop", None, None)], "BA": [("Boeing", None, None)],
           "LMT": [("Lockheed Martin", None, None), ("Lockheed", None, None)], "GD": [("General Dynamics", None, None)],
           "TMUS": [("T-Mobile", None, None)], "KO": [("Coca-Cola Company", None, None)],
           "CMCSA": [("Comcast", None, None)], "CSCO": [("Cisco", None, None)], "INTC": [("Intel", None, None)],
           "AAPL": [("Apple", None, None)], "MSFT": [("Microsoft", None, None)], "DE": [("John Deere", None, None)],
           "DD": [("DuPont", "2019-06-03", None)],
           "LHX": [("L3Harris", None, None), ("Harris Corporation", None, "2019-06-28")]}
# Skip auto names that are ordinary words / ambiguous (aliases above still apply).
SKIP_AUTO = {"Block", "Ball", "Progressive", "Southern", "Visa", "Hartford", "Loews", "Dover", "Target", "Charter",
             "Coherent", "Flex", "Corning", "Gartner", "Amcor", "Everest", "Generac", "Erie", "Q", "Qnity", "Vylor",
             "Vivmark", "Everpure", "Booking", "Copart", "Ventas", "Masco", "Equinix", "Oneok", "Mosaic", "Apollo",
             "Expand", "Ares", "Casey's", "Domino's", "Uber", "Reddit", "Datadog", "Twilio", "Ametek", "Nike",
             "Delta", "Crown", "Old Dominion", "Lennox", "Coca-Cola", "Gen", "Fox", "News", "Cooper", "Lilly",
             "Travelers", "Mondelez", "Moderna", "Waters", "Trimble", "Pentair", "Allegion", "Revvity", "Sandisk",
             "Supermicro", "Solventum", "Veralto", "Kenvue", "Lumentum", "Carvana", "Robinhood", "Coinbase",
             "Skydance", "Bloom Energy", "Comfort Systems USA", "EchoStar", "Interactive Brokers", "AppLovin",
             "Key", "Lab", "Palantir", "CrowdStrike", "DoorDash", "Airbnb", "Vistra", "Corpay", "Ferguson Enterprises"}
SUFFIX = re.compile(r"\s*(\(The\)|\(Class [A-Z]\)|,? Inc\.?|,? Incorporated|Corporation|Corp\.?|Company|plc|Ltd\.?|"
                    r"Limited|Holdings|Group|& Co\.|Co\.)\s*$", re.I)


def get(url, params=None, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(5 * (k + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3 * (k + 1))
    return None


# ------------------------------------------------------------------------------------------------ customers
def customer_names() -> pd.DataFrame:
    spx = pd.read_csv(DATA / "spx_members_2026-10-22.csv")
    rows = []
    for _, r in spx.iterrows():
        tk = r.Symbol.replace(".", "-")
        if tk in ("GOOG", "FOX", "NWS"):  # duplicate share classes
            continue
        base = r.Security
        for _ in range(3):
            base = SUFFIX.sub("", base).strip()
        for n, a, b in ALIASES.get(tk, []):          # dated aliases first so they win the dedupe below
            rows.append((tk, n, a, b))
        if base not in SKIP_AUTO and len(base) >= 3:
            rows.append((tk, base, None, None))
    for tk, names in EXTRA:
        for n in names:
            rows.append((tk, n, None, None))
    df = pd.DataFrame(rows, columns=["cust_ticker", "name", "valid_from", "valid_to"]).drop_duplicates(["cust_ticker", "name"])
    tick = json.loads((DATA / "company_tickers_2026-10-22.json").read_text())
    t2c = {v["ticker"].replace(".", "-"): int(v["cik_str"]) for v in tick.values()}
    df["cust_cik"] = df.cust_ticker.map(t2c)
    return df


# ------------------------------------------------------------------------------------------------ edges crawl
def cmd_edges():
    names = customer_names()
    out = DATA / "fts_hits.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            done.add(json.loads(l)["query"]) if l.startswith('{"query"') else None
    qdone = DATA / "fts_queries_done.txt"
    qd = set(qdone.read_text().splitlines()) if qdone.exists() else set()
    todo = [(r.cust_ticker, r.name, p.format(n=r.name)) for r in names.itertuples() for p in PATTERNS]
    print(f"{len(names)} customer names, {len(todo)} queries, {len(qd)} done")
    with out.open("a") as f, qdone.open("a") as fq:
        for i, (tk, nm, q) in enumerate(todo):
            if q in qd:
                continue
            frm, total = 0, None
            while True:
                r = get("https://efts.sec.gov/LATEST/search-index",
                        dict(q=f'"{q}"', forms=FORMS, dateRange="custom", startdt="2001-01-01", enddt=SEALED_END,
                             **{"from": frm}))
                time.sleep(0.15)
                if r is None or r.status_code != 200:
                    print("FAIL", q, r.status_code if r is not None else None)
                    break
                d = r.json()["hits"]
                total = d["total"]["value"]
                for h in d["hits"]:
                    s = h["_source"]
                    f.write(json.dumps({"query": q, "cust_ticker": tk, "name": nm, "id": h["_id"],
                                        "ciks": s.get("ciks"), "names": s.get("display_names"),
                                        "file_date": s.get("file_date"), "form": s.get("form"),
                                        "sics": s.get("sics"), "period": s.get("period_ending")}) + "\n")
                frm += 100
                if frm >= min(total, 10000) or not d["hits"]:
                    break
            fq.write(q + "\n")
            if i % 200 == 0:
                print(i, q, total, flush=True)
                f.flush(); fq.flush()


def load_edges() -> pd.DataFrame:
    """Hit rows -> one row per (supplier CIK, customer ticker, accession) with dated alias validity applied."""
    rows = [json.loads(l) for l in (DATA / "fts_hits.jsonl").read_text().splitlines()]
    h = pd.DataFrame(rows)
    h["sup_cik"] = h.ciks.map(lambda c: int(c[0]) if c else None)
    h["adsh"] = h.id.str.split(":").str[0]
    h["file_date"] = pd.to_datetime(h.file_date)
    names = customer_names().set_index(["cust_ticker", "name"])
    v = names.reindex(pd.MultiIndex.from_arrays([h.cust_ticker, h.name]))
    vf = pd.to_datetime(v.valid_from.values)
    vt = pd.to_datetime(v.valid_to.values)
    known = v.cust_cik.notna().values | v.index.isin(names.index)
    ok = known & (pd.isna(vf) | (h.file_date.values >= vf)) & (pd.isna(vt) | (h.file_date.values <= vt))
    h = h[ok].copy()
    h["cust_cik"] = names.cust_cik.reindex(pd.MultiIndex.from_arrays([h.cust_ticker, h.name])).values
    h = h[h.sup_cik != h.cust_cik]
    g = (h.groupby(["sup_cik", "cust_ticker", "adsh"])
          .agg(file_date=("file_date", "min"), cust_cik=("cust_cik", "first"), n_patterns=("query", "nunique"),
               patterns=("query", lambda s: "|".join(sorted(set(s)))), sup_name=("names", lambda s: s.iloc[0][0]),
               form=("form", "first"))
          .reset_index())
    tick = json.loads((DATA / "company_tickers_2026-10-22.json").read_text())
    c2t = {}
    for v_ in tick.values():
        c2t.setdefault(int(v_["cik_str"]), v_["ticker"].replace(".", "-"))
    g["sup_ticker"] = g.sup_cik.map(c2t)
    g = g[g.sup_ticker != g.cust_ticker]
    return g


def clean_edges(g: pd.DataFrame) -> pd.DataFrame:
    """Drop self-mentions the CIK test misses (customer's own 10-K under an older CIK, e.g. a re-domiciled parent)."""
    names = customer_names()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    nm = names.groupby("cust_ticker").name.apply(lambda s: [norm(x) for x in s]).to_dict()
    selfm = [any(n and n in norm(sn) for n in nm.get(ct, [])) for sn, ct in zip(g.sup_name, g.cust_ticker)]
    g = g[~np.array(selfm, dtype=bool)]
    vt = DATA / "tgt_verified.csv"
    if vt.exists():                       # Target edges kept only if verified (see cmd_verify_target)
        ok = set(pd.read_csv(vt).query("ok").adsh)
        g = g[(g.cust_ticker != "TGT") | g.adsh.isin(ok)]
    return g


# ------------------------------------------------------------------------------------------------ prices / earnings
def universe() -> list[str]:
    g = clean_edges(load_edges())
    tk = set(g.sup_ticker.dropna()) | set(customer_names().cust_ticker) | {"SPY"}
    return sorted(tk)


def cmd_prices():
    import yfinance as yf
    out = DATA / "prices.parquet"
    tk = universe()
    print(len(tk), "tickers")
    frames = []
    for i in range(0, len(tk), 150):
        d = yf.download(tk[i:i + 150], start="2000-06-01", end="2026-10-22", auto_adjust=True, progress=False,
                        group_by="column", threads=True)
        frames.append(d[["Close", "Volume"]].astype("float32"))
        print(i, flush=True)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, px.notna().any()]
    px.columns = [f"{f}|{t}" for f, t in px.columns]
    px.to_parquet(out)
    print(px.shape, out.stat().st_size / 1e6, "MB")


def cmd_earn():
    """8-K item 2.02 acceptance times + current SIC for every node with a CIK."""
    tick = json.loads((DATA / "company_tickers_2026-10-22.json").read_text())
    t2c = {}
    for v in tick.values():
        t2c.setdefault(v["ticker"].replace(".", "-"), int(v["cik_str"]))
    px_t = {c.split("|")[1] for c in pd.read_parquet(DATA / "prices.parquet", columns=None).columns}
    out = DATA / "earn_events.parquet"
    rows, sic = [], {}
    tks = sorted(t for t in px_t if t in t2c)
    for k, t in enumerate(tks):
        cik = t2c[t]
        r = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        time.sleep(0.12)
        if r is None or r.status_code != 200:
            continue
        j = r.json()
        sic[t] = j.get("sic")
        blocks = [j["filings"]["recent"]]
        for f in j["filings"].get("files", []):
            rr = get(f"https://data.sec.gov/submissions/{f['name']}")
            time.sleep(0.12)
            if rr is not None and rr.status_code == 200:
                blocks.append(rr.json())
        for b in blocks:
            for form, acc, items, fd in zip(b["form"], b["acceptanceDateTime"], b["items"], b["filingDate"]):
                if form == "8-K" and "2.02" in str(items):
                    rows.append((t, cik, fd, acc))
        if k % 100 == 0:
            print(k, t, len(rows), flush=True)
    e = pd.DataFrame(rows, columns=["ticker", "cik", "filing_date", "accepted"])
    e.to_parquet(out)
    pd.Series(sic, name="sic").rename_axis("ticker").reset_index().to_csv(DATA / "sic.csv", index=False)
    print(len(e), "events", e.ticker.nunique(), "firms")


# ------------------------------------------------------------------------------------------------ dev scoring
def nw_t(x: np.ndarray, lags: int):
    import statsmodels.api as sm
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues[0])
    return float(x.mean()), t, float(norm.sf(t)), len(x)


def tier_bps(adv):
    adv = np.asarray(adv, float)
    out = np.full(adv.shape, 60.0)
    for lo, bps in reversed(TIERS):
        out = np.where(adv >= lo, bps, out)
    return np.where(np.isnan(adv), 25.0, out)


class Panel:
    """Dev-window prices, eligibility, edges, events. Everything after DEV_END is dropped here."""

    def __init__(self, end=DEV_END):
        px = pd.read_parquet(DATA / "prices.parquet")
        px = px[px.index <= end]
        close = px[[c for c in px.columns if c.startswith("Close|")]]
        close.columns = [c.split("|")[1] for c in close.columns]
        vol = px[[c for c in px.columns if c.startswith("Volume|")]]
        vol.columns = [c.split("|")[1] for c in vol.columns]
        days = close["SPY"].dropna().index
        self.close = close.loc[days]
        self.vol = vol.reindex(index=days, columns=self.close.columns)
        self.days = days
        self.r = self.close.pct_change(fill_method=None)
        dv = (self.close * self.vol)
        self.adv = dv.rolling(60, min_periods=40).median().shift(1)
        self.elig = (self.close >= 2) & (self.adv >= 1e6)
        g = clean_edges(load_edges())
        g = g[g.file_date <= end].dropna(subset=["sup_ticker"])
        g = g[g.sup_ticker.isin(self.close.columns) & g.cust_ticker.isin(self.close.columns)]
        self.edges = g
        e = pd.read_parquet(DATA / "earn_events.parquet")
        acc = accepted_et(e.accepted)
        after = acc.dt.hour >= 9          # conservative: only pre-9:00 ET counts as same-day (see docstring)
        d = acc.dt.normalize()
        pos = np.searchsorted(days.values, d.values)               # first trading day >= date
        pos = pos + (after.values & (pos < len(days)) & (days.values[np.minimum(pos, len(days) - 1)] == d.values))
        e["day0"] = pos
        e = e[(e.day0 < len(days)) & e.ticker.isin(self.close.columns)].sort_values(["ticker", "day0"])
        keep = []
        for t, s in e.groupby("ticker").day0:                       # first 2.02 per firm per ~30 days (21 trading)
            last = -999
            for v in s.values:
                if v - last > 21:
                    keep.append((t, v))
                    last = v
        self.events = pd.DataFrame(keep, columns=["ticker", "day0"])
        sic = pd.read_csv(DATA / "sic.csv", dtype={"sic": str}).dropna()
        self.sic = dict(zip(sic.ticker, sic.sic.str.zfill(4)))

    def active_pairs(self, date) -> pd.DataFrame:
        g = self.edges
        m = (g.file_date < date) & (g.file_date + pd.Timedelta(days=456) >= date)
        return g.loc[m, ["sup_ticker", "cust_ticker"]].drop_duplicates()

    def month_ends(self):
        s = self.days.to_series()
        me = s[s.dt.month != s.shift(-1).dt.month]
        return me[me >= pd.Timestamp(DEV_START) - pd.Timedelta(days=40)].index


def accepted_et(col: pd.Series) -> pd.Series:
    """EDGAR submissions acceptanceDateTime is NOT plain UTC: checked 2026-10-22 on JPM (06:30-06:41 ET releases shown
    as 14:30Z in EDT months / 16:41Z in EST months), WMT (06:59 ET -> 14:59Z / 16:59Z), AAPL (16:30 ET -> 00:30Z next
    day). All fit ET = shown - 2 x |ET UTC offset|, i.e. apply the UTC->ET conversion twice. Returns naive ET."""
    t = pd.to_datetime(col.str.slice(0, 19)).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)


def ls_quintile(score: pd.Series, fwd: pd.Series, bps: pd.Series):
    s = score.dropna()
    if len(s) < 25:
        return None
    q = pd.qcut(s.rank(method="first"), 5, labels=False)
    lo, hi = s.index[q == 0], s.index[q == 4]
    w = pd.Series(0.0, index=s.index)
    w[hi] = 1 / len(hi)
    w[lo] = -1 / len(lo)
    return w


def run_monthly(P: Panel, scorer, label):
    """scorer(t_idx, t_date) -> pd.Series score (index tickers, already eligible). Returns DataFrame per month."""
    me = P.month_ends()
    mclose = P.close.loc[me]
    mret = mclose.pct_change(fill_method=None)
    rows, prev_w = [], pd.Series(dtype=float)
    for k in range(1, len(me) - 1):
        t, t1 = me[k], me[k + 1]
        if t < pd.Timestamp(DEV_START):
            continue
        sc = scorer(k, t, mret)
        if sc is None:
            rows.append({"t": t1, "gross": 0.0, "net": 0.0, "n": 0}); prev_w = pd.Series(dtype=float); continue
        fwd = mret.loc[t1]
        w = ls_quintile(sc, fwd, None)
        if w is None:
            rows.append({"t": t1, "gross": 0.0, "net": 0.0, "n": 0}); prev_w = pd.Series(dtype=float); continue
        w = w[w != 0]
        ok = fwd.reindex(w.index).notna()
        w = w[ok]
        w[w > 0] = 1 / (w > 0).sum()
        w[w < 0] = -1 / (w < 0).sum()
        gross = float((w * fwd.reindex(w.index)).sum())
        allidx = w.index.union(prev_w.index)
        dw = (w.reindex(allidx, fill_value=0) - prev_w.reindex(allidx, fill_value=0)).abs()
        bps = pd.Series(tier_bps(P.adv.loc[t].reindex(allidx).values), index=allidx)
        cost = float((dw * bps).sum() / 1e4)
        sh = w[w < 0]
        borrow = float((-sh * np.where(P.adv.loc[t].reindex(sh.index).values >= 2e7, 0.003, 0.02)).sum() / 12)
        rows.append({"t": t1, "gross": gross, "net": gross - cost - borrow, "n": int(len(w)), "turnover": float(dw.sum())})
        prev_w = w
    return pd.DataFrame(rows).set_index("t")


def sub_periods(s: pd.Series):
    out = {}
    for a, b in [("2002", "2008"), ("2009", "2015"), ("2016", "2022")]:
        x = s[(s.index >= f"{a}-01-01") & (s.index <= f"{b}-12-31")]
        out[f"{a}-{b[2:]}"] = round(float(x.mean()) * 100, 3) if len(x) else None
    return out


def trial_A(P: Panel, filt=True):
    def scorer(k, t, mret):
        ap = P.active_pairs(t)
        cr = mret.loc[t]
        ap = ap.assign(cr=cr.reindex(ap.cust_ticker).values).dropna()
        s = ap.groupby("sup_ticker").cr.mean()
        if filt:
            el = P.elig.loc[t]
            s = s[el.reindex(s.index).fillna(False).astype(bool)]
        return s
    return run_monthly(P, scorer, "A1")


def trial_A2(P: Panel):
    me = P.month_ends()
    mclose = P.close.loc[me]
    mret = mclose.pct_change(fill_method=None)
    mom = mclose.shift(1) / mclose.shift(12) - 1
    sic2 = pd.Series({t: s[:2] for t, s in P.sic.items()})
    slopes = []
    for k in range(12, len(me) - 1):
        t, t1 = me[k], me[k + 1]
        if t < pd.Timestamp(DEV_START):
            continue
        ap = P.active_pairs(t)
        ap = ap.assign(cr=mret.loc[t].reindex(ap.cust_ticker).values).dropna()
        cr = ap.groupby("sup_ticker").cr.mean()
        cr = cr[P.elig.loc[t].reindex(cr.index).fillna(False).astype(bool)]
        if len(cr) < 25:
            continue
        el = P.elig.loc[t]
        allr = mret.loc[t][el.reindex(mret.columns).fillna(False).astype(bool)].dropna()
        ind = pd.DataFrame({"r": allr, "s2": sic2.reindex(allr.index)}).dropna()
        gs, gn = ind.groupby("s2").r.sum(), ind.groupby("s2").r.count()
        X = pd.DataFrame({"cr": cr.rank(pct=True), "own": mret.loc[t].reindex(cr.index),
                          "mom": mom.loc[t].reindex(cr.index)})
        s2 = sic2.reindex(cr.index)
        X["ind"] = [(gs.get(s, np.nan) - (X.own[i] if i in ind.index else 0)) / (gn.get(s, 0) - (1 if i in ind.index else 0))
                    if gn.get(s, 0) > 1 else np.nan for i, s in zip(cr.index, s2)]
        y = mret.loc[t1].reindex(cr.index)
        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(d) < 25:
            continue
        A = np.column_stack([np.ones(len(d)), d[["cr", "own", "mom", "ind"]].values])
        b = np.linalg.lstsq(A, d.y.values, rcond=None)[0]
        slopes.append((t1, b[1], len(d)))
    return pd.DataFrame(slopes, columns=["t", "b_cr", "n"]).set_index("t")


def propagate(P: Panel, pairs_fn, label):
    """Event sleeves via difference arrays. pairs_fn(event_ticker, day0_date) -> list of target tickers."""
    days, cols = P.days, list(P.close.columns)
    ci = {c: i for i, c in enumerate(cols)}
    T, N = len(days), len(cols)
    R = P.r.values.astype(float)
    spy = R[:, ci["SPY"]]
    X = np.nan_to_num(R - spy[:, None])                 # excess return vs SPY
    cs = np.vstack([np.zeros(N), np.cumsum(X, axis=0)])  # cs[d+1] = sum X[0..d]
    ev = P.events
    ev_by = {t: np.sort(s.values) for t, s in ev.groupby("ticker").day0}
    S = np.zeros((T + 1, N))                             # signed sleeve count, difference array
    SH = np.zeros((T + 1, N))                            # short sleeve count (borrow)
    CNT = np.zeros(T + 1)
    COST = np.zeros(T)
    adv = P.adv.values
    sleeves = []
    first_ok = np.searchsorted(days.values, np.datetime64(DEV_START))
    for t, d0s in ev_by.items():
        for d0 in d0s:
            if d0 < first_ok or d0 + 2 >= T:
                continue
            s = R[d0, ci[t]] - spy[d0]
            if not np.isfinite(s) or abs(s) < 0.02:
                continue
            sg = 1.0 if s > 0 else -1.0
            e = d0 + 1
            for tg in pairs_fn(t, days[d0]):
                j = ci.get(tg)
                if j is None or tg == t:
                    continue
                if not P.elig.values[e, j]:
                    continue
                nx = ev_by.get(tg)
                if nx is None:
                    continue
                k = np.searchsorted(nx, e + 1)           # target's next report strictly after e... day0_s > day0+1
                if k >= len(nx) or nx[k] > e + 60:
                    continue
                x = nx[k] + 1
                if x >= T:
                    continue
                # sleeve earns days e+1 .. x
                S[e + 1, j] += sg; S[x + 1, j] -= sg
                CNT[e + 1] += 1; CNT[x + 1] -= 1
                if sg < 0:
                    SH[e + 1, j] += 1; SH[x + 1, j] -= 1
                c = (tier_bps(adv[e, j]) + 0.5) / 1e4
                COST[e + 1] += c; COST[x] += c
                gross = sg * (cs[x + 1, j] - cs[e + 1, j])
                bor = (0.003 if adv[e, j] >= 2e7 else 0.02) * (x - e) / 252 if sg < 0 else 0.0
                sleeves.append((t, tg, days[d0], sg, x - e, gross, gross - 2 * c - bor))
    S = np.cumsum(S, axis=0)[:T]; SH = np.cumsum(SH, axis=0)[:T]; CNT = np.cumsum(CNT)[:T]
    rate = np.where(adv >= 2e7, 0.003, 0.02) / 252
    gross_d = (S * X).sum(axis=1)
    bor_d = np.nansum(SH * rate, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(CNT > 0, gross_d / CNT, 0.0)
        n = np.where(CNT > 0, (gross_d - COST - bor_d) / CNT, 0.0)
    idx = days
    m = (idx >= DEV_START)
    sl = pd.DataFrame(sleeves, columns=["event", "target", "date", "sign", "hold", "gross", "net"])
    return pd.Series(g[m], idx[m]), pd.Series(n[m], idx[m]), pd.Series(CNT[m], idx[m]), sl


def trial_C(P: Panel):
    me = P.month_ends()
    mclose = P.close.loc[me]
    mom = mclose.shift(1) / mclose.shift(12) - 1
    cache = {}

    def neigh(t):
        ap = P.active_pairs(t)
        a = pd.concat([ap.rename(columns={"sup_ticker": "a", "cust_ticker": "b"}),
                       ap.rename(columns={"sup_ticker": "b", "cust_ticker": "a"})])
        return a

    def base(k, t, mret):
        a = neigh(t)
        m = mom.loc[t]
        a = a.assign(mb=m.reindex(a.b).values).dropna()
        gm = a.groupby("a").mb.mean()
        own = m.reindex(gm.index)
        el = P.elig.loc[t].reindex(gm.index).fillna(False).astype(bool)
        d = pd.DataFrame({"own": own, "gm": gm})[el.values].dropna()
        cache[t] = d
        return d.own.rank(pct=True)

    def graph(k, t, mret):
        d = cache[t]
        return 0.5 * d.own.rank(pct=True) + 0.5 * d.gm.rank(pct=True)
    b = run_monthly(P, base, "C0")
    g = run_monthly(P, graph, "C1")
    return b, g


def ledger_write(recs):
    import hashlib
    lines = LEDGER.read_text().splitlines() if LEDGER.exists() else []
    n0 = len(lines)
    with LEDGER.open("a") as f:
        for i, r in enumerate(recs):
            base = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"cgraph_{r['trial']}",
                    "sha": hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:12],
                    "dataset": "company_graph_edgar_yf", "hypothesis": "pre-registered in scripts/company_graph.py",
                    "source": "company_graph (user 2026-10-22)", "status": "ok", "trial_number": n0 + i + 1,
                    "study_trials": 6, "bonferroni_bar": 0.05 / 6}
            base.update(r)
            base["passed"] = bool(r["p"] < 0.05 / 6 and r["mean"] > 0)
            f.write(json.dumps(base, default=float) + "\n")


def cmd_dev():
    P = Panel()
    print(f"DEV panel {P.days[0].date()}..{P.days[-1].date()}, {P.close.shape[1]} tickers, "
          f"{len(P.edges)} edge-filings, {P.edges[['sup_ticker','cust_ticker']].drop_duplicates().shape[0]} pairs, "
          f"{len(P.events)} report events")
    recs = []
    # A1
    a = trial_A(P, True)
    mu, t, p, n = nw_t(a.net.values, 6)
    a_nf = trial_A(P, False)
    recs.append({"trial": "A1_customer_momentum", "mean": mu, "t": t, "p": p, "months": n,
                 "gross_mo": float(a.gross.mean()), "net_mo": mu, "net_ann": mu * 12,
                 "avg_names": float(a.n[a.n > 0].mean()), "turnover_mo": float(a.turnover.mean()),
                 "gross_p": nw_t(a.gross.values, 6)[2], "sub_net_pct_mo": sub_periods(a.net),
                 "sub_gross_pct_mo": sub_periods(a.gross), "nofilter_gross_mo": float(a_nf.gross.mean()),
                 "nofilter_gross_p": nw_t(a_nf.gross.values, 6)[2]})
    # A2
    fm = trial_A2(P)
    mu, t, p, n = nw_t(fm.b_cr.values, 6)
    recs.append({"trial": "A2_fama_macbeth_cr", "mean": mu, "t": t, "p": p, "months": n,
                 "avg_n": float(fm.n.mean()), "sub_slope_pct": sub_periods(fm.b_cr)})
    # B1-B3
    edges = P.edges

    def mk(direction):
        def fn(tk, date):
            m = (edges.file_date < date) & (edges.file_date + pd.Timedelta(days=456) >= date)
            if direction == "c2s":
                return edges.loc[m & (edges.cust_ticker == tk), "sup_ticker"].unique()
            return edges.loc[m & (edges.sup_ticker == tk), "cust_ticker"].unique()
        return fn
    by_sic = {}
    for tk, s in P.sic.items():
        if tk in P.close.columns and s not in ("0000", "None"):
            by_sic.setdefault(s, []).append(tk)
    for name, fn in [("B1_cust_to_supplier", mk("c2s")), ("B2_supplier_to_cust", mk("s2c")),
                     ("B3_industry_sic4", lambda tk, d: by_sic.get(P.sic.get(tk), []))]:
        g, nt, cnt, sl = propagate(P, fn, name)
        mu, t, p, n = nw_t(nt.values, 10)
        recs.append({"trial": name, "mean": mu, "t": t, "p": p, "days": n, "net_ann": mu * 252,
                     "gross_ann": float(g.mean() * 252), "gross_p": nw_t(g.values, 10)[2],
                     "sleeves": int(len(sl)), "days_with_pos": float((cnt > 0).mean()),
                     "sleeve_gross_mean": float(sl.gross.mean()) if len(sl) else None,
                     "sleeve_net_mean": float(sl.net.mean()) if len(sl) else None,
                     "sleeve_hold_mean": float(sl.hold.mean()) if len(sl) else None,
                     "sub_net_ann_pct": {k: (v * 252 if v is not None else None) for k, v in sub_periods(nt).items()}})
        sl.to_parquet(DATA / f"sleeves_{name}.parquet")
    # C1
    b, gr = trial_C(P)
    diff = (gr.net - b.net)
    mu, t, p, n = nw_t(diff.values, 6)
    recs.append({"trial": "C1_graph_mom_minus_mom", "mean": mu, "t": t, "p": p, "months": n,
                 "graph_net_mo": float(gr.net.mean()), "graph_net_p": nw_t(gr.net.values, 6)[2],
                 "base_net_mo": float(b.net.mean()), "base_net_p": nw_t(b.net.values, 6)[2],
                 "graph_gross_mo": float(gr.gross.mean()), "base_gross_mo": float(b.gross.mean()),
                 "sub_diff_pct_mo": sub_periods(diff)})
    pd.concat({"A1": a, "A1_nofilt": a_nf, "C0": b, "C1": gr}, axis=1).to_parquet(DATA / "dev_monthly.parquet")
    for r in recs:
        print(json.dumps(r, default=lambda v: round(float(v), 5)))
    if "--ledger" in sys.argv:
        ledger_write(recs)
        print("ledger: appended", len(recs))


def cmd_data():
    g = clean_edges(load_edges())
    print(f"edge-filings {len(g)}; pairs {g[['sup_cik','cust_ticker']].drop_duplicates().shape[0]}; "
          f"suppliers {g.sup_cik.nunique()} (with current ticker {g.sup_ticker.notna().groupby(g.sup_cik).first().sum()}); "
          f"customers {g.cust_ticker.nunique()}")
    print(g.groupby(g.file_date.dt.year).size().to_string())
    print(g.cust_ticker.value_counts().head(25).to_string())
    if (DATA / "prices.parquet").exists():
        px = pd.read_parquet(DATA / "prices.parquet")
        c = px[[x for x in px.columns if x.startswith("Close|")]]
        print("priced tickers", c.shape[1], "first dates:", c.apply(lambda s: s.first_valid_index()).dt.year.value_counts().sort_index().to_dict())


def cmd_sample(n=60, seed=7):
    """Precision check: fetch n random edge filings, show the sentence around the customer name."""
    g = clean_edges(load_edges())
    s = g.sample(n, random_state=seed)
    hits = pd.DataFrame([json.loads(l) for l in (DATA / "fts_hits.jsonl").read_text().splitlines()])
    out = []
    for r in s.itertuples():
        h = hits[(hits.id.str.startswith(r.adsh)) & (hits.cust_ticker == r.cust_ticker)].iloc[0]
        adsh, fn = h.id.split(":")
        url = f"https://www.sec.gov/Archives/edgar/data/{r.sup_cik}/{adsh.replace('-', '')}/{fn}"
        resp = get(url)
        time.sleep(0.15)
        txt = re.sub(r"<[^>]+>", " ", resp.text if resp is not None else "")
        txt = re.sub(r"&#?\w+;", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        phr = r.patterns.split("|")[0]
        words = phr.split()
        pat = r"\W+".join(re.escape(w) for w in words)
        m = re.search(pat, txt, re.I)
        snip = txt[max(0, m.start() - 200): m.end() + 200] if m else "(phrase not found in text)"
        out.append({"sup": r.sup_name, "cust": r.cust_ticker, "date": str(r.file_date.date()), "phrase": phr,
                    "snippet": snip, "url": url})
        print(f"\n[{len(out)}] {r.sup_name} -> {r.cust_ticker} {r.file_date.date()} | {phr}\n   {snip}")
    pd.DataFrame(out).to_csv(DATA / "edge_precision_sample.csv", index=False)


def cmd_verify_target():
    """Data check: FTS is case-insensitive, so "sales to target audiences" matches Target. Fetch every TGT edge filing
    and keep it only if a sentence contains case-sensitive 'Target' (not sentence-initial 'Target represented an') next to
    the matched phrase AND a percent / customer / sales word in the same sentence."""
    g = clean_edges(load_edges())
    g = g[g.cust_ticker == "TGT"]
    hits = pd.DataFrame([json.loads(l) for l in (DATA / "fts_hits.jsonl").read_text().splitlines()])
    hits = hits[hits.cust_ticker == "TGT"]
    out = DATA / "tgt_verified.csv"
    done = set(pd.read_csv(out).adsh) if out.exists() else set()
    rows = []
    for r in g.itertuples():
        if r.adsh in done:
            continue
        h = hits[hits.id.str.startswith(r.adsh)].iloc[0]
        adsh, fn = h.id.split(":")
        resp = get(f"https://www.sec.gov/Archives/edgar/data/{r.sup_cik}/{adsh.replace('-', '')}/{fn}")
        time.sleep(0.12)
        txt = re.sub(r"<[^>]+>", " ", resp.text if resp is not None else "")
        txt = re.sub(r"&#?\w+;", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        ok = False
        for sent in re.split(r"(?<=[.;])\s", txt):
            if re.search(r"\bTarget(\s+Corporation|\s+Corp|\s+Stores|,? Inc)?\b", sent) and \
               re.search(r"(sales to|revenues? from|accounted for|represented|comprised|largest customer)", sent, re.I) and \
               re.search(r"(%|percent)", sent) and not re.search(r"Target\s+(audience|market|customer|price|level|range|rate|EBITDA|bonus|amount|date|company|represented an)", sent):
                ok = True
                break
        rows.append({"adsh": r.adsh, "sup_cik": r.sup_cik, "ok": ok})
        if len(rows) % 50 == 0:
            pd.concat([pd.read_csv(out)] if out.exists() else [] + [pd.DataFrame(rows)]).to_csv(out, index=False) if False else None
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["adsh", "sup_cik", "ok"])
    pd.concat([prev, pd.DataFrame(rows)]).to_csv(out, index=False)
    v = pd.read_csv(out)
    print(len(v), "TGT edge filings checked;", int(v.ok.sum()), "kept")


def cmd_export():
    """edges.parquet: customer edges (filing-dated), industry (current SIC, first_known = fetch date), comove (yearly)."""
    g = clean_edges(load_edges())
    cust = g.rename(columns={"sup_cik": "src_cik", "sup_ticker": "src_ticker", "cust_cik": "dst_cik",
                             "cust_ticker": "dst_ticker", "file_date": "first_known"})
    cust = cust.assign(edge_type="customer", source="EDGAR 10-K full-text search", src_name=cust.sup_name,
                       valid_until=cust.first_known + pd.Timedelta(days=456), weight=cust.n_patterns.astype(float),
                       evidence=cust.patterns)[["edge_type", "src_ticker", "src_cik", "src_name", "dst_ticker", "dst_cik",
                                                "first_known", "valid_until", "weight", "source", "adsh", "form", "evidence"]]
    sic = pd.read_csv(DATA / "sic.csv", dtype={"sic": str}).dropna()
    sic["sic"] = sic.sic.str.zfill(4)
    ind = sic.merge(sic, on="sic", suffixes=("_a", "_b"))
    ind = ind[ind.ticker_a != ind.ticker_b]
    ind = pd.DataFrame({"edge_type": "industry_sic4", "src_ticker": ind.ticker_a, "dst_ticker": ind.ticker_b,
                        "first_known": pd.Timestamp("2026-10-22"), "weight": 1.0,
                        "source": "SEC submissions API current SIC (as of fetch date; not point-in-time)",
                        "evidence": "SIC " + ind.sic})
    px = pd.read_parquet(DATA / "prices.parquet")
    close = px[[c for c in px.columns if c.startswith("Close|")]]
    close.columns = [c.split("|")[1] for c in close.columns]
    r = close.pct_change(fill_method=None)
    rows = []
    for y in range(2002, 2026):
        ry = r.loc[str(y)]
        ry = ry.loc[:, ry.notna().sum() >= 200]
        if ry.shape[1] < 10:
            continue
        cm = ry.corr(min_periods=200).to_numpy(copy=True)
        np.fill_diagonal(cm, -9)
        cols = ry.columns
        top = np.argsort(-np.nan_to_num(cm, nan=-9), axis=1)[:, :5]
        for i, a in enumerate(cols):
            for j in top[i]:
                rows.append(("comove_top5", a, cols[j], ry.index[-1], float(cm[i, j])))
    cmv = pd.DataFrame(rows, columns=["edge_type", "src_ticker", "dst_ticker", "first_known", "weight"])
    cmv["source"] = "yfinance daily returns, calendar-year correlation"
    cmv["evidence"] = "corr over year ending first_known"
    e = pd.concat([cust, ind, cmv], ignore_index=True)
    e.to_parquet(DATA / "edges.parquet")
    print(e.edge_type.value_counts().to_string(), (DATA / "edges.parquet").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "data"
    {"edges": cmd_edges, "prices": cmd_prices, "earn": cmd_earn, "data": cmd_data, "dev": cmd_dev,
     "sample": cmd_sample, "export": cmd_export, "verify_target": cmd_verify_target}[cmd]()
