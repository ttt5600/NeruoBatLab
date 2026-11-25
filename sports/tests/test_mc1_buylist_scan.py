"""Offline tests for mc1_buylist_scan (MC1 / T-MTG scanner): strict signal, naming, exclusions, net math, outputs and
the exit-2 network path. Tiny hand-built fixture; no network (downloads and Scryfall are injected)."""
import gzip
import json
import lzma
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mc1_buylist_scan as M  # noqa: E402

D = "2026-11-24"


def mj_rec(ck_buy, tcg, mp, ck_retail=None, fin="foil"):
    """One MTGJSON AllPricesToday record with the given latest prices (None = provider missing)."""
    paper = {"cardkingdom": {"buylist": {fin: {D: ck_buy}}, "retail": {fin: {D: ck_retail}} if ck_retail else {}},
             "tcgplayer": {"buylist": {}, "retail": {fin: {D: tcg}}}}
    if mp is not None:
        paper["manapool"] = {"buylist": {}, "retail": {fin: {D: mp}}}
    return {"paper": paper}


def ck_row(sid, name, buy, retail, foil=True, qty_buying=5, edition="Invasion"):
    return {"sku": "X-1", "scryfall_id": sid, "url": "mtg\\/invasion\\/" + name.lower().replace(" ", "-"),
            "name": name, "variation": "", "edition": edition, "is_foil": "true" if foil else "false",
            "price_retail": f"{retail:.2f}", "qty_retail": 2, "price_buy": f"{buy:.2f}", "qty_buying": qty_buying}


def sf_card(sid, name, usd=None, usd_foil=None, released="2000-10-02", reserved=False, tcgplayer_id=7464):
    return {"id": sid, "name": name, "set": "inv", "set_name": "Invasion", "collector_number": "12",
            "released_at": released, "reserved": reserved, "tcgplayer_id": tcgplayer_id,
            "scryfall_uri": f"https://scryfall.com/card/inv/12/x?utm_source=api",
            "prices": {"usd": usd, "usd_foil": usd_foil}}


# uuid -> (MTGJSON record, expected outcome)
MJ = {
    "A": mj_rec(16.5, 6.49, 6.18, 32.99),            # flag: Crusading Knight INV foil (census row), net 8.41
    "B": mj_rec(10.0, 6.00, 5.00, 19.99),             # net 2.445 < $3 -> not a signal row
    "C": mj_rec(16.5, 6.49, 13.00, 32.99),            # Manapool (second venue) is expensive -> net 1.30 -> no
    "D": mj_rec(80.0, 50.00, 40.00, 159.99),          # max price = $50, not < $50 -> no
    "E": mj_rec(16.5, 6.49, None, 32.99),             # no Manapool price -> no (strict: both venues required)
    "F": mj_rec(55.0, 28.26, 21.87, 109.99, "normal"),  # Bad Moon Unlimited -> pre-1995 exclusion
    "G": mj_rec(40.0, 20.00, 19.00, 79.99),           # Reserved List card -> excluded
    "H": mj_rec(24.0, 12.00, 11.00, 47.99),           # CK qty_buying 0 -> excluded
    "I": mj_rec(36.0, 24.78, 24.67, None),            # no CK retail in MTGJSON -> unresolved (no_ck_retail)
    "J": mj_rec(30.0, 15.00, 14.00, 59.99),           # two CK rows, both Scryfall prices equal -> ambiguous
    "K": mj_rec(13.8, 7.44, 7.75, 22.99),             # Scryfall price differs from MTGJSON TCG -> no_match
    "L": mj_rec(25.0, 18.33, 13.49, 39.99, "normal"),  # flag, nonfoil, qty 16 (Bruna SLD census row), net 3.97
}
CK = [
    ck_row("sA", "Crusading Knight", 16.5, 32.99),
    ck_row("sA2", "Other Tier Card", 16.5, 32.99),       # same tier, different Scryfall price -> filtered by price
    ck_row("sF", "Bad Moon", 55.0, 109.99, foil=False, qty_buying=1, edition="Unlimited"),
    ck_row("sG", "Reserved Card", 40.0, 79.99),
    ck_row("sH", "Zero Qty", 24.0, 47.99, qty_buying=0),
    ck_row("sJ1", "Twin One", 30.0, 59.99),
    ck_row("sJ2", "Twin Two", 30.0, 59.99),
    ck_row("sK", "Moved Price", 13.8, 22.99),
    ck_row("sL", "Bruna, the Fading Light", 25.0, 39.99, foil=False, qty_buying=16, edition="Secret Lair"),
]
SF = {c["id"]: c for c in [
    sf_card("sA", "Crusading Knight", usd="1.49", usd_foil="6.49"),
    sf_card("sA2", "Other Tier Card", usd_foil="9.99"),
    sf_card("sF", "Bad Moon", usd="28.26", released="1993-12-01"),
    sf_card("sG", "Reserved Card", usd_foil="20.00", released="1998-10-12", reserved=True),
    sf_card("sH", "Zero Qty", usd_foil="12.00"),
    sf_card("sJ1", "Twin One", usd_foil="15.00"),
    sf_card("sJ2", "Twin Two", usd_foil="15.00"),
    sf_card("sK", "Moved Price", usd_foil="7.99"),
    sf_card("sL", "Bruna, the Fading Light", usd="18.33", released="2023-08-10", tcgplayer_id=511868),
]}


def fake_post(ids):
    assert len(ids) <= M.SCRY_BATCH
    return {"data": [SF[i] for i in ids if i in SF]}


def run_pipeline():
    cands = M.signal_candidates(MJ)
    idx = M.ck_index(CK, {M.cand_key(c) for c in cands if M.cand_key(c)})
    scry = M.fetch_scryfall(M.needed_ids(cands, idx), post=fake_post, spacing=0)
    return cands, M.resolve(cands, idx, scry)


def test_signal_is_the_strict_census_filter():
    cands = M.signal_candidates(MJ)
    assert sorted(c["uuid"] for c in cands) == ["A", "F", "G", "H", "I", "J", "K", "L"]
    a = next(c for c in cands if c["uuid"] == "A")
    assert a["price"] == 6.49 and a["net"] == 8.41 and a["cost"] == 8.09   # 16.5 - (6.49 x 1.0925 + 1)


def test_signal_uses_latest_date_and_price_cap_edge():
    rec = mj_rec(16.5, 6.49, 6.18, 32.99)
    rec["paper"]["tcgplayer"]["retail"]["foil"]["2026-11-01"] = 1.00   # older, cheaper quote must be ignored
    rec["paper"]["manapool"]["retail"]["foil"] = {"2026-11-23": 1.0, "2026-11-24": 13.0}
    assert M.signal_candidates({"x": rec}) == []        # latest Manapool 13.00 kills it
    ok = mj_rec(60.0, 49.99, 40.0, 119.99)
    assert len(M.signal_candidates({"y": ok})) == 1     # 49.99 < 50 passes, 50.00 (row D) does not


def test_resolve_names_excludes_and_counts():
    cands, res = run_pipeline()
    assert [f["name"] for f in res["flags"]] == ["Crusading Knight", "Bruna, the Fading Light"]   # sorted by net
    exc = {e["name"]: e["excluded"] for e in res["excluded"]}
    assert exc == {"Bad Moon": ["pre1995"], "Reserved Card": ["reserved_list"], "Zero Qty": ["qty_buying_0"]}
    unres = {u["uuid"]: u["reason"] for u in res["unresolved"]}
    assert unres == {"I": "no_ck_retail", "J": "ambiguous", "K": "no_match"}
    s = M.summarize(D, res, len(cands), {})
    assert (s["n_flags"], s["n_signal"], s["n_unresolved"]) == (2, 8, 3)
    assert (s["n_excluded_pre1995"], s["n_excluded_reserved"], s["n_excluded_qty0"]) == (1, 1, 1)


def test_net_math_and_links():
    _, res = run_pipeline()
    ck, br = res["flags"]
    assert ck["buy_price"] == 6.49 and ck["ck_buy_nm_cash"] == 16.5 and ck["ck_qty_buying"] == 5
    assert ck["net"] == 8.41 and ck["net_qty"] == 42.05 and ck["max_ok_price"] == 11.44   # (16.5 - 4) / 1.0925
    assert ck["tcg_url"] == ("https://www.tcgplayer.com/product/7464?Language=English&Printing=Foil"
                             "&Condition=Near+Mint")
    assert ck["ck_url"] == "https://www.cardkingdom.com/mtg/invasion/crusading-knight"
    assert ck["scryfall_uri"] == "https://scryfall.com/card/inv/12/x"
    # Bruna (nonfoil): price used = max(TCG 18.33, Manapool 13.49); net 25 - (18.33 x 1.0925 + 1) = 3.974475
    assert br["buy_price"] == 18.33 and br["net"] == 3.97 and br["net_qty"] == 63.52 and not br["foil"]
    assert "Printing=Normal" in br["tcg_url"]
    s = M.summarize(D, res, 0, {})
    assert s["net_1copy"] == 12.38 and s["net_fullqty"] == 105.57 and s["units_fullqty"] == 21
    # cost totals sum UNROUNDED per-card costs (census convention): 8.090325 + 21.025525 = 29.11585
    assert s["cost_1copy"] == 29.12 and s["cost_fullqty"] == round(8.090325 * 5 + 21.025525 * 16, 2)


def test_spot_sample_is_seeded_and_capped():
    flags = [{"i": i} for i in range(25)]
    a, b = M.spot_sample(D, flags), M.spot_sample(D, flags)
    assert a == b and len(a) == 10 and a == sorted(set(a))
    assert M.spot_sample("2026-11-25", flags) != a
    assert M.spot_sample(D, flags[:3]) == [0, 1, 2]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MIN_CK_ROWS", 1)
    mj = tmp_path / "in" / "AllPricesToday.json.xz"
    mj.parent.mkdir()
    with lzma.open(mj, "wt") as fh:
        json.dump({"meta": {"date": D, "version": "5.3.0+test"}, "data": MJ}, fh)
    ck = tmp_path / "in" / "ck.json.gz"
    with gzip.open(ck, "wt") as fh:
        json.dump({"meta": {"created_at": "2026-11-24 19:07:54"}, "data": CK}, fh)
    return mj, ck, tmp_path / "out"


def test_main_writes_json_md_and_appends_daily(inputs):
    mj, ck, out = inputs
    argv = ["--out", str(out), "--date", D, "--mtgjson-file", str(mj), "--ck-file", str(ck)]
    assert M.main(argv, post=fake_post) == 0
    rep = json.loads((out / f"{D}.json").read_text())
    assert rep["summary"]["n_flags"] == 2 and rep["summary"]["mtgjson_date"] == D and rep["summary"]["warnings"] == []
    need = {"name", "set", "number", "foil", "tcg_price", "tcg_url", "manapool_price", "ck_buy_nm_cash",
            "ck_qty_buying", "net", "net_qty", "cost", "cost_qty"}
    assert all(need <= set(f) for f in rep["flags"])
    assert len(rep["unresolved"]) == 3 and len(rep["excluded"]) == 3 and rep["spot_sample"] == [0, 1]
    md = (out / f"{D}.md").read_text()
    assert "**2 flags.**" in md and "[Crusading Knight](https://scryfall.com/card/inv/12/x)" in md
    assert "https://www.tcgplayer.com/product/511868?" in md and "- [ ] #1 Crusading Knight" in md
    lines = (out / "daily.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert (row["date"], row["n_flags"], row["net_1copy"], row["net_fullqty"]) == (D, 2, 12.38, 105.57)
    assert M.main(argv, post=fake_post) == 0                       # a re-run appends, never rewrites
    assert len((out / "daily.jsonl").read_text().splitlines()) == 2
    assert inputs[0].exists() and inputs[1].exists()                # cached inputs are never deleted


def test_stale_mtgjson_warns(inputs):
    mj, ck, out = inputs
    assert M.main(["--out", str(out), "--date", "2026-11-30", "--mtgjson-file", str(mj), "--ck-file", str(ck)],
                  post=fake_post) == 0
    assert "6 days older" in json.loads((out / "2026-11-30.json").read_text())["summary"]["warnings"][0]


def test_network_failure_exits_2_and_writes_nothing(tmp_path, inputs):
    mj, ck, out = inputs

    def dead_dl(url, dest, headers=None):
        raise M.NetError("offline")

    assert M.main(["--out", str(out), "--date", D], dl=dead_dl, post=fake_post) == 2
    assert not out.exists()

    def dead_post(ids):
        raise M.NetError("scryfall 503")

    assert M.main(["--out", str(out), "--date", D, "--mtgjson-file", str(mj), "--ck-file", str(ck)],
                  post=dead_post) == 2
    assert not out.exists()


def test_truncated_download_exits_2_and_temp_dir_removed(tmp_path, inputs, monkeypatch):
    mj, ck, out = inputs
    made = []
    real_mkdtemp = M.tempfile.mkdtemp
    monkeypatch.setattr(M.tempfile, "mkdtemp", lambda **kw: made.append(real_mkdtemp(dir=tmp_path, **kw)) or made[-1])

    def short_dl(url, dest, headers=None):
        dest.write_bytes(mj.read_bytes()[:200])     # a cut-off .xz stream

    assert M.main(["--out", str(out), "--date", D], dl=short_dl, post=fake_post) == 2
    assert not out.exists() and made and not Path(made[0]).exists()


def test_live_download_path_deletes_files(tmp_path, inputs, monkeypatch):
    mj, ck, out = inputs
    seen = []

    def copy_dl(url, dest, headers=None):
        src = mj if "mtgjson" in url else ck
        dest.write_bytes(src.read_bytes())
        seen.append((url, dest, headers))

    assert M.main(["--out", str(out), "--date", D], dl=copy_dl, post=fake_post) == 0
    assert [u for u, _, _ in seen] == [M.MTGJSON_URL, M.CK_URL]
    assert seen[1][2]["Accept-Encoding"] == "gzip"
    assert all(not d.exists() and not d.parent.exists() for _, d, _ in seen)
