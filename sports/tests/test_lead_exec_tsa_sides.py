"""ops_e4_prep (2026-11-15): LEAD_TSA_SIDES -- the live TSA copy can stop mirroring high-rung NO posts
(notes/tsa_fill_review.md). Reuses the fake exchange / fixtures from test_lead_exec."""
from __future__ import annotations

import pytest

from test_lead_exec import TSA_NO, TSA_YES, ex, jw, lx, orders, run, tsa_post, write_env  # noqa: F401


def _posts(ex):
    jw(lx.TSA_PAPER, [tsa_post(q=0.96), tsa_post(t=TSA_YES, side="yes", q=0.30)])
    ex.no_books[TSA_YES] = [["0.60", "10"]]         # YES ask 0.40 > q 0.30 -> no cross


def test_default_mirrors_both_sides(tmp_path, ex):
    _posts(ex)
    run(ex)
    o = orders()
    assert o[TSA_NO]["status"] == "dry" and o[TSA_YES]["status"] == "dry"
    assert lx.tsa_sides({}) == {"yes", "no"}


def test_yes_only_skips_no_posts_at_zero_cost(tmp_path, ex):
    write_env(tmp_path, LEAD_TSA_SIDES="yes")
    _posts(ex)
    run(ex)
    o = orders()
    assert o[TSA_NO]["status"] == "skip_tsa_side_off" and lx.committed(o[TSA_NO]) == 0
    assert o[TSA_YES]["status"] == "dry" and o[TSA_YES]["count"] == 3


def test_skipped_no_posts_free_the_week_cap(tmp_path, ex):
    # 8 NO posts would use the whole $8 cap and starve the YES post; with LEAD_TSA_SIDES=yes the YES post goes
    posts = [tsa_post(t=f"KXTSAW-26DEC06-A2.{i:02d}") for i in range(8)] + [tsa_post(t=TSA_YES, side="yes", q=0.96)]
    for p in posts:
        ex.markets[p["ticker"]] = {"ticker": p["ticker"], "status": "active", "close_time": ex.markets[TSA_NO]["close_time"]}
        ex.books.setdefault(p["ticker"], [["0.01", "100"]])
    ex.no_books[TSA_YES] = [["0.02", "10"]]         # YES ask 0.98 > q 0.96 -> no cross
    jw(lx.TSA_PAPER, posts)
    run(ex)
    assert orders()[TSA_YES]["status"] == "skip_period"          # control: both sides -> cap used up by NO posts
    for f in tmp_path.glob("**/lead_exec.jsonl"):
        f.write_text("")
    write_env(tmp_path, LEAD_TSA_SIDES="yes")
    run(ex)
    o = orders()
    assert o[TSA_YES]["status"] == "dry" and o[TSA_YES]["count"] == 1
    assert all(o[p["ticker"]]["status"] == "skip_tsa_side_off" for p in posts[:8])


def test_l39_unaffected(tmp_path, ex):
    write_env(tmp_path, LEAD_TSA_SIDES="yes")
    from test_lead_exec import l39_pick
    jw(lx.L39_PICKS, [l39_pick(t=TSA_NO, q=0.96)])   # an L39 NO pick (side "no") must not be filtered
    jw(lx.TSA_PAPER, [])
    run(ex)
    assert orders()[TSA_NO]["strategy"] == "L39" and orders()[TSA_NO]["status"] == "dry"


def test_bad_value_refuses():
    with pytest.raises(ValueError):
        lx.tsa_sides({"LEAD_TSA_SIDES": "high"})
