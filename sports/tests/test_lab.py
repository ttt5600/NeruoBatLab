"""The lab harness: what agents cannot do, and what the scorer never shows them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import datasets as D
from lab import registry, run
from lab import sandbox as S

FIX = Path(__file__).parent / "fixtures"


def test_adversarial_strategy_is_rejected():
    with pytest.raises(S.StrategyRejected, match="read_parquet"):
        S.check_source((FIX / "adversarial_strategy.py").read_text())


@pytest.mark.parametrize("snippet", [
    "import os", "import subprocess", "from pathlib import Path", "x = open('f')",
    "x = __import__('os')", "x = pd.io", "x = np.load", "x = getattr(pd, 'read_csv')",
    "x = '/Users/me/data/lab/nfl_total.parquet'", "x = df.to_csv",
])
def test_static_check_blocks_escape_hatches(snippet):
    src = (f"import pandas as pd\nimport numpy as np\nDATASET='nfl_total'\nHYPOTHESIS='h'\nSOURCE='s'\n"
           f"{snippet}\ndef fit(t): return None\ndef bet(m, s): return pd.Series(0.0, index=s.index)\n")
    with pytest.raises(S.StrategyRejected):
        S.check_source(src)


def test_ordinary_numeric_code_is_allowed():
    S.check_source("import numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
                   "DATASET='nfl_total'\nHYPOTHESIS='h'\nSOURCE='s'\n"
                   "def fit(t): return t['won'].to_numpy().mean()\n"
                   "def bet(m, s): return pd.Series(0.0, index=s.index)\n")


SPY = '''import pandas as pd
DATASET = "fake"
HYPOTHESIS = "spy"
SOURCE = "test"
def fit(train):
    return int(train["fold"].max())
def bet(model, slate):
    hidden = [c for c in ("won", "push", "actual", "home_score", "away_score") if c in slate.columns]
    assert not hidden, f"outcome columns visible to bet(): {hidden}"
    assert model < int(slate["fold"].min()), "fit() saw the fold it is betting on"
    return pd.Series((slate["selection"] == "a").astype(float).to_numpy(), index=slate.index)
'''


def test_walk_forward_hides_outcomes_and_the_future(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(1, 6):
        for i in range(40):
            w = int(rng.integers(0, 2))
            for sel, won in (("a", w), ("b", 1 - w)):
                rows.append(dict(opp_id=f"{fold}_{i}", fold=fold, date=f"2020-0{fold}-{i % 28 + 1:02d}",
                                 selection=sel, dec_odds=1.95, won=won, push=0, actual=w))
    df = pd.DataFrame(rows)
    monkeypatch.setitem(D.SPLITS, "fake", ("fold", [1, 2, 3, 4, 5], []))
    monkeypatch.setitem(D.MIN_TRAIN_FOLDS, "fake", 1)
    monkeypatch.setattr(D, "load", lambda name, split: df)
    strat = tmp_path / "spy.py"
    strat.write_text(SPY)
    res, tested = run.walk_forward(strat, "fake", "dev")
    assert sorted(tested["fold"].unique()) == [2, 3, 4, 5]
    assert res["bets"] == 160
    exp = sum(0.95 if w else -1.0 for w in df[(df.fold > 1) & (df.selection == "a")]["won"])
    assert res["units"] == pytest.approx(exp)


GOOD = {"title": "Wind unders", "sport": "nfl", "market": "total", "mechanism": "totals slow to adjust for wind",
        "claimed_edge": "x", "evidence": "claim", "sources": ["https://example.com/a"], "data_needed": "wind",
        "testable_with": ["nfl_total"], "skeptic_note": "observed not forecast wind"}


def test_registry_validates_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REG", tmp_path / "s.jsonl")
    assert registry.add(json.dumps(GOOD)) == 0
    assert registry.add(json.dumps(GOOD)) == 2, "duplicate must be refused"
    assert registry.add(json.dumps({**GOOD, "title": "Other", "mechanism": "different thing entirely",
                                    "sources": ["not-a-url"]})) == 1
    assert registry.add(json.dumps({**GOOD, "title": "Other2", "mechanism": "another different thing",
                                    "testable_with": ["nba_points"]})) == 1


# ---- the bar needs a sample: one winning day bootstraps to p = 0
def test_one_winning_day_does_not_meet_the_bar():
    df = pd.DataFrame({"date": ["2020-01-01"], "won": [1], "push": [0], "dec_odds": [2.0], "fold": [1]})
    res = run.score(df, pd.Series([1.0], index=df.index), "fold")
    assert res["p"] == 0.0 and res["bet_days"] == 1      # the artefact the rule exists for
    assert not run.meets_bar(res, 1)
    assert run.meets_bar({**res, "bet_days": run.MIN_BET_DAYS}, 1)


def test_status_threshold_matches_run():
    from lab import status
    assert status.MIN_BET_DAYS == run.MIN_BET_DAYS
    assert "## Trials by dataset" in status.build()


# ---- lookahead check: bet() must not read later games in its own slate
HEAD = ('import numpy as np, pandas as pd\nDATASET = "fake"\nHYPOTHESIS = "h"\nSOURCE = "test"\n'
        'def fit(train):\n    return None\n')
CLEAN = HEAD + 'def bet(m, s):\n    return pd.Series((s["line"] > 0.5).astype(float).to_numpy(), index=s.index)\n'
SAME_DAY = HEAD + ('def bet(m, s):\n    # relative to the same DAY only: knowable before kickoff, must pass\n'
                   '    d = s.groupby("date")["line"].transform("mean")\n'
                   '    return pd.Series((s["line"] > d).astype(float).to_numpy(), index=s.index)\n')
SEASON_MEAN = HEAD + ('def bet(m, s):\n    # above the SEASON mean: uses games not yet played\n'
                      '    return pd.Series((s["line"] > s["line"].mean()).astype(float).to_numpy(), index=s.index)\n')
SEASON_RANK = HEAD + ('def bet(m, s):\n    r = s["line"].rank(pct=True)\n'
                      '    return pd.Series((r > 0.8).astype(float).to_numpy(), index=s.index)\n')
UNSEEDED = HEAD + ('def bet(m, s):\n    x = np.random.default_rng().integers(0, 2, len(s)).astype(float)\n'
                   '    return pd.Series(x, index=s.index)\n')


@pytest.fixture
def fake_lines(monkeypatch):
    rng = np.random.default_rng(1)
    rows = []
    for fold in range(1, 5):
        for i in range(60):
            line = float(rng.random()) + 0.3 * (i / 60)      # drifts up through the season
            rows.append(dict(opp_id=f"{fold}_{i}", fold=fold, date=f"2020-{fold:02d}-{i // 2 + 1:02d}",
                             selection="a", dec_odds=1.95, line=line, won=int(rng.integers(0, 2)), push=0))
    df = pd.DataFrame(rows)
    monkeypatch.setitem(D.SPLITS, "fake", ("fold", [1, 2, 3, 4], []))
    monkeypatch.setitem(D.MIN_TRAIN_FOLDS, "fake", 1)
    monkeypatch.setattr(D, "load", lambda name, split: df)
    return df


@pytest.mark.parametrize("name,ok", [("CLEAN", True), ("SAME_DAY", True), ("SEASON_MEAN", False),
                                     ("SEASON_RANK", False), ("UNSEEDED", False)])
def test_lookahead_check(tmp_path, fake_lines, name, ok):
    p = tmp_path / "s.py"
    p.write_text(globals()[name])
    lc = run.lookahead_check(p, "fake")
    assert lc["pass"] is ok, lc
    if name == "UNSEEDED":
        assert lc["nondeterministic"] and not lc["leaks"]
    if name.startswith("SEASON"):
        assert lc["leaks"]
