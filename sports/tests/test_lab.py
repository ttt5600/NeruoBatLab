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
