"""E5 06:00 station-local cutoff (2026-11-12, notes/e5_fix.md F1)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import forecastex_exec as f  # noqa: E402


def test_local_cutoff_is_6am_station_time_in_utc():
    assert f.local_cutoff("ATL", "2026-11-12") == datetime(2026, 11, 12, 11, tzinfo=timezone.utc)   # EST
    assert f.local_cutoff("PHX", "2026-11-12") == datetime(2026, 11, 12, 13, tzinfo=timezone.utc)   # MST, no DST
    assert f.local_cutoff("SFO", "2026-07-01") == datetime(2026, 7, 1, 13, tzinfo=timezone.utc)     # PDT


def test_cutoff_can_be_disabled_and_unknown_station_falls_back():
    assert f.local_cutoff("ATL", "2026-11-12", {"FX_CUTOFF_LOCAL_HOUR": "off"}) is None
    assert f.local_cutoff("XXX", "2026-11-12") is None
    assert f.local_cutoff("ATL", None) is None


def test_every_logged_station_has_a_timezone():
    for code in ["LGA", "LAX", "MSP", "BNA", "DCA", "PHL", "SEA", "DFW", "SFO", "MDW", "ATL", "OKC", "BOS", "SAT",
                 "PHX", "CLT", "AUS", "MIA", "LAS", "MSY", "HOU", "DTW", "JAX", "BKF"]:
        assert code in f.STATION_TZ
