"""READ-ONLY: Open-Meteo forecast precip over the 3 h before kickoff for today's outdoor NFL games (F2 pre-kick rain unders).
F2 fires on >= 1 mm in the last 3 completed hours, 10-50 min before kickoff (live_signals.py)."""
import sys, os
import pandas as pd, requests
sys.path.insert(0, os.path.expanduser("~/.claude/quant-kalshi/scripts"))
import live_signals as ls

g = ls._nfl_schedule()
now = pd.Timestamp.now(tz="UTC")
g = g[(g.start_utc > now) & (g.start_utc < now + pd.Timedelta(hours=40))]
for _, r in g.iterrows():
    out = r.roof_eff in ("outdoors", "open")
    line = f"{r.game_id:<18} {r.start_utc:%m-%d %H:%MZ} roof_eff={r.roof_eff:<9} stadium={r.stadium_id}"
    if out and r.stadium_id in ls.STADIUMS:
        lat, lon = ls.STADIUMS[r.stadium_id]
        d = requests.get(ls.FORECAST, params={"latitude": lat, "longitude": lon, "hourly": "precipitation,wind_speed_10m",
                                              "forecast_days": 3, "timezone": "UTC", "wind_speed_unit": "mph"}, timeout=30).json()
        h = pd.DataFrame(d["hourly"]); h["time"] = pd.to_datetime(h.time, utc=True)
        k = r.start_utc.floor("h")
        w = h[(h.time >= k - pd.Timedelta(hours=3)) & (h.time < k)]
        line += f"  fcst precip 3h pre-kick {w.precipitation.sum():.1f} mm, wind {w.wind_speed_10m.mean():.0f} mph"
    print(line)
