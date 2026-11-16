"""Download Wayback vintages of Zillow Metro ZHVI CSVs listed in raw/wb/metro_list.txt (read-only, polite, retries)."""
import time
from pathlib import Path
import requests

H = Path(__file__).resolve().parent
D = H / "raw" / "wb" / "metro"
D.mkdir(parents=True, exist_ok=True)
def cdx():
    rows = []
    for name in ("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month", "Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month"):
        for i in range(10):
            try:
                r = requests.get("http://web.archive.org/cdx/search/cdx", timeout=120, params=dict(
                    url=f"files.zillowstatic.com/research/public_csvs/zhvi/{name}.csv*", fl="timestamp,original,length",
                    filter="statuscode:200", collapse="digest", output="json"))
                rows += [x for x in r.json()[1:] if int(x[2]) > 100_000]
                break
            except Exception as e:
                print("cdx retry", name, e, flush=True)
                time.sleep(30)
    seen, out = set(), []
    for ts, url, _ in sorted(rows):
        if ts not in seen:
            seen.add(ts)
            out.append(f"{ts} {url}")
    (H / "raw" / "wb" / "metro_list.txt").write_text("\n".join(out))
    print("cdx rows", len(out), flush=True)


cdx()
for line in (H / "raw" / "wb" / "metro_list.txt").read_text().split("\n"):
    if not line.strip():
        continue
    ts, url = line.split()
    kind = "sfr" if "_sfr_tier" in url else "all"
    f = D / f"{kind}_{ts}.csv"
    if f.exists() and f.stat().st_size > 100_000:
        continue
    for i in range(6):
        try:
            r = requests.get(f"http://web.archive.org/web/{ts}id_/{url}", timeout=120)
            if r.status_code == 200 and r.content[:8] == b"RegionID":
                f.write_bytes(r.content)
                print("ok", f.name, len(r.content), flush=True)
                break
            print("bad", ts, r.status_code, r.content[:40], flush=True)
        except requests.RequestException as e:
            print("err", ts, e, flush=True)
        time.sleep(15 * (i + 1))
    time.sleep(3)
