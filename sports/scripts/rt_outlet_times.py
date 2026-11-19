"""rt_review_lag M2 (count-level C1 on history): outlet publication times for a film's critic reviews.

Review list = Metacritic's critic-review feed (the endpoint metacritic.com itself calls; robots allows it; date-only).
For each review URL, fetch the outlet page ONCE (robots.txt checked per domain, 2 s between requests, no logins,
no retries past a 4xx) and read the article's own publication timestamp (article:published_time / ld+json
datePublished / <time datetime>). Output: results/rt_review_lag/outlets_<mc_slug>.json

    python scripts/rt_outlet_times.py how-to-rob-a-bank-2026
"""
import json
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

OUT = Path(__file__).resolve().parents[1] / "results" / "rt_review_lag"
UA = "Mozilla/5.0 (compatible; quant-research-readonly; one fetch per review)"
S = requests.Session()
S.headers["User-Agent"] = UA
MC = "https://backend.metacritic.com/reviews/metacritic/critic/movies/{slug}/web"
_rp = {}

PATS = [
    r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"',
    r'<meta[^>]+content="([^"]+)"[^>]+property="article:published_time"',
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'<meta[^>]+name="(?:parsely-pub-date|pubdate|publish-date|date)"[^>]+content="([^"]+)"',
    r'<time[^>]+datetime="([^"]+)"',
]


def allowed(url):
    d = urlparse(url)
    base = f"{d.scheme}://{d.netloc}"
    if base not in _rp:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = S.get(base + "/robots.txt", timeout=20)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _rp[base] = rp
    return _rp[base].can_fetch(UA, url)


def mc_reviews(slug):
    items, off = [], 0
    while True:
        r = S.get(MC.format(slug=slug), params={"offset": off, "limit": 50}, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        got = d.get("items") or []
        items += got
        off += len(got)
        time.sleep(2)
        if not got or off >= d.get("totalResults", 0):
            return items


def main(slug):
    revs = mc_reviews(slug)
    out = []
    for x in revs:
        url = x.get("url")
        row = {"publication": x.get("publicationName"), "mc_score": x.get("score"), "mc_date": x.get("date"),
               "url": url, "published": None, "status": None, "pattern": None}
        if url and url.startswith("http"):
            if not allowed(url):
                row["status"] = "robots_disallow"
            else:
                try:
                    r = S.get(url, timeout=30)
                    row["status"] = r.status_code
                    if r.status_code == 200:
                        for i, p in enumerate(PATS):
                            m = re.search(p, r.text)
                            if m:
                                row["published"], row["pattern"] = m.group(1), i
                                break
                except requests.RequestException as e:
                    row["status"] = f"err {type(e).__name__}"
                time.sleep(2)
        out.append(row)
        print(row["publication"], row["mc_score"], row["status"], row["published"])
    json.dump(out, open(OUT / f"outlets_{slug}.json", "w"), indent=0)


if __name__ == "__main__":
    main(sys.argv[1])
