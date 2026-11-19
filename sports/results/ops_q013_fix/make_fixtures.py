"""One-off: build trimmed RT page fixtures for tests/test_q013_q014_loggers.py from a live capture.
usage: python make_fixtures.py <captured_ipl.html> <fixtures_dir>"""
import json
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
h = src.read_text()
canon = re.search(r'<link rel="canonical"[^>]*>', h).group(0)
sc = re.search(r'<script[^>]*id="media-scorecard-json"[^>]*>.*?</script>', h, re.S).group(0)
ld = re.search(r'<script type="application/ld\+json">.*?</script>', h, re.S).group(0)
hdr = ("<!-- captured from https://www.rottentomatoes.com/m/i_play_rocky on 2026-11-18 "
       "(trimmed to the tags q013_logger parses) -->\n")
(out / "rt_page_scorepercent_2026-11-18.html").write_text(
    f"<html><head>{hdr}{canon}\n{ld}\n</head><body>\n{sc}\n</body></html>\n")

m = re.search(r'(<script[^>]*id="media-scorecard-json"[^>]*>)(.*?)(</script>)', sc, re.S)
d = json.loads(m.group(2))
c = d["criticsScore"]
c.pop("scorePercent")
c.update({"score": "87", "likedCount": 54, "notLikedCount": 8, "reviewCount": 62, "ratingCount": 62,
          "averageRating": "7.30", "certified": False})
hdr2 = ("<!-- LEGACY shape (criticsScore.score, used by RT until 2026-11-03 ~22:41Z): built from the 2026-11-18 capture "
        "with criticsScore\n     rewritten to the values q013 logged for /m/i_play_rocky at ts 1793735314 "
        "(score 87, n 62, liked 54). Not a raw capture. -->\n")
(out / "rt_page_score_legacy.html").write_text(
    f"<html><head>{hdr2}{canon}\n{ld}\n</head><body>\n{m.group(1)}{json.dumps(d)}{m.group(3)}\n</body></html>\n")
print("ok")
