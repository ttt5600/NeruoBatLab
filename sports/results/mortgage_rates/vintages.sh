#!/bin/bash
# ALFRED Wednesday vintages via curl (python-requests is tarpitted by alfred.stlouisfed.org: read timeouts).
set -u
OUT="$(cd "$(dirname "$0")" && pwd)/vintages"
mkdir -p "$OUT"
d=2022-12-28
while [[ "$d" < "2026-11-19" ]]; do
  for sid in OBMMIC30YF MORTGAGE30US; do
    f="$OUT/${sid}_$d.csv"
    if [ ! -s "$f" ] || ! head -1 "$f" | grep -q observation_date; then
      for k in 1 2 3; do
        curl -s -m 40 -o "$f" "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=$sid&vintage_date=$d" && head -1 "$f" | grep -q observation_date && break
        sleep $((k * 3))
      done
      sleep 0.3
    fi
  done
  d=$(date -j -v+7d -f %Y-%m-%d "$d" +%Y-%m-%d)
done
ls "$OUT" | wc -l
