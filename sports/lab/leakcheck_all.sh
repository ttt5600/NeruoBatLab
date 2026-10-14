#!/bin/bash
# Lookahead-check every strategy whose dataset is available locally. Not a trial: prints no returns.
#   bash lab/leakcheck_all.sh            # -> one line per strategy, full detail for failures
cd "$(dirname "$0")/.." || exit 1          # -> sports/
PY=${PY:-../.venv_quant/bin/python}
for f in lab/strategies/*.py; do
    ds=$(grep -m1 -E '^DATASET *=' "$f" | sed -E 's/.*= *["'"'"']([^"'"'"']+).*/\1/')
    if [ ! -f "data/lab/$ds.parquet" ] && [ ! -f "lab/devdata/$ds.parquet" ]; then
        echo "SKIP  $(basename "$f" .py) ($ds: no local data)"; continue
    fi
    out=$("$PY" lab/run.py leakcheck "$f" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then echo "PASS  $(basename "$f" .py) ($ds)"
    else echo "FAIL  $(basename "$f" .py) ($ds)"; echo "$out" | sed 's/^/      /'; fi
done
