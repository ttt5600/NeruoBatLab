F=/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/ufc_tsa
until ! pgrep -f fetch_tsa_candles.py >/dev/null; do sleep 10; done
wc -l $F/candles_tsad_dev.jsonl
