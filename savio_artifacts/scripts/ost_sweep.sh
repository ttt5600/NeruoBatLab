#!/bin/bash
SC=/global/scratch/users/jonathanswang
for D in $SC/external/fsd50k/FSD50K.dev_audio $SC/external/fsd50k/FSD50K.eval_audio $SC/external/aves $SC/external/dapt; do
  [ -d "$D" ] || continue
  OK=0; BAD=0
  while IFS= read -r f; do
    if head -c 65536 "$f" > /dev/null 2>&1; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "UNREADABLE $f"; fi
  done < <(find "$D" -type f \( -name "*.wav" -o -name "*.pt" -o -name "*.ckpt" \) 2>/dev/null)
  echo "RESULT $D readable=$OK unreadable=$BAD"
done
echo DONE
