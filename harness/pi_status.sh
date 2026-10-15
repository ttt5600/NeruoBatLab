#!/bin/bash
# One-call wake snapshot for the /loop lead (harness/PI.md step 0-2). Prints only what changes between
# wakes, so a freshly /clear-ed session needs this + harness/STATE.md and nothing else to resume.
#   bash harness/pi_status.sh            (also writes the heartbeat)
#   NOHB=1 bash harness/pi_status.sh     (look without claiming the pipeline)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
Q() { grep -v -iE 'post-quantum|store now|upgraded|pq.html|mux_client|ControlSocket'; }

[ -n "${NOHB:-}" ] || date -Iseconds > harness/.pi_heartbeat
echo "== $(date '+%F %H:%M')"

# other agents on this repo (the usage-watch auto-resume has collided twice)
n=$(ps aux | grep -E "claude -p" | grep -v grep | grep -c "vocalizations\|knowledge/CONTEXT\|PI.md\|AGENT.md" || true)
[ "$n" -gt 0 ] && echo "!! $n other 'claude -p' agent(s) running on this project -- stand down / reconcile"

# login + certificate
exp=$(ssh-keygen -L -f ~/.ssh/ssh_certs/brc_cert-cert.pub 2>/dev/null | sed -n 's/.*Valid: from .* to //p')
if ssh -o BatchMode=yes -o ConnectTimeout=15 savio-login true 2>/dev/null; then
    echo "login OK; cert until ${exp:-?}"
    ssh -o BatchMode=yes savio-login '
        echo "-- queue"; squeue -u $USER -h -o "%i %j %T %M %R" | sed "s/^/   /"
        echo "-- ended (3 d)"; sacct -u $USER -S now-3days -X -n -o JobID,JobName%18,State%12,Elapsed,End | grep -v -E "jupyter|^ *$" | tail -8 | sed "s/^/   /"
        for d in /global/scratch/users/$USER/temp_train_*; do
            c=$(ls -t $d/checkpoints_*/ 2>/dev/null | grep -E "^epoch=|^stepstep=" | head -1)
            [ -n "$c" ] && [ $(( $(date +%s) - $(stat -c %Y $d/checkpoints_*/last.ckpt 2>/dev/null || echo 0) )) -lt 43200 ] \
                && echo "   ckpt $(basename $d): $c"
        done' 2>&1 | Q
else
    echo "!! Savio login REFUSED (cert expired ${exp:-?}) -- notify 'run sv login', skip Savio work"
fi

# the search
python3 - <<'PY'
import yaml
d = yaml.safe_load(open("harness/experiments.yaml"))
rows = [f"{e['id']}:{e['status']}" + (f"({e['job']})" if e.get('job') and e['status'] in ('submitted','trained') else "")
        for e in d["experiments"] if e["status"] not in ("done", "rejected")]
print("-- open experiments:", " ".join(rows) or "none", f"| max_concurrent {d['max_concurrent']}")
PY
echo "-- newest finding: $(ls knowledge/findings | tail -1 | cut -c1-60)"

# git: anything left uncommitted from a killed agent?
echo "-- git: $(git log -1 --format='%h %ar %s' | cut -c1-90)"
git status --short | grep -v '^??' | sed 's/^/   dirty /'
(cd pytorchAudio && git status --short | grep -v '^??' | sed 's/^/   pytorchAudio dirty /')
ls harness/digest | tail -1 | sed 's/^/-- last digest: /'
