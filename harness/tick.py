#!/usr/bin/env python3
"""Decide, once per Savio-watcher poll, whether the research agent should run.

The watcher (~/.claude/savio-watch) writes what it saw each poll -- the queue and the jobs that
ended -- and calls this. Bookkeeping happens here; judgement happens in the agent:
  - a `submitted` experiment whose job ended COMPLETED with its artifacts  -> `trained` -> agent
  - one that ended FAILED/TIMEOUT/OOM/CANCELLED, or COMPLETED with nothing written -> `failed` -> agent
  - fewer than max_concurrent jobs in flight and an `approved` experiment waiting -> agent (submit it),
    at most once every IDLE_EVERY seconds
run_agent.sh enforces one agent at a time and a daily cap.

While the long-lived lead agent (harness/PI.md, a /loop session) is alive -- its heartbeat file is
under 2 h old -- this stands down completely, so two agents never act on the queue at once. If the
lead stops, the event-driven agents take over again on the next poll.

  python3 harness/tick.py CYCLE.json [--dry-run]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

H = Path(__file__).resolve().parent
QUEUE = H / "experiments.yaml"
STATE = H / ".tick_state.json"
IDLE_EVERY = 6 * 3600
PI_HEARTBEAT = H / ".pi_heartbeat"
PI_FRESH = 2 * 3600


def load_queue():
    text = QUEUE.read_text()
    header = "".join(l for l in text.splitlines(True) if l.startswith("#"))  # yaml.dump drops comments
    return yaml.safe_load(text), header


def save_queue(doc, header):
    QUEUE.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))


def main():
    dry = "--dry-run" in sys.argv
    if PI_HEARTBEAT.exists() and time.time() - PI_HEARTBEAT.stat().st_mtime < PI_FRESH:
        print("lead agent alive (heartbeat %d min old); standing down"
              % ((time.time() - PI_HEARTBEAT.stat().st_mtime) / 60))
        return
    cycle = json.loads(Path(sys.argv[1]).read_text())
    doc, header = load_queue()
    exps = doc["experiments"]
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    reasons, changed = [], False

    by_job = {str(e["job"]): e for e in exps if e.get("status") == "submitted" and e.get("job")}
    for end in cycle.get("ended", []):
        e = by_job.get(str(end["jid"]))
        if e is None:
            continue
        if end["state"] == "COMPLETED" and end.get("ok") is not False:
            e["status"] = "trained"
            verified = "artifacts verified" if end.get("ok") else "artifact check did not run -- verify"
            reasons.append(f"{e['id']} ({e['tag']}) finished: job {end['jid']} COMPLETED after "
                           f"{end['elapsed']}, {verified}. Score it.")
        else:
            e["status"] = "failed"
            reasons.append(f"{e['id']} ({e['tag']}) job {end['jid']} ended {end['state']} after "
                           f"{end['elapsed']} (artifacts ok: {end.get('ok')}). Investigate.")
        changed = True

    in_flight = [e for e in exps if e.get("status") == "submitted"]
    approved = sorted((e for e in exps if e.get("status") == "approved"),
                      key=lambda e: e.get("priority", 99))
    room = int(doc.get("max_concurrent", 2)) - len(in_flight)
    if approved and room > 0 and time.time() - st.get("last_idle", 0) > IDLE_EVERY:
        reasons.append(f"{len(in_flight)} experiment job(s) in flight, room for {room}; approved "
                       f"and waiting: {', '.join(e['id'] for e in approved)}. Submit the next.")
        if not dry:
            st["last_idle"] = time.time()

    if changed and not dry:
        save_queue(doc, header)
    if reasons:
        msg = " | ".join(reasons)
        if dry:
            print("WOULD START AGENT:", msg)
        else:
            out = subprocess.run(["bash", str(H / "run_agent.sh"), msg], capture_output=True, text=True)
            print(out.stdout.strip() or out.stderr.strip())
    elif dry:
        print("nothing to do")
    if not dry:
        STATE.write_text(json.dumps(st))


if __name__ == "__main__":
    main()
