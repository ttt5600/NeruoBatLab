"""Batch-fetch meta.yaml for every question folder in the public m-ad/claude_forecasting_auto repo via one GraphQL
query per 40 files (read-only). Writes raw/mad_meta.jsonl: title, type, tournament slug, official score, outcome."""
import json, subprocess, sys, yaml
BASE = "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/metaculus_bot_tourney/raw/"
slugs = [s.strip() for s in open(BASE + "mad_question_slugs.txt") if s.strip() and "gitkeep" not in s]
out = open(BASE + "mad_meta.jsonl", "w")
for i in range(0, len(slugs), 20):
    chunk = slugs[i:i + 20]
    fields = "\n".join(f'f{j}: object(expression: "HEAD:questions/{s}/meta.yaml") {{ ... on Blob {{ text }} }}'
                       for j, s in enumerate(chunk))
    q = f'query {{ repository(owner: "m-ad", name: "claude_forecasting_auto") {{ {fields} }} }}'
    repo = None
    for attempt in range(5):
        r = subprocess.run(["gh", "api", "graphql", "-f", f"query={q}"], capture_output=True, text=True, timeout=120)
        try:
            repo = json.loads(r.stdout)["data"]["repository"]; break
        except Exception:
            print("retry", i, attempt, r.stderr[:200], file=sys.stderr)
    if repo is None:
        for s in chunk: out.write(json.dumps({"slug": s, "err": 1}) + "\n")
        continue
    for j, s in enumerate(chunk):
        blob = repo.get(f"f{j}")
        if not blob:
            out.write(json.dumps({"slug": s, "err": 1}) + "\n"); continue
        d = yaml.safe_load(blob["text"]) or {}
        t = d.get("tournament") or {}; res = d.get("resolution") or {}
        out.write(json.dumps({"slug": s, "title": d.get("title"), "type": d.get("type"), "tourn": t.get("slug"),
                              "score": t.get("official_score"), "missed": t.get("missed_window"),
                              "opened": t.get("opened"), "outcome": res.get("outcome")}) + "\n")
    print(i + len(chunk), file=sys.stderr)
out.close()
