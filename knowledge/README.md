# Knowledge base

A findings store for the zebra finch HuBERT project. One YAML file per finding under `findings/`,
validated and rendered by `build.py`.

```bash
python3 build.py     # validates every finding, regenerates index.md and CONTEXT.md
```

## Why this exists rather than a notes file

Over this project several claims that looked obviously correct turned out to be wrong — detection
declining with encoder depth, iteration-2 pretraining helping, max-pooling beating mean-pooling,
the false positives being unannotated calls. Each cost real time, and each was *re-derived* more
than once because the refutation lived only in a chat log.

So the store keeps refuted findings permanently, with the evidence that killed them. `CONTEXT.md`
leads with them for exactly that reason: it is the file to paste at the start of a session so a
cold agent does not retry a dead idea.

## The rules validation enforces

- Every finding has a `claim` (one sentence), a `method` including the split, and `provenance`
  good enough to rerun it.
- Every comparison carries an interval. "X beats Y" without one is not a finding.
- Every accuracy carries its majority-class rate; every AUC carries its split.
- If a claim was selected on the data it is reported against, `caveats` says so.
- Refuted findings are never deleted.

## Files

| path | what |
|---|---|
| `findings/NNN-slug.yaml` | the findings themselves — edit these |
| `schema.json` | required fields and the rules above |
| `build.py` | validator + renderer |
| `index.md` | generated: everything, grouped by status |
| `CONTEXT.md` | generated: compact cold-start digest, refutations first |

`index.md` and `CONTEXT.md` are generated. Do not edit them.

## Retrieval

The tree is plain text and is picked up by the project's `research-rag` MCP server via
`ingest_local_fs`, so findings are searchable alongside the papers and transcripts.
