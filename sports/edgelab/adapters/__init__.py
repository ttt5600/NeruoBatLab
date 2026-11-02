"""Forward adapters: turn what a live executor / paper logger actually did into edgelab bet rows (GOAL.md rule 4).

Each module exposes
    load(params) -> (frame, sources)   frame has `date` (event date, the vault's split key), `bet_id`, the hidden
                                       outcome columns listed in HIDDEN, and a visible `settled` flag
    rows(frame, params) -> bet rows    (no outcome; the vault resolves it from HIDDEN via OUTCOME)
    OUTCOME, HIDDEN
Wired per strategy in vault.yaml `forward_adapters`. Point-in-time is enforced by schema.enforce on the rows (a
violation rejects the whole evaluation, recorded as pit_rejected with counts); adapters never drop a row silently --
everything they leave out is counted in bets.attrs['adapter_report'], which run.evaluate records.
"""
