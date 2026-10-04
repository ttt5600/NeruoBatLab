"""Game and player data for NFL, NBA, MLB and NHL, organised for backtesting.

Every loader returns the same game-level schema (see :mod:`sports.schema`), and
every column carries a timing tag saying *when* it would have been knowable.
That tag is the sports equivalent of the execution lag in ``quant``: a model of
pre-game odds may only see what existed before the game, and the cheapest way
to guarantee that is to make the timing a property of the data rather than a
thing each analysis has to remember.
"""
