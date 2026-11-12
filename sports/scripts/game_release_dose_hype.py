"""Launch-week hype table for scripts/game_release_dose.py -- FROZEN 2026-11-11 before any new outcome was computed.
Rule (notes/game_release_dose.md): H = reported sales within the first 7 days (longest window <= 7 d), nominal USD;
units x US standard price when only units. No rescaling across windows. Sources: researcher subagent (URLs in the notes).
Titles with H < $250M are listed for transparency and dropped by game_release_dose.clusters(); titles with no <= 7-day
figure (Elden Ring, Hogwarts, Cyberpunk, BG3, Starfield, CoD 2020+, Smash/ACNH WW, Overwatch/Starfield "players",
Switch 2 hardware, ...) are NOT in this table.
"""
M = 1e6
LAUNCHES = [
    # --- dev era (<= 2022-12-31)
    ("Halo 3", "2007-09-25", 300 * M, "MS $300M week 1 (day 1 $170M US)"),
    ("GTA IV", "2008-04-29", 500 * M, "$500M 7d WW"),
    ("Fallout 3", "2008-10-28", 300 * M, "4.7M shipped wk1, $300M+ retail (SI)"),
    ("Gears of War 2", "2008-11-07", 120 * M, "2M x $60, 3d"),
    ("CoD MW2", "2009-11-10", 550 * M, "$550M 5d"),
    ("Halo Reach", "2010-09-14", 200 * M, "$200M day 1"),
    ("CoD Black Ops", "2010-11-09", 650 * M, "$650M 5d ST"),
    ("Gears of War 3", "2011-09-20", 180 * M, "3M x $60 wk1"),
    ("Battlefield 3", "2011-10-25", 300 * M, "5M x $60 wk1 (EA est.)"),
    ("CoD MW3", "2011-11-08", 775 * M, "$775M 5d"),
    ("Skyrim", "2011-11-11", 210 * M, "3.5M x $60 2d ST (7M shipped = SI, same window; ST taken)"),
    ("Diablo III", "2012-05-15", 378 * M, "6.3M x $60 wk1"),
    ("Halo 4", "2012-11-06", 220 * M, "$220M day 1 ($300M wk1 was a projection)"),
    ("CoD Black Ops II", "2012-11-13", 500 * M, "$500M day 1 ST"),
    ("GTA V", "2013-09-17", 1000 * M, "$1B 3d (Take-Two)"),
    ("Pokemon X/Y", "2013-10-12", 160 * M, "4M x $40 2d"),
    ("CoD Ghosts", "2013-11-05", 1000 * M, "$1B day 1 SI"),
    ("Watch Dogs", "2014-05-27", 240 * M, "4M x $60 wk1"),
    ("Destiny", "2014-09-09", 500 * M, "$500M day 1 SI (weak source)"),
    ("CoD Black Ops III", "2015-11-06", 550 * M, "$550M 3d ST"),
    ("Fallout 4", "2015-11-10", 750 * M, "$750M day 1 SI"),
    ("The Division", "2016-03-08", 330 * M, "$330M 5d"),
    ("Pokemon Sun/Moon", "2016-11-18", 400 * M, "10M shipped (SI) x $40"),
    ("CoD WWII", "2017-11-03", 500 * M, "$500M 3d"),
    ("God of War", "2018-04-20", 186 * M, "3.1M x $60 3d"),
    ("Marvel's Spider-Man", "2018-09-07", 198 * M, "3.3M x $60 3d"),
    ("CoD Black Ops 4", "2018-10-12", 500 * M, "$500M 3d"),
    ("RDR2", "2018-10-26", 725 * M, "$725M 3d ST"),
    ("CoD MW 2019", "2019-10-25", 600 * M, "$600M 3d ST"),
    ("Pokemon Sword/Shield", "2019-11-15", 360 * M, "6M x $60 wk1 (URL not captured)"),
    ("CoD MW II 2022", "2022-10-28", 800 * M, "$800M 3d ST"),
    ("Pokemon Scarlet/Violet", "2022-11-18", 600 * M, "10M x $60 3d"),
    # --- sealed era (2023-01-01 .. 2026-10-31)
    ("Zelda TotK", "2023-05-12", 700 * M, "10M x $70 3d"),
    ("Diablo IV", "2023-06-06", 666 * M, "$666M 5d ST"),
    ("Marvel's Spider-Man 2", "2023-10-20", 175 * M, "2.5M x $70 1d"),
    ("Palworld", "2024-01-19", 210 * M, "7M x $30 5d Steam"),
    ("Helldivers 2", "2024-02-08", 40 * M, "1M x $40 3d (weak)"),
    ("Black Myth Wukong", "2024-08-20", 600 * M, "10M x $60 3d"),
    ("Monster Hunter Wilds", "2025-02-28", 560 * M, "8M x $70 3d"),
    ("Battlefield 6", "2025-10-10", 490 * M, "7M x $70 3d"),
    ("Pokemon Legends Z-A", "2025-10-16", 348 * M, "5.8M x $60 wk1"),
]
