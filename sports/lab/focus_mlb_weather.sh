#!/bin/bash
# Strategy loop focused on park-specific weather in MLB totals.
#   bash lab/focus_mlb_weather.sh [N] [K]
cd "$(dirname "$0")/.." || exit 1
export LAB_FOCUS="dataset mlb_total. The league-wide weather effect is ~95% priced into the close (temp +0.25%/10F, wind-out +0.26%/10mph beyond the line; see ledger mlb_weather_glm, which lost). Test PARK-SPECIFIC hypotheses the market may underweight: wind direction x park (e.g. Wrigley; open parks vs enclosed), extreme heat/cold tails, condition (rain/overcast), day/night x temperature, umpire (officials) total tendencies learned from training folds only, and closing-line movement (line_move) interactions. Fit everything inside fit() on earlier seasons only."
exec bash lab/loop.sh strategy "${1:-2}" "${2:-3}"
