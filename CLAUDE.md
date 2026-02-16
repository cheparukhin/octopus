# Octopus Energy Heater Analysis

## Project overview

Single-page HTML dashboard (`analysis.html`) analysing why an immersion heater uses ~2x the energy under a new timer regime. Built from `usage.csv` — 18,240 half-hourly smart meter readings (Feb 2025 – Feb 2026) on the Octopus Energy Agile tariff.

## Key facts

- **Element power**: 3.11 kW (measured with a power meter). Max energy per 30-min slot = 1.555 kWh.
- **Timer regimes** (3 identified from data, all times UK local):
  - **Night-only** (Mar 1 – ~Apr 9, 2025): element enabled ~5 h/day in a single evening window (~19:00–23:00 local). Heater on ~3.1 h/day.
  - **Cozy 3-window** (~Apr 10 – ~Aug 25, 2025): element enabled ~9 h/day across three dip windows (04:00–07:00, 13:00–16:00, 22:00–00:00 local). Heater on ~3.4 h/day.
  - **Current** (~Aug 26, 2025 onward): element enabled ~20 h/day, off only 16:00–20:00 local. Heater on ~6.2 h/day in winter.
- **Transition dates**: Night→Cozy ~Apr 10 (sharp jump in daytime firing), Cozy→Current ~Aug 26 (7-day average jumps ~6 kWh/day overnight).
- **DST**: UK clocks forward 2025-03-30, back 2025-10-26. CSV timestamps are UTC; timer operates in local time. All analysis converts UTC→Europe/London.
- **February 2025 excluded**: meter was newly installed / property unoccupied. Readings are anomalously low (24W floor vs 110W+ in March).

## Decomposition methodology

The core problem: a single meter measures total household + heater consumption. We need to separate them.

### Why naive approaches fail

1. **Threshold decomposition** (>0.8 kWh = heater): over-attributes household baseline to the heater during heater-on slots. Showed element power of 2.83 kW (should be 3.11 kW) — proving household contamination.

2. **Flat baseline subtraction** (constant 0.136 kWh per slot): attributes household spikes (cooking, kettle) to the heater. 35% of "heater" energy came from the ambiguous 0.24–0.80 kWh zone, peaking at lunch/dinner times.

3. **"Clean hour" baseline with fixed window**: initially 08:00–10:00 appeared clean (<2% heater firing). But under the NEW regime the heater is enabled during those hours — the baseline appeared to triple from 112W to 340W. This was heater contamination, not a real household change. The user confirmed: "baseline household usage shouldn't have changed much."

### Correct approach: time-varying baseline from regime-specific clean windows

1. **Identify truly clean windows per regime** (hours where heater is guaranteed OFF by timer):
   - Night-only: local hours 07:00–18:00 (daytime, heater off)
   - Cozy: local hours 07:00–12:00 and 16:00–21:00 (between dip windows)
   - Current: local hours 16:00–19:00 (heater explicitly OFF by timer)

2. **Build per-half-hour-of-day baseline**: collect readings from all regime-specific clean windows, take median of readings < 0.5 kWh at each half-hour slot. Missing slots filled by circular linear interpolation. Result: 130W (06:00, overnight minimum) to 450W (19:00, evening peak).

3. **Decompose each slot**: `heater_excess = max(0, reading - baseline[hour])`, capped at 1.555 kWh. Remainder = household.

4. **Validation**: three independent baseline methods (P05, P10, median of <0.5 kWh) converge. Household baseline is stable at ~5.5–7 kWh/day across the full year, consistent with the user's observation.

### Results

| Regime | Total kWh/day | Heater kWh/day | Household kWh/day | On-time h/day |
|---|---|---|---|---|
| Night-only (Mar) | 15.1 | 9.6 | 5.5 | 3.1 |
| Cozy (May–Jun) | 16.1 | 10.4 | 5.7 | 3.4 |
| Summer (Jul–Aug) | 14.6 | 8.9 | 5.7 | 2.9 |
| Current winter (Dec–Feb) | 26.3 | 19.3 | 7.0 | 6.2 |

| Metric | Value |
|---|---|
| Gap (current winter − cozy) | +10.2 kWh/day |
| Heater share of gap | 87% (+8.9 kWh/day) |
| Seasonal household share | 13% (+1.3 kWh/day) |
| Estimated extra cost per year | ~£275 |

Key insight: Night-only and Cozy use almost identical heater energy (~10 kWh/day) despite different timer patterns — both limit enabled time to <9 h/day. The Current regime's 20 h/day eliminates the free temperature setback during off-periods, nearly doubling heater energy consumption.

## Architecture

- `download.py` — fetches consumption + Agile rates from Octopus API, writes `usage.csv`
- `usage.csv` — raw half-hourly data (UTC timestamps)
- `analysis.html` — self-contained dashboard (inline Chart.js, inline JSON data, dark theme)
- `report_data.json` — full chart data (same as inline, kept for reference)
- `og-analysis.png` — Open Graph preview image for Telegram/social sharing
- `.tmp/` — intermediate analysis scripts and data (gitignored)

All chart data is inlined directly into `analysis.html` (no fetch) to avoid CORS issues when opening from `file://`.

Hosted on GitHub Pages: https://cheparukhin.github.io/octopus/analysis.html
