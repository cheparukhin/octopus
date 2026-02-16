# Octopus Energy Heater Analysis

## Project overview

Single-page HTML dashboard (`analysis.html`) analysing why an immersion heater uses ~2x the energy under a new timer regime. Built from `usage.csv` — 18,240 half-hourly smart meter readings (Feb 2025 – Feb 2026) on the Octopus Energy Agile tariff.

## Key facts

- **Element power**: 3.11 kW (measured with a power meter). Max energy per 30-min slot = 1.555 kWh.
- **Timer regimes**:
  - OLD (Mar–May 2025): element enabled ~6 h/day in cheap overnight windows.
  - NEW (Jun 2025 onward): element enabled ~20 h/day, off only 16:00–20:00.
- **Regime change date**: approximately 2025-06-02 (identified from data — daytime heater firing >10% starts that week).
- **February 2025 excluded**: meter was newly installed / property unoccupied. Readings are anomalously low (24W floor vs 110W+ in March).

## Decomposition methodology

The core problem: a single meter measures total household + heater consumption. We need to separate them.

### Why naive approaches fail

1. **Threshold decomposition** (>0.8 kWh = heater): over-attributes household baseline to the heater during heater-on slots. Showed element power of 2.83 kW (should be 3.11 kW) — proving household contamination.

2. **Flat baseline subtraction** (constant 0.136 kWh per slot): attributes household spikes (cooking, kettle) to the heater. 35% of "heater" energy came from the ambiguous 0.24–0.80 kWh zone, peaking at lunch/dinner times.

3. **"Clean hour" baseline with fixed window**: initially 08:00–10:00 appeared clean (<2% heater firing). But under the NEW regime the heater is enabled during those hours — the baseline appeared to triple from 112W to 340W. This was heater contamination, not a real household change. The user confirmed: "baseline household usage shouldn't have changed much."

### Correct approach: time-varying baseline from regime-specific clean windows

1. **Identify truly clean windows per regime**:
   - OLD regime: hours 08:00, 09:00 (heater firing <2%)
   - NEW regime: hours 16:00–19:30 (heater explicitly OFF by timer)

2. **Build per-hour-of-day baseline**: median of readings < 0.5 kWh at clean hours. Missing hours filled by circular linear interpolation. Result: 130W (06:00, overnight minimum) to 450W (19:00, evening peak).

3. **Decompose each slot**: `heater_excess = max(0, reading - baseline[hour])`, capped at 1.555 kWh. Remainder = household.

4. **Validation**: three independent baseline methods (P05, P10, median of <0.5 kWh) converge. Household baseline is stable at ~5.5–7 kWh/day across the full year, consistent with the user's observation.

### Results

| Metric | Value |
|---|---|
| Gap (new − old avg) | +5.4 kWh/day |
| Heater share of gap | 86% |
| Seasonal household share | 14% |
| Extra cost per year | £277 |
| Old regime run-time | 3.5 h/day |
| New regime winter run-time | 6.3 h/day |

The heater uses more energy because the new regime keeps the tank at setpoint ~20 h/day (vs ~6 h/day), eliminating free temperature setback during off-periods and increasing standing heat losses.

## Architecture

- `download.py` — fetches consumption + Agile rates from Octopus API, writes `usage.csv`
- `usage.csv` — raw half-hourly data (UTC timestamps)
- `analysis.html` — self-contained dashboard (inline Chart.js, inline JSON data, dark theme)
- `report_data.json` — full chart data (same as inline, kept for reference)
- `og-analysis.png` — Open Graph preview image for Telegram/social sharing
- `.tmp/` — intermediate analysis scripts and data (gitignored)

All chart data is inlined directly into `analysis.html` (no fetch) to avoid CORS issues when opening from `file://`.

Hosted on GitHub Pages: https://cheparukhin.github.io/octopus/analysis.html
