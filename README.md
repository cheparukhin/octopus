# Octopus Energy Usage Downloader

Downloads half-hourly electricity consumption and [Agile](https://octopus.energy/smart/agile/) unit rates from the Octopus Energy API, joins them by interval, and writes to `usage.csv`.

## Setup

```
uv sync
cp .env.example .env  # then fill in your API key
```

`.env`:
```
OCTOPUS_API_KEY=sk_live_...
```

## Usage

```
uv run python download.py
```

Outputs `usage.csv` with columns:

| Column | Description |
|---|---|
| `interval_start` | Half-hour interval start (UTC) |
| `interval_end` | Half-hour interval end (UTC) |
| `kwh` | Consumption in kWh |
| `rate_p_kwh` | Agile unit rate in p/kWh (inc. VAT) |
| `cost_p` | Cost in pence (`kwh × rate`) |

## Configuration

Edit constants at the top of `download.py`:

- `MPAN` / `SERIAL` — your meter identifiers (find via [account API](https://developer.octopus.energy/rest/guides/))
- `PRODUCT` — your Agile product code
- `SINCE` — earliest date to fetch from
