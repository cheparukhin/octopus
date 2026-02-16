"""Download electricity usage and Agile pricing from Octopus Energy API."""

import csv
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

MPAN = "1200031658314"
SERIAL = "21L3938283"
PRODUCT = "AGILE-24-10-01"
TARIFF = f"E-1R-{PRODUCT}-C"
BASE = "https://api.octopus.energy/v1"
SINCE = "2025-02-01T00:00Z"

CHUNK_DAYS = 30
RATES_PAGE_CAP = 1500


def to_utc(ts: str) -> str:
    """Normalize any ISO timestamp to UTC Z-suffix for consistent keying."""
    return datetime.fromisoformat(ts).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_all(client: httpx.Client, url: str, params: dict[str, str | int]) -> list[dict]:
    """Fetch all results from a paginated Octopus API endpoint."""
    results: list[dict] = []
    p = dict(params)
    while url:
        resp = client.get(url, params=p)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data["results"])
        url = data["next"]
        p = {}
    return results


def fetch_rates(client: httpx.Client, period_from: str, period_to: str) -> dict[str, float]:
    """Fetch unit rates in chunked single-page requests.

    The rates API drops date filters on paginated requests (page 2+
    returns the full unfiltered dataset at 100/page). Work around this
    by fetching in chunks small enough to fit in one page.
    """
    url = f"{BASE}/products/{PRODUCT}/electricity-tariffs/{TARIFF}/standard-unit-rates/"
    rates: dict[str, float] = {}
    start = datetime.fromisoformat(period_from)
    end = datetime.fromisoformat(period_to)

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        resp = client.get(url, params={
            "page_size": RATES_PAGE_CAP,
            "period_from": chunk_start.isoformat(),
            "period_to": chunk_end.isoformat(),
        })
        resp.raise_for_status()
        data = resp.json()
        assert data["next"] is None, (
            f"Rates chunk {chunk_start}→{chunk_end} exceeded {RATES_PAGE_CAP} results "
            f"({data['count']}), reduce CHUNK_DAYS"
        )
        for r in data["results"]:
            rates[to_utc(r["valid_from"])] = r["value_inc_vat"]
        chunk_start = chunk_end

    return rates


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OCTOPUS_API_KEY")
    if not api_key:
        print("OCTOPUS_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(auth=(api_key, ""), timeout=60) as client:
        print("Fetching consumption...", flush=True)
        consumption = fetch_all(
            client,
            f"{BASE}/electricity-meter-points/{MPAN}/meters/{SERIAL}/consumption/",
            {"page_size": 25000, "order_by": "period", "period_from": SINCE},
        )

        if not consumption:
            print("No consumption data found.", file=sys.stderr)
            sys.exit(1)

        period_from = consumption[0]["interval_start"]
        period_to = consumption[-1]["interval_end"]
        print(f"  {len(consumption)} intervals: {period_from} → {period_to}", flush=True)

        print("Fetching unit rates...", flush=True)
        rates = fetch_rates(client, period_from, period_to)
        print(f"  {len(rates)} rate slots", flush=True)

    out = Path("usage.csv")
    matched = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval_start", "interval_end", "kwh", "rate_p_kwh", "cost_p"])
        for c in consumption:
            start_utc = to_utc(c["interval_start"])
            end_utc = to_utc(c["interval_end"])
            rate = rates.get(start_utc)
            if rate is not None:
                cost = round(c["consumption"] * rate, 4)
                matched += 1
            else:
                cost = ""
                rate = ""
            w.writerow([start_utc, end_utc, c["consumption"], rate, cost])

    print(f"Wrote {len(consumption)} rows to {out} ({matched} with rates matched)")


if __name__ == "__main__":
    main()
