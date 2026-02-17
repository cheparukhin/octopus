"""Optimal immersion heater timer schedule.

Energy model: E = base_draws + standby × H + Σ reheat(gap_i)

Cost model (accounts for usage timing):
  cost = (D/N) × Σ eff_rate(t) + maintenance × avg_rate

  where eff_rate(t) = rate(t) × exp(gap_to_usage(t) / τ)

Heating far from the usage window (21:00–00:00) is penalised because
stored heat dissipates exponentially before it's used.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np

LONDON = ZoneInfo("Europe/London")

# ── Physics ───────────────────────────────────────────────────────
BASE_DRAWS_KWH = 5.0
STANDBY_KW = 0.233
REHEAT_C_KWH = 7.85
TAU_HOURS = 33.0
ELEMENT_KW = 3.11
DT = 0.5
SLOTS = 48

USAGE_START = 42  # 21:00

REHEAT_TABLE = np.array([
    REHEAT_C_KWH * (1 - np.exp(-g * DT / TAU_HOURS))
    for g in range(SLOTS + 1)
])

# Gap from each slot to usage start (21:00), in hours
_idx = np.arange(SLOTS)
GAP_TO_USAGE = np.where(
    (_idx >= USAGE_START) & (_idx <= 47),
    0.0,
    ((USAGE_START - _idx) % SLOTS) * DT,
)

USAGE_MULT = np.exp(GAP_TO_USAGE / TAU_HOURS)


def slot_label(i: int) -> str:
    h, m = divmod((i % SLOTS) * 30, 60)
    return f"{h:02d}:{m:02d}"


def maintenance(n: int, gaps: tuple[int, ...]) -> float:
    return STANDBY_KW * n * DT + sum(float(REHEAT_TABLE[g]) for g in gaps)


def energy(n: int, gaps: tuple[int, ...]) -> float:
    return BASE_DRAWS_KWH + maintenance(n, gaps)


# ── Data ──────────────────────────────────────────────────────────
def load_rates(path: str = "usage.csv") -> np.ndarray:
    by_slot: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["interval_start"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            local = ts.astimezone(LONDON)
            if local.year == 2025 and local.month == 2:
                continue
            slot = f"{local.hour:02d}:{local.minute:02d}"
            by_slot[slot].append(float(row["rate_p_kwh"]))
    rates = np.zeros(SLOTS)
    for i in range(SLOTS):
        h, m = divmod(i * 30, 60)
        rates[i] = median(by_slot.get(f"{h:02d}:{m:02d}", [17.0]))
    return rates


# ── Pattern helpers ───────────────────────────────────────────────
def pattern_to_windows(pattern: tuple, rotation: int) -> list[tuple[int, int]]:
    windows = []
    pos = rotation
    for i in range(0, len(pattern), 2):
        g, l = pattern[i], pattern[i + 1]
        pos += g
        start = pos % SLOTS
        pos += l
        windows.append((start, pos % SLOTS))
    return windows


def pattern_to_slots(pattern: tuple, rotation: int) -> list[int]:
    slots = []
    pos = rotation
    for i in range(0, len(pattern), 2):
        g, l = pattern[i], pattern[i + 1]
        pos += g
        for j in range(l):
            slots.append((pos + j) % SLOTS)
        pos += l
    return sorted(slots)


def windows_str(wins: list[tuple[int, int]]) -> str:
    return ", ".join(f"{slot_label(s)}–{slot_label(e)}" for s, e in wins)


# ── Exhaustive search ─────────────────────────────────────────────
def search(rates: np.ndarray, *, max_windows: int = 3,
           min_win: int = 2, max_win: int = 24,
           min_total: int = 10, max_total: int = 40,
           eve_range: tuple[int, int] = (42, 48),
           min_eve: int = 2) -> list[dict]:

    eff_rates = rates * USAGE_MULT

    # Circular prefix sums
    def make_cum(arr: np.ndarray) -> np.ndarray:
        ext = np.concatenate([arr, arr])
        return np.concatenate([[0.0], np.cumsum(ext)])

    cum_r = make_cum(rates)
    cum_e = make_cum(eff_rates)
    eve_ind = np.zeros(SLOTS)
    eve_ind[eve_range[0]:eve_range[1]] = 1.0
    cum_eve = make_cum(eve_ind)

    r = np.arange(SLOTS)

    def rs_vec(off: int, length: int) -> np.ndarray:
        return cum_r[r + off + length] - cum_r[r + off]

    def es_vec(off: int, length: int) -> np.ndarray:
        return cum_e[r + off + length] - cum_e[r + off]

    def ev_vec(off: int, length: int) -> np.ndarray:
        return cum_eve[r + off + length] - cum_eve[r + off]

    def cost_vec(n: int, gaps: tuple[int, ...],
                 rate_s: np.ndarray, eff_s: np.ndarray) -> np.ndarray:
        m = maintenance(n, gaps)
        return (BASE_DRAWS_KWH / n) * eff_s + m * (rate_s / n)

    def best_rot(costs: np.ndarray, eve: np.ndarray) -> int | None:
        valid = eve >= min_eve
        if not valid.any():
            return None
        return int(np.argmin(np.where(valid, costs, np.inf)))

    results: list[dict] = []
    t0 = time.time()
    n1 = n2 = n3 = 0

    # ── 1-window ──────────────────────────────────────────────────
    for l1 in range(max(min_win, min_total), min(max_win + 1, max_total + 1)):
        g1 = SLOTS - l1
        rs = rs_vec(g1, l1)
        es = es_vec(g1, l1)
        ev = ev_vec(g1, l1)
        c = cost_vec(l1, (g1,), rs, es)
        br = best_rot(c, ev)
        n1 += SLOTS
        if br is not None:
            results.append(dict(
                cost_p=float(c[br]), pattern=(g1, l1), rotation=br,
                N=l1, gaps=(g1,), rate_sum=float(rs[br]),
                eff_sum=float(es[br]), n_windows=1))

    # ── 2-window ──────────────────────────────────────────────────
    for g1 in range(1, SLOTS - 2 * min_win):
        for l1 in range(min_win, min(max_win + 1, SLOTS - g1 - min_win)):
            rs1 = rs_vec(g1, l1)
            es1 = es_vec(g1, l1)
            ev1 = ev_vec(g1, l1)
            for l2 in range(min_win, min(max_win + 1, SLOTS - g1 - l1)):
                g2 = SLOTS - g1 - l1 - l2
                if g2 < 1:
                    continue
                N = l1 + l2
                if N < min_total or N > max_total:
                    continue
                off2 = g1 + l1 + g2
                rs = rs1 + rs_vec(off2, l2)
                es = es1 + es_vec(off2, l2)
                ev = ev1 + ev_vec(off2, l2)
                c = cost_vec(N, (g1, g2), rs, es)
                br = best_rot(c, ev)
                n2 += SLOTS
                if br is not None:
                    results.append(dict(
                        cost_p=float(c[br]), pattern=(g1, l1, g2, l2),
                        rotation=br, N=N, gaps=(g1, g2),
                        rate_sum=float(rs[br]), eff_sum=float(es[br]),
                        n_windows=2))

    # ── 3-window ──────────────────────────────────────────────────
    for g1 in range(1, SLOTS - 2 * (min_win + 1) - min_win + 1):
        for l1 in range(min_win, min(max_win + 1,
                        SLOTS - g1 - min_win - 1 - min_win - 1 + 1)):
            p1 = g1 + l1
            rs1 = rs_vec(g1, l1)
            es1 = es_vec(g1, l1)
            ev1 = ev_vec(g1, l1)
            for g2 in range(1, SLOTS - p1 - min_win - 1 - min_win + 1):
                off2 = p1 + g2
                for l2 in range(min_win, min(max_win + 1,
                                SLOTS - off2 - 1 - min_win + 1)):
                    p2 = off2 + l2
                    remain = SLOTS - p2
                    rs2 = rs_vec(off2, l2)
                    es2 = es_vec(off2, l2)
                    ev2 = ev_vec(off2, l2)
                    rs12 = rs1 + rs2
                    es12 = es1 + es2
                    ev12 = ev1 + ev2
                    for l3 in range(min_win, min(max_win + 1, remain)):
                        g3 = remain - l3
                        if g3 < 1:
                            continue
                        N = l1 + l2 + l3
                        if N < min_total or N > max_total:
                            continue
                        off3 = p2 + g3
                        rs = rs12 + rs_vec(off3, l3)
                        es = es12 + es_vec(off3, l3)
                        ev = ev12 + ev_vec(off3, l3)
                        c = cost_vec(N, (g1, g2, g3), rs, es)
                        br = best_rot(c, ev)
                        n3 += SLOTS
                        if br is not None:
                            results.append(dict(
                                cost_p=float(c[br]),
                                pattern=(g1, l1, g2, l2, g3, l3),
                                rotation=br, N=N, gaps=(g1, g2, g3),
                                rate_sum=float(rs[br]),
                                eff_sum=float(es[br]), n_windows=3))

    elapsed = time.time() - t0
    print(f"  Evaluated: {n1:,} 1W + {n2:,} 2W + {n3:,} 3W "
          f"= {n1 + n2 + n3:,} in {elapsed:.1f}s")
    results.sort(key=lambda x: x["cost_p"])
    return results


# ── Named schedules ───────────────────────────────────────────────
NAMED = [
    ("Night-only (19–00)", (38, 10)),
    ("Cozy (04–07, 13–16, 22–00)", (8, 6, 12, 6, 12, 4)),
    ("Current (00–16, 20–00)", (0, 32, 8, 8)),
]


def eval_named(name: str, pattern: tuple, rates: np.ndarray) -> dict:
    """Evaluate a named schedule (pattern with implicit rotation=0)."""
    eff_rates = rates * USAGE_MULT
    slots = pattern_to_slots(pattern, 0)
    N = len(slots)
    gaps = tuple(pattern[i] for i in range(0, len(pattern), 2))
    rs = sum(rates[s] for s in slots)
    es = sum(eff_rates[s] for s in slots)
    m = maintenance(N, gaps)
    cost = (BASE_DRAWS_KWH / N) * es + m * (rs / N)
    return dict(name=name, N=N, gaps=gaps, energy=energy(N, gaps),
                maint=m, rate_sum=rs, eff_sum=es, cost_p=cost,
                slots=slots)


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    rates = load_rates()
    eff_rates = rates * USAGE_MULT

    # ── Rate + effective rate profile ─────────────────────────────
    print("=" * 78)
    print("RATE PROFILE (median Agile p/kWh, UK local time)")
    print("  Effective rate = rate × exp(gap_to_21:00 / τ)")
    print("=" * 78)
    print(f"  {'Slot':>5}  {'Rate':>6}  {'Gap→21':>6}  {'×Mult':>6}  "
          f"{'EffRate':>7}  {'Bar'}")
    print(f"  {'─' * 5}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 20}")
    for i in range(SLOTS):
        gap = GAP_TO_USAGE[i]
        mult = USAGE_MULT[i]
        eff = eff_rates[i]
        bar = "█" * int(eff / 2)
        tag = ""
        if 32 <= i < 38:
            tag = " ◄ PEAK"
        elif USAGE_START <= i <= 47:
            tag = " ◄ USAGE"
        print(f"  {slot_label(i):>5}  {rates[i]:>5.1f}p  {gap:>5.1f}h  "
              f"×{mult:>4.2f}  {eff:>6.1f}p  {bar}{tag}")

    # ── Energy model validation ───────────────────────────────────
    print(f"\n{'=' * 78}")
    print("ENERGY VALIDATION")
    print(f"{'=' * 78}")
    measured = {"Night-only (19–00)": 9.6, "Cozy (04–07, 13–16, 22–00)": 10.4}
    for name, pattern in NAMED[:2]:
        r = eval_named(name, pattern, rates)
        m = measured.get(name, 0)
        print(f"  {name:<35}  model={r['energy']:.1f}  measured={m:.1f}  "
              f"diff={r['energy'] - m:+.1f}")

    # ── Named schedule costs ──────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("NAMED SCHEDULE COSTS (new model with usage-timing penalty)")
    print(f"{'=' * 78}")
    print(f"  {'Name':<35}  {'N':>3}  {'E kWh':>5}  {'AvgR':>5}  "
          f"{'AvgEffR':>7}  {'p/day':>6}  {'£/yr':>6}")
    print(f"  {'─' * 35}  {'─' * 3}  {'─' * 5}  {'─' * 5}  "
          f"{'─' * 7}  {'─' * 6}  {'─' * 6}")
    for name, pattern in NAMED:
        r = eval_named(name, pattern, rates)
        avg_r = r["rate_sum"] / r["N"]
        avg_e = r["eff_sum"] / r["N"]
        ann = r["cost_p"] * 365 / 100
        print(f"  {name:<35}  {r['N']:>3}  {r['energy']:>5.1f}  "
              f"{avg_r:>4.1f}p  {avg_e:>6.1f}p  {r['cost_p']:>6.1f}  £{ann:>5.0f}")

    # ── Exhaustive search ─────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("EXHAUSTIVE SEARCH")
    print("  Constraint: ≥10 slots, ≥2 in 21:00–00:00")
    print(f"  Cost = (D/N)×Σeff_rate + maint×avg_rate")
    print(f"{'=' * 78}\n")

    results = search(rates)

    for nw in (1, 2, 3):
        subset = [x for x in results if x["n_windows"] == nw]
        if not subset:
            continue
        print(f"\n  Top 5 {nw}-window schedules:")
        print(f"  {'#':>2}  {'Schedule':<40}  {'H':>4}  {'E':>4}  "
              f"{'AvgR':>5}  {'AvgEff':>6}  {'p/day':>6}  {'£/yr':>6}")
        print(f"  {'─' * 2}  {'─' * 40}  {'─' * 4}  {'─' * 4}  "
              f"{'─' * 5}  {'─' * 6}  {'─' * 6}  {'─' * 6}")
        seen: set[str] = set()
        rank = 0
        for x in subset:
            key = str(sorted(pattern_to_slots(x["pattern"], x["rotation"])))
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            if rank > 5:
                break
            wins = pattern_to_windows(x["pattern"], x["rotation"])
            desc = windows_str(wins)
            H = x["N"] * DT
            E = energy(x["N"], x["gaps"])
            ar = x["rate_sum"] / x["N"]
            ae = x["eff_sum"] / x["N"]
            ann = x["cost_p"] * 365 / 100
            print(f"  {rank:>2}  {desc:<40}  {H:>4.1f}  {E:>4.1f}  "
                  f"{ar:>4.1f}p  {ae:>5.1f}p  {x['cost_p']:>6.1f}  £{ann:>5.0f}")

    # ── Grand comparison ──────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("COMPARISON")
    print(f"{'=' * 78}")
    rows: list[tuple[str, float]] = []
    for name, pattern in NAMED:
        r = eval_named(name, pattern, rates)
        rows.append((name, r["cost_p"]))
    for nw in (1, 2, 3):
        subset = [x for x in results if x["n_windows"] == nw]
        if not subset:
            continue
        x = subset[0]
        wins = pattern_to_windows(x["pattern"], x["rotation"])
        rows.append((f"★ Best {nw}W: {windows_str(wins)}", x["cost_p"]))

    current_p = 835 * 100 / 365

    print(f"\n  {'Name':<55}  {'p/day':>6}  {'£/yr':>6}  {'Save':>6}")
    print(f"  {'─' * 55}  {'─' * 6}  {'─' * 6}  {'─' * 6}")
    for name, cost_p in rows:
        if "Current" in name:
            ann, cp = 835, current_p
        else:
            ann, cp = cost_p * 365 / 100, cost_p
        save = 835 - ann
        s = f"£{save:+.0f}" if save else "—"
        print(f"  {name:<55}  {cp:>6.1f}  £{ann:>5.0f}  {s:>6}")

    # ── Visualization ─────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("SCHEDULE VISUALIZATION")
    print(f"{'=' * 78}")
    print(f"  Clock: ", end="")
    for i in range(0, SLOTS, 2):
        print(f"{i // 2:02d}", end="")
    print()

    viz: list[tuple[str, list[int]]] = []
    for name, pattern in NAMED:
        short = name.split("(")[0].strip()
        viz.append((short, pattern_to_slots(pattern, 0)))
    for nw in (1, 2, 3):
        subset = [x for x in results if x["n_windows"] == nw]
        if subset:
            x = subset[0]
            viz.append((f"Best {nw}W",
                         pattern_to_slots(x["pattern"], x["rotation"])))

    for label, slots in viz:
        s = set(slots)
        strip = "".join("█" if i in s else "·" for i in range(SLOTS))
        print(f"  {label:>10}: {strip}")

    # Usage window marker
    strip = "".join("▓" if USAGE_START <= i <= 47 else " " for i in range(SLOTS))
    print(f"  {'Usage':>10}: {strip}")


if __name__ == "__main__":
    main()
