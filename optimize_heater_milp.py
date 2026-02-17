#!/usr/bin/env python3
"""Optimize immersion-heater timer schedule for 21:00-00:00 hot-water demand.

Model summary (30-minute slots, one representative day):
- Binary timer schedule x_t (ON/OFF)
- Thermostat-like heating behavior with setpoint cap S_max
- Thermal loss with exponential decay factor alpha
- Fixed demand concentrated between 21:00 and 00:00
- Objective: minimize expected electricity cost under half-hour prices

Uses scipy.optimize.milp (HiGHS) for native-code optimization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, milp
from scipy.sparse import lil_matrix

SLOTS = 48
SLOT_HOURS = 0.5
LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class ThermalParams:
    heater_kw: float
    standby_kw: float
    capacity_kwh: float
    max_windows: int
    evening_demand_kwh: float
    comfort_floor_kwh: float
    switch_penalty_p: float


@dataclass
class Solution:
    success: bool
    message: str
    objective_p: float
    x: np.ndarray
    q: np.ndarray
    s: np.ndarray
    windows: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="usage.csv", help="Path to usage.csv")
    p.add_argument(
        "--timezone",
        default="Europe/London",
        help="Local timezone for slot alignment (default: Europe/London)",
    )
    p.add_argument(
        "--rate-stat",
        choices=("median", "mean"),
        default="median",
        help="Statistic to build expected p/kWh rate by local half-hour slot",
    )
    p.add_argument(
        "--include-feb-2025",
        action="store_true",
        help="Include Feb 2025 rows (excluded by default as anomalous in this project)",
    )
    p.add_argument("--heater-kw", type=float, default=3.11)
    p.add_argument("--standby-kw", type=float, default=0.233)
    p.add_argument(
        "--capacity-kwh",
        type=float,
        default=7.3,
        help="Usable thermal capacity above minimum temperature",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=3,
        help="Maximum ON windows per day",
    )
    p.add_argument(
        "--switch-penalty-p",
        type=float,
        default=0.0,
        help="Penalty (pence) per ON-window start to discourage fragmentation",
    )
    p.add_argument(
        "--evening-demand-kwh",
        type=float,
        default=8.6,
        help="Total thermal demand between 21:00 and 00:00",
    )
    p.add_argument(
        "--demand-profile",
        default="1,1,1,1,1,1",
        help="6 comma-separated nonnegative weights for 21:00..23:30 slots",
    )
    p.add_argument(
        "--comfort-floor-kwh",
        type=float,
        default=0.0,
        help="Minimum storage level required in each 21:00-00:00 slot",
    )
    p.add_argument(
        "--capacity-grid",
        default="",
        help="Optional comma list (e.g. 6,7,8,9,10) to run sensitivity sweep",
    )
    p.add_argument(
        "--estimate-capacity",
        action="store_true",
        help="Estimate capacity from one cool-down/reheat experiment and exit",
    )
    p.add_argument(
        "--off-hours",
        type=float,
        default=0.0,
        help="Experiment: hours tank stayed OFF after full heat",
    )
    p.add_argument(
        "--reheat-kwh",
        type=float,
        default=0.0,
        help="Experiment: kWh used to reheat back to thermostat cutoff",
    )
    p.add_argument(
        "--print-slot-table",
        action="store_true",
        help="Print 48-slot detail (rates, ON flag, charge, state)",
    )
    return p.parse_args()


def slot_label(slot: int) -> str:
    return f"{slot // 2:02d}:{(slot % 2) * 30:02d}"


def parse_demand_weights(raw: str) -> np.ndarray:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 6:
        raise ValueError("--demand-profile must have exactly 6 comma-separated values")
    vals = np.array([float(x) for x in parts], dtype=float)
    if np.any(vals < 0):
        raise ValueError("--demand-profile weights must be nonnegative")
    if np.all(vals == 0):
        raise ValueError("--demand-profile must contain at least one positive weight")
    vals /= vals.sum()
    return vals


def load_rates_by_slot(
    path: str,
    tz_name: str,
    include_feb_2025: bool,
    stat: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tz = ZoneInfo(tz_name)
    by_slot: list[list[float]] = [[] for _ in range(SLOTS)]

    # For optional backtest: local-date x slot matrix with complete 48-slot days only.
    day_slot_rates: dict[tuple[str, int], float] = {}
    day_slots_count: dict[str, int] = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_utc = datetime.fromisoformat(row["interval_start"].replace("Z", "+00:00"))
            ts_local = ts_utc.astimezone(tz)
            if not include_feb_2025 and ts_local.year == 2025 and ts_local.month == 2:
                continue

            slot = ts_local.hour * 2 + ts_local.minute // 30
            rate = float(row["rate_p_kwh"])
            by_slot[slot].append(rate)

            day_key = ts_local.date().isoformat()
            day_slot_rates[(day_key, slot)] = rate
            day_slots_count[day_key] = day_slots_count.get(day_key, 0) + 1

    agg = median if stat == "median" else mean
    rates = np.array([agg(v) if v else 0.0 for v in by_slot], dtype=float)

    full_days = sorted(k for k, cnt in day_slots_count.items() if cnt == SLOTS)
    rates_matrix = np.zeros((len(full_days), SLOTS), dtype=float)
    for i, day in enumerate(full_days):
        for s in range(SLOTS):
            rates_matrix[i, s] = day_slot_rates[(day, s)]

    return rates, rates_matrix, np.array(full_days)


def build_demand(total_kwh: float, weights6: np.ndarray) -> np.ndarray:
    d = np.zeros(SLOTS, dtype=float)
    d[42:48] = total_kwh * weights6
    return d


def extract_windows(x: np.ndarray) -> list[tuple[int, int]]:
    on = np.rint(x).astype(int)
    windows: list[tuple[int, int]] = []

    if on.sum() == 0:
        return windows

    starts = []
    for t in range(SLOTS):
        prev = on[(t - 1) % SLOTS]
        if on[t] == 1 and prev == 0:
            starts.append(t)

    for st in starts:
        t = st
        while on[t] == 1:
            t = (t + 1) % SLOTS
            if t == st:
                break
        windows.append((st, t))

    return windows


def fmt_windows(windows: list[tuple[int, int]]) -> str:
    if not windows:
        return "(none)"
    chunks = []
    for st, en in windows:
        chunks.append(f"{slot_label(st)}-{slot_label(en)}")
    return ", ".join(chunks)


def solve_schedule(prices_p_per_kwh: np.ndarray, params: ThermalParams, demand: np.ndarray) -> Solution:
    n = SLOTS
    q_max = params.heater_kw * SLOT_HOURS
    alpha = float(np.exp(-params.standby_kw * SLOT_HOURS / params.capacity_kwh))

    # Variable layout: [s(0..47), q(0..47), x(0..47), b(0..47), y(0..47)]
    off_s = 0
    off_q = n
    off_x = 2 * n
    off_b = 3 * n
    off_y = 4 * n
    n_vars = 5 * n

    def i_s(t: int) -> int:
        return off_s + t

    def i_q(t: int) -> int:
        return off_q + t

    def i_x(t: int) -> int:
        return off_x + t

    def i_b(t: int) -> int:
        return off_b + t

    def i_y(t: int) -> int:
        return off_y + t

    c = np.zeros(n_vars, dtype=float)
    c[off_q : off_q + n] = prices_p_per_kwh
    c[off_y : off_y + n] = params.switch_penalty_p

    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)

    # s bounds [0, Smax]
    ub[off_s : off_s + n] = params.capacity_kwh
    # q bounds [0, q_max]
    ub[off_q : off_q + n] = q_max
    # x,b,y are binaries in [0,1]
    ub[off_x : off_x + n] = 1.0
    ub[off_b : off_b + n] = 1.0
    ub[off_y : off_y + n] = 1.0

    bounds = Bounds(lb, ub)

    integrality = np.zeros(n_vars, dtype=np.int8)
    integrality[off_x : off_x + n] = 1
    integrality[off_b : off_b + n] = 1
    integrality[off_y : off_y + n] = 1

    # Equality: s_{t+1} - alpha*s_t - q_t = -d_t (cyclic day)
    a_eq = lil_matrix((n, n_vars), dtype=float)
    b_eq = np.zeros(n, dtype=float)
    for t in range(n):
        a_eq[t, i_s((t + 1) % n)] = 1.0
        a_eq[t, i_s(t)] = -alpha
        a_eq[t, i_q(t)] = -1.0
        b_eq[t] = -demand[t]
    eq_con = LinearConstraint(a_eq.tocsr(), b_eq, b_eq)

    # Inequalities
    rows = 6 * n + 1 + 6  # 6 per-slot groups + max-window + 6 comfort rows
    a_ub = lil_matrix((rows, n_vars), dtype=float)
    ub_vec = np.zeros(rows, dtype=float)
    lb_vec = np.full(rows, -np.inf, dtype=float)
    r = 0

    # (1) q_t - q_max*x_t <= 0
    for t in range(n):
        a_ub[r, i_q(t)] = 1.0
        a_ub[r, i_x(t)] = -q_max
        ub_vec[r] = 0.0
        r += 1

    # (2) q_t >= q_max*x_t - q_max*(1-b_t)
    # => -q_t + q_max*x_t + q_max*b_t <= q_max
    for t in range(n):
        a_ub[r, i_q(t)] = -1.0
        a_ub[r, i_x(t)] = q_max
        a_ub[r, i_b(t)] = q_max
        ub_vec[r] = q_max
        r += 1

    # (3) if x=1 and b=0 then s_{t+1}=Smax enforced via lower bound
    # s_{t+1} >= Smax*(x_t - b_t) => -s_{t+1} + Smax*x_t - Smax*b_t <= 0
    for t in range(n):
        a_ub[r, i_s((t + 1) % n)] = -1.0
        a_ub[r, i_x(t)] = params.capacity_kwh
        a_ub[r, i_b(t)] = -params.capacity_kwh
        ub_vec[r] = 0.0
        r += 1

    # (4) Reachability condition for b=0 when x=1:
    # Smax - alpha*s_t + d_t - q_max <= M*(1 - x_t + b_t)
    # => -alpha*s_t + M*x_t - M*b_t <= M - Smax - d_t + q_max
    big_m = params.capacity_kwh + q_max + float(np.max(demand))
    for t in range(n):
        a_ub[r, i_s(t)] = -alpha
        a_ub[r, i_x(t)] = big_m
        a_ub[r, i_b(t)] = -big_m
        ub_vec[r] = big_m - params.capacity_kwh - demand[t] + q_max
        r += 1

    # (5) b_t <= x_t
    for t in range(n):
        a_ub[r, i_b(t)] = 1.0
        a_ub[r, i_x(t)] = -1.0
        ub_vec[r] = 0.0
        r += 1

    # (6) y_t >= x_t - x_{t-1}  => x_t - x_{t-1} - y_t <= 0
    for t in range(n):
        prev = (t - 1) % n
        a_ub[r, i_x(t)] = 1.0
        a_ub[r, i_x(prev)] = -1.0
        a_ub[r, i_y(t)] = -1.0
        ub_vec[r] = 0.0
        r += 1

    # (7) sum y_t <= max_windows
    for t in range(n):
        a_ub[r, i_y(t)] = 1.0
    ub_vec[r] = float(params.max_windows)
    r += 1

    # (8) comfort floor during 21:00-00:00 slots
    for t in range(42, 48):
        a_ub[r, i_s(t)] = -1.0
        ub_vec[r] = -params.comfort_floor_kwh
        r += 1

    assert r == rows

    ub_con = LinearConstraint(a_ub.tocsr(), lb_vec, ub_vec)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[eq_con, ub_con],
        options={"mip_rel_gap": 1e-7},
    )

    if not res.success:
        return Solution(
            success=False,
            message=res.message,
            objective_p=float("nan"),
            x=np.zeros(n),
            q=np.zeros(n),
            s=np.zeros(n),
            windows=[],
        )

    x = res.x[off_x : off_x + n]
    q = res.x[off_q : off_q + n]
    s = res.x[off_s : off_s + n]
    windows = extract_windows(x)

    return Solution(
        success=True,
        message=res.message,
        objective_p=float(res.fun),
        x=x,
        q=q,
        s=s,
        windows=windows,
    )


def simulate_costs(q: np.ndarray, rates_matrix: np.ndarray) -> np.ndarray:
    if rates_matrix.size == 0:
        return np.array([], dtype=float)
    return rates_matrix @ q


def estimate_capacity_from_experiment(off_hours: float, reheat_kwh: float, standby_kw: float) -> float:
    if off_hours <= 0:
        raise ValueError("--off-hours must be > 0")
    if reheat_kwh <= 0:
        raise ValueError("--reheat-kwh must be > 0")
    if standby_kw <= 0:
        raise ValueError("--standby-kw must be > 0")

    # E = S * (1 - exp(-standby_kw * t / S)), solve for S.
    def f(cap: float) -> float:
        return cap * (1.0 - np.exp(-standby_kw * off_hours / cap)) - reheat_kwh

    lo = reheat_kwh + 1e-6
    hi = max(20.0, 10.0 * reheat_kwh)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1000:
            raise RuntimeError("Could not bracket capacity root")

    return float(brentq(f, lo, hi, xtol=1e-9, rtol=1e-9, maxiter=200))


def print_solution(
    title: str,
    rates: np.ndarray,
    demand: np.ndarray,
    params: ThermalParams,
    sol: Solution,
    rates_matrix: np.ndarray,
    day_labels: np.ndarray,
    print_slot_table: bool,
) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)

    if not sol.success:
        print("Status: infeasible / failed")
        print("Solver message:", sol.message)
        return

    on_hours = float(np.rint(sol.x).sum() * SLOT_HOURS)
    charged_kwh = float(sol.q.sum())
    elec_cost_p = float(np.dot(rates, sol.q))
    switch_count = len(sol.windows)
    switch_penalty_total_p = switch_count * params.switch_penalty_p
    avg_rate = elec_cost_p / charged_kwh if charged_kwh > 1e-9 else 0.0

    print(f"Solver status: {sol.message}")
    print(f"Expected heater electricity: {elec_cost_p:.2f} p/day")
    if params.switch_penalty_p > 0:
        print(
            "Switch penalty term:        "
            f"{switch_penalty_total_p:.2f} p/day "
            f"({switch_count} starts x {params.switch_penalty_p:.2f} p)"
        )
    print(f"Objective (electricity+penalty): {sol.objective_p:.2f} p/day")
    print(f"Charged energy:       {charged_kwh:.2f} kWh/day")
    print(f"Avg paid rate:        {avg_rate:.2f} p/kWh")
    print(f"Timer ON duration:    {on_hours:.1f} h/day")
    print(f"ON windows ({len(sol.windows)}):  {fmt_windows(sol.windows)}")

    eve_q = float(sol.q[42:48].sum())
    eve_d = float(demand[42:48].sum())
    print(f"Evening demand modeled (21:00-00:00): {eve_d:.2f} kWh/day")
    print(f"Heating delivered inside window:       {eve_q:.2f} kWh/day")

    costs_hist = simulate_costs(sol.q, rates_matrix)
    if costs_hist.size:
        print(
            "Historical backtest on full 48-slot days: "
            f"n={costs_hist.size}, mean={costs_hist.mean():.2f} p/day, "
            f"median={np.median(costs_hist):.2f} p/day, "
            f"p10={np.percentile(costs_hist, 10):.2f}, "
            f"p90={np.percentile(costs_hist, 90):.2f}"
        )
        min_i = int(np.argmin(costs_hist))
        max_i = int(np.argmax(costs_hist))
        print(
            f"Best historical day: {day_labels[min_i]} ({costs_hist[min_i]:.2f} p) | "
            f"Worst: {day_labels[max_i]} ({costs_hist[max_i]:.2f} p)"
        )

    if print_slot_table:
        print("\nSlot table")
        print(f"{'slot':>5} {'rate':>8} {'x':>3} {'q_kwh':>8} {'s_kwh':>8} {'d_kwh':>8}")
        on = np.rint(sol.x).astype(int)
        for t in range(SLOTS):
            print(
                f"{slot_label(t):>5} {rates[t]:8.2f} {on[t]:3d} "
                f"{sol.q[t]:8.3f} {sol.s[t]:8.3f} {demand[t]:8.3f}"
            )


def parse_capacity_grid(raw: str) -> list[float]:
    if not raw.strip():
        return []
    vals = []
    for token in raw.split(","):
        s = token.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("--capacity-grid provided but no valid numbers parsed")
    return vals


def main() -> None:
    args = parse_args()

    if args.estimate_capacity:
        cap = estimate_capacity_from_experiment(args.off_hours, args.reheat_kwh, args.standby_kw)
        tau_h = cap / args.standby_kw
        alpha = float(np.exp(-args.standby_kw * SLOT_HOURS / cap))
        print("Estimated usable thermal capacity from experiment")
        print(f"  off-hours:     {args.off_hours:.2f} h")
        print(f"  reheat-kwh:    {args.reheat_kwh:.3f} kWh")
        print(f"  standby-kw:    {args.standby_kw:.3f} kW")
        print(f"  capacity-kwh:  {cap:.3f} kWh")
        print(f"  tau:           {tau_h:.2f} h")
        print(f"  alpha/slot:    {alpha:.6f} (30 min)")
        return

    demand_weights = parse_demand_weights(args.demand_profile)
    rates, rates_matrix, day_labels = load_rates_by_slot(
        path=args.csv,
        tz_name=args.timezone,
        include_feb_2025=args.include_feb_2025,
        stat=args.rate_stat,
    )
    demand = build_demand(args.evening_demand_kwh, demand_weights)

    base_params = ThermalParams(
        heater_kw=args.heater_kw,
        standby_kw=args.standby_kw,
        capacity_kwh=args.capacity_kwh,
        max_windows=args.max_windows,
        evening_demand_kwh=args.evening_demand_kwh,
        comfort_floor_kwh=args.comfort_floor_kwh,
        switch_penalty_p=args.switch_penalty_p,
    )

    print(f"Loaded rates from {args.csv}")
    print(
        f"Rate statistic: {args.rate_stat}, local timezone: {args.timezone}, "
        f"full backtest days: {rates_matrix.shape[0]}"
    )
    print(
        f"Slot price summary (p/kWh): min={rates.min():.2f} ({slot_label(int(np.argmin(rates)))}), "
        f"median={np.median(rates):.2f}, max={rates.max():.2f} ({slot_label(int(np.argmax(rates)))})"
    )

    grid = parse_capacity_grid(args.capacity_grid)
    if grid:
        print("\nCapacity sensitivity sweep")
        for cap in grid:
            params = ThermalParams(
                heater_kw=base_params.heater_kw,
                standby_kw=base_params.standby_kw,
                capacity_kwh=cap,
                max_windows=base_params.max_windows,
                evening_demand_kwh=base_params.evening_demand_kwh,
                comfort_floor_kwh=base_params.comfort_floor_kwh,
                switch_penalty_p=base_params.switch_penalty_p,
            )
            sol = solve_schedule(rates, params, demand)
            title = (
                f"Capacity={cap:.2f} kWh | standby={params.standby_kw:.3f} kW | "
                f"max_windows={params.max_windows}"
            )
            print_solution(
                title=title,
                rates=rates,
                demand=demand,
                params=params,
                sol=sol,
                rates_matrix=rates_matrix,
                day_labels=day_labels,
                print_slot_table=False,
            )
        return

    sol = solve_schedule(rates, base_params, demand)
    alpha = float(np.exp(-base_params.standby_kw * SLOT_HOURS / base_params.capacity_kwh))

    title = (
        "Deterministic 21:00-00:00 optimization | "
        f"capacity={base_params.capacity_kwh:.2f} kWh, "
        f"standby={base_params.standby_kw:.3f} kW, alpha={alpha:.6f}, "
        f"max_windows={base_params.max_windows}"
    )
    print_solution(
        title=title,
        rates=rates,
        demand=demand,
        params=base_params,
        sol=sol,
        rates_matrix=rates_matrix,
        day_labels=day_labels,
        print_slot_table=args.print_slot_table,
    )


if __name__ == "__main__":
    main()
