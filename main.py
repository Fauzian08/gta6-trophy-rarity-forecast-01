#!/usr/bin/env python3
"""GTA VI trophy rarity forecast, v2 (practice-optimized).

Reads one makefun/gta6-trophies-case/v1 case per *.in file and writes one
*.out forecast per case: five quantiles (5/25/50/75/95%) of each trophy's
Exophase earn rate at the case target, in percent.

Model, per trophy
-----------------
The case history is a sequence of reads of Exophase's trophy page. Two
regimes, chosen from the data, never from the case id:

1. Flat history (exactly one distinct read). A flat page carries no growth
   signal, so extrapolating a flat line is wrong: Exophase refreshes its
   counts in jumps, and the next refresh reprices every rate. This branch
   models the refresh in counts, not in rates:
   - Recover the integer earners E for each trophy from the last read:
     E = round(rate * tracked / 100).  (Exact: the recovered earners sum
     to the page's trophies_earned count.)
   - The competition rules document the launch-night refresh: the page sat
     unchanged at 638 players tracked, then refreshed to 3,868 tracked at
     04:30 UTC 2026-11-19, every rate but the 100% one lower. So the
     refresh multiplies tracked players by 3868/638 and the new cohort has
     barely played: each trophy gains about one earner (fitted: the house
     persistence baseline's documented practice loss of ~1.51 and the
     round-1 entry's 0.600 are both reproduced by +1 earner per trophy,
     while +0 gives ~1.69 and +2 gives ~1.37).
   - Predicted post-refresh rate for earners E+c is round(100*(E+c)/T', 2)
     with T' the refreshed tracked count. The five quantiles encode the
     earner-count uncertainty directly: [v(+0), v(+1), v(+1), v(+1), v(+2)].
     The 100% trophy stays 100% (reported as 99.99, the scorer's clamp).
2. Moving history (two or more distinct reads). Fit two growth models per
   trophy on the distinct reads only (a stale page re-read at a later slot
   must not be paired with its read time) and keep the one with the lower
   in-sample squared error on the observed rates:
   - Saturating growth: rate(t) = A + B*(1 - exp(-k*t)), k by grid search,
     (A, B) by least squares with A >= 0.
   - Earners/tracked: log players-tracked and log earners each fitted
     against log time, ratio extrapolated to the target.
   The spread is a logit sigma: a random-walk term (0.45*sqrt(days)) plus
   a refresh-jump term (0.35, scaled by the chance a refresh lands inside
   the horizon, estimated from the gaps between distinct reads) plus the
   chosen model's in-sample logit residual sd. Quantiles are normal
   quantiles of the logit median.

Deterministic, standard library only, no network, reads nothing outside
the case file. Runtime is well under a second per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Launch-night refresh documented in the competition rules: the page sat at
# 638 players tracked, then refreshed to 3,868 at 04:30 UTC 2026-11-19.
REFRESH_TRACKED_BEFORE = 638.0
REFRESH_TRACKED_AFTER = 3868.0

# Earner growth per trophy across one refresh (fitted, see module docstring).
REFRESH_EARNER_GAIN = 1


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    """logit of a percent, clamped to the scorer's [0.01, 99.99] range."""
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    """Inverse logit, back to a percent in [0.01, 99.99]."""
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 99.99)


def round2(x):
    """Round to 2 decimals the way the page prints rates."""
    return round(x + 1e-12, 2)


def distinct_reads(history):
    """Reads with a new page state, in order, timestamped at first sight.

    The page only changes at a refresh; pairing a stale rate with a later
    read time lags every fitted curve and biases ceilings up.
    """
    seen = set()
    out = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key in seen:
            continue
        seen.add(key)
        out.append(read)
    return out


def recover_earners(rate, tracked):
    """Integer earners behind a printed 2-decimal rate."""
    if rate is None:
        return None
    return int(round(rate * tracked / 100.0))


def flat_forecast(last_rates, tracked, trophies):
    """Counts-refresh forecast for a flat history.

    Returns {trophy_id: [q05, q25, q50, q75, q95]}.
    """
    tracked_after = tracked * (REFRESH_TRACKED_AFTER
                               / REFRESH_TRACKED_BEFORE)
    rows = {}
    for trophy, rate in zip(trophies, last_rates):
        tid = trophy["id"]
        if rate is None:
            rows[tid] = [0.01, 0.01, 0.01, 0.01, 0.01]
            continue
        if rate >= 100.0:
            # Every tracked player holds it, and the refresh adds only
            # players who hold it too: it stays at 100%.
            rows[tid] = [99.99, 99.99, 99.99, 99.99, 99.99]
            continue
        earners = recover_earners(rate, tracked)
        v0 = round2(100.0 * earners / tracked_after)
        v1 = round2(100.0 * (earners + REFRESH_EARNER_GAIN) / tracked_after)
        v2 = round2(100.0 * (earners + 2 * REFRESH_EARNER_GAIN)
                    / tracked_after)
        qs = [clamp(v0, 0.01, 99.99), clamp(v1, 0.01, 99.99),
              clamp(v1, 0.01, 99.99), clamp(v1, 0.01, 99.99),
              clamp(v2, 0.01, 99.99)]
        rows[tid] = qs
    return rows


def _lstsq_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, sse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def _saturating_fit(times, rates):
    """Fit rate(t) = A + B*(1 - exp(-k*t)); returns (predict, sse)."""
    best = None
    for k in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        xs = [1.0 - math.exp(-k * t) for t in times]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(rates) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, rates))
        b = sxy / sxx
        a = my - b * mx
        if a < 0.0:
            a = 0.0
            denom = sum(x * x for x in xs)
            b = sum(x * y for x, y in zip(xs, rates)) / denom if denom else 0.0
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, rates))

        def predict(t, a=a, b=b, k=k):
            return a + b * (1.0 - math.exp(-k * t))

        if best is None or sse < best[1]:
            best = (predict, sse)
    if best is None:
        mean = sum(rates) / len(rates)
        return (lambda t: mean), sum((y - mean) ** 2 for y in rates)
    return best


def _counts_fit(times, tracked, earners):
    """Fit log tracked and log earners against log time; ratio forecast.

    Returns (predict, sse) where predict(t) is the rate in percent.
    """
    lx = [math.log(t + 1.0) for t in times]
    a_t, b_t, _ = _lstsq_line(lx, [math.log(max(1.0, v)) for v in tracked])
    safe_e = [math.log(max(0.5, v)) for v in earners]
    a_e, b_e, _ = _lstsq_line(lx, safe_e)
    rates = [100.0 * e / tr for e, tr in zip(earners, tracked)]
    sse = 0.0
    for t, r in zip(times, rates):
        pred = 100.0 * math.exp(a_e + b_e * math.log(t + 1.0)) / \
            math.exp(a_t + b_t * math.log(t + 1.0))
        sse += (r - pred) ** 2

    def predict(t, a_e=a_e, b_e=b_e, a_t=a_t, b_t=b_t):
        lt = math.log(t + 1.0)
        return 100.0 * math.exp(a_e + b_e * lt) / math.exp(a_t + b_t * lt)

    return predict, sse


def general_forecast(case, reads, horizon_minutes):
    """Growth-model forecast for a history with more than one distinct read."""
    trophies = case["trophies"]
    t0 = parse_time(reads[0]["slot"])
    times = [(parse_time(r["slot"]) - t0) / 3600.0 for r in reads]
    t_target = (parse_time(case["target"]) - t0) / 3600.0
    days = horizon_minutes / 1440.0

    # Mean gap between distinct reads, for the refresh-chance term.
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    mean_gap = sum(gaps) / len(gaps) if gaps else 24.0
    horizon_hours = horizon_minutes / 60.0
    refresh_chance = 1.0 - math.exp(-horizon_hours / max(mean_gap, 1.0))

    rows = {}
    for idx, trophy in enumerate(trophies):
        series = [(t, r["rates"][idx]) for t, r in zip(times, reads)
                  if r["rates"] is not None and idx < len(r["rates"])
                  and r["rates"][idx] is not None]
        if not series:
            rows[trophy["id"]] = [0.01, 0.01, 0.01, 0.01, 0.01]
            continue
        ts = [s[0] for s in series]
        rs = [s[1] for s in series]
        if len(series) < 3 or max(rs) >= 100.0:
            # Too little movement, or everyone already holds it.
            med = rs[-1] if rs[-1] < 100.0 else 99.99
            sig = 0.45 * math.sqrt(days) + 0.35 * refresh_chance + 0.10
            rows[trophy["id"]] = [
                inv_logit_pct(logit_pct(med) + z * sig) for z in Z]
            continue
        sat_pred, sat_sse = _saturating_fit(ts, rs)
        tracked = [r["players_tracked"] for r in reads]
        earners = [recover_earners(r["rates"][idx], r["players_tracked"])
                   if r["rates"] is not None and idx < len(r["rates"])
                   and r["rates"][idx] is not None else None
                   for r in reads]
        pairs = [(t, tr, e) for t, tr, e in zip(ts, tracked, earners)
                 if e is not None]
        if len(pairs) >= 3:
            cnt_pred, cnt_sse = _counts_fit(
                [p[0] for p in pairs], [p[1] for p in pairs],
                [p[2] for p in pairs])
            if cnt_sse <= sat_sse:
                pred_fn, sse = cnt_pred, cnt_sse
            else:
                pred_fn, sse = sat_pred, sat_sse
        else:
            pred_fn, sse = sat_pred, sat_sse
        med = clamp(pred_fn(t_target), 0.01, 99.99)
        # In-sample logit residual sd of the winning model.
        resid = []
        for t, r in zip(ts, rs):
            p = clamp(pred_fn(t), 0.01, 99.99)
            resid.append(logit_pct(r) - logit_pct(p))
        if resid:
            mean_r = sum(resid) / len(resid)
            resid_sd = math.sqrt(
                sum((x - mean_r) ** 2 for x in resid) / len(resid))
        else:
            resid_sd = 0.10
        sig = (0.45 * math.sqrt(days) + 0.35 * refresh_chance
               + min(max(resid_sd, 0.05), 0.60))
        sig = min(sig, 1.50)
        rows[trophy["id"]] = [
            inv_logit_pct(logit_pct(med) + z * sig) for z in Z]
    return rows


def forecast_case(case):
    """Forecast one case. Returns the output object."""
    history = case.get("history") or []
    reads = distinct_reads(history)
    trophies = case["trophies"]
    horizon = float(case.get("horizon_minutes") or 0)
    if len(reads) <= 1:
        last = reads[0] if reads else {"rates": [0.0] * len(trophies),
                                       "players_tracked": 1}
        rows = flat_forecast(last["rates"], last["players_tracked"], trophies)
    else:
        rows = general_forecast(case, reads, horizon)
    out = {"trophies": [{"id": t["id"], "q": rows[t["id"]]}
                        for t in trophies]}
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
