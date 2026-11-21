#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster, round 3: all-in counts model.

Two regimes, chosen by whether the case history ever changes:

* Flat history (one distinct read, as in both practice cases): the page the
  house reads has not refreshed yet, so the forecast is the counts-refresh
  model. Each trophy's earner count E is recovered exactly from the last
  read (round(rate * tracked / 100); the sum over trophies equals the page's
  trophies_earned). The documented launch-night refresh took the tracked
  players from 638 to 3,868, so the next refresh is modelled as tracked
  growing by that same factor while each trophy gains one earner. All five
  quantiles are the same value, round(100 * (E + 1) / T', 2) with
  T' = tracked * 3868 / 638: zero spread, the whole bet on the counts.
  A trophy already at 100% stays there (99.99, the highest value with a
  logit).

* Moving history (several distinct reads, as the final case will have): a
  real growth model per trophy. Two models are fitted and the one with the
  lower in-sample squared error wins the median: saturating growth
  rate(t) = A + B * (1 - exp(-k * t)), and an earners/tracked model that
  fits log earners and log tracked each against log time and extrapolates
  their ratio. The spread is a logit sigma of a random-walk term
  (0.45 * sqrt(days)) plus a jump term (0.35 scaled by the chance a refresh
  lands inside the horizon, estimated from the gaps between distinct
  reads) plus the winning model's residual spread.

Standard library only. Deterministic: no randomness, no clock, no network,
nothing read outside the case file. Reads $MAKEFUN_INPUT_DIR else /input/,
writes one /output/<id>.out per *.in file to $MAKEFUN_OUTPUT_DIR else
/output/. Outputs are left world-readable (mode 0644).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles at the case levels 0.05, 0.25, 0.5, 0.75, 0.95.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# The documented launch-night refresh: Exophase's tracked players for the
# game went from 638 to 3,868 while every rate but the 100% one fell.
# The flat-history model assumes the next refresh grows tracked by the
# same factor and adds one earner to each trophy.
REFRESH_TRACKED_BEFORE = 638.0
REFRESH_TRACKED_AFTER = 3868.0

# Grid of growth rates (per hour) tried by the saturating-growth model.
K_GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

LO = 0.01
HI = 99.99


def clamp_pct(x):
    """Clamp a percent into the range the scorer can take a logit of."""
    if x < LO:
        return LO
    if x > HI:
        return HI
    return x


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    return clamp_pct(100.0 / (1.0 + math.exp(-l)))


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def distinct_reads(history):
    """Reads with a changed page, timestamped at first occurrence.

    Exophase only changes a game's counts at a refresh, so most reads
    repeat the previous page exactly. Fitting the stale repeats would lag
    the curve, so only the first read of each distinct page is kept.
    """
    out = []
    seen = set()
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(rates))
        if key not in seen:
            seen.add(key)
            out.append(read)
    return out


def last_non_null_rate(history, index):
    """Latest non-null rate for one trophy over the whole history."""
    rate = None
    for read in history:
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            rate = rates[index]
    return rate


def flat_forecast(last_read, trophies):
    """All-in counts model for a history that never changes.

    Returns {trophy_id: [v, v, v, v, v]} with v the refreshed rate.
    """
    tracked = last_read.get("players_tracked") or 0
    t_prime = tracked * REFRESH_TRACKED_AFTER / REFRESH_TRACKED_BEFORE
    out = {}
    for i, trophy in enumerate(trophies):
        rates = last_read.get("rates") or []
        rate = rates[i] if i < len(rates) else None
        if rate is None:
            # The last read lacks this trophy; nothing to refresh from.
            out[trophy["id"]] = [LO, LO, LO, LO, LO]
            continue
        if rate >= 100.0:
            value = HI
        elif tracked <= 0:
            # No tracked count to recover earners from: hold the rate.
            value = clamp_pct(rate)
        else:
            earners = round(rate * tracked / 100.0)
            value = round(100.0 * (earners + 1) / t_prime, 2)
            value = clamp_pct(value)
        out[trophy["id"]] = [value, value, value, value, value]
    return out


def _fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, sse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def _fit_saturating(ts, rs):
    """Least squares for rate(t) = A + B*(1 - exp(-k*t)), A >= 0, B >= 0.

    Grid-searches k; for each k the model is linear in (A, B). Returns
    (sse, predict) where predict(t) gives the fitted rate at hour t.
    """
    best = None
    for k in K_GRID:
        xs = [1.0 - math.exp(-k * t) for t in ts]
        # Solve the 2x2 normal equations for r = A + B*x.
        n = len(xs)
        sx = sum(xs)
        sxx = sum(x * x for x in xs)
        sy = sum(rs)
        sxy = sum(x * r for x, r in zip(xs, rs))
        det = n * sxx - sx * sx
        if abs(det) < 1e-12:
            a, b = sy / n, 0.0
        else:
            a = (sy * sxx - sx * sxy) / det
            b = (n * sxy - sx * sy) / det
        if a < 0.0:
            a = 0.0
            b = sxy / sxx if sxx > 0.0 else 0.0
        if b < 0.0:
            b = 0.0
            a = sy / n

        def predict(t, a=a, b=b, k=k):
            return a + b * (1.0 - math.exp(-k * t))

        sse = sum((r - predict(t)) ** 2 for t, r in zip(ts, rs))
        if best is None or sse < best[0]:
            best = (sse, predict)
    return best


def _fit_counts(ts, earners, tracked):
    """Log-log earners/tracked model. Returns (sse, predict) or None.

    Fits log(E) and log(T) each against log(t + 1) and extrapolates the
    ratio; needs at least 3 usable points.
    """
    pts = [(t, e, tr) for t, e, tr in zip(ts, earners, tracked)
           if e > 0 and tr > 0]
    if len(pts) < 3:
        return None
    us = [math.log(t + 1.0) for t, _, _ in pts]
    a_e, b_e, _ = _fit_line(us, [math.log(e) for _, e, _ in pts])
    a_t, b_t, _ = _fit_line(us, [math.log(tr) for _, _, tr in pts])

    def predict(t):
        u = math.log(t + 1.0)
        e = math.exp(a_e + b_e * u)
        tr = math.exp(a_t + b_t * u)
        if tr <= 0.0:
            return LO
        return clamp_pct(100.0 * e / tr)

    sse = 0.0
    for t, e, tr in pts:
        # The model's fitted rate at this read, from its own fitted E, T.
        u = math.log(t + 1.0)
        r_fit = 100.0 * math.exp(a_e + b_e * u) / math.exp(a_t + b_t * u)
        r_obs = 100.0 * e / tr
        sse += (r_obs - r_fit) ** 2
    return sse, predict


def moving_forecast(case, reads):
    """Growth-model forecast for a history that changes.

    Returns {trophy_id: [q05, q25, q50, q75, q95]}.
    """
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    t0 = parse_time(reads[0]["at"])
    ts = [(parse_time(r["at"]) - t0) / 3600.0 for r in reads]
    t_target = (parse_time(case["target"]) - t0) / 3600.0

    # Chance a refresh lands inside the horizon, from the gaps between
    # distinct reads: refreshes per hour, Poisson-style.
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    gaps = [g for g in gaps if g > 0.0]
    if gaps:
        lam = 1.0 / (sum(gaps) / len(gaps))
        p_refresh = 1.0 - math.exp(-lam * horizon_hours)
    else:
        p_refresh = 0.0

    out = {}
    for i, trophy in enumerate(trophies):
        series = []
        for j, read in enumerate(reads):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                series.append((ts[j], rates[i], read.get("players_tracked") or 0))
        if len(series) < 2:
            # Too little movement for this trophy: fall back to the
            # counts model on its last seen rate.
            last = reads[-1]
            rates = last.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                rate = last_non_null_rate(reads, i) or LO
            sub = flat_forecast(
                {"players_tracked": last.get("players_tracked") or 0,
                 "rates": [rate]}, [{"id": trophy["id"]}])
            out[trophy["id"]] = sub[trophy["id"]]
            continue

        s_ts = [s[0] for s in series]
        s_rs = [s[1] for s in series]
        s_es = [round(s[1] * s[2] / 100.0) if s[2] else 0 for s in series]
        s_tr = [s[2] for s in series]

        sat_sse, sat_pred = _fit_saturating(s_ts, s_rs)
        counts = _fit_counts(s_ts, s_es, s_tr)
        if counts is not None and counts[0] < sat_sse:
            sse, predict = counts
        else:
            sse, predict = sat_sse, sat_pred

        median = clamp_pct(predict(t_target))
        # Residual spread of the winning model, on the logit scale.
        resid = []
        for t, r in zip(s_ts, s_rs):
            fitted = clamp_pct(predict(t))
            resid.append(logit_pct(r) - logit_pct(fitted))
        sd = math.sqrt(sum(x * x for x in resid) / len(resid)) if resid else 0.0
        sigma = 0.45 * math.sqrt(horizon_days) + 0.35 * p_refresh + sd
        sigma = min(max(sigma, 0.15), 2.0)

        lm = logit_pct(median)
        qs = [round(inv_logit_pct(lm + z * sigma), 4) for z in Z]
        qs = [clamp_pct(q) for q in qs]
        # Guarantee non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out[trophy["id"]] = qs
    return out


def forecast_case(case):
    """Forecast one case. Returns {trophy_id: [q05..q95]}."""
    reads = distinct_reads(case.get("history") or [])
    if len(reads) < 2:
        last = (case.get("history") or [None])[-1]
        if last is None:
            return {t["id"]: [LO, LO, LO, LO, LO] for t in case["trophies"]}
        return flat_forecast(last, case["trophies"])
    return moving_forecast(case, reads)


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        os.chmod(out_dir, 0o755)
    except OSError:
        pass

    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        doc = {"trophies": [
            {"id": t["id"], "q": rows[t["id"]]} for t in case["trophies"]
        ]}
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))
            fh.write("\n")
        try:
            os.chmod(out_path, 0o644)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
