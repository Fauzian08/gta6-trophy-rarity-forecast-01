#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster.

Reads one case file per run (/input/<id>.in, schema makefun/gta6-trophies-case/v1)
and writes /output/<id>.out: five quantiles (5/25/50/75/95%) of each trophy's
Exophase earn rate at the case's target, in percent.

Model, per trophy, on the logit scale
--------------------------------------
An earn rate is a ratio: earners / players_tracked. Exophase refreshes its
counts on its own schedule, so the page a read sees is stale between refreshes:
when a refresh lands, players_tracked jumps and every rate drops even though
earners only grew. Two models are fitted per trophy and the one that explains
the trophy's observed history better (lower squared error on its rates) is
used for the median:

  * Saturating growth: rate(t) = A + B * (1 - exp(-k * t)), t hours since
    release, k picked by grid search, (A, B) by least squares. Players reach
    a trophy fast at first and then the share who have it levels off; this
    form cannot run away the way a log-log line does.
  * Earners / tracked: log players_tracked and log earners each fitted
    against log time, the ratio extrapolated. This one handles a rate that
    keeps falling because the denominator outruns the earners.

The spread is a logit sigma that grows with the horizon like a random walk,
plus a jump component for refreshes Exophase has not shown yet, plus the
selected model's residual. The jump component is scaled by the chance a
refresh lands inside the horizon, estimated from the gaps between distinct
reads in the history: a 90-minute horizon rarely holds a refresh, a 4.5-day
one certainly does. Quantiles are the normal quantiles of the logit median.

Flat-history regime (the practice cases): launch night's reads are all the
same page, so there is nothing to fit. The one refresh the house saw took
players_tracked 638 -> 3,868 while earners grew only modestly, so every rate
but the one at 100% fell. The honest forecast there is a median below the
last rate, diluted more the longer the page has sat stale (a refresh is
overdue), with a wide spread: a narrow band around the last rate is exactly
what the refresh punishes.

Deterministic: no randomness, no wall-clock use, no network, stdlib only.
Two runs on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard normal quantiles at the case's levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Flat-history regime: median dilution at a 90-minute horizon when the page
# has sat stale 4+ hours, and the logit sigma there. Dilution weakens with
# horizon (earners catch up over days) and the sigma grows with the horizon.
FLAT_DILUTION_MAX = 0.60     # 1 - DILUTION_MAX = 0.40 median at full staleness
FLAT_SIGMA_90MIN = 1.05

# Real-history regime: logit sigma components.
SIGMA_RW_PER_DAY = 0.45      # random-walk sigma per sqrt(day of horizon)
SIGMA_JUMP = 0.35            # refresh jumps Exophase has not shown yet
SIGMA_FIT_FLOOR = 0.15       # floor on the per-trophy fit residual
SIGMA_FIT_CAP = 0.80         # cap on the per-trophy fit residual
SIGMA_TOTAL_FLOOR = 0.35     # floor on the total logit sigma

RATE_FLOOR = 0.01            # rates are clamped to [0.01, 99.99] for the logit
RATE_CEIL = 99.99


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit(p):
    """logit of a probability p in (0, 1)."""
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def logistic(x):
    """Inverse logit, returning a percent in (0, 100)."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in resid) / (n - 2)
        s = math.sqrt(max(var, 0.0))
    else:
        s = 0.0
    return (a, b, s)


def median_of(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantiles_from_logit(median_pct, sigma):
    """Five quantiles in percent from a logit median and logit sigma."""
    centre = logit(median_pct / 100.0)
    out = []
    prev = None
    for z in Z:
        q = logistic(centre + z * sigma)
        q = min(max(q, RATE_FLOOR), RATE_CEIL)
        q = round(q, 4)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def fit_saturating(ts, rates):
    """Fit rate(t) = A + B * (1 - exp(-k*t)) by grid search over k.

    Returns (predict_fn, sse, resid_std) or None. predict_fn takes hours
    since release and returns a rate in percent.
    """
    n = len(ts)
    if n < 3:
        return None
    # Least squares for rate ~ A + B * basis, with A >= 0: the rate at
    # release cannot be negative (a free A goes negative to chase the lag).
    best = None
    # k in per-hour units, log-spaced from very slow to very fast.
    k = 0.0005
    while k <= 0.5:
        basis = [1.0 - math.exp(-k * max(t, 0.0)) for t in ts]
        # Least squares for rate ~ A + B * basis.
        mb = sum(basis) / n
        mr = sum(rates) / n
        sbb = sum((b - mb) ** 2 for b in basis)
        if sbb > 1e-12:
            sbr = sum((b - mb) * (r - mr)
                      for b, r in zip(basis, rates))
            b_coef = sbr / sbb
            a_coef = mr - b_coef * mb
            if a_coef < 0.0:
                # Refit with A = 0.
                sbr0 = sum(b * r for b, r in zip(basis, rates))
                sbb0 = sum(b * b for b in basis)
                b_coef = sbr0 / sbb0 if sbb0 > 0 else 0.0
                a_coef = 0.0
            sse = sum((r - (a_coef + b_coef * b)) ** 2
                      for b, r in zip(basis, rates))
            if best is None or sse < best[0]:
                best = (sse, a_coef, b_coef, k)
        k *= 1.25
    if best is None:
        return None
    sse, a_coef, b_coef, k = best

    def predict(t):
        v = a_coef + b_coef * (1.0 - math.exp(-k * max(t, 0.0)))
        return v

    resid = math.sqrt(sse / max(n - 3, 1))
    return (predict, sse, resid)


def forecast_case(case):
    """Return the output object for one case dict."""
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")

    def hours_since_release(when):
        return (when - release).total_seconds() / 3600.0

    def x_of(when):
        # log time; the first reads sit just before release midnight.
        return math.log1p(max(hours_since_release(when), 0.0))

    x_target = x_of(target)
    t_target = max(hours_since_release(target), 0.0)

    # Per-read series: (x, hours, players_tracked, rates list).
    reads = []
    for read in history:
        try:
            n = read.get("players_tracked")
            rates = read.get("rates")
            when = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if not n or n <= 0 or not isinstance(rates, list):
            continue
        reads.append((x_of(when), hours_since_release(when),
                      float(n), rates))
    reads.sort(key=lambda r: r[0])

    # Collapse consecutive duplicate reads: the page only changes at a
    # refresh, so a read that repeats the previous one carries no new
    # information, and its value is the truth at the last refresh, not at
    # its own slot. Each distinct value is therefore timestamped at its
    # first occurrence, which is (approximately) its refresh time. Fitting
    # against read times instead would lag the curve and bias the ceiling up.
    obs = []
    prev_sig = None
    for entry in reads:
        sig = (entry[2], tuple(entry[3]))
        if sig != prev_sig:
            obs.append(entry)
            prev_sig = sig

    # Last read's rate per trophy (the persistence anchor).
    last_rates = {}
    if reads:
        last = reads[-1][3]
        for i, trophy in enumerate(trophies):
            if i < len(last) and last[i] is not None:
                last_rates[trophy["id"]] = float(last[i])

    flat = False
    staleness_h = 0.0
    if reads:
        combos = set()
        for _, _, n, rates in reads:
            combos.add((n, tuple(rates)))
        flat = len(combos) == 1
        staleness_h = max(0.0, reads[-1][1] - reads[0][1])
    if not reads:
        flat = True

    rows = []
    if flat:
        # Nothing moved: forecast the refresh, not the flat line. The longer
        # the page has sat stale, the more overdue a refresh is.
        staleness_f = min(staleness_h / 4.0, 1.0)
        dilution = 1.0 - FLAT_DILUTION_MAX * staleness_f
        dilution *= (horizon_days / 0.0625) ** 0.15
        dilution = min(max(dilution, 0.35), 1.0)
        extra_days = max(0.0, horizon_days - 0.0625)
        sigma = math.sqrt(FLAT_SIGMA_90MIN ** 2 + 0.25 * extra_days)
        for trophy in trophies:
            tid = trophy["id"]
            last = last_rates.get(tid)
            if last is None:
                median = 1.0
                sig = 1.5
            elif last >= 99.99:
                median = 99.9
                sig = 0.05
            else:
                median = min(max(last * dilution, 0.02), 99.5)
                sig = sigma
            rows.append({"id": tid,
                         "q": quantiles_from_logit(median, sig)})
    else:
        # Real history. Shared fit: log players_tracked against log time.
        # Estimate the refresh interval from the gaps between distinct reads:
        # a refresh is unlikely inside a horizon much shorter than it.
        distinct_times = []
        prev_sig = None
        for _, t_h, n, rates in reads:
            sig = (n, tuple(rates))
            if sig != prev_sig:
                distinct_times.append(t_h)
                prev_sig = sig
        gaps = [b - a for a, b in zip(distinct_times, distinct_times[1:])]
        refresh_interval_h = median_of(gaps) if gaps else 4.5
        refresh_interval_h = min(max(refresh_interval_h, 1.0), 24.0)
        horizon_h = case["horizon_minutes"] / 60.0
        p_refresh = min(1.0, horizon_h / refresh_interval_h)
        sigma_jump_eff = SIGMA_JUMP * math.sqrt(p_refresh)
        xs = [r[0] for r in obs]
        log_n = [math.log(r[2]) for r in obs]
        fit_n = ols(xs, log_n)
        if fit_n is None:
            a_n, b_n, resid_n = 0.0, 0.0, 0.0
            log_n_target = log_n[-1]
        else:
            a_n, b_n, resid_n = fit_n
            b_n = max(b_n, 0.0)  # tracked players do not shrink
            log_n_target = a_n + b_n * x_target
            log_n_target = min(log_n_target, log_n[-1] + math.log(50.0))
        n_target = math.exp(log_n_target)

        # First pass: per-trophy log-earner fits, for tier median slopes.
        per_trophy = {}
        tier_slopes = {}
        for i, trophy in enumerate(trophies):
            txs, tys = [], []
            for x, _, n, rates in obs:
                if i >= len(rates) or rates[i] is None:
                    continue
                earners = float(rates[i]) / 100.0 * n
                txs.append(x)
                tys.append(math.log(max(earners, 0.5)))
            fit = ols(txs, tys) if len(txs) >= 2 else None
            distinct = len(set(tys))
            per_trophy[trophy["id"]] = (txs, tys, fit, distinct)
            if fit is not None and distinct >= 4:
                tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
        tier_median_slope = {t: median_of(v) for t, v in tier_slopes.items()}

        for i, trophy in enumerate(trophies):
            tid = trophy["id"]
            last = last_rates.get(tid)
            if last is not None and last >= 99.99:
                rows.append({"id": tid,
                             "q": quantiles_from_logit(99.5, 0.15)})
                continue

            # Observed (hours, rate) pairs for this trophy.
            ts, rs = [], []
            for _, t_h, _, rates in obs:
                if i < len(rates) and rates[i] is not None:
                    ts.append(max(t_h, 0.0))
                    rs.append(float(rates[i]))
            txs, tys, fit_e, distinct = per_trophy[tid]

            candidates = []  # (sse, median, sigma_fit)
            # Candidate 1: saturating growth on the rate itself.
            sat = fit_saturating(ts, rs) if len(ts) >= 3 else None
            if sat is not None:
                predict, sse, resid = sat
                med = min(max(predict(t_target), 0.02), 99.5)
                rel = resid / max(median_of(rs) if rs else 1.0, 0.2)
                sig_fit = min(max(rel, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)
                candidates.append((sse, med, sig_fit))
            # Candidate 2: earners / tracked, both log-log fitted.
            if fit_e is not None and distinct >= 2:
                a_e, b_e, s_e = fit_e
                tier_slope = tier_median_slope.get(trophy["tier"])
                if distinct < 4 and tier_slope is not None:
                    b_e = 0.5 * b_e + 0.5 * tier_slope
                log_e_target = a_e + b_e * x_target
                log_e_target = min(log_e_target, tys[-1] + math.log(200.0))
                med = min(max(100.0 * math.exp(log_e_target) / n_target,
                              0.02), 99.5)
                # In-sample squared error of this model's implied rates.
                sse = 0.0
                for x, _, n, rates in obs:
                    if i >= len(rates) or rates[i] is None:
                        continue
                    implied = 100.0 * math.exp(a_e + b_e * x) \
                        / math.exp(a_n + b_n * x)
                    sse += (float(rates[i]) - implied) ** 2
                sig_fit = min(max(s_e, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)
                if distinct < 3:
                    sig_fit = max(sig_fit, 0.5)
                candidates.append((sse, med, sig_fit))

            if not candidates:
                median = last if last is not None else 1.0
                sigma = math.sqrt(SIGMA_RW_PER_DAY ** 2 * horizon_days
                                  + sigma_jump_eff ** 2 + 0.6 ** 2)
            else:
                # The model that explains this trophy's history better wins;
                # a small edge goes to the saturating fit (fewer parameters
                # per trophy). Candidates are ordered saturating first.
                best = candidates[0]
                for cand in candidates[1:]:
                    if cand[0] < 0.9 * best[0]:
                        best = cand
                _, median, sig_fit = best
                sigma = math.sqrt(SIGMA_RW_PER_DAY ** 2 * horizon_days
                                  + sigma_jump_eff ** 2
                                  + sig_fit ** 2
                                  + resid_n ** 2)
                sigma = max(sigma, SIGMA_TOTAL_FLOOR)
            rows.append({"id": tid,
                         "q": quantiles_from_logit(median, sigma)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"))
        with open(os.path.join(out_dir, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
