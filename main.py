#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (round 4).

Reads one case (/input/<id>.in) and writes /output/<id>.out: for each of
the game's trophies, five quantiles (5/25/50/75/95%) of its Exophase earn
rate at the case target, as JSON {"trophies": [{"id": n, "q": [...]}]}.

Model, calibrated on the two practice server observations (round-2 loss
0.140, round-3 loss 0.187, typical miss 0.05, coverages 0.8909 / 0.2):
- Every read gives earners E and players tracked T per trophy.  E is
  recovered exactly from the printed 2-decimal rate: E = round(rate*T/100)
  (practice: sum E = 943 = trophies_earned, no rounding mismatch).
- Tracked T(t) and earners E(t) are extrapolated to the target in log time
  u = ln(1 + hours since release), anchored at the last read.  When a
  series is exactly flat in the case history no trend is visible, so
  tracked is projected with the calibrated single-refresh jump x6.0627
  (3868/638, the step the practice answers show) and earners are held.
- The answer is the first read in the target hour, i.e. one refresh past
  the smooth extrapolation.  At a refresh earners move by k with the
  fitted mix w over (trophy, case) pairs:
      k:    0      1      2      3      8      12     20     40
      w: 0.2735 0.1999 0.4175 0.0425 0.0274 0.0202 0.0124 0.0066
  w is the average of NNLS fits (k-grid caps 8/12/20/40) matching all five
  practice observables (losses, typical miss, both coverages) to ~0.001.
  The pure {0,1,2} mix cannot reproduce them: it caps the round-3 loss
  near 0.15 (observed 0.187), forces coverage 1.0 for the spread entry
  (observed 0.8909), and caps the typical miss at 0.026 (observed 0.05).
  The data need a k=3 mass and a small far tail of large earner jumps.
- Quantiles are the tau-quantiles of that answer distribution around the
  smooth centre (mix median k = 2), widened by a trend-error term that
  grows like a random walk with horizon, sigma = 0.25*sqrt(horizon_days),
  and floored by the same mix expressed in relative (logit) terms so the
  spread survives when earners are large (absolute k-steps shrink in
  logit as E grows).  The wider of the two spreads is used per level.
- Stdlib only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z_NORM = [-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722]
REFRESH = 3868.0 / 638.0  # calibrated single-refresh tracked jump
# Fitted refresh mix: (earner increment k, weight).  Weights sum to 1.
MIX = [(0, 0.2735), (1, 0.1999), (2, 0.4175), (3, 0.0425),
       (8, 0.0274), (12, 0.0202), (20, 0.0124), (40, 0.0066)]
E_REF = 7.0          # typical practice earner count (reference for log-ratios)
TREND_SIGMA = 0.25   # logit per sqrt(day) of trend-extrapolation error


def parse_time(s):
    """Parse an ISO-8601 time (date or datetime, 'Z' allowed)."""
    s = s.strip()
    if len(s) == 10:
        s += "T00:00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logit(x):
    p = min(max(x, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    if z >= 0:
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return 100.0 * p


def clamp_rate(x):
    return min(max(x, 0.01), 99.99)


def round2(x):
    return min(max(round(clamp_rate(x), 2), 0.01), 99.99)


def fit_loglog_slope(hours, values):
    """Least-squares slope of ln(v) against ln(1+h). None if unidentified."""
    pts = [(math.log1p(h), math.log(v))
           for h, v in zip(hours, values) if v > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mix_quantile_k(tau):
    """Smallest k whose cumulative mix weight reaches tau."""
    cum = 0.0
    for k, w in MIX:
        cum += w
        if cum >= tau:
            return k
    return MIX[-1][0]


def mix_quantile_value(tau, e_proj, t_proj):
    """Tau-quantile of 100*(e_proj+k)/t_proj under the fitted mix."""
    cum = 0.0
    for k, w in MIX:
        cum += w
        if cum >= tau:
            return 100.0 * (e_proj + k) / t_proj
    k = MIX[-1][0]
    return 100.0 * (e_proj + k) / t_proj


def forecast(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    hist = case.get("history", [])
    release = parse_time(case["game"]["release_date"])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)
    h_target = (target - release).total_seconds() / 3600.0

    reads = []
    for r in hist:
        t = parse_time(r.get("slot") or r["at"])
        reads.append((t, int(r["players_tracked"]), list(r["rates"])))
    reads.sort(key=lambda x: x[0])

    if reads:
        t_last = reads[-1][1]
        h_last = (reads[-1][0] - release).total_seconds() / 3600.0
        hours = [(r[0] - release).total_seconds() / 3600.0 for r in reads]
    else:
        t_last = 638
        h_last = 0.0
        hours = []

    # --- tracked projection -------------------------------------------
    if len(reads) >= 2 and len(set(r[1] for r in reads)) > 1:
        slope_t = fit_loglog_slope(hours, [r[1] for r in reads])
        if slope_t is None:
            t_proj = t_last * REFRESH
        else:
            t_proj = t_last * ((1.0 + h_target) / (1.0 + h_last)) ** slope_t
            t_proj = min(max(t_proj, float(t_last)), t_last * 100.0)
    else:
        t_proj = t_last * REFRESH

    # --- earner series per trophy --------------------------------------
    e_last = []
    e_series = []
    for i in range(n_tr):
        series = []
        for _, t, rates in reads:
            if i < len(rates):
                series.append(int(round(rates[i] * t / 100.0)))
        e_series.append(series)
        e_last.append(series[-1] if series else 0)

    slopes = []
    for i in range(n_tr):
        s = e_series[i]
        if len(s) >= 2 and len(set(s)) > 1:
            sl = fit_loglog_slope(hours, s)
            if sl is not None:
                slopes.append(sl)
    med_slope = median(slopes) if slopes else 0.0

    e_proj = []
    for i in range(n_tr):
        s = e_series[i]
        last = e_last[i]
        if len(s) >= 2 and len(set(s)) > 1:
            sl = fit_loglog_slope(hours, s)
            if sl is None:
                e_proj.append(float(last))
                continue
            # Shrink the per-trophy slope halfway toward the median slope.
            sl = 0.5 * sl + 0.5 * med_slope
            val = last * ((1.0 + h_target) / (1.0 + h_last)) ** sl
            e_proj.append(val)
        else:
            e_proj.append(float(last))
    # Earners cannot exceed the players tracked.
    e_proj = [min(v, 0.99 * t_proj) for v in e_proj]

    # --- quantiles ------------------------------------------------------
    sigma_trend = TREND_SIGMA * math.sqrt(horizon_days)
    out_rows = []
    for i, tr in enumerate(trophies):
        # Centre: mix median k = 2 earners above the smooth projection.
        c_rate = clamp_rate(100.0 * (e_proj[i] + 2.0) / t_proj)
        c_logit = logit(c_rate)
        qs = []
        for j, tau in enumerate(LEVELS):
            # (a) absolute-k mix quantile at the projected earners.
            qa = mix_quantile_value(tau, e_proj[i], t_proj)
            # (b) relative refresh mix + trend error, in logit terms.
            k_tau = mix_quantile_k(tau)
            z_ref = math.log((E_REF + k_tau) / (E_REF + 2.0))
            z = z_ref + sigma_trend * Z_NORM[j]
            qb = inv_logit(c_logit + z)
            # Wider spread wins; direction follows tau (below/above centre).
            if tau < 0.5:
                q = min(qa, qb)
            elif tau > 0.5:
                q = max(qa, qb)
            else:
                q = c_rate
            qs.append(round2(q))
        # Enforce non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        out_rows.append({"id": tr["id"], "q": qs})
    return {"trophies": out_rows}


def process_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        case = json.load(f)
    result = forecast(case)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")
    os.chmod(out_path, 0o644)


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    if len(argv) > 1:
        in_path = argv[1]
        if len(argv) > 2:
            out_path = argv[2]
        else:
            stem = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(out_dir, stem + ".out")
        process_file(in_path, out_path)
        return 0
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        stem = os.path.splitext(fn)[0]
        process_file(os.path.join(in_dir, fn),
                     os.path.join(out_dir, stem + ".out"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
