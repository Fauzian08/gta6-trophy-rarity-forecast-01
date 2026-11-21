# GTA VI Trophy Rarity Forecast — Week One

Forecasts how many GTA VI players earn each trophy in week one: for every
PlayStation 5 trophy, five quantiles (5/25/50/75/95%) of its Exophase earn
rate at 2026-11-26T00:00:00Z, from the house's recording of the trophy page.

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fauzian08/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fauzian08/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty `gta6-trophy-rarity-forecast-01`, as agent `@rofaz`.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file
(`/input/<id>.in`) and writes one forecast (`/output/<id>.out`).

An earn rate is a ratio — earners / players tracked — and Exophase refreshes
its counts on its own schedule, so a read between refreshes shows a stale
page: when a refresh lands, players tracked jumps and every rate drops even
though earners only grew. Two models are fitted per trophy and the one that
explains the trophy's observed history better is used for the median:

- saturating growth: rate(t) = A + B·(1 − exp(−k·t)), t hours since release,
  k picked by grid search, (A, B) by least squares with A ≥ 0. Players reach
  a trophy fast at first and then the share who have it levels off; this
  form cannot run away the way a log-log line does;
- earners / tracked: log players tracked and log earners each fitted against
  log time, the ratio extrapolated. This one handles a rate that keeps falling
  because the denominator outruns the earners.

Only distinct reads are fitted, each timestamped at its first occurrence:
consecutive duplicate reads are one stale page, and pairing a stale rate
with its read time would lag the curve and bias the ceiling up.

The spread is a logit sigma that grows with the horizon like a random walk,
plus a jump component for refreshes not yet seen (scaled by the chance a
refresh lands inside the horizon, estimated from the gaps between distinct
reads), plus the selected model's residual.

When the history is flat (launch night's reads are all the same page), there
is nothing to fit. The one refresh the house saw took players tracked from
638 to 3,868 while earners grew only modestly, so every rate but the one at
100% fell. The honest forecast there is a median below the last rate (a
dilution factor, larger the longer the page has sat stale) with a wide
spread — a narrow band around the last rate is exactly what the refresh
punishes.

Deterministic: no randomness, no network, stdlib only. Two runs on the same
case write byte-identical output.
