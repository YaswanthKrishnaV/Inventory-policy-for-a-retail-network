# Geo-level marketing mix model with experiment calibration — Project B

What each paid channel truly returns, and how to allocate next quarter's 18M, for the Meridian
sample data (40 geos × 156 weeks). See `PLAN.md` (approach), `DATA_ISSUES.md`, `DECISIONS.md`,
`DECISION_LOG.md` (full record), `docs/dag.md` (causal structure), `memo_B.pdf` (CMO/CFO memo).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, pandas, scipy, matplotlib, pyyaml — no PPL needed
```
The data (`data/geo_all_channels.csv`, Meridian repo `meridian/data/simulated_data/csv`, Apache-2.0) is included.

## Run
```bash
python -m project_b.run_project_b                              # everything; hold-out = final 13 weeks
python -m project_b.run_project_b --holdout-start 2023-07-03   # any 13-week hold-out window
python -m project_b.run_project_b --fast                       # short chains, no sensitivity fits
python -m pytest -q tests                                      # exact-algebra tests
python -m project_b.recovery                                   # sampler recovers known ROIs
```

## Outputs
| file | content |
|---|---|
| `project_b/posterior.nc` | main-model posterior (netCDF3: chains × draws × parameters) |
| `project_b/roi_table.csv` | average and marginal ROI per channel, mean / median / 90% CI |
| `project_b/budget_allocation.csv` | 12M / 18M / 24M allocations, revenue and uplift with 90% CI |
| `project_b/forecast_13w.csv` | hold-out forecast per geo-week and national, with 90% intervals |
| `reports/results.md` | every table quoted in the memo, all model variants, diagnostics |

## How it works (one paragraph)
A hierarchical Bayesian geo model: per-capita conversions = geo intercept + shared trend and
annual seasonality + geo-specific control effects + promo + for each paid channel and organic, a
geo-pooled effect × Hill-saturated geometric adstock of spend. Priors start from Meridian's
defaults and are framed on ROI. Inference is exact where it can be (all linear coefficients are
integrated out / drawn in closed form) and Metropolis for the 12 curve parameters, ROI and geo
spread; validated by recovering known ROIs from simulated data. The vendor's Channel3 lift test
can be added as an extra likelihood term; because the data shows the test was not executed as
described, the main model does not use it.
