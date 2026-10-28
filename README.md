# Inventory policy for a retail network — Project A

Replenishment policy for Walmart's California FOODS business (1 DC → 4 stores, 1,437 items)
that minimises total inventory cost while keeping fill rate ≥ 95%, under censored demand.

| Document | What it is |
|---|---|
| `memo_A.pdf` | 2-page memo for the COO |
| `DECISIONS.md` | the major choices, alternatives, why (≤ 3 pages) |
| `DECISION_LOG.md` | full chronological log of every decision and analysis behind it |
| `DATA_ISSUES.md` | what is wrong with the data and how it is handled |
| `AI_USAGE.md` | how AI was used, including where it was wrong |
| `PLAN.md` | the original plan and requirement map |
| `reports/` | auto-generated audit, backtest, tuning and results tables + figures |
| `project_a/forecasts.parquet` | 28-day quantile forecasts (q05…q95) per item × store and item × DC |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # macOS: LightGBM (optional backend) needs `brew install libomp`
```
Put the Kaggle M5 files `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv` in `data/`.

## Run
```bash
make all                                    # tests + full pipeline at the default cutoff d_1885
make project_a CUTOFF=1857                  # any cutoff: retrains on d <= cutoff only
python -m project_a.run_project_a --cutoff 1885 --fast   # quick look: 20 sims, no backtest/tuning
```
Expected runtime of `make all` on a recent laptop: well under 2 hours (forecast fits dominate;
fitted models are cached in `cache/` so re-runs are fast).

Individual steps:
```bash
python -m project_a.audit_data --cutoff 1885     # data audit        -> reports/data_audit.md
python -m project_a.backtest                     # forecast backtests -> reports/backtest.md
python -m project_a.simulate --policy baseline   # replay one policy
pytest -q tests                                  # simulator invariants
```

## Layout
```
config/project_a.yaml     every assumption (costs, lead times, caps, quantiles, knobs, seeds)
project_a/
  run_project_a.py        end-to-end entry point (--cutoff)
  simulate.py             replay simulator (Task 3)
  policy.py               policy(state) -> orders: baseline and recommended (Task 4)
  stress.py               lead-time shock scenarios (Task 5)
  tune.py                 knob tuning on past windows only
  backtest.py             rolling forecast backtests (Task 2)
  audit_data.py           data audit (Task 1)
  report.py               results tables and figures
  src/data.py             M5 -> CA FOODS panel (wide arrays, cached)
  src/demand_recovery.py  stockout / listing / closure detection and imputation (Task 1)
  src/features.py         lag>=28 feature matrices
  src/forecast.py         global Poisson GBM + two-level gamma-Poisson paths, coherent DC (Task 2)
tests/                    simulator invariants
```

## No-leakage guarantees
- Training, demand recovery, capacity limits, store caps and initial stock use data with d ≤ cutoff only.
- Policy knobs are tuned on 56-day replays that end on or before the cutoff (cutoff − 56/84/112).
- Inside the replay the model is never retrained; every 28 days it re-forecasts from demand observed so far.
- Test-period prices are used only to price costs (not as forecast features, unless `prices.use_future_prices`).
