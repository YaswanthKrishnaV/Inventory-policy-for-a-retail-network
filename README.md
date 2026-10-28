# Senior Data Scientist take-home — inventory policy (A) and geo MMM (B)

- **Project A:** a replenishment policy for Walmart's California FOODS business (1 DC → 4 stores,
  1,437 items) that minimises total inventory cost while keeping fill rate ≥ 95%, under censored
  demand.
- **Project B:** what each paid channel truly returns, and how to allocate next quarter's 18M,
  from a hierarchical Bayesian geo-level marketing mix model checked against a vendor geo-lift
  test (Meridian sample data, 40 geos × 156 weeks).

| Document | What it is |
|---|---|
| `memo_A.pdf`, `memo_B.pdf` | 2-page memos for the COO (A) and the CMO/CFO (B) |
| `DECISIONS.md` | the major choices, alternatives and why, for both projects (≤ 3 pages) |
| `DECISION_LOG.md` | full chronological log of every decision and the analysis behind it |
| `DATA_ISSUES.md` | what is wrong with each dataset and how it is handled |
| `AI_USAGE.md` | how AI was used, including where it was wrong and how that was caught |
| `PLAN.md`, `PLAN_B.md` | the original plans and requirement maps (B includes validation against published practice) |
| `docs/dag.md` | Project B causal structure |
| `reports/`, `reports/project_b/` | auto-generated audits, backtests, results tables and figures |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # macOS: LightGBM (optional backend for A) needs `brew install libomp`
```
- Project A: put the Kaggle M5 files `sales_train_evaluation.csv`, `sell_prices.csv` and `calendar.csv` in `data/`.
- Project B: the data (`data/geo_all_channels.csv`, from the Meridian repo `meridian/data/simulated_data/csv`, Apache-2.0) is included.

## Run everything (one command)
```bash
./run_all.sh                                       # A at cutoff d_1885, B with hold-out = final 13 weeks
CUTOFF=1857 HOLDOUT_START=2023-07-03 ./run_all.sh  # any cutoff / hold-out window
```
This runs both pipelines and the tests, then rebuilds the memo PDFs if `pandoc` and
`wkhtmltopdf` are installed.

Measured runtime is about 85 minutes on a 2-core machine: Project A about 63 minutes, Project B
about 22 minutes. It is faster on a recent laptop.

## Project A — run and outputs
```bash
python -m project_a.run_project_a --cutoff 1885          # full pipeline; retrains on d <= cutoff only
python -m project_a.run_project_a --cutoff 1885 --fast   # quick look: 20 sims, no backtest/tuning
python -m project_a.audit_data --cutoff 1885             # data audit        -> reports/data_audit.md
python -m project_a.backtest                             # forecast backtests -> reports/backtest.md
python -m project_a.simulate --policy baseline           # replay one policy
```
| Output | Content |
|---|---|
| `project_a/policy.py` | `policy(state) -> orders`: baseline and recommended |
| `project_a/simulate.py` | replay simulator that enforces every rule |
| `project_a/forecasts.parquet` | 28-day quantile forecasts (q05…q95) per item × store and item × DC |
| `reports/results_<cutoff>.md` | cost, fill rate and stress results quoted in memo A |

```
config/project_a.yaml     every assumption (costs, lead times, caps, quantiles, knobs, seeds)
project_a/
  run_project_a.py        end-to-end entry point (--cutoff)
  simulate.py, policy.py  simulator and policies;  stress.py  lead-time shocks;  tune.py  knob tuning
  backtest.py, audit_data.py, report.py
  src/demand_recovery.py  stockout / listing / closure detection and imputation
  src/forecast.py         global Poisson GBM + two-level gamma-Poisson paths, coherent DC
```
**No leakage.** Training, demand recovery, caps and initial stock use only d ≤ cutoff. Knobs are
tuned on replays that end on or before the cutoff. Inside the replay the model is never
retrained; it re-forecasts every 28 days from the demand observed so far. Test-period prices are
used only to price costs.

## Project B — run and outputs
```bash
python -m project_b.run_project_b                              # hold-out = final 13 weeks
python -m project_b.run_project_b --holdout-start 2023-07-03   # any 13-week hold-out window
python -m project_b.run_project_b --fast                       # short chains, no sensitivity fits
python -m project_b.run_project_b --reuse-fits                 # regenerate outputs from cached posteriors (< 1 min)
python -m project_b.recovery                                   # sampler recovers known ROIs
```
| Output | Content |
|---|---|
| `project_b/posterior.nc` | main-model posterior (netCDF3: chains × draws × parameters) |
| `project_b/roi_table.csv` | average and marginal ROI per channel, mean / median / 90% CI |
| `project_b/budget_allocation.csv` | 12M / 18M / 24M allocations, revenue and uplift with 90% CI |
| `project_b/forecast_13w.csv` | hold-out forecast per geo-week and national, with 90% intervals |
| `reports/project_b/results.md` | every table quoted in memo B, all model variants, diagnostics |

```
config/project_b.yaml     every assumption (priors, experiment, sampler, budget, next-experiment grid)
project_b/
  run_project_b.py        end-to-end entry point (--holdout-start)
  optimise.py             greedy water-filling budget optimiser, value of information
  experiment_design.py    power by simulation for the next geo test
  audit.py, recovery.py, report.py
  src/model.py            hierarchical geo MMM and sampler;  src/linear.py  exact Gaussian block
  src/roi.py              average / marginal ROI and response curves
```
**No leakage.** The model is trained strictly on weeks before `--holdout-start`. ROI, budget and
experiment design all come from that fit; the 13 held-out weeks are used only to score the
forecast.

## Tests
```bash
python -m pytest -q tests     # A: simulator invariants;  B: exact algebra vs dense matrices
```
