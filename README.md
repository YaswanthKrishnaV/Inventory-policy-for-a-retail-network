# Inventory policy for a retail network — Project A

Replenishment policy for Walmart CA FOODS (1 DC -> 4 stores) under censored demand.
See `PLAN.md` for the full approach, `DATA_ISSUES.md` for what is wrong with the data, `DECISIONS.md` for choices.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv` (Kaggle M5) in `data/`.

## Run
```bash
python -m project_a.src.data            # build cached panel (~10 s)
python -m project_a.audit_data --cutoff 1885   # data audit -> reports/data_audit.md
```
(Forecast, simulation and policy entry points are added as they are built.)
