# Project A — Plan: Inventory policy for Walmart CA FOODS under censored demand

> Working plan for the Fire AI Senior Data Scientist take-home (Project A only).
> Purpose: lock down *what is being asked*, *what decisions we have to make*, and *how we will build it end to end*, before writing code.

---

## 0. TL;DR

- **The real objective is a cost number, not a forecast score.** We are graded on total inventory cost and fill rate (≥ 95%) when *they* re-run our policy at *their* cutoff, versus a base-stock baseline. Forecasting and demand recovery only matter through that lens.
- **The cost structure is extremely lopsided.** Holding one unit for a day costs `0.25/365 ≈ 0.07%` of price; losing a sale costs `35%` of price. One lost sale ≈ **511 unit-days of holding**. So the cost-optimal policy wants very high service; the 95% fill-rate floor is unlikely to be the binding constraint. **What actually binds are the physical limits:** DC outbound capped at 1.25× average daily sales, the 21-day store cap, weekly DC ordering, and long, noisy FOODS_3 lead times (p95 ≈ 17.5 days).
- **Biggest data trap: zeros in M5 are not demand.** They mix (a) item not yet listed / delisted, (b) store closed (Christmas), (c) stockouts, (d) true zero demand. Training on raw sales biases demand down → under-stocking → lost sales that cost 500× more than the holding we "saved".
- **Recommended solution:** censoring-aware demand recovery → one global LightGBM model with a count distribution on top → coherent **sample paths** (store paths sum to DC by construction) → **two-echelon, capacity-aware, forecast-driven base-stock policy** whose few safety knobs are tuned by simulation on *backtest* cutoffs → Monte Carlo replay simulator with random lead times and censored-day demand imputation.
- Everything is driven by `--cutoff` and a single `config.yaml`, so we can change any assumption live in the defence interview.

---

## 1. Business problem, in plain terms

Walmart California runs **1 DC → 4 stores** (CA_1..CA_4) for ~1,437 FOODS items (~5,700 item×store series = 1,437 × 4 stores — verify exact count on load). Suppliers ship to the DC; the DC ships to stores.

The COO wants: *"How much should the DC order from suppliers each Monday, and how much should each store get each day, so that we spend the least on inventory while keeping at least 95% of customer demand served?"*

Total cost to minimise over the 56-day test window:

| Component | Rule | Notes |
|---|---|---|
| Holding | 25% of selling price / year / unit → `0.25·p/365` per unit-day | At DC and stores. Decision: charge on end-of-day on-hand; in-transit not charged (documented in DECISIONS.md, switchable flag). |
| Lost sales | 35% of selling price per unit of unmet store demand | Lost sales, not backorders (grocery). |
| Purchase orders | $50 per department per PO | 3 departments (FOODS_1/2/3). Max 8 Mondays × 3 = $1,200 — small; skipping weeks is a minor lever. |
| DC→store transfer | $0.05 per unit | Paid on every unit shipped; only "wasted" on units that never sell. For a $1 item this is 5% of price — not negligible vs holding. |

Constraints: DC outbound ≤ 1.25 × avg daily CA FOODS sales (last 365 training days); store on-hand ≤ 21 × item's avg daily sales; DC orders only on Mondays; store→DC lead time fixed 2 days; supplier→DC lead time lognormal (mean 5/7/10 days for FOODS_1/2/3, CV 0.4).

**Fill rate definition (decision):** units sold ÷ units demanded, summed over all stores, items and test days (customer-facing, volume-weighted). We will also report it per department and per store, and an item-weighted version, because a 95% volume fill rate can hide terrible service on slow movers.

**What "uncertainty ranges" means here:** we run the replay many times (random lead times × draws of imputed demand on censored days) and report mean and 90% interval of cost and fill rate, plus the probability fill rate ≥ 95%.

---

## 2. Requirements map (nothing gets missed)

| # | Requirement from the brief | Where it's satisfied | Done when |
|---|---|---|---|
| T1 | Find stockout/missing-listing periods, estimate unconstrained demand, quantify how wrong a raw-sales model would be | `src/demand_recovery.py`, `DATA_ISSUES.md`, notebook figs | Flags + imputed demand table; bias in units **and in $ cost via simulator** |
| T2 | 28-day-ahead daily quantiles 0.05–0.95 per item×store; store sums to DC; ≥3 rolling backtests before d_1885 | `src/forecast.py`, `forecasts.parquet` | Quantile columns q05…q95 for store and DC rows; pinball/coverage at ≥3 cutoffs |
| T3 | `simulate.py` replays a policy through the test period, random lead times, capacity limits, explains censored-day handling | `project_a/simulate.py` | Deterministic given seed; docstring + DECISIONS section on censoring |
| T4 | `policy(state) -> orders` for DC POs and DC→store transfers; compare to simple base-stock baseline | `project_a/policy.py`, `src/baselines.py` | Table: cost, fill rate, CIs, policy vs baseline |
| T5 | Stress test: FOODS_3 lead times double for 3 weeks; impact + what to change with 2 weeks' warning | `src/stress.py`, memo section | Three runs: base / shock unwarned / shock with 2-week warning mitigation |
| T6 | COO memo ≤ 2 pages: policy, expected cost & fill rate with ranges, 3 biggest risks, data wishlist | `memo_A.pdf` | ≤ 2 pages, CFO-readable, every number reproducible |
| I1 | `--cutoff` arg retrains on data ≤ cutoff only | `run_project_a.py --cutoff` | Run at d_1885 and at e.g. d_1857 and d_1913 without code change |
| D1 | `project_a/forecasts.parquet` with quantile columns | pipeline output | — |
| D2 | `DECISIONS.md` (≤3 pages, shared with B), `DATA_ISSUES.md`, `AI_USAGE.md` | repo root | Every major choice: alternatives + why rejected |
| D3 | Single command, < 2 hours on a laptop, pinned deps | `Makefile` (`make all`), `requirements.txt` with `==` pins | Timed end-to-end run logged |

**Hidden requirements implied by "we re-run at a cutoff we choose":**
- The test window must be `cutoff+1 … min(cutoff+56, 1941)`, not hard-coded d_1886.
- Nothing may be tuned on d_1886–d_1941 (no peeking). All tuning happens on backtest windows *before* the cutoff.
- The 365-day capacity window, average-sales normalisations (21-day cap, 7-day starting stock) and the first Monday must all be computed relative to the cutoff.
- Must handle a test window of <56 days and a cutoff that doesn't fall on a Sunday (first DC order = first Monday on/after cutoff+1).

---

## 3. Data facts worth knowing before we start (to verify on load)

- d_1 = 2011-01-29 (Sat). **Test window d_1886 = 2016-03-28 (Mon) → d_1941 = 2016-05-22 (Sun)**. Easter is the day before the test starts; window contains Orthodox Easter, Cinco de Mayo, Mother's Day; SNAP days for CA are the 1st–10th of each month.
- `sell_prices.csv` is weekly (`wm_yr_wk`). **No price row ⇒ item not on sale in that store that week** — this is the cleanest "not listed" signal M5 gives us.
- Stores close on Christmas (d_331, d_697, d_1062, d_1427, d_1792) → network-wide zeros that are not demand.
- FOODS_3 is high-volume; FOODS_1/2 contain many intermittent, slow movers. Intermittency matters for both detection (zero runs are normal for slow items) and distributions (need count distributions, not Gaussian).
- Strong weekly seasonality (weekends noticeably higher than mid-week) — relevant because the DC outbound cap is 1.25× the *average*, so peak days can exceed it.
- Known store-level anomalies in CA (e.g. a CA store with a structural level shift around 2015) — to be confirmed in EDA, not assumed.
- Prices exist through d_1941. Using test-period prices in the forecast is arguably fine (planned prices are known in advance) but it's a leakage risk the panel may probe. Decision: use prices only up to the cutoff week + carry-forward by default; flag to switch on "known future prices".

---

## 4. Task 1 — Recover true demand (the "deliberately hard" part)

### 4.1 Taxonomy of a zero (or a low number)

| Type | Signal | Treatment |
|---|---|---|
| Not yet launched | Zeros before first non-zero sale; no price rows | Drop from training (not a demand observation). |
| Not listed / delisted week | No `sell_price` row for that week | Missing, not zero. Drop. If delisted at cutoff → zero forecast & no replenishment. |
| Store closed | Christmas; any day where *the whole store* sells ~0 | Missing. Drop. |
| Store-wide disruption | Store total far below its expected level on a day (and not a holiday) | Down-weight / treat as censored for all items that day. |
| Item stockout (full-day) | Zero run with price present that is implausible given item's recent rate | Censored → impute. |
| Item stockout (partial-day) | Sales low but non-zero, often at end of a run of zeros or on peak days | Mostly undetectable from daily data; quantify via sensitivity, list in DATA_ISSUES and memo "data we'd need". |
| Genuine zero demand | Short zero runs on slow movers | Keep as real zeros. |

### 4.2 Detection — options

| Approach | How | Pros | Cons |
|---|---|---|---|
| **A. Zero-run probability test** *(recommended core)* | For each zero run of length k with price present, estimate local rate λ from surrounding non-censored days (robust, e.g. trimmed mean of ±56 days excl. zeros-in-runs, adjusted for day-of-week). Flag if P(k consecutive zeros \| NB(λ, α)) < 1% | Simple, explainable to a CFO, adapts to slow movers, fast on 5,700 series | Needs a threshold; misses partial stockouts; λ estimate itself biased if neighbours are censored (iterate twice) |
| B. Two-state HMM per series (in-stock / out-of-stock) | Hidden state; emission NB(λ) vs point mass at 0 | Principled, gives P(stockout) per day, handles ambiguity | Slower, fiddly to fit robustly at scale, harder to defend live |
| C. Fixed rule ("≥ N zero days in a row") | Hard threshold | Trivial | Wrong for both fast (1 zero day is suspicious) and slow movers (7 zeros normal) |
| D. Cross-store/sister signal | Item sells in 3 stores, zero in the 4th | Good corroboration | Not a standalone detector |

**Plan:** A as the detector, D as corroboration feature, store-level disruption check on top. B only as a stretch / cross-check on a sample.

### 4.3 Imputation — options

| Approach | Pros | Cons |
|---|---|---|
| **Model-based (EM-style)**: train forecast model on non-censored days only (censored rows masked out of the loss), predict on censored days, iterate once | Uses all features (price, events, SNAP, DOW); consistent with forecaster | Needs care to avoid leakage across cutoffs |
| Local interpolation (λ × DOW factor) | Fast, transparent | Ignores price/events |
| Tobit / censored likelihood (treat sales as `min(demand, stock)`) | The theoretically right likelihood | We don't observe stock, so the censoring point is unknown — collapses to masking anyway |

**Plan:** for training — **mask** censored rows (don't fabricate targets); for analysis and for the simulator's censored test days — impute with the model's predictive distribution (draws, not a point), so imputation uncertainty propagates into cost ranges.

### 4.4 Validating the detector without ground truth (senior-level point)

**Synthetic injection test:** take clean in-stock windows, artificially insert stockout runs of known length (and partial-day cuts), run the detector, report precision/recall by velocity band and imputation error. This is how we "quantify how well we found where sales understate demand" honestly.

### 4.5 "How wrong would a raw-sales model be?" — quantify three ways

1. **Bias in units:** forecast trained on raw sales vs recovered demand, evaluated on non-censored days of backtest windows → % under-forecast, by department/velocity.
2. **Share of demand hidden:** imputed lost units ÷ total demand, by dept/store/time.
3. **In dollars (the one the panel cares about):** run the *same* policy driven by the raw-sales forecast vs the recovered-demand forecast through the simulator → Δ lost-sale cost, Δ fill rate. This is the number for the memo.

---

## 5. Task 2 — Probabilistic, coherent forecast

### 5.1 Model options

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Global LightGBM (Tweedie/Poisson mean) + count distribution (NB with modelled dispersion) → sample paths** | Proven on M5 (top solutions were LightGBM), fast on a laptop, one model for all series, distribution gives coherent samples | Need a dispersion model; temporal correlation needs explicit handling | **Recommended** |
| LightGBM quantile regression (one model per quantile) | Direct pinball optimisation | 19 models, quantile crossing, **no way to aggregate quantiles to DC** (quantiles don't add) | Use only as a calibration benchmark |
| DeepAR / TFT (GluonTS, neuralforecast) | Native probabilistic, sample paths | Slow on CPU, harder to explain/change live, 2-day budget | Reject (mention as future) |
| Classical per-series (ETS/Croston/TSB) | Interpretable, good for intermittent | 5,700 fits, weak on price/event effects, poor covariates | Fallback for zero-history/new items |
| Empirical residual bootstrap around LightGBM mean | Distribution-free | Needs lots of residual history per series; weak for slow movers | Alternative to NB if NB miscalibrates |

### 5.2 Design

- **Target:** recovered demand with censored rows masked; unlisted/closed days dropped.
- **Features:** lags ≥ 28 (direct, no recursion) + rolling means/std over 7/28/56/112 days, day-of-week, month, SNAP_CA, event flags & days-to-event, price, price relative to item's own median, price change, item/dept/store IDs (categorical), store-day level index. Lags computed on recovered demand.
- **Horizon:** the test is 56 days but the brief asks for 28-day forecasts. Plan: **re-forecast at day 28 of the replay using data observed up to then** (in a real operation you would; the policy only sees *sales*, so the in-sim refit uses observed sales + the same recovery logic). Fallback flag: single 56-day horizon forecast with lag ≥ 56 features.
- **Distribution:** NB with mean from LightGBM and dispersion from a second small model on squared Pearson residuals (by velocity band at minimum). Check calibration; switch to bootstrap if coverage is off.
- **Coherence (store → DC):** generate **S sample paths** (e.g. 200) per item×store with a shared item-level shock (Gaussian copula / common multiplicative factor estimated from cross-store residual correlation) so store paths are realistically correlated. **DC path = sum of store paths** → coherent by construction; DC quantiles are taken from summed paths, never by adding quantiles. Means can additionally be checked against a MinT/bottom-up reconciliation.
- **Outputs:** `forecasts.parquet` with `item_id, store_id (incl. 'DC'), dept_id, d, date, horizon, mean, q05, q10, …, q95`. Sample paths kept in a separate binary file (npz/parquet) for the policy and simulator.

### 5.3 Backtest

- Rolling cutoffs before d_1885, e.g. **d_1801, d_1829, d_1857** (28-day windows), plus d_1885 itself as the final fit. Configurable list.
- Metrics: weighted pinball loss over all quantiles, empirical coverage of 50%/90% intervals, bias, WRMSSE-style scaled error; reported per dept, velocity band, and at DC level.
- Compare against: seasonal-naive, raw-sales LightGBM (for the bias story), quantile-LightGBM.
- **The same cutoffs are used to tune policy knobs** (Section 7.4), keeping the test window clean.

---

## 6. Task 3 — Replay simulator (`simulate.py`)

### 6.1 Daily event order (for each day t in the test window)

1. **Arrivals:** supplier→DC POs whose sampled lead time ends today; DC→store transfers shipped at t−2.
2. **Demand realises** at stores (see 6.2). `sales = min(on_hand, demand)`; `lost = demand − sales`.
3. **Policy call:** `policy(state)` sees only the past (observed sales, not true demand; inventory positions; pipeline; forecasts made at the last cutoff/refit; calendar). Returns transfers for today and, on Mondays, POs by department.
4. **Constraint enforcement (simulator, not trusted to the policy):** DC outbound ≤ capacity (if violated, scale down via the same allocation rule and log); can't ship more than DC on-hand; store order-up-to ≤ 21-day cap counting inventory in transit to the store (decision: cap applies to on-hand + inbound; documented).
5. **Ship:** transfers leave DC (arrive t+2); POs get a lead time draw `L ~ LogNormal(μ, σ)` with `σ² = ln(1+0.4²) = 0.148`, `μ = ln(mean) − σ²/2`. Orders may cross (documented).
6. **Costs:** holding on end-of-day on-hand at DC and stores, lost-sale cost, $50 per department with a PO, $0.05 × units shipped.

**Initial state:** each store holds 7 × item's avg daily sales; DC holds 7 × network avg daily sales per item (decision: "each location" includes the DC); **no pipeline** at start (decision, flagged — this makes the first ~1–2 weeks tight for FOODS_3, which is realistic and worth showing). "Average" = last 28 days before cutoff of recovered demand (decision; alternative: 365-day average — configurable).

### 6.2 Demand on censored test days (the question they will definitely ask)

On test days, recorded sales are themselves censored by *Walmart's* historical stock, not ours. If our policy has stock on a day Walmart sold 0 due to a stockout, true demand was higher than recorded.

| Option | Pros | Cons |
|---|---|---|
| Use recorded sales as demand | Simple, no modelling | Rewards policies that happen to stock out when Walmart did; under-states demand; inflates fill rate |
| **Recorded sales on clean days + draws from the predictive/imputation distribution on flagged-censored days** *(recommended)* | Honest, propagates uncertainty, consistent with Task 1 | Relies on detector; imputation model also used by policy → mild optimism (mitigate: impute with a model fit with *all* data incl. test-period uncensored days, separate from the policy's forecaster) |
| Fully synthetic demand from the forecast distribution | Clean for stress tests | Divorced from reality; flatters the forecaster |

**Plan:** option 2 as main; option 1 and 3 as sensitivity rows. Also handle not-listed test days (demand = 0, no cost) and store closures.

### 6.3 Monte Carlo

N runs (e.g. 50–200; config) over (lead-time draws × censored-day demand draws), seeded. Report mean, 5–95% interval of total cost and fill rate, cost breakdown, P(fill ≥ 95%). Vectorised over items with numpy (5,700 items × 56 days is small) so 100 runs stay within minutes.

---

## 7. Task 4 — Policy design

### 7.1 What the economics tell us (put this in the memo)

- Critical ratio at the store ≈ `0.35 / (0.35 + 0.000685·Δt)` → > 99% for any reasonable Δt. ⇒ **Run out as rarely as physically possible; overstock is cheap.**
- So the trade-offs that actually matter: (i) the 21-day store cap, (ii) DC outbound cap at 1.25× average — **weekend/event peaks will exceed it, so stores must be pre-filled on weekdays (lookahead)**, (iii) FOODS_3's protection interval: review 7 days + lead time mean 10, p95 ≈ 17.5 ⇒ the DC must cover ~25+ days of network demand at high quantiles, (iv) $0.05/unit transfer on very cheap items.
- Purchase cost is *not* in the objective — so buying a lot only costs holding. We must report **ending inventory** too, and not let the policy look good by under-stocking at the end or exploit free purchasing. (CFO honesty.)

### 7.2 Policy families

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| Simple base-stock (mean of last 28 days, normal safety stock, z fixed) | Required baseline; transparent | Ignores censoring, seasonality, capacity, lead-time tail | **Baseline** |
| (s, S) | Handles fixed order costs | Only fixed cost is $50/dept/week — negligible; adds parameters without gain | Reject (explain why) |
| **Forecast-driven two-echelon base-stock, capacity-aware, quantile-based levels** | Explainable, fast, uses the probabilistic forecast directly, few knobs to tune and to change live | Myopic; heuristics for allocation | **Recommended** |
| Stochastic optimisation / SAA over sample paths (e.g. LP per week) | Optimises jointly under capacity using the same samples | More code, LP size (5,700 items × days × samples), harder live edits | Stretch: SAA only for the capacity-allocation step |
| Decision-focused learning | Trains forecaster on downstream cost | Heavy, unstable, little gain when cost is so lopsided | Reject (mention) |
| Reinforcement learning | Flexible | Sample-hungry, unexplainable to COO, unverifiable in 2 days, overfits simulator | Reject (mention) |

### 7.3 Recommended policy, concretely

**Stores (daily):**
- Target `S_store(i,s,t) = Q_{α_d}( demand over [t+2, t+2+1+lookahead] )` from sample paths (lead 2 + review 1, plus a lookahead buffer for upcoming capacity-constrained peaks), capped at `21 × avg_daily(i,s)`.
- Order = `max(0, S − (on_hand + in_transit))`.

**DC (Mondays, per department):**
- Echelon base-stock: `S_DC(i) = Q_{β_d}( network demand over [t, t + 7 + L] )` with L drawn from the lognormal and convolved with demand sample paths (Monte Carlo inside the policy), minus echelon position (DC on-hand + DC pipeline + store on-hand + store pipeline).
- Optional: skip a department's PO if expected marginal benefit < $50 (tiny effect; shows we read the cost table).

**Allocation when DC stock or outbound capacity is short:**
- Greedy by marginal expected lost-sale saving per unit: `0.35·p · P(demand > position)` ranked across all item×store requests; fill until capacity or DC stock runs out. Equivalent to a fair-share-with-priority rule, cheap to compute, easy to explain ("ship where a missing unit costs most").
- **Capacity lookahead:** compute next 7 days of expected outbound requirement; if any day exceeds the cap, pull shipments forward into earlier slack days (bounded by the 21-day cap). This is the single most important non-obvious piece.

**Knobs:** `α_d`, `β_d` (per department), lookahead length. Tuned by grid/Bayesian search in the simulator on the **backtest cutoffs**, objective = expected cost with a chance constraint P(fill ≥ 95%) ≥ 0.9.

### 7.4 Comparison table to produce

| Policy | Cost (mean, 90% CI) | Fill rate (mean, 90% CI) | Lost / holding / PO / transfer | Ending inventory value |
|---|---|---|---|---|
| Base-stock baseline (raw sales) | | | | |
| Base-stock baseline (recovered demand) | | | | |
| Recommended policy (raw-sales forecast) | | | | |
| **Recommended policy (recovered demand)** | | | | |
| Recommended − capacity lookahead (ablation) | | | | |

The ablations explain *where* the savings come from (demand recovery vs probabilistic levels vs capacity handling).

---

## 8. Task 5 — Stress test (FOODS_3 lead times double for 3 weeks)

- **Interpretation (decision):** POs placed in weeks 3–5 of the test window for FOODS_3 get lead-time mean 20 (CV 0.4 kept). Alternative interpretation (in-flight orders also delayed) run as a sensitivity.
- **Runs:** (a) base, (b) shock, policy unaware, (c) shock, policy warned 2 weeks earlier.
- **Mitigation with 2 weeks' warning:** policy switches FOODS_3 lead-time distribution to the shocked one for orders affected → pre-buys at the DC on the two Mondays before (DC storage is uncapped; outbound cap is the bottleneck so pre-position stores up to the 21-day cap during slack days), prioritise allocation to high-margin/high-velocity items, accept lower service on slow movers. Report Δcost and Δfill, by dept, and the day-by-day fill-rate trough chart.
- Memo framing: "unwarned, fill rate drops to X% for ~N weeks and costs $Y; with two weeks' notice we recover Z% of that by pre-buying ~K days of FOODS_3 stock at a holding cost of $W."

---

## 9. Task 6 — COO memo (≤ 2 pages)

1. Recommendation in one paragraph (what the policy does, in operational words).
2. Expected 56-day cost and fill rate with 90% ranges; vs baseline; where savings come from.
3. Three biggest risks (likely: supplier lead-time shocks on FOODS_3; DC outbound capacity at peaks; demand under-estimation from undetectable partial stockouts / new or delisted items).
4. Stress-test result and the 2-week-warning playbook.
5. Data we'd need: actual on-hand / stockout logs, intraday POS timestamps, supplier lead-time history, planned promotions, DC capacity by day.
6. What the model can't tell them.

Built as markdown → PDF (pandoc/weasyprint, or reportlab), one script, part of `make all`.

---

## 10. Repo layout

```
.
├── Makefile                  # make data-check | backtest | project_a | all
├── requirements.txt          # pinned (==)
├── config/project_a.yaml     # every assumption: costs, LTs, CV, caps, quantiles, knobs, seeds, N sims
├── data/raw/                 # M5 csvs (gitignored; README says how to get them)
├── project_a/
│   ├── run_project_a.py      # entry point: --cutoff, --config, --n-sims, --fast
│   ├── policy.py             # policy(state) -> orders  (required interface)
│   ├── simulate.py           # replay simulator (required)
│   ├── forecasts.parquet     # output (required)
│   └── src/
│       ├── data.py           # load, melt, scope to CA FOODS, calendar/prices join, memory-efficient dtypes
│       ├── demand_recovery.py
│       ├── features.py
│       ├── forecast.py       # LightGBM + dispersion + sample paths + reconciliation
│       ├── backtest.py
│       ├── baselines.py
│       ├── tuning.py         # policy knob search on backtest cutoffs
│       ├── stress.py
│       └── report.py         # tables, figures, memo_A.pdf
├── tests/                    # simulator invariants (stock conservation, caps never violated, cost accounting)
├── memo_A.pdf
├── DECISIONS.md  DATA_ISSUES.md  AI_USAGE.md
└── README.md
```

**Simulator invariants to unit-test** (cheap insurance for the live defence): inventory conservation; no negative stock; caps never exceeded; cost = sum of components; same seed → same result; a "perfect information" policy gives fill ≈ 100% when capacity allows.

---

## 11. Timeline (2 calendar days covers both projects → ~1 day for A)

| Block | Hours | Output |
|---|---|---|
| 1. Data load + EDA + issue hunting | 2 | `data.py`, first DATA_ISSUES entries, figures |
| 2. Demand recovery + injection test | 2 | flags, imputation, bias numbers |
| 3. Forecast + backtests + coherence | 3 | `forecasts.parquet`, backtest table |
| 4. Simulator + tests | 2 | `simulate.py`, invariants passing |
| 5. Baseline + recommended policy + tuning | 2.5 | comparison table |
| 6. Stress test | 1 | 3-run table + chart |
| 7. Memo, DECISIONS, AI_USAGE, end-to-end timed run | 1.5 | PDFs, `make all` < 2 h |

Keep a running `AI_USAGE.md` log as we go — the brief explicitly wants one place where the AI was wrong and how it was caught; we'll capture real ones (e.g. a sum-of-quantiles reconciliation, or a lognormal parameterised with σ = CV instead of `sqrt(ln(1+CV²))`).

---

## 12. Defence-interview readiness (likely live changes)

Make each of these a one-line config/flag change, and rehearse the reasoning:
- Fill-rate target 95% → 98%; lost-sale cost 35% → 10% (does the 95% constraint start binding?).
- Lead-time CV 0.4 → 0.8; FOODS_1 mean 5 → 10.
- DC orders twice a week; DC→store 2 → 4 days.
- Capacity 1.25× → 1.1×; store cap 21 → 10 days.
- Backorders instead of lost sales.
- "Turn off demand recovery — what happens to cost?" (we already have the ablation).
- "Show me where the policy decides allocation" / "change the priority rule".

---

## 13. Open decisions to confirm with you before coding

1. Fill rate = volume-weighted network fill rate at stores (plus per-dept view). OK?
2. DC starts with 7 days of *network* average sales; no initial pipeline.
3. Holding cost on on-hand only (not in-transit).
4. 21-day cap applies to on-hand + inbound.
5. Re-forecast at day 28 of the replay (vs a single 56-day forecast).
6. Prices: carry forward from cutoff by default (no future prices).

## 14. Data access

M5 needs a Kaggle account (competition rules accepted). The sandbox can't reach Kaggle, so either download `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv` and attach them / put them in a connected folder, or run the pipeline locally. `data/raw/` stays out of git.
