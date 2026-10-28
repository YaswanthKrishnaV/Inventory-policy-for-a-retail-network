# DECISION_LOG.md

The full chronological record behind `DECISIONS.md`: Project A (IDs A-xx / D-xx), then Project B (IDs B-Axx / B-Dxx).

# Project A — inventory policy

Chronological log of every decision and every analysis behind it. `DECISIONS.md` (the ≤3-page deliverable)
will be distilled from this file. Entries are never deleted; superseded ones are marked **[superseded by X]**.

- **A-xx** = analysis (what we looked at, what it showed)
- **D-xx** = decision (what we chose, alternatives, why, where it lives in code/config)

Reproduce any number here with `python -m project_a.audit_data --cutoff 1885` unless noted.

---

## 2026-10-27 — Understanding the brief and planning

### A-01 · How lopsided is the cost structure?
- Holding = 25%/yr of price → `0.25/365 = 0.000685 × price` per unit-day. Lost sale = `0.35 × price`.
- **One lost sale ≈ 511 unit-days of holding** (`0.35 / 0.000685`).
- Newsvendor critical ratio at a store with a ~3-day protection interval ≈ `0.35 / (0.35 + 0.002) ≈ 99.4%`.
- PO cost ≤ 8 Mondays × 3 depts × $50 = $1,200 over the window: negligible. Transfer $0.05/unit is ~2% of a $2.50 item: not negligible for cheap items but second-order.
- **Conclusion:** the cost-optimal policy wants near-100% service; the 95% fill-rate floor will rarely bind. What binds are the physical limits (DC outbound cap, 21-day store cap, weekly DC orders, long FOODS_3 lead times).

### A-02 · Supplier lead-time distribution
- Lognormal with CV 0.4 → `σ² = ln(1 + 0.4²) = 0.148`, `σ = 0.385`, `μ = ln(mean) − σ²/2`.
- 95th percentile = `exp(σ·1.645 − σ²/2) × mean ≈ 1.75 × mean` → FOODS_1 8.7d, FOODS_2 12.2d, **FOODS_3 17.5d**.
- With weekly ordering, FOODS_3's DC protection interval at p95 is ~7 + 17.5 ≈ 25 days of network demand.
- Pitfall noted: setting `σ = CV` directly is a common error (gives the wrong spread).

### A-03 · Test window
- d_1 = 2011-01-29 (Sat). Default test window d_1886 = **2016-03-28 (Mon)** → d_1941 = **2016-05-22 (Sun)**, 56 days, 8 DC order Mondays.
- Events inside: Pesach End, Orthodox Easter, Cinco de Mayo, Mother's Day. Easter Sunday is the day before the window.

### D-01 · Framing: optimise cost, treat forecasting as an input
- Grading = total cost + fill rate on *their* cutoff vs a base-stock baseline. Every modelling choice is judged by its effect on simulated cost, not on forecast metrics alone.
- Consequence: we quantify the raw-sales vs recovered-demand difference **in dollars through the simulator**, not only in forecast error.

### D-02 · Everything driven by `--cutoff` and one config file
- No hard-coded d_1886; the test window is `cutoff+1 … min(cutoff+56, 1941)`; capacity window, average sales, initial stock and the first Monday are all relative to the cutoff.
- No tuning on the test window; policy knobs tuned only on backtest cutoffs (default 1801, 1829, 1857).
- All assumptions in `config/project_a.yaml` so any live change in the defence is a one-line edit.

### D-03 … D-08 · Open decisions from PLAN.md §13 — confirmed by Yaswanth 2026-10-27

| ID | Decision | Alternatives considered | Why | Config |
|---|---|---|---|---|
| D-03 | Fill rate = units sold ÷ units demanded, summed over all stores × items × test days (network, volume-weighted). Also report per dept/store and item-weighted. | Item-average fill; per-store minimum | Customer-facing and what a COO means by "95% of demand served"; item-weighted shown because volume-weighting hides slow movers | `fill_rate.definition` |
| D-04 | DC also starts with 7 days of stock (7 × network avg daily sales per item); no pipeline at start | DC starts empty; seed a pipeline | Brief says "each location"; no pipeline is the literal reading and stresses FOODS_3 realistically in week 1–2 | `capacity.initial_stock_days`, `initial_pipeline` |
| D-05 | Holding charged on end-of-day on-hand only, not in-transit | Charge in-transit too | Supplier-owned until receipt is standard; switchable | `costs.hold_in_transit` |
| D-06 | 21-day store cap applies to on-hand + inbound | On-hand only | Prevents the policy "hiding" stock in transit to dodge the cap | `capacity.store_cap_days` |
| D-07 | Re-forecast inside the replay every 28 days using data observed up to then | One 56-day-horizon forecast | Matches the brief's 28-day forecast and real operations; fallback flag kept | `calendar.refit_every` |
| D-08 | Prices after cutoff: carry forward last known price (no future prices in features) | Use actual test-period prices as "planned prices" | Avoids a leakage question in the defence; costs in the simulator still use actual test prices | `prices.use_future_prices` |

### D-09 · Recommended architecture (from PLAN.md, chosen over alternatives listed there)
- Forecast: one global LightGBM mean model + negative-binomial dispersion → correlated sample paths; DC = sum of store paths (coherent by construction). Rejected: per-quantile LightGBM (quantiles don't add to DC), DeepAR/TFT (slow on CPU, hard to change live).
- Policy: forecast-driven two-echelon base-stock with capacity lookahead and marginal-value allocation. Rejected: (s,S) (fixed cost negligible), RL (unexplainable, overfits simulator), decision-focused learning (heavy, little gain given cost asymmetry).

---

## 2026-10-27 — Environment

### D-10 · Where code lives and runs
- Repo root: `~/Inventory-policy-for-a-retail-network` on Yaswanth's Mac; data in `data/` (gitignored).
- Yaswanth chose to install dependencies locally (`.venv`, `requirements.txt` pinned). Data stays on the Mac (not copied to cloud).
- pandas/numpy-only steps (data load, recovery, audit) are run by Claude in the Mac sandbox; LightGBM steps are run by Yaswanth.

### D-11 · Data representation
- Wide arrays `(series × days)`: sales `int16`, price `float32` (NaN = no price row). 5,748 × 1,941 fits in a few MB each.
- Rows sorted item-major, store-minor (the 4 stores of an item are adjacent → DC aggregation is a reshape).
- Cache as compressed `.npz` (no pyarrow needed for the cache). Panel builds in ~7s.
- Code: `project_a/src/data.py`.

---

## 2026-10-27 — Data analysis and demand recovery

### A-04 · Scope check
- 1,437 FOODS items × 4 CA stores = **5,748 series**: FOODS_1 216, FOODS_2 398, FOODS_3 823 items per store. Every series has at least one sale.

### A-05 · Listing / price analysis
- **No sales ever occur without a price row** (0 units) → missing price = not on sale, reliably.
- 23.7% of series-days have no price row.
- **No series loses its price row once it appears** → price gaps only mark pre-launch; **delistings are invisible in prices** (they show as long zero runs with a price present).
- Price row before first sale: median 1 day, p90 6 days, max 1,578 days.

### A-06 · Department economics (last 365 days before cutoff, recorded sales)

| Dept | avg units/series/day | median | zero-sale share | median price | network $ sales/day |
|---|---|---|---|---|---|
| FOODS_1 | 1.62 | 0.93 | 52% | $2.68 | $3,639 |
| FOODS_2 | 1.18 | 0.74 | 56% | $2.98 | $6,643 |
| FOODS_3 | 2.27 | 1.09 | 48% | $2.50 | $18,252 |

- FOODS_3 is ~64% of $ volume and has the longest lead time → it dominates both cost and risk.
- About half of all series-days are zeros → intermittent demand; need count distributions and a detector that adapts to velocity.

### A-07 · Store-level totals
- Christmas closures in all 4 stores every year (d_331, 697, 1062, 1427, 1792); totals 0–7 units.
- **CA_2 structural break, June 2015:** items selling/day ~450 → 650 (Jun 2015) → 900 (May 2016); daily units ~1,300 → ~2,700–3,200. Looks like a remodel/range expansion.
- **CA_2 disruption 19–20 Mar 2016** (1,773 and 1,406 units vs ~3,900 on a normal weekend; other stores normal), one week before the cutoff.
- Figure: `reports/fig_store_totals.png`.

### D-12 · Status taxonomy for every series-day

| Code | Status | Meaning | Training | Demand |
|---|---|---|---|---|
| 0 | ok | observed, uncensored | used | = sales |
| 1 | prelaunch | before first ever sale | dropped | 0 |
| 2 | unlisted | no price row | dropped | 0 |
| 3 | closed | store closed (Christmas / near-zero store day) | dropped | 0 |
| 4 | disrupted | store-wide shortfall | masked | imputed |
| 5 | stockout | zero run implausible for the item's rate | masked | imputed |
| 6 | dormant | zero run ≥ 56 days with price present | dropped | 0 |

- Key principle: **masking, not fabricating** — censored days are removed from training targets; imputed values are used only for lag features, averages and caps.
- Code: `project_a/src/demand_recovery.py`.

### D-13 · Store-day closure / disruption rule
- Compare store total to the median of the same weekday ±1…4 weeks. `< 0.05×` → closed; `< 0.5×` → disrupted. Christmas forced to closed.
- Alternatives: rolling 28-day median (mixes weekdays, false alarms on Mondays); fixed holiday list only (misses CA_2 Mar 2016).
- Result: 3 disrupted store-days (CA_2 2013-03-31 Easter; CA_2 2016-03-19/20) + Christmas.

### D-14 · Stockout detector design (zero-run test)
- For each maximal run of k zeros (counted over available days only, so a Christmas inside a run doesn't break it):
  - local rate λ = mean of de-seasonalised sales over ≤56 clean days before + ≤56 after the run;
  - day-of-week factors per store × dept from the last 2 years of clean days;
  - NB size r per series by method of moments, clipped to [0.5, 50];
  - `log P(run) = Σ r·log(r/(r+μ_t))`, flag if below threshold and ≥14 neighbour days exist;
  - 2 passes: the second recomputes λ excluding runs flagged in the first (they bias λ down).
- Uses only t < cutoff → no leakage; can be re-run "as of" any date.
- Alternatives (PLAN.md §4.2): fixed "N zeros in a row" rule (wrong for both fast and slow movers), 2-state HMM per series (slower, harder to defend live; kept as possible cross-check), cross-store signal (corroboration only).

### A-08 · Validating the detector without ground truth
Two tests, run at three thresholds:
- **Injection (recall):** plant zero runs of length 1/3/7/14 in clean stretches, rerun the detector, measure share of planted days flagged; also imputed ÷ true units removed.
- **Null (false alarms):** simulate stockout-free NB sales per series (same availability, DOW, local level, dispersion), share of days flagged.

| p threshold | stockout share (all days) | hidden demand last 365 (F1/F2/F3) | recall 7-day (bands <0.5 / 0.5–2 / 2–5 / >5 units/day) | false alarms (same bands) |
|---|---|---|---|---|
| 0.01 | 6.9% | 10.1% / 8.2% / 8.8% | 0.36 / 0.90 / 1.00 / 1.00 | **8.8%** / 3.4% / 1.3% / 0.6% |
| **0.001** | 4.5% | 7.4% / 5.4% / 7.2% | 0.11–0.09 / 0.50–0.57 / 0.96–0.97 / 0.99–1.00 | 2.0% / 0.8% / 0.4% / 0.2% |
| 0.0001 | 3.5% | 6.1% / 4.1% / 6.4% | 0.04 / 0.24 / 0.84 / 0.98 | 1.1% / 0.2% / 0.2% / 0.2% |

(Recall ranges at 0.001 are across two runs with 200 and 400 plants per cell.)
- Imputation accuracy on fully detected runs (p = 0.001): imputed/true 0.95–1.05 for runs ≥3 days on items ≥0.5 units/day.
- Single-day stockouts are essentially undetectable except on the fastest movers (21% recall).

### D-15 · Detector threshold p = 0.001 **[supersedes initial p = 0.01]**
- p = 0.01 was the first choice; the null test showed 8.8% false-alarm days on slow movers (multiple testing over hundreds of zero runs per series) → would overstate hidden demand by ~3 points and inflate stock.
- p = 0.001 keeps false alarms ≤2% while still catching ≥92% of 2-week stockouts on items selling ≥0.5/day and ≥96% of week-long ones on items ≥2/day.
- p = 0.0001 loses half the 7-day recall on 2–5/day items for little false-alarm gain.
- Cost logic: missed stockouts on slow movers are cheap (low volume); false alarms on slow movers are also cheap. Fast-mover accuracy is what matters and is good at 0.001.
- **To do:** report total-cost sensitivity to the threshold once the simulator exists.
- Logged in `AI_USAGE.md` as the "AI was wrong" example.

### D-16 · Dormant = zero run ≥ 56 days with price present → treat as not ranged
- 10.0% of series-days (9–12% by dept in the last year); 159 series dormant on the cutoff day.
- Reasoning: a 2-month run of zeros is far more likely off-shelf / discontinued than demand waiting. Treating it as censored would impute large phantom demand and make the policy stock items nobody can buy.
- Policy consequence: no replenishment for a dormant item until it sells again. Risk: genuinely long stockouts get no demand → listed as a risk; **to do:** sensitivity at 28 / 56 / 112 days.

### D-17 · Imputation for censored days
- Stockout days: run-level λ × DOW factor (same λ the test used).
- Disrupted store-days: local clean-rate ±56 days × DOW, and at least the recorded sales (demand ≥ sales).
- Point imputation for features/averages now; the simulator will draw demand on censored test days from the predictive distribution so imputation uncertainty propagates to cost ranges.

### A-09 · Audit results at cutoff d_1885 (p = 0.001)
- All history: ok 61.3%, unlisted 23.7%, dormant 10.0%, stockout 4.5%, prelaunch 0.2%, closed 0.2%, disrupted <0.1%.
- Last 365 days, demand hidden by stockouts: FOODS_1 7.4%, FOODS_2 5.4%, FOODS_3 7.2%; CA_1 6.7%, CA_2 5.7%, **CA_3 8.2%**, CA_4 5.5%.
- On the cutoff day: 5,272 ok, **317 in stockout**, 159 dormant → the policy inherits 317 empty shelves.
- **Test window d_1886–1941:** recorded 665,355 units vs ~702,889 recovered → **5.3% of test demand is hidden**; 4.4% of test series-days are stockouts. A simulator that used recorded sales as demand would flatter every policy.
- Figure: `reports/fig_stockout_examples.png` — detections line up with visible gaps; also shows stretches of 1s between stockouts (partial-day stockouts, not detectable).

### A-10 · DC outbound capacity vs demand
- Cap = 1.25 × avg daily recorded CA FOODS sales over the last 365 days = 1.25 × 10,754 = **13,442 units/day**.
- Recovered demand averages 11,548/day; by weekday Mon 11,218, Tue 10,235, Wed 10,015, Thu 10,069, Fri 11,040, **Sat 13,867, Sun 14,338**.
- Days above cap in the last year: 49 on recorded sales, **73 on recovered demand**.

### D-18 · Capacity lookahead is mandatory in the policy
- Because Sat/Sun demand exceeds the cap, stores must be pre-filled Mon–Fri (bounded by the 21-day cap). A day-by-day policy without lookahead will stock out every weekend.
- Cap is computed on recorded sales as the brief states; noted that this makes it tighter than it looks.

---

## Open items / to revisit
- Cost sensitivity to detector threshold (D-15) and dormant length (D-16) — after simulator.
- CA_3's higher stockout share: real process issue vs more detection power (highest volume).
- CA_2 regime change: how much pre-June-2015 history to use / weight in the forecast.
- Partial-day stockouts: unquantifiable; goes into the memo's "data we'd need".
- Event effects in the test window (Orthodox Easter, Cinco de Mayo, Mother's Day).

---

## 2026-10-28 — Environment change

### D-19 · Model backend: scikit-learn HistGradientBoosting by default, LightGBM optional
- Claude runs the pipeline in its cloud workspace (Yaswanth delegated running everything). PyPI is blocked there, so LightGBM can't be installed; scikit-learn's `HistGradientBoostingRegressor` is available.
- Same algorithm family (histogram GBM, leaf-wise trees, native categoricals) and the same Poisson objective. Config switch `forecast.backend: sklearn | lightgbm`; hyper-parameters map 1:1.
- Default = sklearn so every reported number is reproducible bit-for-bit on the Mac with one command. LightGBM 4.6.0 is installed on the Mac and can be compared later.
- Data copied to the cloud workspace for execution; code and results are synced back to the Mac repo and committed there.
- Row subsample 0.5 of ~3.2M training rows (2-core workspace; ~2–2.5 min per fit).

## 2026-10-28 — Forecast (Task 2), first version

### D-20 · Feature design
- **Lag ≥ 28 everywhere**: one model forecasts every day 1..28 after the origin with no recursion; the same feature value is valid for training rows and forecast rows.
- Features (31): lags 28/35/42/49/56; NaN-aware rolling means 7/28/56/112/364 and std 28 (all ending at t−28); same-weekday mean of 8 weeks; availability share (56d); CV; item-level mean across its 4 stores and this store's share; store×dept trend (28d vs 364d level — for the CA_2 step change); age since first sale; price, price vs 28 days earlier, vs historical max, vs the item's mean across stores; calendar (DOW, month, day of month, SNAP, event type, days to/since an event); dept and store as categoricals. Item id is excluded (1,437 levels; captured through its own lags).
- Lag inputs = recovered demand (imputed on censored days), NaN on no-demand days so rolling stats average only real selling days.
- Training: last 728 target days before the origin, **uncensored (OK) days only**; Poisson loss.
- Forecast = 0 for series dormant at the origin and for items without a price at the origin (not ranged).

### D-21 · Uncertainty model: two-level gamma-Poisson
- `demand = Poisson(mu × G_item × G_day)`. `G_item` (variance s_p²) is shared by the item's 4 stores and persists over the horizon — the forecast-level error. `G_day` (variance s_d²) is independent day noise.
- Both estimated **out of sample**: refit the model 28 days before the origin, forecast the last 28 observed days, match moments — daily cells by predicted-mean band (1/r), 28-day item-network sums by item-volume band (s_p²); s_d² follows from 1/r and s_p².
- Why two levels: with independent daily noise only, DC (sum of 4 stores × many days) intervals would be far too narrow; the shared level shock gives realistic correlation across stores and over time.
- DC quantiles come from summed store sample paths (coherent by construction).
- Rejected: per-quantile models (quantiles don't add to the DC); a single NB per cell (no cross-store/temporal correlation).

### A-11 · Backtest v1 (cutoffs d_1801, d_1829, d_1857; scored on uncensored days)

Store level (bias = actual / forecast; >1 = under-forecast):

| cutoff | recovered bias | raw bias | ma28 bias | recovered pinball | raw pinball | ma28 pinball | recovered 90% cov |
|---|---|---|---|---|---|---|---|
| d_1801 (Jan) | 1.095 | 1.180 | 1.184 | 0.237 | 0.247 | 0.269 | 0.95 |
| d_1829 (Feb) | 1.083 | 1.145 | 1.099 | 0.229 | 0.236 | 0.239 | 0.96 |
| d_1857 (Mar) | 1.036 | 1.096 | — | 0.233 | 0.236 | — | 0.95 |

DC level: recovered bias 1.098 / 1.095 / 1.042, 90% coverage 0.93 / 0.94 / 0.94; ma28 DC 90% coverage only 0.75–0.79.
- **Training on raw sales roughly doubles the under-forecast** (≈ +6–8 points of bias at every cutoff) — the quantified cost of ignoring censoring at forecast level; the $ version comes from the simulator.
- The recovered model is best on every metric, but is still **under-forecasting by 4–10%**. ma28 is also low → demand was rising through Q1 2016 and lag-28 features can't see the last 27 days.
- Store-level calibration curve looks bad at low quantiles only because ~40% of store-days are zero (discrete ties): needs a randomised PIT, not a "share ≤ quantile" curve. DC curve (large counts) is close to the diagonal.
- Saved as `reports/backtest_v1.md`. Next: diagnose bias (dormant items resuming? trend? horizon?) and test a recent-bias correction + yearly lag.

## 2026-10-28 — Simulator (Task 3) and baseline

### D-22 · Simulator design (`project_a/simulate.py`)
- Daily order of events: receive → demand/sales/lost → policy (sees end-of-day state and its **own** sales, never true demand) → simulator enforces all rules → ship → holding cost.
- **The simulator enforces rules itself** (POs only Mondays; store cap on on-hand + inbound; transfers ≤ DC stock; outbound cap) with pro-rata cuts, and counts violations — a policy can't cheat.
- One lognormal lead-time draw **per PO** (a PO is one shipment), arrival `round(L)` days later (≥1); orders may cross.
- **Demand on censored test days:** recorded sales on clean days; NB draw around the imputed rate on stockout/disrupted days (at least the recorded sales), fresh per Monte Carlo run; 0 on unlisted/dormant days. Sensitivity modes: `recorded`, `synthetic`.
- Monte Carlo over lead-time draws × censored-day demand draws; report mean, 5–95% range, P(fill ≥ 95%), ending inventory value.
- Unit tests (`tests/test_simulate.py`): no negative stock, caps never exceeded, sold + lost = demand, costs add up, POs only on Mondays, deterministic per seed, perfect supply → 100% fill, lognormal mean/CV correct. All pass.

### D-23 · "Average daily sales" for the store cap and initial stock = recorded sales over the last 365 training days, on days the item was sellable
- Brief says "sales"; recorded is the literal reading (recovered would give the policy more room and could look like gaming). 365 days matches the only window the brief names (DC capacity). Config `capacity.avg_sales_basis`, `avg_sales_window_days`.
- Consequence: a series with no sales in the last year gets a zero cap and zero initial stock.

### A-12 · Baseline smoke test at cutoff d_1885 (3 runs)
- Total cost ≈ **$136–160k**, fill rate **0.84–0.87** (item-average 0.74–0.80). Lost sales ≈ 70% of cost, transfers ≈ $28k (unavoidable ~ $0.05 × units sold), holding ≈ $12k, POs $1.2k.
- DC outbound capacity cut the baseline's shipments on ~47 of 56 days; DC stock-outs on ~6–8k item-days (no initial pipeline + FOODS_3 lead time).
- **Replaying recorded sales as demand** (ignoring censoring in the test window) would report the same baseline at fill 0.90 and cost ~$110k — i.e. ~20% too optimistic. Confirms D-22's censored-day treatment matters.
- Simulator speed: ~0.2 s per 56-day run → 100 runs in ~20 s.

### A-13 · Why the first forecast under-forecast (diagnosis at d_1829, d_1857)
- Items **dormant at the origin** (forecast 0) produced 1.8% / 2.3% of next-28-day uncensored demand → ~2 points of the bias. In the simulator that is ~$12k of lost-sale cost (~10% of the total).
- Bias grows with horizon week: d_1829 weeks 1–4 = 1.01 / 1.09 / 1.12 / 1.12 → recent upward trend that lag ≥ 28 features cannot see. At d_1857 week 4 (Easter week) = 1.12.
- Out-of-sample calibration-window bias (model refit 28 days earlier) was 1.089 / 1.077 — a usable signal of current under-forecasting.
- Store × dept level correction from that window: shrink 0.5 → bias 1.036 / 0.995 (from 1.083 / 1.036); WAPE +0.4–0.8% worse. Full correction overshoots (0.994 / 0.957).

### D-24 · Post-model adjustments (measured only on the 28 days before the origin)
- **Level:** multiply by `1 + 0.5 × (actual/forecast − 1)` per store × dept from the calibration window, clipped to [0.85, 1.2]. Half-strength because full strength overshoots and trends don't persist fully. Bias matters more than WAPE here: under-forecasting costs ~500× more than over-forecasting.
- **Dormant resume:** dormant-at-origin items (still priced) are forecast at `resume × pre-dormancy daily rate`, where `resume` = share of their pre-dormancy rate that dormant items actually sold in the calibration window. Measured values: 0.11 (d_1801), 0.49 (d_1885) — volatile, but zero is clearly wrong.
- Config `forecast.adjust`; both can be turned off for ablation.

### D-25 · Forecast cache and tuning cutoffs
- Fitted models + forecasts cached by (origin, forecast/recovery/price config, seed) → backtest, tuning and final run share fits.
- **Policy tuning cutoffs = cutoff − 56/84/112** (d_1829, d_1801, d_1773 for d_1885): each 56-day tuning replay ends on or before the cutoff. (The forecast backtest's d_1857 cutoff cannot be used for 56-day replays — it would run into the test window.)

## 2026-10-28 — Recommended policy (Task 4), first runs at d_1885

### A-14 · Recommended policy vs baseline, default knobs (α = β = 0.97, store window 3, pre-build on)
- 2 runs: cost **$80–83k**, fill **0.964–0.968**, zero rule violations; baseline ≈ $136–160k, fill 0.84–0.87.
- Cost mix: transfers $34.5k (≈ fixed: $0.05 × units sold), lost $21–23k, DC holding $16–17k, store holding $7k, POs $1.2k. Ending inventory ~$865k (CFO note: the policy builds stock because purchase cost isn't in the objective — reported, not hidden).
- Days 6–10 dip to 77–88% fill: DC starts with 7 days, no pipeline, FOODS_3 lead ≈ 10 days. Largely forced by the starting rules.
- Outbound at the 13,442 cap on most days → capacity really is the binding constraint.

### A-15 · Policy engineering bugs found and fixed
- Speed: scipy NB quantiles every day made one replay take ~38 s → precomputed quantile tables per (forecast block, window) at 15 probability levels, interpolated; ~1 s per replay. Store targets verified identical to the direct computation.
- **Bug 1:** the capacity allocation capped stores at their quantile level and ignored the pre-build extra → pre-shipping cut exactly when needed. Fixed (allocation level = quantile level + pre-build).
- **Bug 2:** interpolated levels are fractional; flooring 5,748 shipments stranded ~2,000 units/day of outbound capacity (fill 0.933). Fixed with floor + largest-remainder rounding to the capacity budget.
- **Mis-diagnosis (mine):** a remaining "gap" (0.952 vs 0.966) turned out to be comparing different Monte Carlo seeds. Like-for-like seeds reproduce 0.964 / 0.968 exactly. Lesson: always compare policies on common random numbers.

### A-16 · Run-to-run spread is large (12 runs, default knobs, d_1885)
- Fill 0.896–0.974 (mean **0.940**, sd 0.024); cost $76k–$119k (mean $95k). Driven by FOODS_3 (fill 0.869–0.973): **one lead-time draw per PO**, so a single slow FOODS_3 PO (p95 ≈ 17.5 days) starves the DC.
- At default knobs the policy misses 95% on average → DC protection must go up; tuning uses common random numbers and a fill-rate margin on every tuning cutoff.

### A-17 · Where the recommended policy loses sales (d_1885, one bad and one good seed)
- **Bad seed (fill 0.909):** the day-0 FOODS_3 PO took **23 days** (mean 10). With no starting pipeline and 7 days of DC stock, FOODS_3 weekly fill fell to 0.60 / 0.72 in weeks 2–3, then recovered to ~0.97. The day-7 order was automatically larger (100k vs 69k units) because the day-0 PO was overdue — but nothing ordered on day 0 can arrive sooner. **Start-up risk is structural under the stated starting conditions**; it dominates the run-to-run spread.
- **Steady state (days 21–55, good seed):** 2.8% of demand lost. ~15% of that on series whose 21-day cap is 0 (no recorded sales in the last year) — 75 series; ~16% on items dormant at the cutoff; ~30% on fast-growing items whose cap (21 × last-year average) is tight vs. current demand; the rest ordinary shortfall spread over weekdays. Lost share by velocity: 4.0% (<0.5/day), 3.0%, 2.2%, 3.0% (>5/day).
- Raising the DC quantile 0.97 → 0.99 moved fill only 0.939 → 0.942 at +$4k DC holding → order *size* was not the problem.

### D-26 · DC order sized path by path, with open POs that may be late **[supersedes the echelon target in A-14]**
- Old: echelon position (DC + all open POs + stores) vs a quantile of demand to the next arrival. Counted every open PO as if already available.
- New, per sample path: draw each open PO's remaining lead time *conditional on it not having arrived yet* (lognormal truncated at elapsed days), today's order arrival L0, and the next delivery A = earliest of the next 4 weekly orders (handles crossing). Required order `Q_s = max over τ in [L0, A) of [demand to τ + 2 − (stock now + open POs arrived by τ)]`; order the β-quantile of Q_s.
- Effect at default knobs: mean fill unchanged within noise (0.939), range 0.892–0.974 → 0.905–0.973, DC holding −$0.8k. Kept because it is the correct logic for random lead times and it is what makes an announced lead-time shock (Task 5) translate into pre-buying.

### A-18 · Backtest v2 (with D-24 adjustments)

| cutoff | store bias rec. (v1 → v2) | store pinball rec. (v1 → v2) | DC bias rec. | DC 90% cov |
|---|---|---|---|---|
| d_1801 | 1.095 → 1.090 | 0.237 → 0.236 | 1.094 | 0.93 |
| d_1829 | 1.083 → 1.044 | 0.229 → 0.227 | 1.055 | 0.95 |
| d_1857 | 1.036 → 0.999 | 0.233 → 0.230 | 1.005 | 0.94 |

- Better on every cutoff; the January window (d_1801) remains ~9% under (post-holiday rebound the calibration window didn't show).
- By dept (mean of cutoffs): FOODS_1 bias 1.027, FOODS_2 1.030, FOODS_3 1.053; 90% coverage 0.95–0.97.

### D-28 · The raw-sales benchmark must be fully naive (bug in my own comparison)
- In backtest v2 the raw-sales model's level correction and dispersion were calibrated against *recovered* demand, and it inherited the dormancy flags → it quietly received part of the benefit of demand recovery (raw bias gap shrank from ~6–8 to ~2–4 points).
- Fixed: the raw benchmark is calibrated against recorded sales on all sellable days, gets no dormancy zeroing and no resume factor. Raw backtests and the raw-forecast ablation are re-run.

### D-27 · Stress-test interpretation (Task 5)
- "FOODS_3 lead times double for 3 weeks" = POs **placed** in test weeks 3–5 (days 14–34) get mean 20 days (CV 0.4 kept). Orders already in flight are unaffected (variant noted as a limitation).
- Three scenarios on common random numbers: base / unwarned (policy believes 10 days) / warned 14 days ahead (policy plans each future order with the lead time it will face → pre-buys on the Mondays before). Baseline policy is run through the same scenarios for comparison.

### A-19 · First tuning attempt: the quantile knobs barely matter (stopped early)
- On the tuning windows (cutoffs d_1829 / d_1801 / d_1773), default knobs gave fill **0.945 / 0.912 / 0.926** and cost ~$89k. Changing the store quantile for one department moved fill by ≤ 0.002. Stopped after 8 settings (`reports/tuning_round0_partial.log`).

### A-20 · Root cause on the d_1801 window: the DC runs out of hundreds of items every week
- Daily trace: from Thursday to Sunday the DC is short on 300–1,000 items, shipments collapse from the 12,965 cap to 4–6k units/day, lost sales jump to 1–2k units/day. DC total stock is not zero (20–60k units) — it is item-level.
- Outbound capacity used only 82–85% on average → capacity is *not* what binds on this window; DC stock is.
- Cause: the DC rule required DC + store stock to cover cumulative demand to the next delivery — which lets DC **and** stores hit zero together. Stores can't share stock and need their working level to sell; that stock never flows back to the DC.

### D-29 · Echelon base-stock with store working stock (Clark–Scarf logic)
- DC order now requires `demand to τ + (store target levels still in place at τ) − stock − arrived POs ≤ Q`; knob `store_buffer` (share of store targets, default 1.0).
- A/B on 6 common-random-number runs: d_1801 fill **0.912 → 0.932**, cost $96.5k → $93.9k; d_1885 fill **0.945 → 0.949**, cost $90.0k → $95.5k (DC holding +$8.5k, lost −$3.4k).

### A-21 · Longer store cover windows do not help
- Store window 3 → 5 → 7 days: d_1801 fill 0.932 → 0.938 → 0.935, cost $93.9k → $96.3k → $101.4k; d_1885 fill 0.949 → 0.946 → 0.935, cost $95.5k → $102.5k → $110.9k.
- Bigger store targets inflate the DC order (D-29 term) and compete for outbound capacity on peak days; the allocation then spreads capacity across building cover rather than urgent refills. Window 3 kept; window 2–4 left to tuning.

### D-30 · Tuning pass 2
- One coordinate-descent pass over: DC quantile per dept {0.95, 0.97, 0.99, 0.995}, store quantile per dept {0.95, 0.97, 0.99}, store_buffer {0.5, 1, 1.5}, store window {2, 3, 4}; 6 CRN runs × 3 tuning cutoffs per setting; objective = mean cost + $20k per fill point below 95.5% on any tuning cutoff.

### A-22 · Tuning results (pass 1 full grid, pass 2 at the grid edges; `reports/tuning_pass1.log`, `tuning_pass2.log`, `tuning_1885.json`)
Mean over tuning cutoffs d_1829 / d_1801 / d_1773, 6 CRN runs each:

| setting | cost | fill (three windows) |
|---|---|---|
| defaults (α 0.97, β 0.97, window 3, buffer 1.0) | $90.2k | 0.950 / 0.932 / 0.939 |
| + β FOODS_1 0.995, FOODS_3 0.995 | $93.6k | 0.952 / 0.936 / 0.941 |
| + α 0.99 all depts | $94.1k | 0.955 / 0.941 / 0.946 |
| + store buffer 1.5 | $98.8k | 0.956 / 0.943 / 0.948 |
| + store window 4 | $102.0k | 0.957 / 0.945 / 0.952 |
| pass 2: α FOODS_1/3 0.995, β FOODS_1 0.99 (**chosen**) | **$103.1k** | **0.958 / 0.946 / 0.953** |
| pass 2: buffer 2.0 / 2.5 | $110.2k / $117.6k | 0.958 / 0.947 / 0.953 → 0.959 / 0.948 / 0.954 |
| pass 2: window 5 | $108.3k | 0.956 / 0.946 / 0.954 |

- Fill plateaus at ≈ 95.8 / 94.7 / 95.3%: beyond the chosen point each extra 0.1 fill point costs $3–7k. The remaining shortfall is the start-up (no pipeline, 7 days of stock) and the cap rules (A-17), which no knob can fix.
- The d_1801 window (Jan–Feb) stays below 95%: it is also where the forecast under-forecasts most (bias 1.09).

### D-31 · Final knobs
α (store) = 0.995 / 0.99 / 0.995, β (DC) = 0.99 / 0.97 / 0.995 for FOODS_1/2/3, store window 4 days, store buffer 1.5, pre-build on. Saved as config defaults; the full run re-tunes from them on windows before whatever cutoff it is given (one pass over a grid around them), so other cutoffs are never tuned on their own test period. `--fast` / `--skip-tune` use these defaults as-is (tuned for d_1885).

## 2026-10-28 — Final run at d_1885 (`python -m project_a.run_project_a --cutoff 1885 --skip-tune`, 63 min on 2 cores)

### A-23 · Results (100 Monte Carlo runs per policy; `reports/results_1885.md`)

| policy | cost | 5–95% | fill | 5–95% | P(≥95%) | lost | holding | transfer | ending inv. |
|---|---|---|---|---|---|---|---|---|---|
| baseline | $141.8k | $119–178k | 85.7% | 79.4–89.7% | 0.00 | $101.9k | $10.7k | $28.1k | $0.44M |
| **recommended** | **$114.1k** | $95–150k | **95.2%** | 89.5–98.0% | **0.63** | $29.3k | $47.6k | $35.9k | $1.50M |
| no weekend pre-build | $113.8k | | 95.3% | | 0.64 | | | | |
| no forecast adjustments | $118.8k | | 94.5% | | 0.58 | $35.3k | | | |
| raw-sales forecast | $118.7k | | 94.1% | | 0.50 | $35.8k | | | |
| recorded test sales as demand (sensitivity) | $105.8k | | 96.4% | | 0.80 | | | | |
| baseline, recorded test sales | $113.2k | | 89.2% | | 0.00 | | | | |

- **−$27.8k (−20%) and +9.6 fill points vs the baseline**; zero rule violations (baseline: ~16k cut requests per run).
- **Value of demand recovery in $:** the same policy driven by a raw-sales forecast costs +$4.7k and −1.1 fill points; P(≥95%) 0.63 → 0.50. Forecast corrections (D-24) are worth +$4.7k / +0.7 points.
- **Censoring in evaluation:** replaying recorded test sales overstates fill by 1.2 (recommended) to 3.5 points (baseline) and understates cost by $8–29k.
- **Pre-build is redundant** at the tuned settings (4-day store window + 99%+ store levels already carry weekend cover). Kept on as a safeguard; reported honestly.
- Fill by dept: 97.3% / 98.0% / 94.2% — FOODS_3 (long lead time) is the weak spot.
- Forecast backtest with the fully naive raw benchmark (D-28): raw store bias 1.128 / 1.084 / 1.057 vs recovered 1.090 / 1.044 / 0.999.

### A-24 · Inventory build-up and the lean variant
- DC cover jumps to ~48 days in week 2 and drains to ~29 by week 8; stores 5 → 9 days. Ending inventory $1.50M (≈ 44 days of sales at retail value) vs baseline $0.44M. Cause: with no pipeline at the start, early orders insure against a late first PO (500:1 cost ratio + 95% floor).
- **Lean variant** (α = β = 0.97, window 3, buffer 1.0), 40 runs on the same seeds: $95.6k, fill 94.8%, P(≥95%) 0.625, ending inv. $1.00M. **Recommended on the same 40 seeds: $115.2k, fill 95.0%, P 0.625.** On this window the tuned insurance buys only +0.2 points for ~$19.6k; on the three past windows it bought +0.8–1.4 points (A-22).
- Decision: keep the tuned knobs (chosen without seeing the test window; meets 95% on 2 of 3 past windows vs 1 of 3 for lean) and give the COO the lean option and its trade-off explicitly in the memo. Re-tuning on the test window would be peeking.

### A-25 · Stress test (FOODS_3 lead ×2 for POs placed days 14–34)

| policy | base | unwarned | warned (14 days) |
|---|---|---|---|
| recommended (100 runs) | $114.1k / 95.2% | $113.1k / 95.1% | $116.3k / 95.2% |
| lean (40 runs) | $95.6k / 94.8% | $97.4k / 94.2% | $98.3k / 94.7% |
| baseline (20 runs) | $138.7k / 86.2% | $171.6k / 79.8% | same as unwarned (can't use a warning) |

- The recommended policy's 30–40 days of DC cover absorb a 10-day slip; extra warned pre-buying adds $2.2k for nothing. The warning is worth most to a lean policy (recovers 0.5 of 0.6 lost points for +$0.9k). The baseline has no buffer: −6.4 points, +$32.9k.

### D-32 · Deliverable format
- `forecasts.parquet` is written via pandas/pyarrow. The analysis environment had no parquet engine, so this run wrote `forecasts.csv.gz` (same columns); the Parquet file is produced by running on the Mac (pyarrow in `.venv`).
- Memo: `memo/memo_A.md` → `memo_A.pdf` (pandoc + wkhtmltopdf, `python memo/build_memo.py`), 2 pages.

---

# Project B — geo MMM with experiment calibration

Chronological log of every decision and the analysis behind it. `DECISIONS.md` (≤ 3 pages) is
distilled from this file. **A-xx** = analysis, **D-xx** = decision. IDs prefixed B to keep them
distinct from Project A's log.

---

## 2026-10-28 — Brief, data and plan

### B-A01 · What is asked
CMO/CFO decision: true return per paid channel (average and marginal ROI with 90% intervals) and
an 18M allocation for the next 13 weeks with every channel within ±30% of its last-13-week
spend; allocations at 12M and 24M; experiment calibration; next experiment; memo. Pipeline must
take `--holdout-start`.

### B-A02 · Data facts (`python -m project_b.audit` → `reports/project_b/data_audit.md`)
- 40 geos × 156 weeks (2021-01-25 → 2024-01-15, Mondays), no missing values.
- **Cost per impression is constant within every channel** (CV ≈ 3e-8): impressions = spend ÷ fixed CPM.
- Zero-spend geo-weeks: Ch0 12%, Ch1 28%, **Ch2 64%**, Ch3 3%, Ch4 13%.
- Last 13 weeks spend: 3.80 / 3.20 / 1.44 / 7.95 / 4.37 M = **20.77M** → ±30% range **14.54M–26.99M**. 18M = −13% (feasible), 24M = +16% (feasible), **12M infeasible**.
- Spend per capita over 3 years nearly identical across geos (CV 0.04–0.13) **but 65–79% of weekly per-capita spend variance is geo × week** (common week 20–35%, geo < 1%).
- Spend is seasonal (Q4 ≈ 1.4–1.6× Q1); weekly revenue flat (8.0–8.8M).
- Revenue per conversion ≈ 0.0200 (sd 0.0001); conversions ≈ 19 per capita per week (synthetic units).

### B-A03 · The Channel3 experiment is not visible in the data
- Reported: Channel3 switched off in Geo0–7 for 8 weeks from 2023-06-05; vendor iROAS 0.8 (90% CI 0.3–1.3).
- Data: **2 of 64** test geo-weeks have zero Channel3 spend. Test-geo spend per 1,000 people: 19.5 in the window vs 17.9 in the 8 weeks before. Test ÷ control spend per capita: 0.88 before, **0.94 during**, 1.00 in all other weeks. Difference-in-differences of spend per capita ≈ +0.001 (no drop).
- Revenue per capita in test geos fell ~4% relative to control in the window (DiD −0.016 on ~0.39) — with no spend change, that cannot be caused by Channel3 being switched off.
- Figure: `reports/project_b/fig_experiment_check.png`.

### B-D01 · Plan decisions confirmed by Yaswanth (2026-10-28, "go with the defaults")
1. Hand-written sampler in numpy/scipy (no PPL can be installed where the analysis runs), validated by parameter recovery.
2. Train strictly on weeks before `--holdout-start`; every reported number comes from that fit.
3. Vendor result treated as unverified; budget built on the uncalibrated model; calibrated variants as sensitivity.
4. Model conversions per capita; report revenue (× revenue per conversion).
5. Optimise channel totals; each channel keeps its last-13-week geo × week pattern.
6. 12M: report the ±30% floor and a bounds-relaxed optimum, clearly labelled.
7. Separate repo for Project B (Yaswanth's choice; the brief asks for one repo, so the two will be merged before submission).

### B-D02 · Priors start from Meridian's published defaults (validated against Meridian docs, PLAN §15)
ROI ~ LogNormal(0.2, 0.9); Hill slope 1; half-saturation ~ TruncNormal(0.8, 0.8, [0.1, 10]) × median non-zero spend per capita; adstock decay ~ Uniform(0, 1), geometric, 8-week max lag; geo intercepts ~ N(0, 5). Departures: geo spread of media effects is additive-relative `b[m,g] ~ N(β_m, (η_m β_m)²)`, η ~ HalfNormal(0.5) (Meridian: log-normal geo effects) — keeps the coefficients conditionally Gaussian so they can be sampled exactly; baseline/controls/promo ~ N(0, 1) on the standardised scale (Meridian N(0, 5) is effectively flat); σ_g ~ HalfNormal(1).

### B-D03 · Baseline: extrapolable trend + seasonality (main), weekly fixed effects (robustness)
Meridian's geo default (one effect per week) cannot forecast an end-of-window hold-out; Meridian's docs advise against such hold-outs and say it isn't designed for KPI forecasting. The brief requires one. Main model: linear-spline trend with knots every 8 weeks (held flat beyond the training data) + 3 annual Fourier harmonics; a weekly-fixed-effects variant is fitted for ROI comparison only.

## 2026-10-28 — Engine

### B-D04 · Partially collapsed Gibbs sampler (`project_b/src/model.py`, `linear.py`)
- Given the 12 curve parameters (adstock decay and half-saturation for 5 paid + organic), every other coefficient is linear with Gaussian priors → integrated out **exactly** for the Metropolis steps on curve parameters, ROI and geo spread, then drawn exactly from its Gaussian conditional; σ_g by Metropolis.
- The posterior precision is arrowhead-shaped (40 small per-geo blocks + one shared block) → solved via the Schur complement; ~5 ms per marginal likelihood, ~80 ms per sweep.
- The lift test enters as one extra Gaussian observation (rank-one update), so calibration uses the same exact machinery.
- ROI parameterisation: β_m = ROI_m × c_m(α, ec), c_m = training spend ÷ training revenue per unit effect — the prior is on the number a CMO understands.

### B-A04 · The linear algebra matches brute force (`tests/test_linear.py`)
Posterior mean, marginal likelihood and draws (40k) match dense-matrix computations, with and without the extra lift-test observation (rtol 1e-8; sample covariance within 0.02).

### B-A05 · Parameter recovery (`python -m project_b.recovery`, `reports/project_b/recovery.json`)
Simulated KPI from the model with known parameters using the real spend/organic/promo/control series; 2 replicates × 2 chains × 600 draws.

| channel | true ROI | rep 1 posterior [90% CI] | rep 2 posterior [90% CI] |
|---|---|---|---|
| Channel0 | 0.90 | 0.80 [0.42, 1.36] | 1.47 [0.88, 2.15] |
| Channel1 | 1.60 | 1.77 [1.49, 2.14] | 1.71 [1.40, 2.00] |
| Channel2 | 0.60 | 0.81 [0.35, 1.27] | 0.94 [0.44, 1.49] |
| Channel3 | 2.20 | 1.91 [1.63, 2.26] | 2.35 [1.98, 2.82] |
| Channel4 | 1.30 | 1.26 [0.99, 1.60] | 1.00 [0.80, 1.22] ✗ |

- **Coverage 9/10 = 90%**, as a calibrated sampler should give. Adstock decay recovered well; half-saturation weakly identified (trades off against ROI) → slow mixing (R-hat up to 1.06, ESS ~33 with short chains). Real fits use 4 chains × 1,500 warm-up × 1,000 draws.
- Channel2 (64% zero weeks) and Channel0 have the widest intervals — expected to be the least certain on real data too.

## 2026-10-28 — First real fits

### B-A06 · Prior-predictive check
Before seeing the KPI, the priors imply paid + organic media drive a median **22%** of conversions (90% range 12–44%; P(> 50%) = 0.03) — plausible, and close to Meridian's own fallback assumption (40% ± 20%). Simulated national KPI per capita spans 1.7–52.6 (observed 15.7–22.6): wide but covering.

### B-A07 · First geo fit had implausible negative geo effects → controls must vary by geo
- First full fit (shared control coefficients; 4 × 1,000 draws; R-hat ≤ 1.03): ROI 1.01 / 1.21 / 1.84 / 1.06 / 0.78, but the geo spread of media effects came out at **1.3–1.6× the channel mean** for Ch0/2/3/4, and **11–18 of 40 geos had negative mean effects** (e.g. Channel4 lowering sales in 18 geos). Not credible.
- Test 1 — tighter pooling (η ~ HalfNormal(0.15)): the data pushed η for Ch2/3/4 back to ~0.8; 9–11 negative geos remained; hold-out unchanged → the heterogeneity is structural, not a prior problem.
- Test 2 — **control effects vary by geo** (Meridian's γ_gc; competitor sales is correlated 0.39–0.48 with Ch2/3/4 spend): η fell to 0.33–0.49, **0 negative geos**, residual sd 0.857 → 0.787, geo hold-out WAPE 0.229 → 0.220, in-sample geo WAPE 0.221 → 0.205. National hold-out WAPE slightly worse (0.060 → 0.067).
- Interpretation: competitor sales and sentiment act differently by geo; forcing one coefficient pushed that geo-specific confounding into the media coefficients.

### B-D05 · Controls get geo-specific coefficients (main specification) **[supersedes shared controls]**
Per-geo control coefficients ~ N(0, 1) (Meridian uses geo-varying control effects too). ROIs moved materially (e.g. Ch2 1.84 → 1.28, Ch4 0.78 → 1.89 in a short test fit), so this is one of the most consequential choices; it is logged with both results.

### B-D06 · Budget optimiser: greedy water-filling instead of SLSQP
SLSQP failed on the piecewise-linear interpolated response curves. The curves are concave (slope-1 Hill after adstock), so allocating small budget slices (0.25% of a channel's last-13-week spend) to the channel with the highest marginal revenue per unit of spend is exact; at the optimum, marginal ROIs are equal across channels not at a ±30% bound.

### B-D07 · Sampler settings for the deliverable run
4 chains × 1,000 warm-up × 600 draws (the first fit with 1,500 × 1,000 reached R-hat ≤ 1.03, ESS ≥ 109; shorter chains keep the full 7-model pipeline within the 2-hour budget on a laptop). Convergence is reported for every variant.

### B-A08 · Mixing problem after adding geo controls, and the fix
- With geo-specific controls, the one-parameter-at-a-time Metropolis moves mixed poorly: worst R-hat **1.078**, min ESS **30** (4 × 1,000 + 600).
- Cause: each channel's adstock decay, half-saturation and ROI are strongly correlated in the posterior (a slower decay or later saturation can be traded for a different ROI).
- Fix (B-D08): one **joint 3-D adaptive Metropolis** block per channel (Haario et al.: the proposal covariance is learned from the chain's own warm-up history; adaptation stops after warm-up so the chain is valid). Same budget: worst R-hat **1.042**, min ESS **130**; all ROI parameters ≤ 1.03. Remaining R-hat > 1.01 is on half-saturation (`ec`), a weakly identified nuisance.

### B-D08 · Sampler: adaptive joint block per channel; 4 chains × 1,000 warm-up × 800 draws

## 2026-10-29 — First full run (85 min on 2 cores) and fixes

### B-A09 · Results of the first complete run (superseded numerically by B-A11 after the fixes below; fits identical in spec)
- Average ROI (last 52 training weeks): Ch0 1.30 [0.29, 2.97], Ch1 1.03 [0.31, 2.27], Ch2 1.20 [0.46, 2.21], Ch3 0.94 [0.38, 1.71], Ch4 1.62 [0.36, 3.35].
- **Marginal ROI below 1 for every channel** (0.51–0.77): at current spend the next unit returns less than it costs → the 13% cut to 18M is directionally right; the question is where to cut.
- 18M plan: Ch1 −30%, Ch3 −30%, Ch2 −15%, Ch0 −5%, Ch4 +23%; uplift vs today's mix +316k [−1.09M, +1.76M], P(> 0) = 0.63 — weak evidence; report honestly.
- Hold-out: national WAPE 6.3%, bias −6.2% (under-forecast of the Q4-2023 level; trend held flat), 90% coverage 0.85; geo WAPE 21.9%.
- Calibration barely matters: uncalibrated model already implies **1.10 [0.38, 2.01]** for the vendor's test (vendor 0.8 [0.3, 1.3]); full trust moves 18M allocation by < 0.4M in any channel.
- Without sentiment, Channel3 ROI collapses to 0.27 — sentiment is an important confounder (but that fit did not converge, R-hat 1.18).

### B-A10 · Problems found in the first run's outputs, and fixes
1. **12M "floor" uplift was wrong** (+2.27M, P = 1.00): it compared spending 14.54M with today's mix scaled to 12M. Fixed: uplift always compares at equal total spend (at the floor it is today's mix by construction → reported as "—").
2. **EVPPI tie** (Ch0 and Ch4 both 316k, equal to the 18M uplift): the per-bin allocation used the coarse 5% grid while the baseline used the fine optimiser, so coarse-grid losses leaked into the comparison. Fixed: the fine optimiser everywhere.
3. **No design reached useful power** (≤ 0.18; minimum detectable iROAS 5–20). Geo-week revenue noise is large (residual ≈ 26% of a geo-week's revenue), so 8-week tests on 8–16 geos cannot see a channel whose ROI is ~1. Grid widened (10 or 20 treated geos; 8/12/16 weeks; switch-off / double / triple spend) and pre-period lengthened 8 → 26 weeks (a longer baseline averages out noise).
4. Weekly-fixed-effects fit took 56 min: the 145×145 shared-block cross-products were recomputed on every likelihood call though they only change with σ. Cached (`linear.shared_terms`): 13 ms per call instead of tens.
5. No-sentiment sensitivity did not converge: robustness fits now 2 × 800 + 600.
6. Added `--reuse-fits` (regenerate all outputs from cached posteriors without refitting).

## 2026-10-29 — Final run (18 min with cached main fits; 22 min from scratch on 2 cores)

### B-A11 · Final results (`reports/project_b/results.md`, hold-out 2023-10-23 → 2024-01-15)
| channel | ROI (52w) | marginal ROI | P(ROI > 1) | 18M plan |
|---|---|---|---|---|
| Channel0 | 1.30 [0.29, 2.97] | 0.64 [0.12, 1.46] | 0.55 | 3.61M (−5%) |
| Channel1 | 1.03 [0.31, 2.27] | 0.54 [0.14, 1.17] | 0.40 | 2.24M (−30%) |
| Channel2 | 1.20 [0.46, 2.21] | 0.71 [0.20, 1.41] | 0.55 | 1.23M (−15%) |
| Channel3 | 0.94 [0.38, 1.71] | 0.51 [0.20, 0.95] | 0.36 | 5.56M (−30%) |
| Channel4 | 1.62 [0.36, 3.35] | 0.77 [0.16, 1.47] | 0.73 | 5.36M (+23%) |
- 18M: incremental revenue 23.03M [14.57, 33.05]; uplift vs today's mix scaled to 18M **+316k [−1.09M, +1.76M]**, P(> 0) 0.63. 24M: 26.77M [17.38, 38.41] (extra 6M buys ~3.7M). 12M relaxed: 18.12M; ±30% floor (14.54M): 20.03M.
- Hold-out: national WAPE 6.3%, bias −6.2%, 90% coverage 0.85, R² −1.46 (weekly variance small; level under-forecast in Dec–Jan); geo WAPE 21.9%, R² 0.73, coverage 0.86.
- Variants agree on direction: weekly fixed effects and no-organic within ~0.3 of the main ROIs; national centres similar with wider intervals for Ch2/Ch3; without sentiment Channel3 drops to 0.26 (that fit mixes poorly, R-hat 1.145 — indicative only).
- Experiment: model-implied iROAS for the vendor's test 1.10 [0.38, 2.01] (uncalibrated), 0.96 (discounted), 0.86 [0.49, 1.26] (full). Calibrated allocations differ from the main plan by ≤ 0.38M per channel.
- Value of information at 18M: EVPI 895k; EVPPI Ch0 339k ≈ Ch4 337k > Ch3 216k > Ch1 180k > Ch2 127k.
- Power: best design (Channel0 switched off in 20 geos for 16 weeks, 26-week pre-period) **32%**, minimum detectable iROAS 2.7, ~2M spend forgone. No design in the grid reaches 80%.
- Prior-predictive media share with the final spec: median 0.24 [0.13, 0.59].
- Convergence: main R-hat ≤ 1.039 (ESS ≥ 143); calibrated ≤ 1.042; national ≤ 1.036; weekly FE ≤ 1.058; no-organic ≤ 1.034; no-sentiment 1.145.

### B-D09 · Experiment recommendation: no small test
The most valuable uncertainty (Ch0/Ch4, ~0.34M each) cannot be resolved by a feasible geo test: weekly geo revenue noise is ~26% of a geo-week's revenue. A 16-week, 20-geo switch-off detects the effect ~1 time in 3 and forgoes ~2M spend (≈ 2.6M revenue at the estimated ROI) — more than the information is worth. Recommend gradual, monitored reallocation and, if testing, ≥ 20 geos × ≥ 16 weeks with weekly execution checks (the Channel3 test's failure mode).

### B-A12 · Recovery test re-run with the final specification and sampler
2 replicates: **10/10** true ROIs inside their 90% intervals (posterior means 0.61–2.25 vs truths 0.6–2.2). Across both validation rounds 19/20 = 95%.

### B-A13 · `--holdout-start` works on another window
`--fast --holdout-start 2023-07-03` (train to 2023-06-26, 127 weeks; hold-out Jul–Sep 2023): national WAPE 3.2%, bias −1.6%, 90% coverage 1.00, R² 0.59; geo WAPE 21.5%. Same qualitative ranking (Ch4 highest marginal ROI 1.05, Ch3 lowest 0.46). Runs in 3.5 min in fast mode.
