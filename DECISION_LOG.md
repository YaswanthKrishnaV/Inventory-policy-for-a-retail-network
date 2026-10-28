# DECISION_LOG.md — Project A

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
