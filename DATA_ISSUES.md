# DATA_ISSUES.md

Each problem found in the data, how it was detected, and how it is handled. Project A first, then Project B.

# Project A — CA FOODS, cutoff d_1885

Scope: 1,437 FOODS items x 4 CA stores = 5,748 series, d_1 (2011-01-29) .. d_1941 (2016-05-22).
Numbers below are reproduced by `python -m project_a.audit_data --cutoff 1885` (full tables in `reports/data_audit.md`).
Every series-day is given one status (`project_a/src/demand_recovery.py`); censored days are masked from
training targets, no-demand days are dropped.

| # | Issue | How found | Size | Treatment |
|---|---|---|---|---|
| 1 | **Recorded sales are censored by stockouts.** Zero runs that are too long for the item's own selling rate. | Zero-run test: P(k zeros \| local NB rate, DOW factors) < 0.001 | 4.5% of all series-days; last 365 days hides **5.4–7.4% of demand** by dept (FOODS_1 7.4%, FOODS_2 5.4%, FOODS_3 7.2%); CA_3 worst (8.2%). 317 series are out of stock on the cutoff day. | Masked from training targets; imputed for lags, averages and caps. |
| 2 | **Test-period sales are censored too.** | Same detector run on d_1886–1941 | 4.4% of test series-days; recorded 665k units vs ~703k recovered (**5.3% hidden**). | Simulator uses recorded sales on clean days and draws demand on censored days (see DECISIONS). |
| 3 | **"Not listed" is only visible as a missing price row.** | `sell_prices` has no row for that item/store/week | 23.7% of series-days. Never any sales without a price row. No series ever loses its price row once it appears, so price gaps only mark pre-launch. | Dropped (no demand). |
| 4 | **Delisting / long non-replenishment is NOT marked by price.** Price rows continue through months of zeros. | Zero runs >= 56 days with price present | 10.0% of series-days (9–12% by dept in the last year); 159 series are dormant at cutoff. | Status DORMANT = not ranged: no demand, not in training, not replenished until it sells again. A judgment call: an item with no sales for 2+ months is far more likely off-shelf than out of stock with demand waiting. |
| 5 | **Listed before first sale.** Price row exists but item never sold yet. | First sale vs first price week | Median 1 day, p90 6 days, max 1,578 days. | Days before first sale = PRELAUNCH (no demand). |
| 6 | **Store closures.** | Store total ~0 vs same-weekday norm | Christmas every year (d_331, 697, 1062, 1427, 1792), all 4 stores. | CLOSED (no demand). Not relevant inside the test window. |
| 7 | **Store-wide disruption right before cutoff.** CA_2 sold 1,773 and 1,406 units on 19–20 Mar 2016 vs ~3,900 on a normal weekend (other stores normal). Also 31 Mar 2013 (Easter). | Store total < 0.5 x median of +-4 same weekdays | 3 store-days, 1 week before cutoff. | DISRUPTED = censored; imputed from local rate. Matters because short lags/rolling means would otherwise drag CA_2's forecast down. |
| 8 | **CA_2 structural break, June 2015.** Items selling per day jump ~450 -> 650-900; volume ~1,300 -> ~2,700 units/day and still rising into 2016. | Monthly store totals (`reports/fig_store_totals.png`) | Whole store. | Looks like a remodel/range expansion. Pre-break history under-represents CA_2: forecast uses store-level recent-level features and recent-weighted training; flagged as a risk. |
| 9 | **Partial-day stockouts are undetectable.** A shelf that empties mid-afternoon records low-but-positive sales. Visible in examples as stretches of 1s between full stockouts. | Visual (`reports/fig_stockout_examples.png`) | Unknown; likely biases recovered demand down on fast movers / peak days. | Not imputed. Listed under "data we'd need" (on-hand / intraday POS). Policy safety levels absorb some of it. |
| 10 | **Detector blind spots.** Single-day and short stockouts on slow movers look like normal zeros. | Injection test (planted stockouts) | Recall for 7-day stockouts: 11% (<0.5 units/day), 50% (0.5–2), 96% (2–5), 100% (>5). 14-day: 35% / 92% / 100% / 100%. 3-day on fast movers: 86%. | Accepted: missed stockouts on slow movers are cheap in $ (low volume). Hidden-demand estimates are therefore a lower bound. |
| 11 | **Detector false alarms.** | Null test: simulate stockout-free NB sales per series, run detector | Flagged days: 2.0% (<0.5/day), 0.8% (0.5–2), 0.4% (2–5), 0.2% (>5). At the first threshold tried (p<0.01) slow movers were 8.8% — see AI_USAGE.md. | Threshold p<0.001 chosen as the balance; sensitivity of total cost to the threshold reported in the simulator. |
| 12 | **DC outbound cap binds on weekends.** Cap = 1.25 x 10,754 = 13,442 units/day is set on *recorded* sales. Recovered network demand averages 11,548/day; Sat 13,867, Sun 14,338. | Capacity check in audit | 73 of the last 365 days exceed the cap on recovered demand (49 on recorded). | Policy must pre-position store stock on weekdays (capacity lookahead). Using recorded sales for the cap is per the brief; it means the cap is tighter than it looks. |
| 13 | **Strong weekly seasonality.** Weekend network demand ~1.4x mid-week. | DOW means | — | DOW factors in detector, DOW features in forecast, weekday pre-shipping in policy. |
| 14 | **Prices exist for the test window.** | sell_prices through wk 11621 | — | Not used by default (decision: carry forward last known price; flag `prices.use_future_prices`). Prices used for costs in the simulator are the actual test-week prices. |

Items to check later as the model is built: price-change / promotion effects, event effects (test window has
Pesach End, Orthodox Easter, Cinco de Mayo, Mother's Day), and whether CA_3's high stockout share is a
process issue or detector bias (it has the highest volume, so more power to detect).

# Project B — Meridian geo sample data

Reproduced by `python -m project_b.audit` (`reports/project_b/data_audit.md`, `reports/project_b/fig_experiment_check.png`).

| # | Issue | How found | Size | Handling |
|---|---|---|---|---|
| 1 | **The Channel3 geo-lift test is not visible in the spend data.** The test supposedly switched Channel3 off in Geo0–Geo7 for 8 weeks from 2023-06-05. | Weekly Channel3 spend per capita, test vs control geos, around the window | Only **2 of 64** test geo-weeks have zero spend; test ÷ control spend per capita went from 0.88 (8 weeks before) to **0.94 during** (1.00 in all other weeks); spend diff-in-diff ≈ +0.001 per capita | The vendor's iROAS (0.8, 90% CI 0.3–1.3) cannot be verified from this data → main model is **uncalibrated**; calibrated versions (full and ×3-discounted uncertainty) are reported as sensitivities; the memo recommends re-running the test with execution checks |
| 2 | Test-geo revenue fell ~4% relative to control during the window although spend did not change | Revenue per capita diff-in-diff | −0.016 on ~0.39 per capita per week | Not attributable to Channel3; noted as an unexplained shift (or part of whatever the vendor measured) |
| 3 | **Impressions carry no information beyond spend** | spend ÷ impressions per row | cost per impression constant within each channel (CV ≈ 3e-8) | Use spend only (needed for ROI anyway) |
| 4 | Channel2 is off most of the time | zero-spend share | 64% of geo-weeks (Ch1 28%, Ch0 12%, Ch4 13%, Ch3 3%) | Kept; expect wide intervals; Ch2's saturation curve is least identified |
| 5 | Spend per capita over 3 years is near-identical across geos | CV of per-capita totals | 0.04–0.13 | Geo *levels* say little; identification comes from geo × week variation (item 6); strong pooling of geo media effects |
| 6 | Most spend variation is geo × week | two-way variance decomposition of weekly spend per capita | 65–79% geo × week, 20–35% common week, < 1% geo | This is what separates channels; national totals discard it (geo vs national comparison) |
| 7 | Spend is seasonal while revenue is flat | quarterly sums | Q4 spend ≈ 1.4–1.6× Q1; weekly revenue 8.0–8.8M | Trend + annual seasonality in the baseline; weekly fixed effects as a robustness check |
| 8 | Controls are correlated with media and act differently by geo | correlations; first fit with shared control effects gave 11–18 geos with *negative* media effects | competitor sales vs Ch2/3/4: 0.39–0.48 | Control coefficients vary by geo (DECISION_LOG B-A07) |
| 9 | Population differs 7× across geos | population column (constant per geo) | 137k–994k | Model per capita; revenue scaled back by population |
| 10 | KPI scale is synthetic | conversions per capita | ≈ 19 conversions per person per week | Model per capita and standardised; report revenue = conversions × revenue per conversion (≈ 0.0200, sd 0.0001) |
| 11 | **12M budget is infeasible under the ±30% rule** | last-13-week spend 20.77M | ±30% allows 14.54M–26.99M | Report the ±30% floor (14.54M) and a bounds-relaxed 12M allocation, clearly labelled |
| 12 | Hold-out at the end of the series vs time effects | Meridian guidance: per-week time effects can't be estimated for held-out final weeks | — | Extrapolable baseline (trend held flat beyond training + Fourier seasonality) |
