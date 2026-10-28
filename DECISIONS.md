# DECISIONS.md — Project A

The major choices, the alternatives, and why. Every entry points to the full evidence in
`DECISION_LOG.md` (IDs in brackets). Numbers at cutoff d_1885 unless stated.

## 1. Framing: optimise cost, use forecasts only as inputs [A-01, D-01]
One lost sale (35% of price) costs as much as holding the unit for **511 days** (25%/yr). The
cost-optimal policy therefore wants near-100% service; the 95% floor rarely binds. What binds are
the physical limits: the DC outbound cap (1.25 × average daily sales, set on *recorded* sales,
below recovered weekend demand), the 21-day store cap, weekly DC ordering, and FOODS_3 lead
times (mean 10, p95 17.5 days). Every modelling choice was judged by simulated cost, not by
forecast metrics alone. *Rejected:* tuning the forecast for accuracy first and the policy after.

## 2. Demand recovery: classify every series-day, mask rather than fabricate [D-12…D-17]
Seven statuses: ok / pre-launch / unlisted (no price row) / closed / store-disrupted / stockout /
dormant. Censored days (stockout, disrupted) are **masked from training targets**; imputed values
are only used for lags, averages and in the simulator. Stockouts are found with a zero-run test:
P(k zeros | local NB rate × weekday factors) < 0.001, two passes.
- *Threshold:* p < 0.01 was my first choice; a null test (stockout-free simulated sales) showed
  8.8% false-alarm days on slow movers. p < 0.001 keeps false alarms ≤ 2% and still catches ≥ 92%
  of two-week stockouts on items selling ≥ 0.5/day [A-08, D-15].
- *Dormant (≥ 56 zero days with a price):* treated as not on the shelf, not as waiting demand —
  imputing 10% of series-days as phantom demand would make the policy stock items nobody can buy.
  Dormant items are forecast at an empirical "resume" rate, not zero [D-16, D-24].
- *Rejected:* fixed "N zeros in a row" rules (wrong for both fast and slow movers); a per-series
  HMM (slower, harder to defend live); Tobit (the censoring point — stock — is unobserved).

**Size of the problem:** stockouts hide 5.4–7.4% of demand by department in the last year and
**5.3% of demand inside the test window itself**; 317 series are out of stock on the cutoff day.

## 3. Forecast: one global Poisson GBM + two-level gamma-Poisson spread [D-20, D-21, D-24]
- Lag ≥ 28 features only → one model covers every day of the 28-day horizon, no recursion.
- Spread: `demand = Poisson(μ × G_item × G_day)`. G_item is shared by the item's 4 stores and
  persists over the horizon (forecast-level error); G_day is daily noise. Both estimated out of
  sample on the 28 days before the origin. **DC paths are sums of store paths**, so store and DC
  forecasts are coherent by construction (quantiles are never added).
- Post-model corrections, measured only on that pre-origin window: half-strength store × dept
  level correction (lag-28 features miss trends) and the dormant-resume factor.
- Backtests (d_1801/1829/1857, uncensored days): store bias 1.090 / 1.044 / 0.999, DC 90%
  interval coverage 0.93–0.95; better than a 28-day moving average on every metric (DC 90%
  coverage 0.75–0.79) [A-18].
- *Rejected:* per-quantile models (DC quantiles impossible), DeepAR/TFT (CPU time, hard to
  change live), independent daily NB (DC intervals far too narrow).
- *Backend:* scikit-learn HistGradientBoosting (same algorithm family and Poisson loss as
  LightGBM) because LightGBM could not be installed where the analysis ran; LightGBM is a config
  switch [D-19].

## 4. Simulator: enforce every rule, never trust the policy [D-22]
Daily: receive → demand → policy (sees its own sales, not demand) → **simulator enforces** POs on
Mondays only, store cap on on-hand + inbound, DC stock, outbound cap (pro-rata cuts, counted as
violations) → ship → cost. One lognormal lead-time draw per PO (σ² = ln(1+CV²)); orders may cross.
**Censored test days:** recorded test sales are capped by Walmart's own stock-outs, so demand on
those days is drawn around the imputed rate (fresh each Monte Carlo run). Replaying recorded
sales instead flatters the baseline by ~3 fill points and ~20% of cost [A-12].

## 5. Policy: two-echelon, capacity-aware base-stock [D-26, D-29, A-14…A-21]
- **Stores (daily):** order up to the α-quantile of demand over the next 3 days, plus a weekend
  pre-build when the DC outbound cap would bind later in the week; capped at 21 days.
- **DC (Mondays, per dept):** per sample path, draw when each open PO will really arrive
  (conditional on not having arrived yet), when today's order and the next deliveries arrive;
  order the β-quantile of the shortfall in the window that today's order must cover — including
  the stock stores must still hold at the end of it (Clark–Scarf echelon logic). Fixes found by
  tracing: counting open POs as already available, and ignoring store working stock, each left
  the DC empty of hundreds of items every Thursday–Sunday.
- **Allocation when outbound is capped:** one water-filling threshold — ship where an extra unit
  saves the most expected lost-sale cost (price × P(demand > position)).
- **Knobs** (α, β by dept, store window, store buffer) tuned on 56-day replays ending on or before
  the cutoff (cutoff − 56/84/112), common random numbers, objective = cost + penalty below 95.5%
  fill on any tuning window [D-25, D-30].
- *Rejected:* (s,S) — the only fixed cost is $50/dept/week; RL — unexplainable and would overfit
  the simulator; decision-focused learning — heavy, little gain given the cost asymmetry; longer
  store cover windows — tested, worse [A-21].

## 6. Results at d_1885 (100 runs; `reports/results_1885.md`) [A-23, A-24]
| | cost | fill | P(fill ≥ 95%) |
|---|---|---|---|
| baseline base-stock | $141.8k | 85.7% | 0.00 |
| **recommended** | **$114.1k** | **95.2%** | **0.63** |
| same, raw-sales forecast | $118.7k | 94.1% | 0.50 |
| same, no forecast corrections | $118.8k | 94.5% | 0.58 |
| lean knobs (40 runs) | $95.6k | 94.8% | 0.63 |

Stress (FOODS_3 lead ×2, weeks 3–5): baseline −6.4 fill pts / +$32.9k; recommended −0.1 pts (buffered);
lean −0.6 pts unwarned, −0.1 pts with two weeks' warning. The tuned knobs are insurance chosen on past
windows; on this spring window the lean knobs did as well for ~$19k less — reported to the COO as a choice,
not re-tuned (that would be peeking).

## 7. Interpretations of the brief [D-03…D-08, D-23, D-27]
| Question | Choice | Why |
|---|---|---|
| Fill rate | units sold ÷ demanded, all stores × items × days | what "95% of demand served" means to a COO; item-average also reported |
| Initial stock | 7 days at every location incl. DC; no pipeline | literal reading; makes the first two weeks the riskiest |
| "Average daily sales" for caps | recorded sales, last 365 training days, sellable days | the brief says sales; 365 is the only window it names |
| Holding cost | end-of-day on-hand, not in transit | switchable |
| Store cap | on-hand + inbound | stops "hiding" stock in transit |
| Re-forecast | every 28 days with the same trained model | no retraining on test data |
| Future prices | not used as features | avoids a leakage argument; costs use actual prices |
| Stress test | FOODS_3 POs placed in weeks 3–5 get mean 20 days | orders in flight unaffected (limitation) |

## 8. What I would do with more time
Horizon-specific models (lags ≥ 7/14/21) for the trend the lag-28 model misses; explicit
promotion/price-elasticity features; a smarter start-up (it is the largest single source of
lost sales and run-to-run variance); re-estimating the stockout detector with real on-hand data.
