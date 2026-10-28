# DECISIONS.md

Every major modelling choice, the alternatives considered, and why they were rejected. Full evidence
is in `DECISION_LOG.md` (IDs in brackets: A-/D- for Project A, B-A/B-D for Project B).

# Project A — inventory policy under censored demand (cutoff d_1885)

**A1. Optimise cost; forecasts are only inputs [A-01, D-01].** One lost sale (35% of price) costs
as much as holding the unit for 511 days, so the cost-optimal policy wants near-100% service and
the 95% floor rarely binds. What binds are the physical limits: the DC outbound cap (1.25 × average
*recorded* sales, below recovered weekend demand), the 21-day store cap, weekly DC orders and
FOODS_3 lead times (mean 10, p95 17.5 days). Every choice was judged by simulated cost.
*Rejected:* tuning forecast accuracy first and the policy afterwards.

**A2. Demand recovery: classify every series-day, mask rather than fabricate [D-12…D-17].** Seven
statuses (ok / pre-launch / unlisted / closed / store-disrupted / stockout / dormant). Censored
days are masked from training targets; imputed values feed only lags, averages and the simulator.
Stockouts are detected by a zero-run test, P(k zeros | local NB rate × weekday) < 0.001, in two
passes. p < 0.01 was the first choice, but a null test showed 8.8% false-alarm days on slow
movers; 0.001 keeps false alarms ≤ 2% and still catches ≥ 92% of two-week stockouts on items
selling ≥ 0.5/day [A-08, D-15]. Dormant series (≥ 56 zero days with a price) are treated as off
the shelf and forecast at an empirical resume rate; otherwise 10% of series-days would become
phantom demand [D-16, D-24]. *Rejected:* fixed "N zeros" rules (wrong for both fast and slow
movers), a per-series HMM (slow, hard to defend), Tobit (stock is unobserved). **Size:** stockouts
hide 5.4–7.4% of demand by department, and 5.3% inside the test window.

**A3. Forecast: one global Poisson GBM + two-level gamma-Poisson spread [D-20, D-21, D-24].**
Lag ≥ 28 features give one model for the whole 28-day horizon. Demand = Poisson(μ × G_item × G_day):
G_item is shared by the item's stores and persists (forecast-level error), G_day is daily noise,
both estimated out of sample. **DC paths are sums of store paths**, so store and DC forecasts are
coherent. Pre-origin level and dormant-resume corrections. Backtests at 3 origins: store bias
0.999–1.090, DC 90% coverage 0.93–0.95, against 0.75–0.79 for a moving average [A-18]. *Rejected:*
per-quantile models (no coherent DC quantiles), DeepAR/TFT (CPU, hard to change live),
independent daily NB (DC intervals far too narrow). Backend scikit-learn HistGB, with LightGBM as a
config switch [D-19].

**A4. Simulator enforces every rule [D-22].** Each day: receive → demand → policy (sees its sales,
not demand) → the simulator enforces Monday-only POs, the store cap on on-hand + inbound, DC stock
and the outbound cap → ship → cost. One lognormal lead time per PO (orders may cross). On
censored test days demand is drawn around the imputed rate. Replaying recorded sales instead
flatters the baseline by ~3 fill points and ~20% of cost [A-12].

**A5. Policy: two-echelon, capacity-aware base-stock [D-26, D-29, A-14…A-21].** *Stores (daily):*
order up to the α-quantile of 3-day demand, plus a weekend pre-build when the outbound cap will
bind. *DC (Mondays):* per sample path, simulate when open POs and today's order arrive, and order
the β-quantile of the shortfall over the window today's order must cover. That window includes
the stock stores must still hold (Clark–Scarf echelon logic). Capped outbound is allocated by one
water-filling threshold on expected lost-sale cost. Knobs are tuned on 56-day replays ending on
or before the cutoff, with common random numbers. The objective is cost plus a penalty below 95.5%
fill [D-25, D-30]. *Rejected:* (s,S) (the only fixed cost is $50/dept/week); RL (unexplainable,
overfits the simulator); decision-focused learning (heavy, little gain given the cost asymmetry);
longer store windows (tested, worse) [A-21].

**A6. Results at d_1885, 100 runs [A-23, A-24].**

| | cost | fill | P(fill ≥ 95%) |
|---|---|---|---|
| baseline base-stock | $141.8k | 85.7% | 0.00 |
| **recommended** | **$114.1k** | **95.2%** | **0.63** |
| same, raw-sales forecast | $118.7k | 94.1% | 0.50 |
| lean knobs (40 runs) | $95.6k | 94.8% | 0.63 |

Stress test (FOODS_3 lead time ×2 in weeks 3–5): the baseline loses 6.4 fill points and $32.9k;
the recommended policy loses 0.1 points. The lean knobs did as well here for ~$19k less. They are
reported to the COO as a choice and were not re-tuned on the test window (that would be peeking).

**A7. Interpretations of the brief [D-03…D-27].**
- **Fill rate:** units sold ÷ units demanded, network-wide.
- **Start:** 7 days of initial stock everywhere, with no pipeline.
- **Caps:** based on recorded sales over the last 365 days.
- **Costs:** holding is charged on end-of-day on-hand stock.
- **Store cap:** includes inbound stock.
- **Re-forecasting:** every 28 days, without retraining.
- **Prices:** future prices are not used as features.
- **Stress test:** FOODS_3 POs placed in weeks 3–5 get a mean lead time of 20 days.

# Project B — geo MMM with experiment calibration (hold-out = final 13 weeks)

**B1. Framing: a budget decision with honest ranges [B-A01, B-A02].** The output is an 18M
allocation for 13 weeks within ±30% of each channel's last-13-week spend. It is backed by
**marginal** ROI (where the next unit should go) and average ROI (was it worth it), both with 90%
intervals. Last-13-week spend was 20.77M, so 18M is a 13% cut. 12M is below the rule's floor
(14.54M), so it is reported both at the floor and with the rule relaxed.

**B2. Causal structure [docs/dag.md].**
- **Treatment:** media spend.
- **Adjusted confounders:** season/trend, competitor sales and sentiment, all correlated with both spend and sales.
- **Also included:** promo (a separate treatment) and organic impressions (a possible mediator); organic and sentiment are each dropped in sensitivity fits.
- **Excluded:** paid impressions. Cost per impression is constant within every channel, so they duplicate spend.

**B3. Model: hierarchical geo MMM, Meridian-style [B-D02, B-D03, B-D05].** Per-capita conversions
= geo intercept + shared trend (knots every 8 weeks) + annual Fourier seasonality + **geo-specific
control effects** + promo + per channel a geo-pooled effect × Hill(adstock(spend)). Priors are
Meridian's defaults, framed on ROI: ROI ~ LogNormal(0.2, 0.9), Hill slope 1, half-saturation
~ TN(0.8, 0.8), decay ~ U(0, 1), 8-week lag. The prior predictive gives media a median 24% share
of conversions [B-A06]. *Geo controls rather than shared ones:* shared coefficients gave 11–18 of
40 geos negative media effects, which tighter pooling did not fix [B-A07]. *Extrapolable baseline
rather than Meridian's per-week effects:* per-week effects cannot forecast an end-of-window
hold-out; a weekly-fixed-effects fit is kept as an ROI robustness check. *Geo rather than national
totals:* 65–79% of spend variation is geo × week, which national totals throw away (143 points
for ~40 parameters). *Rejected:* S-shaped Hill (weakly identified), log-normal geo effects (break
exact Gaussian sampling), Weibull adstock (more parameters than the data identifies).

**B4. Inference: own exact-where-possible sampler [B-D04, B-D08, B-A04, B-A05, B-A08].** No PPL
could be installed where the analysis ran. Given the curve parameters, all other coefficients are
linear-Gaussian, so they are integrated out exactly and drawn exactly (an arrowhead solve, ~5 ms).
Each channel's decay, saturation and ROI move together in one adaptive joint Metropolis block.
*Validated:* the algebra matches dense matrices, known ROIs are recovered from simulated data
(19/20 inside their 90% intervals), and R-hat/ESS are reported for every model (main fit R-hat ≤ 1.039).
*Rejected:* point-estimate ridge MMM (no honest intervals); one-parameter-at-a-time Metropolis
(R-hat 1.08, ESS 30).

**B5. The Channel3 experiment: checked, not trusted [B-A03, B-D01].** Channel3 spend in Geo0–7 did
**not** drop during the test window (2 of 64 geo-weeks at zero; the per-capita test/control ratio
went 0.88 → 0.94). Three fits are compared: uncalibrated; calibrated with the test's uncertainty
×3; and fully calibrated. Calibration adds the vendor's iROAS as an extra likelihood term on the
model's own counterfactual for those geos and weeks, as PyMC-Marketing does. The budget uses the
**uncalibrated** fit; the full fit moves it by ≤ 0.38M per channel. *Rejected:* full trust (an
unverifiable input would drive the money); silently ignoring the test; prior-only calibration
(cannot express "these geos and weeks").

**B6. Budget and next experiment [B-D06, B-D07].** Response curves cover 13 weeks plus carry-over
for each posterior draw, keeping each channel's geo × week pattern. Allocation maximises
posterior-mean revenue by exact greedy water-filling on the concave curves; SLSQP failed on them.
The uplift is reported against a uniform cut at equal spend. The next experiment targets the
channel with the highest expected value of partial perfect information. Power is simulated on the
model's own residuals with a difference-in-differences read-out.

**B7. Results [B-A11, B-D09].**

| channel | ROI | marginal ROI | 18M plan |
|---|---|---|---|
| Ch0 | 1.30 [0.29, 2.97] | 0.64 [0.12, 1.46] | −5% |
| Ch1 | 1.03 [0.31, 2.27] | 0.54 [0.14, 1.17] | −30% |
| Ch2 | 1.20 [0.46, 2.21] | 0.71 [0.20, 1.41] | −15% |
| Ch3 | 0.94 [0.38, 1.71] | 0.51 [0.20, 0.95] | −30% |
| Ch4 | 1.62 [0.36, 3.35] | 0.77 [0.16, 1.47] | +23% |

Every marginal ROI is below 1, so the robust decision is the cut itself. The re-mix adds +0.32M
[−1.09, +1.76] (P = 0.63) over a uniform cut. Hold-out: national WAPE 6.3% (bias −6%), 90%
coverage 0.85. The costliest uncertainty (Ch0 ≈ Ch4, ~0.34M each) is not worth a feasible geo
test: the best design has 32% power for ~2M of forgone spend. We recommend gradual, monitored
reallocation instead.
