# DECISION_LOG.md — Project B (geo MMM with experiment calibration)

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
