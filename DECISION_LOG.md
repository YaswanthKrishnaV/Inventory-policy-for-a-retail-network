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

### B-A02 · Data facts (`python -m project_b.audit` → `reports/data_audit.md`)
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
- Figure: `reports/fig_experiment_check.png`.

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

### B-A05 · Parameter recovery (`python -m project_b.recovery`, `reports/recovery.json`)
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
