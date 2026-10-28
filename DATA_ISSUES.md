# DATA_ISSUES.md — Project B (Meridian geo sample data)

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
