# Project B — Plan: geo-level marketing mix model with experiment calibration

> Working plan for Project B of the take-home. Purpose: pin down what is asked, what the data
> actually shows, which decisions we must make, and how to build it end to end before writing code.
> Data checks quoted below were run on `geo_all_channels.csv` from the Meridian repo (commit `ef585b5`).

---

## 0. TL;DR

- **The deliverable is a budget decision, not a model.** The CMO/CFO want: what each channel truly returns (average and *marginal* ROI, with 90% intervals), and how to split **18M over the next 13 weeks** with no channel moving more than ±30% from its last-13-week spend. Everything else (DAG, priors, checks, geo vs national, calibration) exists to make that recommendation defensible.
- **The planted trap: the Channel3 "experiment" is not in the data.** Spend was supposedly switched off in Geo0–Geo7 for 8 weeks from 2023-06-05. In the data, those geos kept spending (avg 8.0k/week in the window vs 6.8k in the 8 weeks before; only 2 of 64 geo-weeks are zero), and their ratio to control geos barely moves (0.57 → 0.66). The vendor's 0.8 ROAS cannot have come from that switch-off. **Default stance: do not let it drive the model**; show calibrated vs uncalibrated as a sensitivity.
- **The 12M scenario is infeasible under the ±30% rule.** Last-13-week spend is 20.76M → floor 14.53M, ceiling 26.99M. 18M (−13%) and 24M (+16%) are feasible; for 12M we report the allocation with the rule relaxed and say so.
- **Where the signal is:** geo *totals* of spend per capita are almost identical (CV 4–13%), but **65–79% of the weekly variation in each channel's spend per capita is geo-specific** (geos don't spend in lock-step). That geo × week variation is what separates channels and protects against seasonal confounding (spend is Q4-heavy, revenue flat) — and it is exactly what national totals throw away. Channels are still correlated (0.3–0.6), so priors matter and intervals will be honest rather than narrow.
- **Engine:** no PyMC / NumPyro / Stan / JAX can be installed in the environment where I run code (PyPI blocked). Recommended: a **hand-written, exact Bayesian sampler in numpy/scipy** that exploits the model's structure (coefficients are conditionally linear → sampled in closed form; only ~15 nonlinear parameters need MCMC), validated by a **parameter-recovery test on simulated data**. Optional PyMC cross-check on your Mac.

---

## 1. Business problem in plain terms

The company spends ~1.6M/week across 5 paid channels in 40 regions (geos), ~219M over 3 years, against ~8.5M/week of revenue. Next quarter's budget is 18M — **13% less than the last quarter's 20.76M**. The CMO wants to know where to cut (or move money) so revenue drops the least (or rises). The CFO wants every number with a range and a reason.

Two quantities matter and are different:
- **Average ROI** = revenue the channel produced ÷ what it cost ("was it worth it overall?").
- **Marginal ROI** = revenue from the *next* unit of spend at today's level ("where should the next dollar go?"). Budget reallocation is driven by marginal ROI; a channel can have high average ROI but be saturated.

---

## 2. Requirements map

| # | Requirement | Where | Done when |
|---|---|---|---|
| T1 | DAG before modelling; use it to justify variables | `project_b/dag.md` + figure; DECISIONS | Each variable labelled confounder / mediator / precision; inclusion argued |
| T2 | Hierarchical Bayesian geo model: adstock + saturation per paid channel, + organic + promo; justify every prior; prior- & posterior-predictive checks | `project_b/src/model.py`, `sampler.py`, `checks.py` | Prior table with reasons; PPC figures; R-hat < 1.01, ESS > 400 |
| T3 | Same model on national totals; explain disagreement; which to trust | `project_b/national.py` | Side-by-side ROI table + explanation |
| T4 | Calibrate with Channel3 experiment (prior or likelihood); ROIs with vs without; **check spend actually drops** | `project_b/src/calibration.py`, DATA_ISSUES | Data check figure; 3 variants (none / discounted / full) |
| T5 | Average ROI + marginal ROI at current spend per channel, 90% CrI; accuracy on 13 held-out weeks | `roi_table.csv`, `forecast_13w.csv`, report | Table + holdout WAPE / R² / 90% PI coverage, geo & national |
| T6 | Allocate 18M over Channel0–4 for next 13 weeks, ±30% per channel; expected incremental revenue with uncertainty; allocations at 12M and 24M | `budget_allocation.csv`, `project_b/optimise.py` | Allocation + revenue distribution; 12M infeasibility handled explicitly |
| T7 | Next experiment: which channel's uncertainty costs most money; geos, duration, power calc | `project_b/experiment_design.py` | Value-of-information ranking; geo split; simulated power curve |
| T8 | CMO/CFO memo ≤ 2 pages incl. what the model can't tell | `memo_B.pdf` | Non-technical, actionable, every number with a range |
| I1 | `--holdout-start` argument; re-run with different window | `run_project_b.py --holdout-start YYYY-MM-DD` | Runs at default (2023-10-23) and at another date unchanged |
| D1 | `project_b/posterior.nc` (or similar), `roi_table.csv`, `budget_allocation.csv`, `forecast_13w.csv` | pipeline outputs | Written by one command |
| D2 | Shared `DECISIONS.md` (≤ 3 pages for the whole repo), `DATA_ISSUES.md`, `AI_USAGE.md` | repo root | Project A text compressed to make room |

**Hidden requirements from "we re-run with a different hold-out window":** nothing may be hard-coded to the final 13 weeks; priors and settings can't be tuned on the hold-out; the hold-out may sit earlier in the series (e.g. covering the experiment window or Q4), so the code must handle any 13-week window.

---

## 3. What the data shows (checked)

| Fact | Evidence | Consequence |
|---|---|---|
| 40 geos × 156 weeks (Mondays, 2021-01-25 → 2024-01-15), 6,240 rows, no missing values | load check | — |
| **Impressions = spend ÷ a constant CPM per channel** (cost per impression has zero variance in every channel) | spend/impression per row | The two columns carry identical information → use **spend** (needed for ROI); impressions add nothing |
| Revenue = conversions × revenue_per_conversion; rpc ≈ 0.0200 (sd 0.0001) | describe | Model conversions (the KPI) and convert to revenue with rpc, as Meridian does; equivalent here |
| Conversions ≈ 19 per capita per week | ratio | Units are synthetic; model per-capita KPI so geos of 137k–994k people are comparable |
| Population constant per geo; Geo0–7 are mixed sizes (137k–892k) | pop check | Scale everything per capita; geo weights by population |
| **Spend per capita nearly identical across geos in total** (CV 0.04–0.13) … | per-geo totals | Geo *levels* say little; geo-level slopes should be strongly pooled |
| … **but 65–79% of weekly per-capita spend variance is geo × week** (common-week share 20–35%, geo share < 1%); KPI per capita: 86% geo × week | two-way variance decomposition | Plenty of within-week cross-geo variation → channels are identifiable at geo level; national aggregation discards most of it |
| Channel spend correlated 0.30–0.63 (Ch3–Ch4 0.63) | correlation | Collinearity → priors matter; report joint uncertainty |
| Channel2 has spend in only 36% of geo-weeks | zero share | Its saturation curve is poorly identified; wide intervals expected |
| **Spend is seasonal** (Q4 ≈ 1.4–1.6× Q1 every year); revenue flat-ish (8.0–8.8M/week) | quarterly sums | Seasonality/trend terms must not be so flexible they absorb media, nor so rigid media absorbs seasonality — key modelling tension |
| Controls correlated with media (competitor 0.39–0.48 with Ch2/3/4; sentiment 0.24–0.40 with all) | correlation | They behave like **confounders** → include (DAG §4) |
| Organic impressions correlated 0.38–0.55 with paid | correlation | Possible mediator (paid → organic) — see DAG |
| **Experiment not visible**: test geos' Channel3 spend in the 8-week window ≈ their own pre-period; 62 of 64 geo-weeks non-zero | pivot by week | The vendor result can't be verified from this data (§7) |
| Last 13 weeks spend: Ch0 3.80M, Ch1 3.20M, Ch2 1.44M, Ch3 7.95M, Ch4 4.37M = 20.76M | sums | ±30% bounds; 18M = −13% |
| Total revenue 1,318.7M vs spend 219.5M over 156 weeks | sums | Even if *all* revenue were media-driven, blended ROAS ≤ 6; realistic incremental share is far smaller |

Still to check during the build: outliers/structural breaks per geo, whether controls are standardised per geo, the provided `national_all_channels.csv` vs our own geo sums, holiday weeks (public holiday calendars are allowed as outside data).

---

## 4. Task 1 — Causal DAG

```
            Season / trend (time) ──────────────┐
             │            │                     │
             ▼            ▼                     ▼
   Competitor sales   Sentiment ───────────► Conversions ──► Revenue (× rpc)
        │    │           │  │                   ▲   ▲   ▲
        │    └──────┐    │  └──► Paid media 0–4 ─┘   │   │
        └──────► Paid media 0–4 (planner reacts)     │   │
                       │                             │   │
                       └──► Organic impressions? ────┘   │
                                      Promo ─────────────┘
   Population ──► (scales everything; modelled per capita)
```

| Variable | Role | Decision |
|---|---|---|
| Paid media 0–4 (spend) | treatments | in, with adstock + saturation |
| Season / trend | confounder (drives budget planning and demand) | in, as shared time effects (Fourier + smooth trend) |
| Competitor sales | confounder (correlated with media and with KPI) | in, linear, geo-pooled coefficient |
| Sentiment | likely confounder (macro mood drives both planning and buying); could be partly a mediator of brand media | in; **sensitivity: drop it** and see ROI change |
| Organic impressions | required by the brief; may be a **mediator** of paid (paid → search/organic) | in, as its own adstock+saturation channel (as briefed); **sensitivity: drop it** — if paid ROIs rise a lot, organic was absorbing paid effects |
| Promo | non-media treatment, little correlation with media | in, linear (with adstock check) |
| Population | scale | per-capita modelling |
| Impressions | duplicate of spend | out (documented) |
| revenue_per_conversion | outcome conversion factor | used to turn conversions into revenue |

Rule from the DAG: adjust for confounders, **not** mediators; where the role is ambiguous, include as the brief requires and quantify the sensitivity.

---

## 5. Task 2 — Hierarchical Bayesian geo model

### 5.1 Specification (per-capita KPI, Gaussian likelihood — Meridian-style)

`y[g,t] = conversions[g,t] / population[g]`, standardised per geo.

```
y[g,t] = τ_g                                  geo baseline
       + s(t)  + δ_g · s(t)?                   shared seasonality (Fourier, K=3) + smooth trend (knots every ~8 weeks); geo scaling optional
       + Σ_m β_{m,g} · Hill( Adstock(x[m,g,t]; α_m, L=8) ; K_m, n_m )     paid channels m = 0..4, x = spend per capita
       + β_org,g · Hill( Adstock(organic) ; … )
       + γ_promo · Adstock(promo)? + γ_comp · comp[g,t] + γ_sent · sent[g,t]
       + ε[g,t],   ε ~ Normal(0, σ_g)
β_{m,g} ~ TruncNormal⁺(μ_m, η_m)   (partial pooling across geos)
```

- **Adstock:** geometric, max lag 8 weeks, normalised so total carry-over weight = 1. *Alternative:* delayed/Weibull adstock — more flexible, poorly identified here.
- **Saturation:** Hill on spend scaled by the channel's median non-zero spend per capita. *Alternative:* exponential `1 − e^(−x/λ)` (one parameter fewer). Pick by prior-predictive plausibility and posterior identifiability.
- **Priors framed on ROI** (what a CMO can reason about), starting from Meridian's published defaults and departing only with a stated reason: channel ROI ~ **LogNormal(0.2, 0.9)** (Meridian default; median 1.22, 90% between ~0.28 and 5.4) translated to `μ_m`; Hill **slope fixed at 1** (Meridian default for impression channels — concave, no S-curve; S-curve as a sensitivity); half-saturation ec_m ~ **TruncNormal(0.8, 0.8, 0.1, 10)** × median spend (Meridian default); adstock decay α_m ~ Uniform(0, 1) (Meridian default) — or Beta(2, 5) if the prior-predictive check shows Uniform implies implausible 8-week carry-over; geo spread of media effects η_m ~ HalfNormal on the relative scale (Meridian uses HalfNormal(1) on log-normal geo effects; we use an additive-relative form to keep the conjugate block — documented difference); controls ~ Normal(0, 5) on standardised scale (Meridian default; weakly informative); σ_g ~ HalfNormal. Every prior gets a one-line justification and a prior-predictive check (implied ROI and implied media share of revenue must be plausible; Meridian's fallback when ROI can't be set is "paid media ≈ 40% ± 20% of KPI", a useful sanity anchor).
- **Time effects — a deliberate departure from Meridian's geo default.** Meridian gives every week its own shared effect in geo models. That is the strongest protection against seasonal confounding, but those effects **cannot be extrapolated**, and Meridian's docs explicitly advise against end-of-window holdouts and say it "isn't designed for forecasting the KPI". The brief requires exactly that (final 13 weeks, `--holdout-start`). So the **main model uses an extrapolable baseline** (smooth trend with knots every ~8 weeks + annual Fourier seasonality, K = 3), and a **robustness variant with weekly fixed effects** (identifiable here because 65–79% of spend variation is geo × week) is fitted for ROI only. If the two agree on ROIs, the extrapolable baseline isn't absorbing or leaking media effects; if they disagree, that is reported.

### 5.2 Inference engine — options

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Google Meridian** | Exactly this model class; built-in calibration with ROI priors; reviewers know it | TensorFlow Probability; cannot install here; heavy on a laptop; harder to change live | Reject for the main pipeline; mention |
| **PyMC-Marketing** | Standard, NUTS sampler, nice diagnostics | Can't install here (PyPI blocked) → I couldn't run or verify anything; you'd run every step | Optional cross-check on your Mac |
| NumPyro / Stan | Fast NUTS | Same install problem | Reject |
| **Own sampler (numpy/scipy), exploiting structure** *(recommended)* | Runs everywhere (incl. here); fast; fully explainable line by line (the interview asks for live code changes); the brief explicitly allows own code | Correctness burden on us → must be validated | **Recommend** |

**Recommended sampler — collapsed Gibbs + Metropolis:**
1. Given the ~15 nonlinear parameters (adstock decays, saturation K and n for 5 paid + organic channels, promo decay), every other parameter enters **linearly** (geo baselines, seasonality/trend coefficients, media coefficients, control coefficients). With Gaussian likelihood and Gaussian(-truncated) priors, their conditional posterior is (truncated) multivariate normal → **sampled exactly** in one block (media coefficients' positivity via a Gibbs sweep of truncated normals).
2. The nonlinear parameters are updated by **adaptive random-walk Metropolis** on transformed scales (logit/log), one channel at a time, using the conditional likelihood.
3. Hyper-parameters (μ_m, η_m, σ_g) have conjugate or 1-D updates.
4. 4 chains, R-hat / bulk-ESS computed ourselves (Vehtari rank-normalised), trace and rank plots.
5. **Validation:** simulate a dataset from the model with known parameters (same spend matrices), fit, and check that true ROIs fall inside the 90% intervals (~90% of the time across several replicates). This is the evidence that the hand-written sampler is correct — and it doubles as a statement of what this data *can* identify.

### 5.3 Checks to show
Prior-predictive (implied KPI ranges, ROI, media share), posterior-predictive (fit by geo and national, residual ACF, coverage), convergence (R-hat, ESS, traces), hold-out accuracy, sensitivity to sentiment/organic inclusion and to the ROI prior width.

---

## 6. Task 3 — Geo vs national

- Fit the same model to national weekly sums (156 rows; single geo; no pooling level).
- Expected disagreements and why: (1) **far less information** (156 vs 6,240 rows; seasonal, collinear national spend) → wider and more prior-driven ROIs; (2) **aggregation bias**: saturation is nonlinear, so the curve of summed spend ≠ sum of geo curves (Jensen) — national tends to mis-state saturation and marginal ROI; (3) geo-level timing variation (geos don't spend in lock-step week to week) is exactly what separates channels — it disappears in the national sum.
- Trust: geo, unless diagnostics say otherwise; show where they disagree channel by channel.

---

## 7. Task 4 — Experiment calibration

**Data check (already done):** Channel3 spend in Geo0–7 during 2023-06-05 → 2023-07-31 did **not** drop (figure to produce: weekly Channel3 spend, test vs control, with the window shaded; also the implied "lift" in revenue vs a synthetic control). Possible explanations to state, not choose between: the switch-off was not executed (or partially), the geos listed are wrong, or our data is not the data the vendor measured. Either way the reported 0.8 is **not an estimate of what this data says**.

**How to calibrate (both offered, one recommended):**
- *Prior route (Meridian's way):* Channel3 ROI prior ← LogNormal matching 0.8 (90% CI 0.3–1.3); Meridian's CalibrationBuilder additionally widens the uncertainty for duration (8 weeks vs adstock), spend scale and recency, and warns that lift tests must measure against *zero* spend to match the MMM's counterfactual.
- *Likelihood route (PyMC-Marketing's way, preferred technically):* PyMC-Marketing scores a lift test (x, Δx, Δy, σ) against the model's own saturation curve. We do the same with the full model (adstock included): compute inside the model the experiment's estimand — incremental revenue in Geo0–7 over the window (+ adstock tail) if Channel3 had been zero vs actual, ÷ spend removed — and add `0.8 ~ Normal(lift_model, se)` with se = 0.5/1.645 = 0.30. Because spend did *not* drop in the data, the counterfactual "spend removed" is the actual Channel3 spend in those geos/weeks — the estimand is well defined even though the test wasn't run as described.
- **Trust decision (recommended):** fit three variants — *uncalibrated*, *discounted* (se inflated ×3 ≈ 0.9 — the same idea as Meridian's uncertainty inflation / "se widening"), *full* — and report all three ROIs. **Use the uncalibrated (or discounted) model for the budget**, because the data contradicts the test's design. Show how much the allocation would move under full trust: if it barely moves, the question is moot; if it moves a lot, that's the argument for re-running the test properly (feeds T7).

---

## 8. Task 5 — Effectiveness and hold-out accuracy

- **Average ROI (per channel, per posterior draw):** incremental revenue with the channel at actual spend vs. channel at 0 (carry-over included) over the last 52 training weeks ÷ spend. Report mean, median, 90% CrI.
- **Marginal ROI at current spend:** increase the channel's last-13-week spend by 1% (same weekly/geo pattern), Δrevenue ÷ Δspend, per draw → 90% CrI. Also full **response curves** (0–200% of current) for the memo.
- **Hold-out:** train strictly on weeks before `--holdout-start`; forecast the next 13 weeks with actual media and controls in that window (a conditional forecast — controls are not forecast); report WAPE, MAPE, R², bias and 90%-interval coverage at geo and national level; `forecast_13w.csv` with mean and 5/50/95% per geo-week and national.
- All reported ROIs come from the same training fit (hold-out untouched). Flag `--refit-all` refits on all weeks for the final allocation (default off, so every memo number is from the validated fit) — decision to confirm.

---

## 9. Task 6 — Budget optimisation

- Decision variables: 13-week total per channel `b_m`; geo and weekly split of each channel kept proportional to its last-13-week pattern (the decision being made is channel mix, not geo targeting — flagged as a limitation/upside).
- Constraints: Σ b_m = B; 0.7·last13_m ≤ b_m ≤ 1.3·last13_m.
- Objective: posterior **mean** incremental revenue over the 13 weeks (+ carry-over tail into the following weeks, reported separately). Solved with SLSQP on the posterior-mean response; also checked with a risk-aware variant (maximise the 10th percentile) — if the two agree, say so; if not, show both.
- Outputs per budget (12M, 18M, 24M): allocation, % change vs last 13 weeks, expected incremental revenue with 90% CrI, and the **uplift vs the "same mix scaled to B" baseline** (with its own CrI — the honest measure of what optimising buys).
- **12M:** infeasible (floor 14.53M). Report (a) the floor allocation (every channel −30%) and (b) the optimum with bounds relaxed, clearly labelled.
- Marginal-ROI equalisation check: at the optimum, unconstrained channels should have ≈ equal marginal ROI; channels at a bound will not — that tells the CMO which rule is binding.

---

## 10. Task 7 — Next experiment

- **Which channel's uncertainty costs most money:** expected value of perfect information per channel. For each posterior draw, compute the revenue of the allocation we would choose if that channel's response curve were known exactly vs. the allocation we actually choose; the average shortfall (regret) is the money at stake. Rank channels.
- **Design for the top channel:** geo-level holdout/boost test. Choose ~8–12 treatment geos by stratified matching on population and pre-period KPI trend (reuse everything learned from the failed Channel3 test: verify execution weekly); duration 6–8 weeks + 4-week cool-down for carry-over; spend change large enough to detect (e.g. −100% or +50%).
- **Power:** simulation using the posterior predictive (geo-week noise, cross-geo correlation, carry-over): for candidate designs (number of geos × weeks × spend change), probability the test's 90% CI excludes zero and the MDE in ROI units. Pick the cheapest design with ≥ 80% power. Report its cost (lost/extra spend × ROI).

---

## 11. Task 8 — CMO/CFO memo (≤ 2 pages)

What to do with the 18M (table), what each channel returns (average and marginal, ranges), what we gain vs keeping today's mix, what the experiment does and doesn't tell us, what we'd test next and why, and **what the model cannot tell them** (e.g. creative quality, effects beyond 13 weeks, geo targeting, anything driven by unmeasured confounders).

---

## 12. Repo layout (same repo as Project A)

```
data/project_b/geo_all_channels.csv           (gitignored; README says where to get it)
config/project_b.yaml                         all priors, adstock length, knots, budgets, bounds, seeds
project_b/
  run_project_b.py        entry point: --holdout-start, --config, --fast
  src/data.py             load, per-capita scaling, checks
  src/transforms.py       adstock, Hill, scaling (shared by fit, ROI, optimiser)
  src/model.py            design matrices, priors, likelihood
  src/sampler.py          collapsed Gibbs + adaptive Metropolis, diagnostics
  src/calibration.py      experiment estimand + likelihood term; data check
  src/roi.py              average / marginal ROI, response curves
  optimise.py             budget allocation at 12 / 18 / 24M
  national.py             national-totals model + comparison
  experiment_design.py    value of information + power simulation
  report.py               tables, figures
  posterior.nc            (scipy netCDF3 writer — no xarray needed)
  roi_table.csv  budget_allocation.csv  forecast_13w.csv
tests/test_project_b.py   adstock/saturation unit tests; sampler recovers known parameters
memo_B.pdf
```

---

## 13. Timeline (~1 working day, like Project A)

| Block | Hours | Output |
|---|---|---|
| Data checks, DAG, DATA_ISSUES entries (incl. experiment check figure) | 1.5 | `dag.md`, figures |
| Transforms + model + sampler + **recovery test on simulated data** | 3 | sampler validated |
| Fit geo model, diagnostics, prior/posterior predictive, hold-out accuracy | 1.5 | fit report, `forecast_13w.csv` |
| National model + comparison; calibration variants | 1 | comparison tables |
| ROI table, optimiser (12/18/24M), value of information, power simulation | 1.5 | CSVs, figures |
| Memo B, DECISIONS (compress A + add B to ≤ 3 pages), AI_USAGE, end-to-end timed run | 1.5 | `memo_B.pdf` |

---

## 14. Decisions to confirm before coding

1. **Engine:** hand-written numpy/scipy sampler (runs here; validated by parameter recovery) rather than PyMC/Meridian (can't run here). OK?
2. **Hold-out:** train strictly on weeks before `--holdout-start`; every reported number (ROI, budget) comes from that fit; `--refit-all` optional, off by default.
3. **Experiment:** treat the vendor result as unverified because spend did not drop; budget built on the uncalibrated (or ×3-discounted) model; full calibration shown as sensitivity.
4. **Outcome:** model conversions per capita, report revenue (× revenue_per_conversion).
5. **Budget scope:** optimise channel totals; keep each channel's geo and weekly pattern as in the last 13 weeks.
6. **12M:** report the ±30% floor (14.53M) and a bounds-relaxed optimum, clearly labelled.
7. **Repo docs:** one shared `DECISIONS.md` (≤ 3 pages total) — Project A's section gets compressed to make room; full detail stays in `DECISION_LOG.md`.

---

## 15. Validation against published practice (checked 2026-10-28)

| Plan element | What established practice does | Verdict |
|---|---|---|
| Geo-level hierarchical model, per-capita KPI, Gaussian likelihood, geometric adstock with max lag 8, Hill saturation | Meridian: geometric adstock default, `max_lag` default 8, Hill with half-saturation `ec` and slope; geo hierarchy with geo-level media effects | **Aligned** |
| ROI-parameterised priors | Meridian parameterises media priors on ROI (default LogNormal(0.2, 0.9)), mROI or contribution | **Aligned** — adopting Meridian's default numbers as the starting point |
| Geo beats national | Google's geo-level Bayesian hierarchical MMM paper: geo models give tighter intervals and lower error because national data has small samples and little spend variation | **Aligned** — and our variance decomposition (65–79% geo × week) shows why it applies strongly here |
| Time effects | Meridian geo default: one effect per week; warns against end-of-window holdouts; not designed for KPI forecasting | **Deliberate departure** (brief requires an end-of-window forecast): extrapolable trend + seasonality in the main model, weekly fixed effects as a robustness check |
| Experiment calibration | Meridian: experiment → ROI prior (with uncertainty widening; zero-spend lift tests only). PyMC-Marketing: lift test added to the likelihood, scored against the model's saturation curve | **Aligned** — likelihood route as primary, prior route as an alternative, widening as the "discount" knob |
| Checking the experiment against the data | Meridian guidance: review calibrated output against expectations; test design must match the counterfactual | **Aligned, and stronger** — we test whether the treatment happened at all (it didn't) |
| Budget optimiser: fixed budget, ±30% per channel | Meridian's optimiser default spend constraint is exactly ±30% of historical spend, fixed budget, maximising ROI / incremental revenue | **Aligned** (the brief's rule mirrors Meridian's default) |
| Next experiment chosen by value of information | Practitioner guidance (e.g. Recast): test where wide intervals sit behind large budget decisions and the result would change the allocation | **Aligned** — EVPI formalises the same idea |
| Power by simulation, matched geos, cool-down for carry-over | GeoLift / GeoX style designs: synthetic-control or matched-market selection with simulated power and MDE | **Aligned** |
| Hand-written sampler | Not standard (practice uses NUTS via Meridian / PyMC / NumPyro) | **Departure, forced by the environment** — mitigated by a parameter-recovery test; PyMC cross-check optional on the Mac |

**Changes made to this plan after the review:** (1) priors now start from Meridian's defaults (ROI LogNormal(0.2, 0.9), Hill slope 1, ec TruncNormal(0.8, 0.8, 0.1, 10), adstock Uniform(0, 1)); (2) the claim that identification is weak was corrected — geo × week variation is large; (3) added the weekly-fixed-effects robustness model and the reason for not using it as the main model; (4) calibration now explicitly mirrors both Meridian's prior route and PyMC-Marketing's likelihood route.
