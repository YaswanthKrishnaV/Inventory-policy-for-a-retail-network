# DECISIONS.md — Project B

Major modelling choices, the alternatives, and why. Full evidence in `DECISION_LOG.md` (IDs in
brackets). Results quoted are from `reports/project_b/results.md` (hold-out = final 13 weeks).

## 1. Framing: a budget decision with honest ranges [B-A01]
The output is an 18M allocation for 13 weeks within ±30% of last-13-week spend per channel,
backed by **marginal** ROI (where the next unit should go) and average ROI (was it worth it), all
with 90% intervals. Last-13-week spend is 20.77M, so 18M is a 13% cut: the question is mostly
*where to cut*. 12M is below the rule's floor (14.54M) and is reported both at the floor and with
the rule relaxed [B-A02].

## 2. Causal structure [docs/dag.md]
Media spend is the treatment; season/trend, competitor sales and sentiment are confounders
(correlated with both spend and sales) and are adjusted for; organic impressions may be a mediator
of paid media and promo is a separate treatment — both included as briefed, with organic and
sentiment dropped in sensitivity fits. Impressions are excluded: cost per impression is constant
in every channel, so they duplicate spend. *Rejected:* adding impressions as separate drivers
(perfectly collinear with spend); dropping controls (they are correlated 0.24–0.48 with spend).

## 3. Model: hierarchical geo MMM, Meridian-style [B-D02, B-D03, B-D05]
Per-capita conversions = geo intercept + shared trend (knots every 8 weeks) and annual
seasonality + **geo-specific control effects** + promo + per channel a geo-pooled effect ×
Hill(adstock(spend)). Priors start from Meridian's published defaults (ROI ~ LogNormal(0.2, 0.9),
Hill slope 1, half-saturation ~ TruncNormal(0.8, 0.8), adstock decay ~ U(0, 1), 8-week lag) and
are framed on ROI, which a CMO can reason about. The prior-predictive check gives media a median
24% share of conversions (13–59%) before seeing the data [B-A06, B-A11].
- *Geo controls instead of shared:* with one shared coefficient, 11–18 of 40 geos got negative
  media effects; tighter pooling did not fix it; geo-specific controls (as Meridian does) did,
  and improved fit [B-A07].
- *Extrapolable baseline instead of Meridian's per-week effects:* per-week effects cannot forecast
  an end-of-window hold-out (Meridian's own docs warn against such hold-outs); a weekly-fixed-effects
  version is fitted as a robustness check on ROI.
- *Rejected:* S-shaped Hill (slope > 1) — weakly identified, kept as a possible sensitivity;
  log-normal geo effects — break the exact conditional-Gaussian sampling; Weibull adstock — more
  parameters than this data can identify.

## 4. Inference: our own exact-where-possible sampler [B-D04, B-D08, B-A04, B-A05, B-A08]
No probabilistic-programming library can be installed where the analysis runs, and the brief
allows own code. Given the 12 curve parameters, every other coefficient is linear-Gaussian, so it
is integrated out exactly for the Metropolis steps and drawn exactly afterwards; the arrowhead
structure (40 geo blocks + shared block) makes this ~5 ms per step. Each channel's decay,
saturation and ROI move together in one adaptive joint proposal. **Validated** three ways: the
algebra matches brute-force matrices; known ROIs are recovered from simulated data (19/20 inside
their 90% intervals over two rounds); R-hat / ESS reported for every model (main ≤ 1.039). *Rejected:* Meridian / PyMC / NumPyro
(not installable here — a PyMC cross-check is possible on a laptop); point-estimate MMM (Robyn-style
ridge) — no honest intervals.

## 5. The Channel3 experiment: checked, not trusted [B-A03, B-D01]
The data contradicts the test's design: Channel3 spend in Geo0–7 did **not** drop in the 8-week
window (2 of 64 geo-weeks at zero; test/control spend ratio 0.88 → 0.94). So the vendor's 0.8
cannot be an estimate of what this data shows. We fit three versions — uncalibrated, calibrated
with the test's uncertainty tripled, and fully calibrated (the test enters as an extra likelihood
term: the model's own counterfactual iROAS for those geos and weeks, PyMC-Marketing style) — and
build the budget on the **uncalibrated** model, reporting how the others would change it.
*Rejected:* full trust (unverifiable input would drive the money); ignoring it silently (the
reader needs to see what trusting it would imply); prior-only calibration (cannot express "this
specific test, these geos and weeks").

## 6. Geo vs national
Fitting national totals throws away the 65–79% of spend variation that is geo × week; the national
model has 143 training points for ~40 parameters. Where the two disagree we trust the geo model,
whose intervals are narrower for a reason (more independent variation), not because of priors.

## 7. Budget and next experiment [B-D06]
Response curves per channel (13 weeks + carry-over, each channel keeping its last-13-week geo ×
week pattern) per posterior draw; allocation maximises posterior-mean incremental revenue by
exact greedy water-filling (curves are concave), so marginal ROIs equalise across channels not at
a bound. Uncertainty is reported for the allocation's revenue and for its uplift over "today's mix
scaled to the budget". The next experiment targets the channel with the highest expected value of
partial perfect information (money lost by not knowing its curve), with power by simulation on the
model's own residuals and a difference-in-differences read-out.

## 8. Results and what they mean [B-A11, B-D09]
| channel | ROI | marginal ROI | 18M plan |
|---|---|---|---|
| Ch0 | 1.30 [0.29, 2.97] | 0.64 [0.12, 1.46] | −5% |
| Ch1 | 1.03 [0.31, 2.27] | 0.54 [0.14, 1.17] | −30% |
| Ch2 | 1.20 [0.46, 2.21] | 0.71 [0.20, 1.41] | −15% |
| Ch3 | 0.94 [0.38, 1.71] | 0.51 [0.20, 0.95] | −30% |
| Ch4 | 1.62 [0.36, 3.35] | 0.77 [0.16, 1.47] | +23% |

Every channel's marginal ROI is below 1, so the robust decision is the cut itself; the re-mix adds
+0.32M [−1.09, +1.76] (P = 0.63) over a uniform cut. Hold-out: national WAPE 6.3% (bias −6%), 90%
coverage 0.85. Calibration with the vendor's test changes the plan by ≤ 0.38M per channel. The
most valuable uncertainty (Ch0 ≈ Ch4, ~0.34M each) is not worth a feasible geo test (best design:
32% power for ~2M forgone spend), so we recommend gradual monitored reallocation instead.
