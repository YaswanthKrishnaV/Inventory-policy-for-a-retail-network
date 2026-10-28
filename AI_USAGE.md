# AI_USAGE.md — Project B

**Tooling.** Claude (Anthropic) as a pair programmer: reading the brief and data, drafting the
plan, checking it against Meridian / PyMC-Marketing documentation, writing the code, running
fits, and drafting documents. Interpretation choices were confirmed by me (`PLAN.md` §14).

**How AI output was checked.** The exact Gaussian algebra is tested against brute-force dense
matrices; the sampler is tested by recovering known ROIs from simulated data (9/10 true ROIs in
their 90% intervals); every model change was compared on hold-out accuracy and on the plausibility
of geo-level effects, not just on headline ROIs.

## Where the AI was wrong, and how it was caught

| # | What the AI did | How it was caught | Fix |
|---|---|---|---|
| 1 | Claimed in the first plan that channels were weakly identified because spend per capita is almost identical across geos | A two-way variance decomposition showed 65–79% of spend variation is geo × week — the opposite conclusion | Plan corrected; this variation is now the core argument for geo over national |
| 2 | Gave control variables one shared coefficient | The fitted model had 11–18 of 40 geos with *negative* media effects; tighter pooling did not remove them | Geo-specific control effects (as in Meridian); negatives disappear, fit improves |
| 3 | Used SLSQP for the budget optimiser | It failed on piecewise-linear response curves in the end-to-end smoke test | Exact greedy water-filling (curves are concave) |
| 4 | Initially compared the experiment geos using absolute spend ratios (0.57 → 0.66) | Test geos are smaller; per-capita ratios (0.88 → 0.94) are the right comparison | Audit uses per-capita spend |
| 5 | Reported a 12M "floor" plan with +2.3M uplift and P = 1.00 | Implausibly certain; traced to comparing 14.54M of spend against today's mix scaled to 12M | Uplift always compared at equal total spend |
| 6 | Drafted the memo saying Channel3's return "could be higher" if sentiment were a mediator | Re-reading against the numbers: dropping sentiment *lowers* Channel3's ROI, so the mediator reading would mean advertising lowers sales via sentiment | Memo corrected; the same argument now supports treating sentiment as a confounder |
| 7 | One-parameter-at-a-time Metropolis after adding geo controls | R-hat 1.078, ESS 30 | Joint adaptive block per channel: R-hat 1.039, ESS 143 |
