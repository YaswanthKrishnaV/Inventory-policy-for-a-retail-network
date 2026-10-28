# AI_USAGE.md

**Tooling.** Claude (Anthropic) was used as a pair programmer on both projects. It read the brief
and the data, drafted the plans (`PLAN.md`, `PLAN_B.md`), and checked the Project B plan against the
Meridian and PyMC-Marketing documentation. It wrote most of the code, ran the pipelines and drafted
the documents. I reviewed the plans against the brief, confirmed every open interpretation
(`PLAN.md` §13, `PLAN_B.md` §14) and checked the results. Every number in the memos is produced by
code in this repo.

**How AI output was checked.** Nothing was accepted because it looked plausible.
- *Project A:* each component had an independent check: the stockout detector against planted stockouts and stockout-free simulations; the forecast against rolling backtests and calibration curves; the simulator against unit-tested invariants; every policy change against an A/B on common random numbers; and every surprising result against a day-by-day trace.
- *Project B:* the exact Gaussian algebra is tested against brute-force dense matrices; the sampler is tested by recovering known ROIs from simulated data (19 of 20 true ROIs inside their 90% intervals over two rounds); every model change was compared on hold-out accuracy and on the plausibility of geo-level effects, not just on headline ROIs.

## Where the AI was wrong, and how it was caught

### Project A
| # | What the AI did | How it was caught | Fix | Impact if missed |
|---|---|---|---|---|
| 1 | Set the stockout detector threshold at p < 0.01 | A **null test** — simulate stockout-free sales for each series, run the detector — flagged 8.8% of slow-mover days (multiple testing over hundreds of zero runs per series) | Threshold grid scored on recall (planted stockouts) vs false alarms; p < 0.001 | "Hidden demand" overstated by ~3 points → systematic overstocking |
| 2 | Policy speed-up: replaced exact quantiles with an interpolated table, then floored every shipment | Fill fell from 0.966 to 0.933 after a change that should have been neutral; a daily trace showed ~2,000 units/day of outbound capacity unused | Largest-remainder rounding to the capacity budget | ~3 fill points lost to an arithmetic artefact |
| 3 | Concluded a remaining 1.4-point gap was a second bug | Re-running both versions on the **same seeds** reproduced the old numbers exactly; the "gap" was Monte Carlo noise from comparing different seeds | Every comparison since uses common random numbers | Time spent "fixing" a non-bug; wrong conclusions about what matters |
| 4 | Calibrated the raw-sales benchmark against *recovered* demand and let it use dormancy flags | Reviewing why the raw-vs-recovered gap shrank between backtest versions | Benchmark made fully naive (recorded sales only) and re-run | The value of demand recovery would have been understated |
| 5 | DC order rule counted every open PO as already available and required only DC + store stock ≥ cumulative demand | A daily trace on a tuning window showed the DC short of 300–1,000 items every Thursday–Sunday while tuning knobs did nothing | Path-wise order with open POs' remaining lead times, plus store working stock (Clark–Scarf) | ~2 fill points on some windows; tuning would have chased the wrong knobs |
| 6 | First forecast under-forecast by 4–10% | Backtests split by horizon week and by dormant items showed trend and dormant-restart effects | Pre-origin level correction + dormant-resume factor | Systematic under-stocking, where errors cost 500× more |

### Project B
| # | What the AI did | How it was caught | Fix |
|---|---|---|---|
| 1 | Claimed in the first plan that channels were weakly identified because spend per capita is almost identical across geos | A two-way variance decomposition showed 65–79% of spend variation is geo × week — the opposite conclusion | Plan corrected; this variation is now the core argument for geo over national |
| 2 | Gave control variables one shared coefficient | The fitted model had 11–18 of 40 geos with *negative* media effects; tighter pooling did not remove them | Geo-specific control effects (as in Meridian); negatives disappear, fit improves |
| 3 | Used SLSQP for the budget optimiser | It failed on piecewise-linear response curves in the end-to-end smoke test | Exact greedy water-filling (curves are concave) |
| 4 | Initially compared the experiment geos using absolute spend ratios (0.57 → 0.66) | Test geos are smaller; per-capita ratios (0.88 → 0.94) are the right comparison | Audit uses per-capita spend |
| 5 | Reported a 12M "floor" plan with +2.3M uplift and P = 1.00 | Implausibly certain; traced to comparing 14.54M of spend against today's mix scaled to 12M | Uplift always compared at equal total spend |
| 6 | Drafted the memo saying Channel3's return "could be higher" if sentiment were a mediator | Re-reading against the numbers: dropping sentiment *lowers* Channel3's ROI, so the mediator reading would mean advertising lowers sales via sentiment | Memo corrected; the same argument now supports treating sentiment as a confounder |
| 7 | One-parameter-at-a-time Metropolis after adding geo controls | R-hat 1.078, ESS 30 | Joint adaptive block per channel: R-hat 1.039, ESS 143 |

## What the AI was not used for
Choosing the business interpretation (fill-rate definition, start-up assumptions, what counts as a
"sale" for caps) — those were confirmed by me. Final judgement on what to recommend to the COO.
