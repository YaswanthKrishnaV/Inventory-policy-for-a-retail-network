# AI_USAGE.md

**Tooling.** Claude (Anthropic) was used as a pair programmer throughout: reading the brief and
drafting the plan, writing most of the code, running the pipeline, and drafting the documents.
I reviewed the plan against the brief, confirmed every open interpretation (PLAN.md §13), and
checked results. Every number in the memo is produced by code in this repo.

**How AI output was checked.** Nothing was accepted because it looked plausible. Each component
had an independent check: the stockout detector against planted stockouts and stockout-free
simulations; the forecast against rolling backtests and calibration curves; the simulator against
unit-tested invariants; every policy change against an A/B on common random numbers; and every
surprising result against a day-by-day trace.

## Where the AI was wrong, and how it was caught

| # | What the AI did | How it was caught | Fix | Impact if missed |
|---|---|---|---|---|
| 1 | Set the stockout detector threshold at p < 0.01 | A **null test** — simulate stockout-free sales for each series, run the detector — flagged 8.8% of slow-mover days (multiple testing over hundreds of zero runs per series) | Threshold grid scored on recall (planted stockouts) vs false alarms; p < 0.001 | "Hidden demand" overstated by ~3 points → systematic overstocking |
| 2 | Policy speed-up: replaced exact quantiles with an interpolated table, then floored every shipment | Fill fell from 0.966 to 0.933 after a change that should have been neutral; a daily trace showed ~2,000 units/day of outbound capacity unused | Largest-remainder rounding to the capacity budget | ~3 fill points lost to an arithmetic artefact |
| 3 | Concluded a remaining 1.4-point gap was a second bug | Re-running both versions on the **same seeds** reproduced the old numbers exactly; the "gap" was Monte Carlo noise from comparing different seeds | Every comparison since uses common random numbers | Time spent "fixing" a non-bug; wrong conclusions about what matters |
| 4 | Calibrated the raw-sales benchmark against *recovered* demand and let it use dormancy flags | Reviewing why the raw-vs-recovered gap shrank between backtest versions | Benchmark made fully naive (recorded sales only) and re-run | The value of demand recovery would have been understated |
| 5 | DC order rule counted every open PO as already available and required only DC + store stock ≥ cumulative demand | A daily trace on a tuning window showed the DC short of 300–1,000 items every Thursday–Sunday while tuning knobs did nothing | Path-wise order with open POs' remaining lead times, plus store working stock (Clark–Scarf) | ~2 fill points on some windows; tuning would have chased the wrong knobs |
| 6 | First forecast under-forecast by 4–10% | Backtests split by horizon week and by dormant items showed trend and dormant-restart effects | Pre-origin level correction + dormant-resume factor | Systematic under-stocking, where errors cost 500× more |

## What the AI was not used for
Choosing the business interpretation (fill-rate definition, start-up assumptions, what counts as a
"sale" for caps) — those were confirmed by me. Final judgement on what to recommend to the COO.
