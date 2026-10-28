# AI_USAGE.md

Tooling: Claude (Anthropic) as a pair programmer for planning, code drafting and review; all numbers
produced by code in this repo and checked by me.

## Log

| When | What the AI did | Where it was wrong / how it was caught | Fix |
|---|---|---|---|
| Plan | Drafted PLAN.md: requirement map, options and trade-offs | Reviewed against the brief line by line; open decisions confirmed by me (§13) | — |
| Demand recovery | Wrote the zero-run stockout detector with p<0.01 threshold | **Wrong threshold.** A null test (simulate stockout-free sales, run the detector) showed 8.8% of slow-mover days flagged — multiple testing over hundreds of zero runs per series. It would have inflated "hidden demand" from ~6% to ~9% and pushed stock levels up. | Grid over thresholds with planted-stockout recall vs null false-alarm rate; chose p<0.001 (false alarms <=2%). |
