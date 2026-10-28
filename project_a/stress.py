"""Task 5 stress test: FOODS_3 supplier lead times double for three weeks.

Interpretation (DECISION_LOG D-27): FOODS_3 POs *placed* in test days [from_day, to_day)
(default weeks 3-5, days 14-34) get twice the mean lead time (CV unchanged). Orders already in
flight are not affected (a variant that also delays them is a stated limitation).

Three runs on common random numbers:
  base      - no shock
  unwarned  - shock happens, the policy keeps believing the normal lead time
  warned    - the policy is told `warn_days` (14) before the shock starts; it then plans every
              order with the lead time it will actually face (the DC order logic looks for the
              next supplier delivery among future orders, so it pre-buys before the slow weeks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import simulate as sm


def run_stress(make_policy, net, panel, cfg, n_sims, seed, frame, context=None, shock=None, log=print):
    sc = dict(cfg["stress"])
    if shock:
        sc.update(shock)
    base_shock = dict(dept=sc["dept"], from_day=sc["from_day"], to_day=sc["to_day"], multiplier=sc["multiplier"])
    scenarios = {
        "base": None,
        "unwarned": dict(base_shock, warn_days=None),
        "warned": dict(base_shock, warn_days=sc["warn_days"]),
    }
    out, daily = {}, {}
    for name, shk in scenarios.items():
        df, runs = sm.monte_carlo(make_policy, net, panel, cfg, n_sims, seed, shock=shk, frame=frame, context=context)
        out[name] = df
        d = runs[0]["daily"]
        daily[name] = 1 - np.array(d["lost"]) / np.maximum(np.array(d["demand"]), 1)
        log(f"  stress {name:9s}: cost {df.total_cost.mean():,.0f}  fill {df.fill_rate.mean():.4f}  "
            f"FOODS_3 fill {df.fill_FOODS_3.mean():.4f}")
    return out, daily


def summarise(out: dict) -> pd.DataFrame:
    rows = []
    for name, df in out.items():
        rows.append(dict(scenario=name, cost_mean=df.total_cost.mean(), cost_p05=df.total_cost.quantile(.05),
                         cost_p95=df.total_cost.quantile(.95), fill_mean=df.fill_rate.mean(),
                         fill_p05=df.fill_rate.quantile(.05), fill_FOODS_3=df.fill_FOODS_3.mean(),
                         lost_cost=df.cost_lost.mean(), holding_dc=df.cost_holding_dc.mean(),
                         ending_inventory=df.ending_inventory_value.mean()))
    t = pd.DataFrame(rows).set_index("scenario")
    t["delta_cost_vs_base"] = t["cost_mean"] - t.loc["base", "cost_mean"]
    t["delta_fill_vs_base"] = t["fill_mean"] - t.loc["base", "fill_mean"]
    return t
