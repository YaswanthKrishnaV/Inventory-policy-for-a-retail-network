"""Tune the recommended policy's knobs by simulation on past windows only.

For a cutoff c, tuning replays use cutoffs c - offset (config calendar.tuning_offsets, default
56/84/112 days), so every tuning window ends on or before c: nothing from the evaluation window
is ever seen. Each replay uses forecasts trained on data up to its own tuning cutoff.

Search: coordinate descent over a small grid (store quantile, DC quantile per department,
store window), objective = mean total cost across tuning cutoffs, subject to every tuning
cutoff reaching the fill-rate target with a safety margin (penalty otherwise).
"""
from __future__ import annotations

import copy
import itertools
import json
import time

import numpy as np

from . import simulate as sm
from .policy import RecommendedPolicy, build_forecasts
from .src.config import path


def _evaluate(env, knobs, cfg, n_sims, seed):
    costs, fills = [], []
    for e in env:
        pol = RecommendedPolicy(e["panel"], e["net"], cfg, e["forecasts"], knobs, seed=cfg["seed"])
        def make():
            pol.lt_rng = np.random.default_rng(cfg["seed"] + 99)
            return pol
        df, _ = sm.monte_carlo(make, e["net"], e["panel"], cfg, n_sims, seed, frame=e["frame"])
        costs.append(df["total_cost"].mean())
        fills.append(df["fill_rate"].mean())
    return float(np.mean(costs)), [float(f) for f in fills]


def objective(cost, fills, target, margin, penalty_per_point):
    short = sum(max(0.0, target + margin - f) for f in fills)
    return cost + penalty_per_point * 100 * short


def tune(panel, cutoff, cfg, log=print):
    tc = cfg["tuning"]
    env = []
    for off in cfg["calendar"]["tuning_offsets"]:
        c = cutoff - off
        net = sm.build_network(panel, c, cfg)
        forecasts, _ = build_forecasts(panel, net, cfg, log=log)
        env.append(dict(panel=panel, net=net, forecasts=forecasts, frame=sm.test_demand_frame(panel, c, cfg), cutoff=c))
    knobs = copy.deepcopy(cfg["policy"]["knobs"])
    target = cfg["fill_rate"]["target"]
    hist = []

    def score(k):
        t0 = time.time()
        cost, fills = _evaluate(env, k, cfg, tc["n_sims"], cfg["simulation"]["seed"] + 1000)
        obj = objective(cost, fills, target, tc["fill_margin"], tc["penalty_per_point"])
        hist.append(dict(knobs=copy.deepcopy(k), cost=cost, fills=fills, objective=obj, seconds=time.time() - t0))
        log(f"    cost {cost:,.0f} fills {[round(f, 3) for f in fills]} obj {obj:,.0f}  "
            f"{json.dumps({kk: k[kk] for kk in ('alpha_store', 'beta_dc', 'store_window')})}")
        return obj

    best = score(knobs)
    for rnd in range(tc["rounds"]):
        improved = False
        for name, grid in tc["grid"].items():
            per_dept = name in ("alpha_store", "beta_dc")
            targets = [(name, d) for d in ("FOODS_1", "FOODS_2", "FOODS_3")] if per_dept else [(name, None)]
            for key, dept in targets:
                cur = knobs[key][dept] if dept else knobs[key]
                for v in grid:
                    if v == cur:
                        continue
                    k = copy.deepcopy(knobs)
                    if dept:
                        k[key][dept] = v
                    else:
                        k[key] = v
                    log(f"  round {rnd} {key}{'[' + dept + ']' if dept else ''} = {v}")
                    o = score(k)
                    if o < best - 1e-6:
                        best, knobs, improved = o, k, True
        if not improved:
            break
    out = dict(cutoff=cutoff, tuning_cutoffs=[e["cutoff"] for e in env], best_knobs=knobs, best_objective=best, history=hist)
    (path(cfg, "reports_dir") / f"tuning_{cutoff}.json").write_text(json.dumps(out, indent=1))
    return knobs, out
