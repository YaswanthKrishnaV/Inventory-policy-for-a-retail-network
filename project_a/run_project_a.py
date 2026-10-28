"""Project A end to end.

    python -m project_a.run_project_a --cutoff 1885            # full run
    python -m project_a.run_project_a --cutoff 1857 --fast     # fewer sims, no backtest/tuning

Everything is computed from data with d <= cutoff, except the replay itself, which plays the
56 days after the cutoff (and re-forecasts every 28 days with the same, already-trained model).

Steps
  1. data audit (status of every series-day, detector validation)      -> reports/data_audit.md
  2. forecast backtests at cutoff - 84/56/28                           -> reports/backtest.md
  3. policy knob tuning on 56-day replays at cutoff - 56/84/112         -> reports/tuning_<c>.json
  4. 28-day probabilistic forecast at the cutoff                       -> project_a/forecasts.parquet
  5. Monte Carlo replays: baseline, recommended, ablations, sensitivity  -> reports/results_<c>.json/.md
  6. stress test: FOODS_3 lead times x2 for 3 weeks, with/without warning
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import audit_data, backtest, report, stress, simulate as sm
from .policy import BaselinePolicy, RecommendedPolicy, build_forecasts
from .src import forecast as fc
from .src.config import load_config, path
from .src.data import load_panel

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _factory(policy_cls, *args, **kw):
    if policy_cls is BaselinePolicy:
        return lambda: BaselinePolicy(*args)
    pol = RecommendedPolicy(*args, **kw)
    seed = args[2]["seed"]

    def make():
        pol.lt_rng = np.random.default_rng(seed + 99)
        return pol
    return make


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=int, default=None, help="last training day d (default from config)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-sims", type=int, default=None)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--skip-tune", action="store_true", help="use knobs from config instead of tuning")
    ap.add_argument("--knobs", default=None, help="JSON file with knobs (e.g. reports/tuning_1885.json)")
    ap.add_argument("--fast", action="store_true", help="20 sims, skip backtest and tuning")
    a = ap.parse_args(argv)
    T0 = time.time()
    cfg = load_config(a.config)
    c = a.cutoff or cfg["calendar"]["default_cutoff"]
    if c + 28 > cfg["calendar"]["last_data_day"]:
        raise SystemExit(f"cutoff d_{c} leaves fewer than 28 test days (data ends d_{cfg['calendar']['last_data_day']})")
    n_sims = a.n_sims or (20 if a.fast else cfg["simulation"]["n_sims"])
    seed = cfg["simulation"]["seed"]
    rep = path(cfg, "reports_dir")
    log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
    timings = {}

    log(f"Project A — cutoff d_{c}")
    panel = load_panel(cfg)

    # 1) data audit
    if not a.skip_audit:
        t = time.time(); audit_data.main(["--cutoff", str(c)] + (["--config", a.config] if a.config else [])); timings["audit"] = time.time() - t
        log("data audit written")

    # 2) forecast backtests (28-day windows ending <= cutoff)
    if not (a.skip_backtest or a.fast):
        t = time.time()
        cuts = [c - o for o in cfg["calendar"]["backtest_offsets"]]
        res = backtest.run(cfg, cuts, log=log)
        backtest.write_report(res, cfg)
        timings["backtest"] = time.time() - t
        log(f"backtests at {cuts} written")

    # 3) policy knobs
    knobs = copy.deepcopy(cfg["policy"]["knobs"])
    if a.knobs:
        knobs = json.load(open(a.knobs)).get("best_knobs", knobs)
    elif not (a.skip_tune or a.fast):
        from .tune import tune
        t = time.time()
        knobs, _ = tune(panel, c, cfg, log=log)
        timings["tune"] = time.time() - t
    log(f"knobs: {json.dumps(knobs)}")

    # 4) forecast at the cutoff -> deliverable file
    t = time.time()
    net = sm.build_network(panel, c, cfg)
    frame = sm.test_demand_frame(panel, c, cfg)
    blocks, model = build_forecasts(panel, net, cfg, log=log)
    table = fc.forecast_table(blocks[0], panel, cfg)
    out_file = fc.write_forecasts(table, path(cfg, "out_dir") / "forecasts.parquet")
    timings["forecast"] = time.time() - t
    log(f"forecasts written: {out_file.name} ({len(table):,} rows)")

    # 5) Monte Carlo replays
    t = time.time()
    runs = {}
    def mc(name, make, **kw):
        df, r = sm.monte_carlo(make, net, panel, cfg, n_sims, seed, frame=frame, **kw)
        runs[name] = (df, r[0])
        log(f"  {name:34s} cost {df.total_cost.mean():>10,.0f}  fill {df.fill_rate.mean():.4f}  "
            f"P(fill>={cfg['fill_rate']['target']:.0%}) {(df.fill_rate >= cfg['fill_rate']['target']).mean():.2f}")
    mc("baseline", _factory(BaselinePolicy, panel, net, cfg))
    mc("recommended", _factory(RecommendedPolicy, panel, net, cfg, blocks, knobs, seed=cfg["seed"]))
    # ablations: where do the savings come from?
    k2 = dict(knobs, prebuild=False)
    mc("ablation: no weekend pre-build", _factory(RecommendedPolicy, panel, net, cfg, blocks, k2, seed=cfg["seed"]))
    blocks_noadj, _ = build_forecasts(panel, net, cfg, log=log, model=model, adjust=False)
    mc("ablation: no forecast adjustments", _factory(RecommendedPolicy, panel, net, cfg, blocks_noadj, knobs, seed=cfg["seed"]))
    blocks_raw, _ = build_forecasts(panel, net, cfg, log=log, raw=True)
    mc("ablation: raw-sales forecast", _factory(RecommendedPolicy, panel, net, cfg, blocks_raw, knobs, seed=cfg["seed"]))
    # evaluation sensitivity: what if censored test days were replayed as recorded?
    df_r, r_r = sm.monte_carlo(_factory(RecommendedPolicy, panel, net, cfg, blocks, knobs, seed=cfg["seed"]),
                               net, panel, cfg, max(5, n_sims // 5), seed, frame=frame, demand_mode="recorded")
    runs["sensitivity: recorded test sales as demand"] = (df_r, r_r[0])
    df_b, r_b = sm.monte_carlo(_factory(BaselinePolicy, panel, net, cfg), net, panel, cfg, max(5, n_sims // 5), seed,
                               frame=frame, demand_mode="recorded")
    runs["sensitivity: baseline, recorded test sales"] = (df_b, r_b[0])
    timings["simulate"] = time.time() - t

    # 6) stress test
    t = time.time()
    st_out, st_daily = stress.run_stress(_factory(RecommendedPolicy, panel, net, cfg, blocks, knobs, seed=cfg["seed"]),
                                         net, panel, cfg, n_sims, seed, frame, log=log)
    st_bl, _ = stress.run_stress(_factory(BaselinePolicy, panel, net, cfg), net, panel, cfg, max(5, n_sims // 5), seed, frame, log=log)
    timings["stress"] = time.time() - t

    timings["total"] = time.time() - T0
    report.write(c, cfg, net, runs, st_out, st_daily, st_bl, knobs, blocks, frame, timings)
    log(f"done in {timings['total'] / 60:.1f} min — see reports/results_{c}.md")


if __name__ == "__main__":
    main()
