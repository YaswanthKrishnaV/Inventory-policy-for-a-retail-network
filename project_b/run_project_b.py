"""Project B end to end.

    python -m project_b.run_project_b                              # default hold-out: final 13 weeks
    python -m project_b.run_project_b --holdout-start 2023-07-03   # any 13-week hold-out window
    python -m project_b.run_project_b --fast                       # shorter chains, fewer variants

The model is trained strictly on weeks before --holdout-start; every reported number (ROI, budget,
experiment design) comes from that fit, and the 13 held-out weeks are only used to score the forecast.

Outputs (project_b/): posterior.nc, roi_table.csv, budget_allocation.csv, forecast_13w.csv
Reports (reports/project_b/):   results.md / results.json, figures, data audit, model comparison
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time

import numpy as np
import pandas as pd

from . import audit, experiment_design, optimise, report
from .src import checks, roi as roi_mod
from .src.data import load, load_config, out_path, set_holdout
from .src.fit import fit, flat
from .src.model import Model, build_design


def calibration_spec(d, des, cfg, se_mult):
    ex = cfg["experiment"]
    se = (ex["ci90"][1] - ex["ci90"][0]) / 2 / 1.645
    return dict(channel=d.channels.index(ex["channel"]), geos=np.isin(d.geos, ex["geos"]),
                t0=d.week_index(ex["start"]), t1=d.week_index(ex["start"]) + ex["weeks"], tail=ex["tail_weeks"],
                obs=ex["iroas"], se=se * se_mult)


def fit_variant(name, d, cfg, log, overrides=None, se_mult=None, national=False, chains=None, warmup=None, draws=None):
    c = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        c[k] = {**c[k], **v}
    des = build_design(d, c, national=national)
    cal = calibration_spec(d, des, c, se_mult) if se_mult is not None else None
    m = Model(des, c, calibration=cal, fixed_eta=1e-3 if national else None)
    t = time.time()
    post, diag = fit(m, c, chains=chains, warmup=warmup, draws=draws, log=None)
    worst = max(diag.values(), key=lambda v: v["rhat"])["rhat"]
    log(f"  fitted {name:28s} in {time.time() - t:5.0f}s  worst R-hat {worst:.3f}  "
        f"min ESS {min(v['ess'] for v in diag.values()):.0f}")
    return m, post, diag


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-start", default=None, help="first hold-out week (YYYY-MM-DD); default: final 13 weeks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fast", action="store_true", help="short chains; skip sensitivity variants")
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)   # tiny chains, for testing the pipeline
    ap.add_argument("--reuse-fits", action="store_true", help="load fitted posteriors from cache/ instead of refitting")
    a = ap.parse_args(argv)
    T0 = time.time()
    log = lambda s: print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)
    cfg = load_config(a.config)
    hs = a.holdout_start or cfg["holdout"]["default_start"]
    if a.fast:
        cfg["sampler"].update(chains=2, warmup=500, draws=400)
        cfg["next_experiment"]["n_sims"] = 100
    if a.smoke:
        cfg["sampler"].update(chains=2, warmup=20, draws=20)
        cfg["next_experiment"].update(n_sims=5, n_treat_geos=[8], durations_weeks=[6])
    d = load(cfg)
    t0, t1 = set_holdout(d, hs, cfg["holdout"]["weeks"])
    log(f"Project B — train on {d.dates[0].date()}..{d.dates[t0 - 1].date()} ({t0} weeks); "
        f"hold-out {d.dates[t0].date()}..{d.dates[t1 - 1].date()}")
    out, rep = out_path(cfg, "out_dir"), out_path(cfg, "reports_dir")
    audit.main([])

    # ---------------------------------------------------------------- fits
    budget_variant = cfg["experiment"]["use_for_budget"]
    rb = cfg["sampler"]["robustness"]
    rkw = dict(chains=rb["chains"], warmup=rb["warmup"], draws=rb["draws"])
    specs = {"uncalibrated": ("geo, uncalibrated (main)", {}, None, False, {})}
    for name, mult in cfg["experiment"]["variants"].items():
        if mult is not None:
            specs[name] = (f"geo, calibrated ({name})", {}, mult, False, {})
    specs["national"] = ("national totals", {}, None, True, {})
    if not (a.fast or a.smoke):
        specs["weekly_fe"] = ("geo, weekly fixed effects", {"model": {"baseline": "weekly_fe"}}, None, False, rkw)
        specs["no_sentiment"] = ("geo, without sentiment", {"model": {"controls": ["competitor_sales_control"]}}, None, False, rkw)
        specs["no_organic"] = ("geo, without organic", {"model": {"include_organic": False}}, None, False, rkw)
    import pickle as _pk
    cache_file = out_path(cfg, "cache_dir") / f"fits_{hs}{'_fast' if a.fast else ''}{'_smoke' if a.smoke else ''}.pkl"
    cached = _pk.loads(cache_file.read_bytes()) if (a.reuse_fits and cache_file.exists()) else {}
    fits = {}
    for name, (label, ov, mult, nat, kw) in specs.items():
        if name in cached:
            c = copy.deepcopy(cfg)
            for k, v in ov.items():
                c[k] = {**c[k], **v}
            des = build_design(d, c, national=nat)
            m = Model(des, c, calibration=calibration_spec(d, des, c, mult) if mult is not None else None,
                      fixed_eta=1e-3 if nat else None)
            fits[name] = (m, *cached[name])
            log(f"  loaded {label} from cache")
        else:
            fits[name] = fit_variant(label, d, cfg, log, overrides=ov, se_mult=mult, national=nat, **kw)
    cache_file.write_bytes(_pk.dumps({k: (v[1], v[2]) for k, v in fits.items()}))
    main_m, main_post, main_diag = fits[budget_variant]
    f_main = flat(main_post)

    # ---------------------------------------------------------------- ROI for every variant
    n_roi = 600 if not (a.fast or a.smoke) else 40
    w0 = max(0, t0 - cfg["roi"]["window_weeks"])
    roi_tabs, roi_draws = {}, {}
    for name, (m, post, diag) in fits.items():
        f, _ = roi_mod.draws_subset(flat(post), n_roi, seed=1)
        rd = roi_mod.roi_draws(m, f, w0, t0, cfg["roi"]["mroi_step"])
        roi_draws[name] = rd
        roi_tabs[name] = rd
    log("ROI computed for all variants")

    # ---------------------------------------------------------------- hold-out forecast (main model)
    f_sub, idx = roi_mod.draws_subset(f_main, 400 if not a.smoke else 30, seed=2)
    pred = checks.predict(main_m, f_sub, noise=True, seed=3)
    acc = checks.accuracy(d.kpi, pred, np.arange(t0, t1))
    acc_train = checks.accuracy(d.kpi, pred, np.arange(max(0, t0 - 52), t0))
    rev_pred = pred * d.rpk[None]
    fc_rows = []
    for g in range(d.G):
        for t in range(t0, t1):
            v = rev_pred[:, g, t]
            fc_rows.append(dict(geo=d.geos[g], week=d.dates[t].date(), actual_conversions=d.kpi[g, t],
                                pred_conversions_mean=pred[:, g, t].mean(), pred_conversions_p05=np.quantile(pred[:, g, t], .05),
                                pred_conversions_p95=np.quantile(pred[:, g, t], .95), actual_revenue=d.revenue[g, t],
                                pred_revenue_mean=v.mean(), pred_revenue_p05=np.quantile(v, .05), pred_revenue_p95=np.quantile(v, .95)))
    nat = pred[:, :, t0:t1].sum(1)
    for k, t in enumerate(range(t0, t1)):
        fc_rows.append(dict(geo="NATIONAL", week=d.dates[t].date(), actual_conversions=d.kpi[:, t].sum(),
                            pred_conversions_mean=nat[:, k].mean(), pred_conversions_p05=np.quantile(nat[:, k], .05),
                            pred_conversions_p95=np.quantile(nat[:, k], .95), actual_revenue=d.revenue[:, t].sum(),
                            pred_revenue_mean=(rev_pred[:, :, t].sum(1)).mean(), pred_revenue_p05=np.quantile(rev_pred[:, :, t].sum(1), .05),
                            pred_revenue_p95=np.quantile(rev_pred[:, :, t].sum(1), .95)))
    pd.DataFrame(fc_rows).to_csv(out / "forecast_13w.csv", index=False)
    log(f"hold-out accuracy: national WAPE {acc['national']['wape']:.3f}, R2 {acc['national']['r2']:.3f}, "
        f"90% coverage {acc['national']['coverage90']:.2f}; geo WAPE {acc['geo']['wape']:.3f}")

    # ---------------------------------------------------------------- budget (main model, data to the end of training)
    bc = cfg["budget"]
    f_b, _ = roi_mod.draws_subset(f_main, 400 if not (a.fast or a.smoke) else 30, seed=4)
    # history = all weeks observed before the plan starts (the plan follows the last data week)
    curves = roi_mod.response_curves(main_m, f_b, optimise.GRID, horizon=bc["horizon_weeks"], hist_end=d.T)
    last13 = roi_mod.last_spend(main_m, bc["horizon_weeks"], hist_end=d.T)
    scen = {}
    for B in bc["totals"]:
        scen[B] = optimise.scenario(curves, last13, B, bc["bound"])
        if not scen[B]["feasible"]:
            scen[f"{B}_relaxed"] = optimise.scenario(curves, last13, B, bc["bound"], relaxed=True)
    voi = optimise.value_of_information(curves, last13, bc["main_total"], bc["bound"])
    log(f"budget: 18M uplift vs today's mix {np.mean(scen[bc['main_total']]['uplift']) / 1e3:,.0f}k; "
        f"EVPI {voi['evpi'] / 1e3:,.0f}k; per channel {np.round(voi['evppi'] / 1e3, 0).tolist()}")

    # calibrated-model budget (sensitivity): does trusting the vendor change the allocation?
    alt = {}
    for name in ("full", "discounted"):
        if name in fits:
            fa, _ = roi_mod.draws_subset(flat(fits[name][1]), 200 if not a.smoke else 20, seed=4)
            ca = roi_mod.response_curves(fits[name][0], fa, optimise.GRID, horizon=bc["horizon_weeks"], hist_end=d.T)
            alt[name] = optimise.scenario(ca, last13, bc["main_total"], bc["bound"])

    # ---------------------------------------------------------------- next experiment
    top = int(np.argmax(voi["evppi"]))
    f_e, _ = roi_mod.draws_subset(f_main, 200 if not a.smoke else 20, seed=5)
    design = experiment_design.design_grid(main_m, f_e, top, cfg, seed=6)
    log(f"experiment design for {d.channels[top]}: {len(design)} designs simulated")

    # ---------------------------------------------------------------- write outputs
    checks.write_netcdf(out / "posterior.nc", main_post, main_m,
                        dict(holdout_start=hs, variant=budget_variant, train_weeks=t0))
    report.write(cfg, d, t0, t1, fits, roi_draws, acc, acc_train, scen, alt, voi, design, top, last13, curves, out, rep,
                 prior_share=checks.prior_predictive(main_m, n=150)[0], elapsed=time.time() - T0, pred=pred)
    log(f"done in {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
