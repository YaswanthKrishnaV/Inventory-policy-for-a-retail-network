"""Rolling-origin backtests of the 28-day probabilistic forecast.

    python -m project_a.backtest            # cutoffs from config (default d_1801, d_1829, d_1857)

For each cutoff c the model is trained on d <= c only, forecasts d_{c+1}..d_{c+28}, and is scored
against demand on *uncensored* days (status OK in a recovery run on all data). Models compared:

* recovered  - GBM on recovered demand (the proposed model)
* raw        - same GBM trained on raw recorded sales (what you'd get ignoring stockouts)
* ma28       - 28-day moving average of recorded sales, Poisson spread (the baseline's view)

Metrics: bias = actual/forecast (>1 = under-forecast), WAPE, mean pinball loss over the 19
quantiles scaled by mean demand, and empirical coverage of the 50% / 90% intervals — at store
level and at DC level (DC = sum of the item's 4 store paths).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .src import demand_recovery as dr
from .src import forecast as fc
from .src.config import load_config, path
from .src.data import load_panel

warnings.filterwarnings("ignore", category=RuntimeWarning)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, GRID, INK = "#6b6a63", "#e6e5e0", "#1f1f1e"


def pinball(q_pred: np.ndarray, y: np.ndarray, qs) -> np.ndarray:
    """Mean pinball loss per quantile. q_pred (Q, n), y (n,)."""
    qs = np.asarray(qs)[:, None]
    d = y[None, :] - q_pred
    return np.mean(np.maximum(qs * d, (qs - 1) * d), axis=1)


def score(paths: np.ndarray, mu: np.ndarray, y: np.ndarray, ok: np.ndarray, qs, meta) -> dict:
    """Store- and DC-level scores on OK cells."""
    out = {}
    qp = fc.quantiles(paths, qs)                        # (Q, N, H)
    yv, mv, qv = y[ok], mu[ok], qp[:, ok]
    out["store"] = _metrics(yv, mv, qv, qs)
    dept = np.repeat(meta["dept_id"].to_numpy()[:, None], y.shape[1], 1)[ok]
    out["store_by_dept"] = {d: _metrics(yv[dept == d], mv[dept == d], qv[:, dept == d], qs)
                            for d in sorted(np.unique(dept))}
    # DC: only item-days where all 4 stores are observed
    N, H = y.shape
    ok4 = ok.reshape(N // 4, 4, H).all(1)
    ydc = y.reshape(N // 4, 4, H).sum(1)
    mdc = mu.reshape(N // 4, 4, H).sum(1)
    qdc = fc.quantiles(fc.to_dc(paths), qs)
    out["dc"] = _metrics(ydc[ok4], mdc[ok4], qdc[:, ok4], qs)
    out["coverage_curve_store"] = [float((yv <= qv[k]).mean()) for k in range(len(qs))]
    out["coverage_curve_dc"] = [float((ydc[ok4] <= qdc[k][ok4]).mean()) for k in range(len(qs))]
    return out


def _metrics(y, mu, q, qs) -> dict:
    qs = list(qs)
    i05, i25, i75, i95 = qs.index(0.05), qs.index(0.25), qs.index(0.75), qs.index(0.95)
    ym = max(y.mean(), 1e-9)
    return dict(n=int(len(y)), mean_demand=float(y.mean()),
                bias=float(y.sum() / max(mu.sum(), 1e-9)),
                wape=float(np.abs(y - mu).sum() / max(y.sum(), 1e-9)),
                spl=float(pinball(q, y, qs).mean() / ym),
                cov50=float(((y >= q[i25]) & (y <= q[i75])).mean()),
                cov90=float(((y >= q[i05]) & (y <= q[i95])).mean()))


def ma28_forecast(panel, origin, H, rec):
    """Baseline view: mean of recorded sales over the last 28 days, flat over the horizon."""
    x = panel.sales[:, origin - 28:origin].astype(np.float32)
    mu = np.repeat(x.mean(1, keepdims=True), H, 1)
    mu[rec.status[:, origin - 1] == dr.DORMANT] = 0
    return mu


def run(cfg, cutoffs, log=print):
    p = load_panel(cfg)
    H = cfg["calendar"]["forecast_horizon"]
    qs = cfg["forecast"]["quantiles"]
    S = cfg["forecast"]["n_paths"]
    full = dr.recover(p, cfg["calendar"]["last_data_day"], cfg)
    results = []
    for c in cutoffs:
        log(f"cutoff d_{c}")
        y = full.demand[:, c:c + H]
        ok = full.status[:, c:c + H] == dr.OK
        row = {"cutoff": c}
        for name, raw in (("recovered", False), ("raw", True)):
            t0 = time.time()
            f, _ = fc.run_forecast(p, c, cfg, raw=raw, log=log)
            paths = f.paths(S, cfg["seed"] + c)
            row[name] = score(paths, f.mu, y, ok, qs, p.meta)
            row[name]["seconds"] = time.time() - t0
            row[name]["dispersion"] = dict(inv_r=f.disp.inv_r.round(4).tolist(), s_p2=f.disp.s_p2.round(4).tolist())
            del paths
        rec_c = dr.recover(p, c, cfg)
        mu = ma28_forecast(p, c, H, rec_c)
        paths = np.random.default_rng(c).poisson(np.broadcast_to(mu, (S,) + mu.shape)).astype(np.int16)
        row["ma28"] = score(paths, mu, y, ok, qs, p.meta)
        del paths
        results.append(row)
        for m in ("recovered", "raw", "ma28"):
            s, d = row[m]["store"], row[m]["dc"]
            log(f"  {m:9s} store bias {s['bias']:.3f} wape {s['wape']:.3f} spl {s['spl']:.3f} "
                f"cov50 {s['cov50']:.2f} cov90 {s['cov90']:.2f} | DC bias {d['bias']:.3f} "
                f"spl {d['spl']:.3f} cov50 {d['cov50']:.2f} cov90 {d['cov90']:.2f}")
    return results


def write_report(results, cfg):
    rep = path(cfg, "reports_dir")
    (rep / "backtest.json").write_text(json.dumps(results, indent=1))
    qs = cfg["forecast"]["quantiles"]
    lines = ["# Forecast backtests", "",
             "Auto-generated by `python -m project_a.backtest`. Scored on uncensored days only.",
             "bias = actual / forecast (>1 means under-forecast); spl = mean pinball loss over 19 quantiles / mean demand.", ""]
    for level in ("store", "dc"):
        lines += [f"## {level.upper()} level", "",
                  "| cutoff | model | bias | WAPE | scaled pinball | 50% cov | 90% cov |", "|---|---|---|---|---|---|---|"]
        for r in results:
            for m in ("recovered", "raw", "ma28"):
                s = r[m][level]
                lines.append(f"| d_{r['cutoff']} | {m} | {s['bias']:.3f} | {s['wape']:.3f} | {s['spl']:.3f} | {s['cov50']:.2f} | {s['cov90']:.2f} |")
        lines.append("")
    lines += ["## Store level by department (mean over cutoffs)", "",
              "| dept | model | bias | WAPE | scaled pinball | 90% cov |", "|---|---|---|---|---|---|"]
    for d in ("FOODS_1", "FOODS_2", "FOODS_3"):
        for m in ("recovered", "raw", "ma28"):
            v = [r[m]["store_by_dept"][d] for r in results]
            lines.append(f"| {d} | {m} | {np.mean([x['bias'] for x in v]):.3f} | {np.mean([x['wape'] for x in v]):.3f} | "
                         f"{np.mean([x['spl'] for x in v]):.3f} | {np.mean([x['cov90'] for x in v]):.2f} |")
    lines += ["", "![](fig_calibration.png)", ""]
    (rep / "backtest.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, lvl, title in ((axes[0], "store", "Store level"), (axes[1], "dc", "DC level")):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        for k, m in enumerate(("recovered", "raw", "ma28")):
            cov = np.mean([r[m][f"coverage_curve_{lvl}"] for r in results], 0)
            ax.plot(qs, cov, color=SERIES[k], lw=2, marker="o", ms=4, label=m)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("nominal quantile", fontsize=8, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("share of actuals below quantile", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Quantile calibration, mean over backtest cutoffs (on the diagonal = calibrated)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(rep / "fig_calibration.png", dpi=140)
    plt.close(fig)
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cutoffs", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cut = a.cutoffs or cfg["calendar"]["backtest_cutoffs"]
    res = run(cfg, cut)
    write_report(res, cfg)


if __name__ == "__main__":
    main()
