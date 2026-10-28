"""Data audit for Project A: status classification, stockout detector validation, figures.

    python -m project_a.audit_data --cutoff 1885

Writes reports/data_audit.md (+ .json, figures). Numbers feed DATA_ISSUES.md.
Uses data with d <= cutoff only, except the "test window" section which, like the simulator,
needs to know which recorded test-period sales are themselves censored.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .src import demand_recovery as dr
from .src.config import load_config, path
from .src.data import load_panel

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_store_totals(p, rec, out):
    cal = p.cal.iloc[:rec.end]
    stores = p.meta["store_id"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, s in enumerate(sorted(np.unique(stores))):
        tot = pd.Series(p.sales[stores == s, :rec.end].sum(0), index=cal["date"])
        wk = tot.rolling(7, center=True).mean()
        ax.plot(wk.index, wk.values, color=SERIES[k], lw=1.6, label=s)
        ax.annotate(s, (wk.index[-4], wk.values[-4]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
        bad = np.flatnonzero(np.isin(rec.status[stores == s][0], [dr.DISRUPTED, dr.CLOSED]))
        ax.scatter(cal["date"].iloc[bad], tot.iloc[bad], s=16, color=SERIES[k], edgecolor="white", lw=0.8, zorder=3)
    ax.set_xlim(pd.Timestamp("2014-01-01"), cal["date"].iloc[-1] + pd.Timedelta(days=40))
    ax.set_title("Store daily FOODS units (7-day mean); dots = closed / disrupted days", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_examples(p, rec, out, n=3, seed=3):
    rng = np.random.default_rng(seed)
    st = rec.status
    last = slice(rec.end - 365, rec.end)
    so = (st[:, last] == dr.STOCKOUT).sum(1)
    vel = p.sales[:, last].mean(1)
    cand = np.flatnonzero((so >= 20) & (vel > 2))
    pick = rng.choice(cand, n, replace=False)
    dates = p.cal["date"].iloc[:rec.end].to_numpy()[last]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    for ax, i in zip(axes, pick):
        x = p.sales[i, last].astype(float)
        cen = st[i, last] == dr.STOCKOUT
        ax.bar(dates, x, width=1.0, color=SERIES[0], label="recorded sales")
        ax.bar(dates[cen], rec.demand[i, last][cen], width=1.0, color=SERIES[1], alpha=0.85, label="imputed demand (stockout)")
        dor = st[i, last] == dr.DORMANT
        if dor.any():
            ax.bar(dates[dor], np.full(dor.sum(), x.max() * 0.04), width=1.0, color=MUTED, label="dormant")
        ax.set_title(p.meta["id"].iloc[i].replace("_evaluation", ""), loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cutoff = a.cutoff or cfg["calendar"]["default_cutoff"]
    rep = path(cfg, "reports_dir")
    p = load_panel(cfg)
    rec = dr.recover(p, cutoff, cfg)          # d <= cutoff  <=> t < cutoff
    summ = dr.summarise(p, rec)
    inj = dr.injection_test(p, rec, cfg)
    nul = dr.null_test(p, rec, cfg)

    # status counts at the cutoff day (what the policy inherits)
    on_cut = {dr.STATUS_NAMES[c]: int((rec.status[:, -1] == c).sum()) for c in range(7)}

    # test window: how much of the recorded test-period sales is itself censored
    test_end = min(cutoff + cfg["calendar"]["test_days"], cfg["calendar"]["last_data_day"])
    full = dr.recover(p, test_end, cfg)
    tw = full.status[:, cutoff:test_end]
    tw_sales = p.sales[:, cutoff:test_end].sum()
    tw_dem = full.demand[:, cutoff:test_end].sum()
    test = dict(window=[cutoff + 1, test_end],
                status_share={dr.STATUS_NAMES[c]: float((tw == c).mean()) for c in range(7)},
                recorded_units=int(tw_sales), recovered_units=float(tw_dem),
                hidden_share=float(1 - tw_sales / tw_dem))

    # network capacity context
    S = p.sales[:, :cutoff]
    W = cfg["capacity"]["dc_outbound_window_days"]
    net_raw = S[:, -W:].sum(0)
    net_rec = rec.demand[:, -W:].sum(0)
    cap = cfg["capacity"]["dc_outbound_multiple"] * net_raw.mean()
    capacity = dict(avg_daily_recorded=float(net_raw.mean()), avg_daily_recovered=float(net_rec.mean()),
                    cap_units_per_day=float(cap),
                    days_recorded_above_cap=int((net_raw > cap).sum()),
                    days_recovered_above_cap=int((net_rec > cap).sum()),
                    recovered_by_dow={int(k): float(v) for k, v in
                                      pd.Series(net_rec).groupby(p.cal["dow"].to_numpy()[cutoff - W:cutoff]).mean().items()})

    out = dict(cutoff=cutoff, n_series=p.n_series, summary=summ, on_cutoff_day=on_cut,
               test_window=test, capacity=capacity,
               injection=inj.to_dict("records"), null=nul.astype({"band": str}).to_dict("records"))
    (rep / "data_audit.json").write_text(json.dumps(out, indent=1, default=float))
    fig_store_totals(p, rec, rep / "fig_store_totals.png")
    fig_examples(p, rec, rep / "fig_stockout_examples.png")

    pct = lambda v: f"{100 * v:.1f}%"
    md = [f"# Data audit (cutoff d_{cutoff})", "",
          "Auto-generated by `python -m project_a.audit_data`. Interpretation lives in DATA_ISSUES.md.", "",
          "## Status of every series-day (d_1 .. cutoff)", "",
          "| status | share |", "|---|---|"]
    md += [f"| {k} | {pct(v)} |" for k, v in summ["status_share"].items()]
    md += ["", "## Last 365 days before cutoff", "", "| group | stockout days | dormant days | demand hidden by stockouts |", "|---|---|---|---|"]
    for key in ["dept_id", "store_id"]:
        for g, v in summ["last_365"][key].items():
            md.append(f"| {g} | {pct(v['stockout_day_share'])} | {pct(v['dormant_share'])} | {pct(v['hidden_demand_share'])} |")
    md += ["", f"## Status on the cutoff day (inherited by the policy)", "", json.dumps(on_cut), "",
           f"## Test window d_{test['window'][0]}..d_{test['window'][1]} (recorded sales are censored too)", "",
           f"Recorded units {test['recorded_units']:,}; recovered demand {test['recovered_units']:,.0f}; "
           f"hidden share {pct(test['hidden_share'])}. Stockout days {pct(test['status_share']['stockout'])}, "
           f"dormant {pct(test['status_share']['dormant'])}.", "",
           "## DC outbound capacity", "",
           f"Cap = {cfg['capacity']['dc_outbound_multiple']} x {capacity['avg_daily_recorded']:,.0f} = "
           f"{capacity['cap_units_per_day']:,.0f} units/day. Recovered demand averages "
           f"{capacity['avg_daily_recovered']:,.0f}/day. Days above cap in last {W}: recorded "
           f"{capacity['days_recorded_above_cap']}, recovered {capacity['days_recovered_above_cap']}.", "",
           "Recovered network demand by weekday (Mon=0): " +
           ", ".join(f"{k}: {v:,.0f}" for k, v in capacity["recovered_by_dow"].items()), "",
           "## Detector validation", "", "Recall on planted stockouts (share of planted days flagged):", "",
           inj.pivot(index="band", columns="run_len", values="recall").round(2).to_markdown(), "",
           "Imputed / true units on fully detected planted runs:", "",
           inj.pivot(index="band", columns="run_len", values="imputed_over_true").round(2).to_markdown(), "",
           "False-alarm share of days on simulated stockout-free series:", "",
           nul.round(4).to_markdown(index=False), "",
           "![](fig_store_totals.png)", "", "![](fig_stockout_examples.png)", ""]
    (rep / "data_audit.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
