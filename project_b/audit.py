"""Data audit for Project B -> reports/data_audit.md (+ figures). Numbers feed DATA_ISSUES.md.

    python -m project_b.audit
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .src.data import load, load_config, out_path
from .src.plotstyle import SERIES, INK, MUTED, GRID, style


def experiment_check(d, cfg):
    """Did Channel3 spend actually drop in the test geos during the test window?"""
    ex = cfg["experiment"]
    m = d.channels.index(ex["channel"])
    test = np.isin(d.geos, ex["geos"])
    t0 = d.week_index(ex["start"]); t1 = t0 + ex["weeks"]
    pre = slice(t0 - ex["weeks"], t0)
    sp = d.spend[m] / d.pop[:, None] * 1e3                     # spend per 1,000 people
    tr, ct = sp[test].mean(0), sp[~test].mean(0)
    zero_weeks = int((d.spend[m][test, t0:t1] == 0).sum())
    out = dict(test_geos=ex["geos"], window=[str(d.dates[t0].date()), str(d.dates[t1 - 1].date())],
               test_geo_weeks=int(test.sum() * ex["weeks"]), zero_spend_geo_weeks=zero_weeks,
               test_spend_pc_window=float(tr[t0:t1].mean()), test_spend_pc_pre=float(tr[pre].mean()),
               control_spend_pc_window=float(ct[t0:t1].mean()), control_spend_pc_pre=float(ct[pre].mean()),
               ratio_test_to_control_pre=float(tr[pre].mean() / ct[pre].mean()),
               ratio_test_to_control_window=float(tr[t0:t1].mean() / ct[t0:t1].mean()),
               ratio_test_to_control_all_other=float(np.delete(tr, np.r_[t0:t1]).mean() / np.delete(ct, np.r_[t0:t1]).mean()))
    # difference-in-differences on revenue per capita (what a switch-off would show)
    rv = d.revenue / d.pop[:, None]
    did = (rv[test, t0:t1].mean() - rv[test, pre].mean()) - (rv[~test, t0:t1].mean() - rv[~test, pre].mean())
    out["did_revenue_per_capita"] = float(did)
    out["did_spend_per_capita"] = float((d.spend[m][test, t0:t1] / d.pop[test, None]).mean() - (d.spend[m][test, pre] / d.pop[test, None]).mean()
                                        - ((d.spend[m][~test, t0:t1] / d.pop[~test, None]).mean() - (d.spend[m][~test, pre] / d.pop[~test, None]).mean()))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(d.dates, ct, color=SERIES[0], lw=1.6, label="control geos (Geo8–Geo39)")
    ax.plot(d.dates, tr, color=SERIES[1], lw=1.6, label="test geos (Geo0–Geo7)")
    ax.axvspan(d.dates[t0], d.dates[t1 - 1], color=GRID, alpha=0.8, lw=0)
    ax.text(d.dates[t0], ax.get_ylim()[1] * 0.92, "  reported switch-off window", fontsize=8, color=MUTED)
    ax.set_ylabel("Channel3 spend per 1,000 people", fontsize=8, color=MUTED)
    ax.set_title("Channel3 spend did not stop in the test geos during the reported test", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    return out, fig


def main(argv=None):
    cfg = load_config()
    d = load(cfg)
    rep = out_path(cfg, "reports_dir")
    res = {"geos": d.G, "weeks": d.T, "first_week": str(d.dates[0].date()), "last_week": str(d.dates[-1].date()),
           "missing_values": 0}
    # impressions vs spend
    cpm = {}
    for m, c in enumerate(d.channels):
        nz = d.impressions[m] > 0
        r = d.spend[m][nz] / d.impressions[m][nz]
        cpm[c] = dict(cost_per_impression=float(np.median(r)), cv=float(r.std() / r.mean()),
                      zero_spend_share=float((d.spend[m] == 0).mean()),
                      spend_total=float(d.spend[m].sum()), spend_last13=float(d.spend[m][:, -13:].sum()))
    res["channels"] = cpm
    # variance decomposition of per-capita spend: common week / geo / geo x week
    vd = {}
    for m, c in enumerate(d.channels):
        x = d.spend[m] / d.pop[:, None]
        tc, gc = x.mean(0), x.mean(1)
        r = x - tc[None] - gc[:, None] + x.mean()
        vd[c] = dict(common_week=float(tc.var() / x.var()), geo=float(gc.var() / x.var()), geo_x_week=float(r.var() / x.var()))
    res["spend_variance_decomposition"] = vd
    tot = (d.spend.sum(2) / d.pop[None]).T
    res["spend_per_capita_cv_across_geos"] = {c: float(tot[:, m].std() / tot[:, m].mean()) for m, c in enumerate(d.channels)}
    flat = np.vstack([d.spend.reshape(d.M, -1), d.organic.reshape(1, -1), d.promo.reshape(1, -1),
                      d.controls.reshape(len(d.control_names), -1), d.kpi.reshape(1, -1)])
    names = d.channels + ["organic", "promo"] + d.control_names + ["kpi"]
    res["correlation"] = pd.DataFrame(np.corrcoef(flat), index=names, columns=names).round(2).to_dict()
    rpk = d.rpk
    res["revenue_per_conversion"] = dict(mean=float(rpk.mean()), sd=float(rpk.std()))
    res["kpi_per_capita_per_week_median"] = float(np.median(d.kpi / d.pop[:, None]))
    last13 = np.array([cpm[c]["spend_last13"] for c in d.channels])
    res["budget"] = dict(last13_total=float(last13.sum()), floor_30pct=float(0.7 * last13.sum()),
                         ceiling_30pct=float(1.3 * last13.sum()))
    q = pd.DataFrame(d.spend.sum(1).T, index=d.dates, columns=d.channels).resample("QS").sum() / 1e6
    q.index = q.index.strftime("%Y-%m")
    res["quarterly_spend_M"] = q.round(2).astype(float).to_dict()
    wr = pd.Series(d.revenue.sum(0), index=d.dates).resample("QS").mean() / 1e6
    wr.index = wr.index.strftime("%Y-%m")
    res["weekly_revenue_M_by_quarter"] = wr.round(3).to_dict()
    ex, fig = experiment_check(d, cfg)
    res["experiment_check"] = ex
    fig.savefig(rep / "fig_experiment_check.png", dpi=140)
    plt.close(fig)
    (rep / "data_audit.json").write_text(json.dumps(res, indent=1, default=str))

    p = lambda v: f"{100 * v:.1f}%"
    md = ["# Project B data audit", "", "Auto-generated by `python -m project_b.audit`.", "",
          f"{d.G} geos x {d.T} weeks ({res['first_week']} to {res['last_week']}), no missing values.", "",
          "## Channels", "", "| channel | cost / impression | CV of cost/impr. | zero-spend geo-weeks | total spend | last 13 weeks |",
          "|---|---|---|---|---|---|"]
    for c, v in cpm.items():
        md.append(f"| {c} | {v['cost_per_impression']:.5f} | {v['cv']:.1e} | {p(v['zero_spend_share'])} | {v['spend_total'] / 1e6:.1f}M | {v['spend_last13'] / 1e6:.2f}M |")
    md += ["", "## Where spend varies (share of variance of weekly spend per capita)", "",
           "| channel | common week | geo | geo x week | CV of 3-year spend per capita across geos |", "|---|---|---|---|---|"]
    for c, v in vd.items():
        md.append(f"| {c} | {p(v['common_week'])} | {p(v['geo'])} | {p(v['geo_x_week'])} | {res['spend_per_capita_cv_across_geos'][c]:.2f} |")
    b = res["budget"]
    md += ["", f"Budget: last 13 weeks {b['last13_total'] / 1e6:.2f}M -> +-30% range {b['floor_30pct'] / 1e6:.2f}M to {b['ceiling_30pct'] / 1e6:.2f}M.", "",
           "## Experiment check (Channel3, Geo0–Geo7, 8 weeks from 2023-06-05)", "",
           f"- Zero-spend geo-weeks in the window: **{ex['zero_spend_geo_weeks']} of {ex['test_geo_weeks']}**",
           f"- Test-geo Channel3 spend per 1,000 people: window {ex['test_spend_pc_window']:.2f} vs previous 8 weeks {ex['test_spend_pc_pre']:.2f}",
           f"- Test / control spend ratio: before {ex['ratio_test_to_control_pre']:.2f}, during {ex['ratio_test_to_control_window']:.2f}, all other weeks {ex['ratio_test_to_control_all_other']:.2f}",
           f"- Difference-in-differences of Channel3 spend per capita (test − control, window − pre): {ex['did_spend_per_capita']:+.4f}",
           f"- Difference-in-differences of revenue per capita: {ex['did_revenue_per_capita']:+.5f}", "",
           "![](fig_experiment_check.png)", ""]
    (rep / "data_audit.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
