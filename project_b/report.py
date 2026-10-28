"""Tables, CSV deliverables and figures for Project B (numbers quoted in memo_B come from here)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import optimise
from .src.plotstyle import GRID, INK, MUTED, SERIES, style

VARIANT_LABEL = {"uncalibrated": "geo model (main)", "discounted": "geo + experiment, discounted (se x3)",
                 "full": "geo + experiment, full trust", "national": "national totals", "weekly_fe": "geo, weekly fixed effects",
                 "no_sentiment": "geo, without sentiment", "no_organic": "geo, without organic"}


def q(x, p):
    return float(np.quantile(x, p))


def roi_table(d, roi_draws, last13):
    rows = []
    for v, rd in roi_draws.items():
        for m, c in enumerate(d.channels):
            r, mr, inc = rd["roi"][:, m], rd["mroi"][:, m], rd["incremental"][:, m]
            rows.append(dict(variant=v, channel=c, spend_window=float(rd["spend"][0, m]), last13_spend=float(last13[m]),
                             roi_mean=r.mean(), roi_median=np.median(r), roi_p05=q(r, .05), roi_p95=q(r, .95),
                             mroi_mean=mr.mean(), mroi_median=np.median(mr), mroi_p05=q(mr, .05), mroi_p95=q(mr, .95),
                             incremental_revenue_mean=inc.mean(), incremental_revenue_p05=q(inc, .05), incremental_revenue_p95=q(inc, .95),
                             p_roi_gt_1=float((r > 1).mean())))
    return pd.DataFrame(rows)


def budget_table(d, scen, last13):
    rows = []
    for key, s in scen.items():
        for m, c in enumerate(d.channels):
            rows.append(dict(scenario=(f"{s['total'] / 1e6:.0f}M" + (" (bounds relaxed)" if s["relaxed"] else "")
                                       + ("" if s["feasible"] or s["relaxed"] else " (infeasible: floor of ±30% rule)")),
                             channel=c, last13_spend=float(last13[m]), recommended_spend=float(s["spend"][m]),
                             change_pct=float(100 * (s["k"][m] - 1)),
                             mroi_at_recommendation_mean=float(s["mroi"][:, m].mean()),
                             mroi_at_recommendation_p05=q(s["mroi"][:, m], .05), mroi_at_recommendation_p95=q(s["mroi"][:, m], .95)))
        rows.append(dict(scenario=rows[-1]["scenario"], channel="TOTAL", last13_spend=float(last13.sum()),
                         recommended_spend=float(s["spend"].sum()), change_pct=float(100 * (s["spend"].sum() / last13.sum() - 1)),
                         incremental_revenue_mean=float(s["revenue"].mean()), incremental_revenue_p05=q(s["revenue"], .05),
                         incremental_revenue_p95=q(s["revenue"], .95), uplift_vs_same_mix_mean=float(s["uplift"].mean()),
                         uplift_vs_same_mix_p05=q(s["uplift"], .05), uplift_vs_same_mix_p95=q(s["uplift"], .95),
                         p_uplift_positive=float((s["uplift"] > 0).mean())))
    return pd.DataFrame(rows)


def write(cfg, d, t0, t1, fits, roi_draws, acc, acc_train, scen, alt, voi, design, top, last13, curves, out, rep,
          prior_share, elapsed, pred):
    rt = roi_table(d, roi_draws, last13)
    rt[rt.variant == cfg["experiment"]["use_for_budget"]].drop(columns="variant").to_csv(out / "roi_table.csv", index=False)
    rt.to_csv(rep / "roi_all_variants.csv", index=False)
    bt = budget_table(d, scen, last13)
    bt.to_csv(out / "budget_allocation.csv", index=False)
    dz = pd.DataFrame(design)
    dz.to_csv(rep / "experiment_design.csv", index=False)
    rnd = lambda v, n: None if not np.isfinite(v) else round(float(v), n)
    diag = {v: {k: dict(rhat=rnd(x["rhat"], 3), ess=rnd(x["ess"], 0)) for k, x in f[2].items()} for v, f in fits.items()}

    # lift implied by each model for the Channel3 test (what the vendor claims to have measured)
    lift = {}
    for v, (m, post, _) in fits.items():
        if v == "national":
            continue
        from .run_project_b import calibration_spec
        from .src.fit import flat
        f = flat(post)
        idx = np.random.default_rng(0).choice(f["roi"].shape[0], min(300, f["roi"].shape[0]), replace=False)
        cal = calibration_spec(d, m.d, cfg, 1.0)
        m.cal = cal
        vals = [float((f["u"][i, :, 1 + cal["channel"]] * m.lift_coefs(cal["channel"], f["alpha"][i, cal["channel"]], f["ec"][i, cal["channel"]])).sum())
                for i in idx]
        lift[v] = dict(mean=float(np.mean(vals)), p05=q(vals, .05), p95=q(vals, .95))

    res = dict(train=[str(d.dates[0].date()), str(d.dates[t0 - 1].date())], holdout=[str(d.dates[t0].date()), str(d.dates[t1 - 1].date())],
               accuracy_holdout=acc, accuracy_last52_train=acc_train, diagnostics=diag, implied_channel3_test_iroas=lift,
               evpi=voi["evpi"], evppi=dict(zip(d.channels, voi["evppi"].tolist())), next_experiment_channel=d.channels[top],
               prior_media_share=dict(median=float(np.median(prior_share)), p05=q(prior_share, .05), p95=q(prior_share, .95)),
               alt_budget={k: dict(spend=v["spend"].tolist(), revenue_mean=float(v["revenue"].mean())) for k, v in alt.items()},
               runtime_min=elapsed / 60)
    (rep / "results.json").write_text(json.dumps(res, indent=1, default=float))

    # ---------------------------------------------------------------- markdown
    f0 = lambda v: f"{v:,.0f}"
    f2 = lambda v: f"{v:.2f}"
    md = [f"# Project B results — hold-out {res['holdout'][0]} to {res['holdout'][1]}", "",
          f"Trained on {res['train'][0]} – {res['train'][1]}. Generated by `python -m project_b.run_project_b`. "
          "90% credible intervals in brackets. ROI = incremental revenue ÷ spend over the last 52 training weeks "
          "(carry-over included); marginal ROI = revenue from the next 1% of spend ÷ that spend.", "",
          "## ROI by channel (main model)", "", "| channel | spend (52w) | ROI | marginal ROI | P(ROI > 1) |", "|---|---|---|---|---|"]
    main = rt[rt.variant == cfg["experiment"]["use_for_budget"]]
    for _, r in main.iterrows():
        md.append(f"| {r.channel} | {r.spend_window / 1e6:.2f}M | {f2(r.roi_mean)} [{f2(r.roi_p05)}, {f2(r.roi_p95)}] | "
                  f"{f2(r.mroi_mean)} [{f2(r.mroi_p05)}, {f2(r.mroi_p95)}] | {r.p_roi_gt_1:.2f} |")
    md += ["", "## ROI across model variants (mean [90% CI])", "", "| channel | " + " | ".join(VARIANT_LABEL.get(v, v) for v in roi_draws) + " |",
           "|---|" + "---|" * len(roi_draws)]
    for c in d.channels:
        cells = []
        for v in roi_draws:
            r = rt[(rt.variant == v) & (rt.channel == c)].iloc[0]
            cells.append(f"{f2(r.roi_mean)} [{f2(r.roi_p05)}, {f2(r.roi_p95)}]")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    md += ["", "Model-implied iROAS of the reported Channel3 test (Geo0–7, 8 weeks + carry-over), vs vendor 0.8 [0.3, 1.3]: " +
           "; ".join(f"{VARIANT_LABEL.get(k, k)} {v['mean']:.2f} [{v['p05']:.2f}, {v['p95']:.2f}]" for k, v in lift.items()), "",
           "## Hold-out forecast accuracy (13 weeks)", "", "| level | WAPE | MAPE | R² | bias | 90% interval coverage |", "|---|---|---|---|---|---|"]
    for lvl in ("national", "geo"):
        a = acc[lvl]
        md.append(f"| {lvl} | {a['wape']:.3f} | {a['mape']:.3f} | {a['r2']:.3f} | {100 * a['bias']:+.1f}% | {a['coverage90']:.2f} |")
    md += ["", f"In-sample (last 52 training weeks): national WAPE {acc_train['national']['wape']:.3f}, R² {acc_train['national']['r2']:.3f}, "
           f"coverage {acc_train['national']['coverage90']:.2f}.", "",
           "## Budget — next 13 weeks", "", "| scenario | " + " | ".join(d.channels) + " | total | incremental revenue | uplift vs today's mix | P(uplift > 0) |",
           "|---|" + "---|" * (len(d.channels) + 4)]
    for key, s in scen.items():
        lbl = f"{s['total'] / 1e6:.0f}M" + (" relaxed" if s["relaxed"] else "") + ("" if s["feasible"] or s["relaxed"] else " (±30% floor)")
        cells = [f"{s['spend'][m] / 1e6:.2f}M ({100 * (s['k'][m] - 1):+.0f}%)" for m in range(len(d.channels))]
        same = not s["feasible"] and not s["relaxed"]
        md.append(f"| {lbl} | " + " | ".join(cells) + f" | {s['spend'].sum() / 1e6:.2f}M | {s['revenue'].mean() / 1e6:.2f}M "
                  f"[{q(s['revenue'], .05) / 1e6:.2f}, {q(s['revenue'], .95) / 1e6:.2f}] | " +
                  ("— (today's mix by construction) | — |" if same else
                   f"{s['uplift'].mean() / 1e3:,.0f}k [{q(s['uplift'], .05) / 1e3:,.0f}, {q(s['uplift'], .95) / 1e3:,.0f}] | {(s['uplift'] > 0).mean():.2f} |"))
    md += ["", "Calibrated-model allocations at 18M (sensitivity): " + "; ".join(
        f"{k}: " + ", ".join(f"{c} {v['spend'][m] / 1e6:.2f}M" for m, c in enumerate(d.channels)) for k, v in alt.items()), "",
        "## Value of information and next experiment", "",
        f"EVPI (all channels) at 18M: {voi['evpi'] / 1e3:,.0f}k. Per channel (EVPPI): " +
        ", ".join(f"{c} {v / 1e3:,.0f}k" for c, v in zip(d.channels, voi["evppi"])) + f". Next test: **{d.channels[top]}**.", "",
        "| treated geos | weeks | spend change | power | MDE (iROAS) | spend moved |", "|---|---|---|---|---|---|"]
    for r in sorted(design, key=lambda r: (-r["power"], abs(r["spend_moved"]))):
        md.append(f"| {r['n_treat_geos']} | {r['weeks']} | {100 * r['spend_change']:+.0f}% | {r['power']:.2f} | {r['mde_iroas']:.2f} | {r['spend_moved'] / 1e3:,.0f}k |")
    md += ["", "## Convergence (worst R-hat / min ESS per variant)", ""]
    for v, dg in diag.items():
        rh = [x['rhat'] for x in dg.values() if x['rhat'] is not None]; es = [x['ess'] for x in dg.values() if x['ess'] is not None]
        md.append(f"- {VARIANT_LABEL.get(v, v)}: R-hat ≤ {max(rh):.3f}, ESS ≥ {min(es):.0f}")
    md += ["", f"Prior-predictive media share of conversions: median {np.median(prior_share):.2f} [{q(prior_share, .05):.2f}, {q(prior_share, .95):.2f}].",
           "", f"Runtime: {elapsed / 60:.1f} min.", "",
           "![](fig_roi.png)", "", "![](fig_holdout.png)", "", "![](fig_response.png)", ""]
    (rep / "results.md").write_text("\n".join(md))

    # ---------------------------------------------------------------- figures
    variants = [v for v in ("uncalibrated", "discounted", "full", "national") if v in roi_draws]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    M = len(d.channels)
    wdt = 0.8 / len(variants)
    for k, v in enumerate(variants):
        r = roi_draws[v]["roi"]
        xs = np.arange(M) - 0.4 + wdt * (k + 0.5)
        ax.errorbar(xs, r.mean(0), yerr=[r.mean(0) - np.quantile(r, .05, 0), np.quantile(r, .95, 0) - r.mean(0)],
                    fmt="o", ms=5, color=SERIES[k], lw=1.6, capsize=0, label=VARIANT_LABEL[v])
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.text(M - 0.45, 1.02, "break-even", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(range(M), d.channels, fontsize=8)
    ax.set_ylabel("ROI (revenue per unit spend)", fontsize=8, color=MUTED)
    ax.set_title("ROI by channel, mean and 90% credible interval", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    style(ax); fig.tight_layout(); fig.savefig(rep / "fig_roi.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    nat_a = d.revenue.sum(0)
    rp = (pred * d.rpk[None]).sum(1)
    lo, hi = np.quantile(rp, [0.05, 0.95], 0)
    ws = np.arange(max(0, t0 - 39), t1)
    ax.fill_between(d.dates[ws], lo[ws] / 1e6, hi[ws] / 1e6, color=SERIES[0], alpha=0.18, lw=0, label="model 90% interval")
    ax.plot(d.dates[ws], rp.mean(0)[ws] / 1e6, color=SERIES[0], lw=1.6, label="model mean")
    ax.plot(d.dates[ws], nat_a[ws] / 1e6, color=INK, lw=1.2, label="actual")
    ax.axvspan(d.dates[t0], d.dates[t1 - 1], color=GRID, alpha=0.7, lw=0)
    ax.text(d.dates[t0], ax.get_ylim()[1], "  hold-out", fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("weekly revenue (M)", fontsize=8, color=MUTED)
    ax.set_title("National weekly revenue: fit and 13-week hold-out forecast", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax); fig.tight_layout(); fig.savefig(rep / "fig_holdout.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, M, figsize=(11, 2.6), sharey=False)
    g = optimise.GRID
    s18 = scen[cfg["budget"]["main_total"]]
    for m, ax in enumerate(axes):
        c = curves[:, m, :] / 1e6
        ax.fill_between(g * last13[m] / 1e6, np.quantile(c, .05, 0), np.quantile(c, .95, 0), color=SERIES[m], alpha=0.2, lw=0)
        ax.plot(g * last13[m] / 1e6, c.mean(0), color=SERIES[m], lw=1.8)
        ax.axvline(last13[m] / 1e6, color=MUTED, lw=1, ls=":")
        ax.axvline(s18["spend"][m] / 1e6, color=INK, lw=1.2)
        ax.axvspan(0.7 * last13[m] / 1e6, 1.3 * last13[m] / 1e6, color=GRID, alpha=0.5, lw=0)
        ax.set_title(d.channels[m], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("13-week spend (M)", fontsize=7, color=MUTED)
        style(ax)
    axes[0].set_ylabel("incremental revenue (M)", fontsize=7, color=MUTED)
    fig.suptitle("Response curves, next 13 weeks (band = 90% CI; dotted = last 13 weeks; solid = 18M plan; shaded = ±30%)",
                 fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(rep / "fig_response.png", dpi=140); plt.close(fig)
    print("\n".join(md))
