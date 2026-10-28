"""Parameter-recovery test: does the hand-written sampler find known ROIs?

    python -m project_b.recovery [--reps 3]

Simulates KPI data from the model itself, using the real spend, organic, promo and control
series and the real training window, with known parameters; fits the model; and reports whether
each true ROI falls in its 90% credible interval. Across channels and replicates about 90% of
intervals should cover. This is the evidence that the sampler is correct, and it also shows how
well *this* spend data can separate the channels.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .src.data import load, load_config, out_path, set_holdout
from .src.fit import fit, flat
from .src.model import Model, build_design

TRUE = dict(roi=[0.9, 1.6, 0.6, 2.2, 1.3], alpha=[0.5, 0.3, 0.6, 0.4, 0.2, 0.5], ec=[1.0, 0.7, 1.5, 1.2, 0.9, 1.0],
            eta=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], beta_org=0.15, sigma=0.25)


def simulate(model, rng, true=TRUE):
    d = model.d
    J, M = model.J, model.M
    H = np.stack([model.curve(j, true["alpha"][j], true["ec"][j]) for j in range(J)])
    beta = [true["roi"][j] * model.c_factor(j, H[j]) for j in range(M)] + [true["beta_org"]]
    b = np.array([rng.normal(beta[j], true["eta"][j] * beta[j], model.G) for j in range(J)])   # (J, G)
    tau = rng.normal(0, 1.0, model.G)
    s = rng.normal(0, 0.3, model.q)
    mean = tau[:, None] + np.einsum("gtq,q->gt", d.W, s) + np.einsum("jgt,jg->gt", H, b)
    y = mean + rng.normal(0, true["sigma"], mean.shape)
    return y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--draws", type=int, default=600)
    a = ap.parse_args(argv)
    cfg = load_config()
    d = load(cfg)
    set_holdout(d, cfg["holdout"]["default_start"], cfg["holdout"]["weeks"])
    rows = []
    for rep in range(a.reps):
        des = build_design(d, cfg)
        m0 = Model(des, cfg)
        des.y = simulate(m0, np.random.default_rng(100 + rep))
        m = Model(des, cfg)
        post, diag = fit(m, cfg, chains=2, warmup=a.warmup, draws=a.draws, seed=500 + rep)
        f = flat(post)
        for j, n in enumerate(des.names[:m.M]):
            r = f["roi"][:, j]
            lo, hi = np.quantile(r, [0.05, 0.95])
            rows.append(dict(rep=rep, channel=n, true=TRUE["roi"][j], mean=float(r.mean()), lo=float(lo), hi=float(hi),
                             covered=bool(lo <= TRUE["roi"][j] <= hi), rhat=diag[f"roi[{n}]"]["rhat"]))
            print(f"rep {rep} {n}: true {TRUE['roi'][j]:.2f}  posterior {r.mean():.2f} [{lo:.2f}, {hi:.2f}]  "
                  f"{'covered' if rows[-1]['covered'] else 'MISSED'}  R-hat {rows[-1]['rhat']:.3f}", flush=True)
        for j, n in enumerate(des.names):
            print(f"   alpha {n}: true {TRUE['alpha'][j]:.2f} post {f['alpha'][:, j].mean():.2f};  ec true {TRUE['ec'][j]:.2f} post {f['ec'][:, j].mean():.2f}")
    cov = np.mean([r["covered"] for r in rows])
    print(f"coverage of 90% intervals: {cov:.0%} ({len(rows)} intervals)")
    (out_path(cfg, "reports_dir") / "recovery.json").write_text(json.dumps(dict(true=TRUE, rows=rows, coverage=cov), indent=1))


if __name__ == "__main__":
    main()
