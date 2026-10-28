"""Task 7: design the next geo experiment and its power, by simulation.

For a channel and a candidate design (number of treated geos, weeks, spend change):
  1. noise   = the fitted model's weekly residuals per geo (real autocorrelation and geo noise);
  2. timing  = a random start week in the last year of training data;
  3. geos    = stratified random assignment (pair geos by population, treat one of each pair);
  4. effect  = the revenue change the model implies for that spend change in the treated geos,
               for a posterior draw (so ROI uncertainty enters the power), incl. 4 cool-down weeks;
  5. analysis = difference-in-differences: per-geo (test + cool-down mean − 26-week pre mean),
               treated vs control, t-test across geos (geo-clustered), two-sided at alpha.
Power = share of simulations where the lift is detected. Also reported: the minimum detectable
iROAS at 80% power and the test's cost (spend moved x model ROI).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .src.transforms import adstock, hill


def residuals(model, f_sub):
    """Actual minus posterior-mean predicted revenue per geo-week (training weeks), revenue units."""
    d = model.d
    pred = np.zeros_like(d.y)
    for i in range(f_sub["roi"].shape[0]):
        H = np.stack([hill(adstock(d.x[j], f_sub["alpha"][i, j], d.max_lag), f_sub["ec"][i, j]) for j in range(model.J)])
        pred += np.einsum("gtp,gp->gt", model.Z_of(H), f_sub["u"][i]) + np.einsum("gtq,q->gt", d.W, f_sub["s"][i])
    pred /= f_sub["roi"].shape[0]
    res = (d.y - pred) * d.rev_scale
    return res[:, d.mask[0]]


def simulate_power(model, f_sub, m, n_treat, weeks, change, n_sims, alpha, rng, pre=26, cool=4):
    d = model.d
    res = residuals(model, f_sub) if not hasattr(model, "_res_cache") else model._res_cache
    model._res_cache = res
    G, Tr = res.shape
    order = np.argsort(d.pop)
    pairs = [order[i:i + 2] for i in range(0, G - 1, 2)]
    span = pre + weeks + cool
    starts = np.arange(max(Tr - 52, d.max_lag + pre), Tr - weeks - cool)
    hits, est, removed_spend, true_iroas = [], [], [], []
    for _ in range(n_sims):
        i = rng.integers(f_sub["roi"].shape[0])
        s = int(rng.choice(starts))
        chosen = rng.permutation(len(pairs))[:n_treat]
        treat = np.zeros(G, bool)
        for c in chosen:
            treat[rng.choice(pairs[c])] = True
        a, e, b = f_sub["alpha"][i, m], f_sub["ec"][i, m], f_sub["u"][i, :, 1 + m]
        x = d.x[m][:, :Tr]
        x1 = x.copy()
        tw = np.arange(s, s + weeks)
        x1[np.ix_(treat, tw)] *= (1 + change)
        h0, h1 = hill(adstock(x, a, d.max_lag), e), hill(adstock(x1, a, d.max_lag), e)
        effect = b[:, None] * (h1 - h0) * d.rev_scale[:, :Tr]
        y = res + effect
        prew = np.arange(s - pre, s)
        postw = np.arange(s, s + weeks + cool)
        delta = y[:, postw].mean(1) - y[:, prew].mean(1)
        diff = delta[treat].mean() - delta[~treat].mean()
        se = np.sqrt(delta[treat].var(ddof=1) / treat.sum() + delta[~treat].var(ddof=1) / (~treat).sum())
        dfree = G - 2
        p = 2 * stats.t.sf(abs(diff / se), dfree)
        spend_moved = (d.spend[m][np.ix_(treat, tw)] * change).sum()
        lift_total = diff * treat.sum() * len(postw)
        hits.append(p < alpha)
        est.append(lift_total / spend_moved)
        removed_spend.append(spend_moved)
        true_iroas.append(effect[:, postw].sum() / spend_moved)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)
    se_iroas = np.std(np.array(est) - np.array(true_iroas))
    return dict(power=float(np.mean(hits)), mde_iroas=float(z * se_iroas), se_iroas=float(se_iroas),
                spend_moved=float(np.mean(removed_spend)), true_iroas_mean=float(np.mean(true_iroas)))


def design_grid(model, f_sub, m, cfg, seed=0):
    nc = cfg["next_experiment"]
    rng = np.random.default_rng(seed)
    rows = []
    for n_treat, weeks, change in itertools.product(nc["n_treat_geos"], nc["durations_weeks"], nc["spend_change"]):
        r = simulate_power(model, f_sub, m, n_treat, weeks, change, nc["n_sims"], nc["alpha"], rng)
        rows.append(dict(n_treat_geos=n_treat, weeks=weeks, spend_change=change, **r))
    return rows
