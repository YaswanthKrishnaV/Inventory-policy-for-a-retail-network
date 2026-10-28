"""Budget allocation across paid channels for the next 13 weeks (Task 6) and the value of
information for choosing the next experiment (Task 7, part 1).

Decision: a multiplier k_m on each channel's last-13-week spend (its geo x week pattern is kept).
Constraint: sum_m k_m * last13_m = B,  1 - bound <= k_m <= 1 + bound.
Objective: posterior-mean incremental revenue over the 13 weeks + carry-over tail.
Response curves f_m(k) are computed per posterior draw on a grid of k and interpolated; the model
is additive across channels, so total revenue = sum_m f_m(k_m).
"""
from __future__ import annotations

import numpy as np

GRID = np.round(np.arange(0.0, 1.61, 0.05), 2)


def interp(curves, k):
    """curves (..., M, len(GRID)); k (M,) -> (..., M)"""
    out = np.empty(curves.shape[:-1])
    for m in range(curves.shape[-2]):
        out[..., m] = np.array([np.interp(k[m], GRID, c) for c in curves[..., m, :].reshape(-1, len(GRID))]).reshape(curves.shape[:-2])
    return out


def optimise(curves_mean, last13, total, lo, hi, step=0.0025):
    """Maximise sum_m f_m(k_m) s.t. sum_m k_m last13_m = total, lo <= k <= hi.

    The response curves are concave (Hill with slope 1 after adstock), so greedy water-filling is
    exact: start every channel at its lower bound and repeatedly give the next small slice of
    budget (step x last13_m) to the channel with the highest marginal revenue per unit of spend,
    until the budget is used. Marginal ROIs end up equal across channels not at a bound."""
    M = len(last13)
    k = lo.astype(float).copy()
    left = total - (k * last13).sum()
    if left < -1e-6 or total > (hi * last13).sum() + 1e-6:
        raise ValueError("budget outside the feasible range")
    val = lambda m, km: np.interp(km, GRID, curves_mean[m])
    while left > 1e-6:
        gains = np.full(M, -np.inf)
        for m in range(M):
            if k[m] < hi[m] - 1e-12:
                dk = min(step, hi[m] - k[m], left / last13[m])
                gains[m] = (val(m, k[m] + dk) - val(m, k[m])) / (dk * last13[m])
        m = int(np.argmax(gains))
        if not np.isfinite(gains[m]):
            break
        dk = min(step, hi[m] - k[m], left / last13[m])
        k[m] += dk
        left -= dk * last13[m]
    return k


def greedy_alloc(curves, last13, total, lo, hi, step=0.05):
    """Fast allocation for one draw (used inside value-of-information loops): start at the lower
    bounds and add budget in GRID steps to the channel with the best marginal return."""
    M = len(last13)
    gi = np.array([np.searchsorted(GRID, l - 1e-9) for l in lo])
    gmax = np.array([np.searchsorted(GRID, h + 1e-9) - 1 for h in hi])
    spent = (GRID[gi] * last13).sum()
    while True:
        best, bm = 0.0, -1
        for m in range(M):
            if gi[m] < gmax[m]:
                cost = (GRID[gi[m] + 1] - GRID[gi[m]]) * last13[m]
                if spent + cost <= total + 1e-6:
                    gain = (curves[m, gi[m] + 1] - curves[m, gi[m]]) / cost
                    if gain > best:
                        best, bm = gain, m
        if bm < 0:
            break
        spent += (GRID[gi[bm] + 1] - GRID[gi[bm]]) * last13[bm]
        gi[bm] += 1
    return GRID[gi]


def scenario(curves, last13, total, bound, relaxed=False):
    """Allocation + revenue distribution for one budget. curves: (draws, M, len(GRID))."""
    M = len(last13)
    feasible = (1 - bound) * last13.sum() <= total <= (1 + bound) * last13.sum()
    if feasible or not relaxed:
        lo, hi = np.full(M, 1 - bound), np.full(M, 1 + bound)
    else:
        lo, hi = np.zeros(M), np.full(M, 1 + bound)
    if not feasible and not relaxed:
        # infeasible: the closest feasible point (every channel at the binding bound)
        k = np.full(M, 1 - bound if total < last13.sum() else 1 + bound)
    else:
        k = optimise(curves.mean(0), last13, total, lo, hi)
    k_same = np.full(M, (k * last13).sum() / last13.sum())      # today's mix scaled to the SAME total spend
    rev = interp(curves, k).sum(1)
    rev_same = interp(curves, np.clip(k_same, 0, GRID[-1])).sum(1)
    # marginal ROI at the optimum (finite difference on each draw's curve)
    h = 0.02
    mroi = np.stack([(interp(curves, k + h * (np.arange(M) == m))[:, m] - interp(curves, k)[:, m]) / (h * last13[m])
                     for m in range(M)], 1)
    return dict(total=total, feasible=bool(feasible), relaxed=bool(relaxed and not feasible), k=k, spend=k * last13,
                revenue=rev, revenue_same_mix=rev_same, uplift=rev - rev_same, mroi=mroi)


def value_of_information(curves, last13, total, bound, n_bins=10):
    """Expected value of perfect information, total and per channel (EVPPI by binning).
    EVPI   = E[ max_k V_d(k) ] - max_k E[V_d(k)]
    EVPPI_m = E_bins[ max_k E[V_d(k) | channel m's curve in bin] ] - max_k E[V_d(k)]"""
    M = len(last13)
    lo, hi = np.full(M, 1 - bound), np.full(M, 1 + bound)
    k_star = optimise(curves.mean(0), last13, total, lo, hi)
    base = interp(curves, k_star).sum(1).mean()
    perfect = np.mean([interp(c[None], optimise(c, last13, total, lo, hi, step=0.01)).sum() for c in curves])
    evpi = perfect - base
    evppi = np.zeros(M)
    for m in range(M):
        # summarise channel m's curve by its slope around current spend (what drives allocation)
        key = curves[:, m, np.searchsorted(GRID, 1.3)] - curves[:, m, np.searchsorted(GRID, 0.7)]
        edges = np.quantile(key, np.linspace(0, 1, n_bins + 1))
        val = 0.0
        for b in range(n_bins):
            sel = (key >= edges[b]) & (key <= edges[b + 1]) if b == n_bins - 1 else (key >= edges[b]) & (key < edges[b + 1])
            if not sel.any():
                continue
            cm = curves[sel].mean(0)
            kb = optimise(cm, last13, total, lo, hi)
            val += sel.mean() * interp(curves[sel], kb).sum(1).mean()
        evppi[m] = val - base
    return dict(evpi=float(evpi), evppi=np.maximum(evppi, 0.0), k_star=k_star)
