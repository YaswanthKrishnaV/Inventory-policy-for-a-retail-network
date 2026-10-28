"""ROI, marginal ROI and response curves from posterior draws.

All quantities are counterfactual and include carry-over (adstock tail):

  average ROI_m   = [revenue(actual) - revenue(channel m set to 0 in the window)] / spend_m(window)
  marginal ROI_m  = [revenue(spend_m x 1.01 in the window) - revenue(actual)] / (0.01 x spend_m(window))
  response f_m(k) = revenue from spending k x (last-13-week pattern of channel m) over the next 13 weeks,
                    relative to spending nothing on m in those weeks (history's carry-over held fixed)

Revenue is summed over the window plus `max_lag` weeks after it, so spend near the end of the
window gets credit for the sales it causes later. Revenue per unit of standardised KPI is
y_sd x population x revenue_per_conversion.
"""
from __future__ import annotations

import numpy as np

from .transforms import adstock, hill


def draws_subset(f, n, seed=0):
    N = f["roi"].shape[0]
    idx = np.random.default_rng(seed).choice(N, size=min(n, N), replace=False)
    return {k: v[idx] for k, v in f.items()}, idx


def _rev(model, j, alpha, ec, b_g, x, rev_scale, weeks):
    h = hill(adstock(x, alpha, model.d.max_lag), ec)
    return (b_g[:, None] * h[:, weeks] * rev_scale[:, weeks]).sum()


def roi_draws(model, f, t_start, t_end, step=0.01):
    """Per draw and paid channel: average ROI, marginal ROI, incremental revenue, over weeks [t_start, t_end)."""
    d = model.d
    L = d.max_lag
    T = d.y.shape[1]
    pad = max(0, t_end + L - T)
    X = np.concatenate([d.x, np.zeros(d.x.shape[:2] + (pad,))], axis=2)
    RS = np.concatenate([d.rev_scale, np.repeat(d.rev_scale[:, -1:], pad, 1)], axis=1)
    weeks = np.arange(t_start, t_end + L)
    win = np.arange(t_start, t_end)
    nd = f["roi"].shape[0]
    out = {k: np.zeros((nd, model.M)) for k in ("roi", "mroi", "incremental", "spend")}
    for m in range(model.M):
        x = X[m]
        x0 = x.copy(); x0[:, win] = 0
        x1 = x.copy(); x1[:, win] *= (1 + step)
        spend = d.spend[m][:, win].sum()
        for i in range(nd):
            a, e, b = f["alpha"][i, m], f["ec"][i, m], f["u"][i, :, 1 + m]
            base = _rev(model, m, a, e, b, x, RS, weeks)
            inc = base - _rev(model, m, a, e, b, x0, RS, weeks)
            up = _rev(model, m, a, e, b, x1, RS, weeks) - base
            out["roi"][i, m] = inc / spend
            out["mroi"][i, m] = up / (step * spend)
            out["incremental"][i, m] = inc
            out["spend"][i, m] = spend
    return out


def response_curves(model, f, grid, horizon=13, hist_end=None, rpk_future=None):
    """f_m(k) per draw: incremental revenue over the next `horizon` weeks (+ carry-over tail) from
    running channel m at k x its last-`horizon`-week geo x week pattern. Returns (draws, M, len(grid))."""
    d = model.d
    L = d.max_lag
    T = d.y.shape[1] if hist_end is None else hist_end
    Xh = d.x[:, :, :T]
    pattern = Xh[:, :, T - horizon:T]                     # (J, G, horizon)
    fut = horizon + L
    RSf = d.y_sd * d.pop[:, None] * (rpk_future if rpk_future is not None else
                                     (d.rev_scale[:, T - horizon:T] / (d.y_sd * d.pop[:, None])).mean(1, keepdims=True))
    RSf = np.repeat(RSf, fut, 1)
    weeks = np.arange(T, T + fut)
    nd = f["roi"].shape[0]
    out = np.zeros((nd, model.M, len(grid)))
    for m in range(model.M):
        for gi, k in enumerate(grid):
            xf = np.concatenate([Xh[m], k * pattern[m], np.zeros((d.x.shape[1], L))], axis=1)
            x0 = np.concatenate([Xh[m], np.zeros((d.x.shape[1], fut))], axis=1)
            for i in range(nd):
                a, e, b = f["alpha"][i, m], f["ec"][i, m], f["u"][i, :, 1 + m]
                h1 = hill(adstock(xf, a, L), e)[:, weeks]
                h0 = hill(adstock(x0, a, L), e)[:, weeks]
                out[i, m, gi] = (b[:, None] * (h1 - h0) * RSf).sum()
    return out


def last_spend(model, horizon=13, hist_end=None):
    T = model.d.y.shape[1] if hist_end is None else hist_end
    return model.d.spend[:, :, T - horizon:T].sum((1, 2))


def summarise(x, axis=0):
    return dict(mean=np.mean(x, axis), median=np.median(x, axis), p05=np.quantile(x, 0.05, axis), p95=np.quantile(x, 0.95, axis))
