"""Prior- and posterior-predictive checks, hold-out accuracy, and the posterior file."""
from __future__ import annotations

import numpy as np
from scipy import special, stats
from scipy.io import netcdf_file

from .transforms import adstock, hill


# ----------------------------------------------------------------------------- prior predictive
def prior_predictive(model, n=300, seed=0):
    """Draw parameters from the priors, compute implied ROI-consistent effects and KPI paths.
    Returns implied media share of KPI (training weeks) and simulated national KPI per capita."""
    rng = np.random.default_rng(seed)
    pr, d = model.pr, model.d
    J, M, G = model.J, model.M, model.G
    shares, nat = [], []
    tr = d.mask[0]
    for _ in range(n):
        alpha = rng.beta(pr.a_a, pr.a_b, J)
        ec = pr.ec.rvs(J, random_state=rng)
        roi = np.exp(rng.normal(pr.roi_mu, pr.roi_sd, M))
        eta = np.abs(rng.normal(0, pr.eta_scale, J))
        borg = abs(rng.normal(0, pr.org_scale))
        H = np.stack([hill(adstock(d.x[j], alpha[j], d.max_lag), ec[j]) for j in range(J)])
        beta = [roi[j] * model.c_factor(j, H[j]) for j in range(M)] + [borg]
        b = np.array([rng.normal(beta[j], eta[j] * abs(beta[j]), G) for j in range(J)])
        media = np.einsum("jgt,jg->gt", H, b)
        tau = rng.normal(0, pr.tau_sd, G)
        s = rng.normal(0, 1, model.q)
        sig = np.abs(rng.normal(0, pr.sigma_scale, G))
        y = tau[:, None] + np.einsum("gtq,q->gt", d.W, s) + media + rng.normal(0, 1, media.shape) * sig[:, None]
        if d.Zc is not None:
            y = y + np.einsum("gtc,gc->gt", d.Zc, rng.normal(0, pr.ctrl_sd, (G, d.Zc.shape[2])))
        kpi = (y * d.y_sd + d.y_mean) * d.pop[:, None]
        mk = media * d.y_sd * d.pop[:, None]
        shares.append(mk[:, tr].sum() / max(kpi[:, tr].sum(), 1e-9))
        nat.append(kpi.sum(0) / d.pop.sum())
    return np.array(shares), np.array(nat)


# ----------------------------------------------------------------------------- posterior predictive
def predict(model, f, idx=None, noise=True, seed=0):
    """Posterior predictive KPI (conversions) per draw: (draws, G, T)."""
    rng = np.random.default_rng(seed)
    d = model.d
    idx = np.arange(f["roi"].shape[0]) if idx is None else idx
    out = np.zeros((len(idx), model.G, model.T))
    for k, i in enumerate(idx):
        H = np.stack([hill(adstock(d.x[j], f["alpha"][i, j], d.max_lag), f["ec"][i, j]) for j in range(model.J)])
        Z = model.Z_of(H)
        mu = np.einsum("gtp,gp->gt", Z, f["u"][i]) + np.einsum("gtq,q->gt", d.W, f["s"][i])
        if noise:
            mu = mu + rng.standard_normal(mu.shape) * f["sigma"][i][:, None]
        out[k] = (mu * d.y_sd + d.y_mean) * d.pop[:, None]
    return out


def accuracy(actual, pred_draws, weeks):
    """WAPE, MAPE, R^2, bias and 90% interval coverage, at geo-week and national-week level."""
    res = {}
    for level in ("geo", "national"):
        if level == "geo":
            a = actual[:, weeks]; p = pred_draws[:, :, weeks]
        else:
            a = actual[:, weeks].sum(0); p = pred_draws[:, :, weeks].sum(1)
        m = p.mean(0)
        lo, hi = np.quantile(p, [0.05, 0.95], axis=0)
        res[level] = dict(wape=float(np.abs(a - m).sum() / np.abs(a).sum()), mape=float(np.mean(np.abs(a - m) / np.abs(a))),
                          r2=float(1 - ((a - m) ** 2).sum() / ((a - a.mean()) ** 2).sum()), bias=float(m.sum() / a.sum() - 1),
                          coverage90=float(((a >= lo) & (a <= hi)).mean()))
    return res


# ----------------------------------------------------------------------------- posterior file
def write_netcdf(path, post, model, attrs: dict):
    """posterior.nc (netCDF3 via scipy — no xarray needed). Dims: chain, draw, channel, geo, coef."""
    C, D = post["roi"].shape[:2]
    with netcdf_file(path, "w") as nc:
        nc.createDimension("chain", C); nc.createDimension("draw", D)
        nc.createDimension("paid_channel", model.M); nc.createDimension("media", model.J)
        nc.createDimension("geo", model.G); nc.createDimension("geo_coef", model.p); nc.createDimension("shared_coef", model.q)
        def var(name, arr, dims, desc):
            v = nc.createVariable(name, "f8", dims)
            v[:] = arr
            v.description = desc
        var("roi_prior_param", post["roi"], ("chain", "draw", "paid_channel"), "ROI parameter (training window), per paid channel")
        var("adstock_alpha", post["alpha"], ("chain", "draw", "media"), "geometric adstock decay")
        var("half_saturation", post["ec"], ("chain", "draw", "media"), "Hill half-saturation (x median non-zero spend per capita)")
        var("geo_spread", post["eta"], ("chain", "draw", "media"), "relative sd of geo media effects")
        var("sigma", post["sigma"], ("chain", "draw", "geo"), "residual sd (standardised KPI)")
        var("geo_coefs", post["u"], ("chain", "draw", "geo", "geo_coef"), "per geo: [intercept, media effects...]")
        var("shared_coefs", post["s"], ("chain", "draw", "shared_coef"), "shared: trend, seasonality, controls, promo")
        nc.media_names = ",".join(model.d.names)
        nc.shared_coef_names = ",".join(model.d.w_names)
        nc.y_mean = model.d.y_mean
        nc.y_sd = model.d.y_sd
        for k, v in attrs.items():
            setattr(nc, k, str(v))
