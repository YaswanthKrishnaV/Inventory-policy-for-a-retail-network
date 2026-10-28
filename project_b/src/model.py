"""Hierarchical Bayesian geo MMM and its sampler.

Model (per geo g, week t; y = standardised KPI per capita):

    y[g,t] = tau_g + baseline(t) + gamma . controls[g,t] + gamma_p . promo[g,t]
             + sum_m b[m,g] * hill(adstock(x[m,g,.]; alpha_m); ec_m)       (5 paid + 1 organic)
             + e[g,t],   e ~ N(0, sigma_g^2)

    b[m,g] ~ N(beta_m, (eta_m * beta_m)^2)                                  geo pooling
    beta_m = roi_m * c_m(alpha_m, ec_m)  for paid channels                  ROI parameterisation
    roi_m ~ LogNormal(0.2, 0.9); alpha_m ~ U(0,1); ec_m ~ TN(0.8, 0.8, [0.1, 10]); eta_m ~ HN(0.5)
    beta_organic ~ HN(0.5); tau_g ~ N(0, 5); baseline, controls, promo ~ N(0, 1); sigma_g ~ HN(1)

c_m converts ROI into an effect size: if every geo had b = beta_m, the channel's revenue over the
training weeks divided by its spend would equal roi_m.

Sampler (partially collapsed Gibbs):
  1. per channel, (alpha, ec) by adaptive random-walk Metropolis on (logit, log) scale using the
     marginal likelihood with all linear coefficients integrated out exactly (src/linear.py);
  2. per channel, log roi (paid) / log beta (organic) and log eta, same way;
  3. all linear coefficients (tau, b, baseline, controls, promo) drawn exactly from their Gaussian
     conditional;
  4. sigma_g by Metropolis given the residuals.
An optional lift-test observation enters steps 1-3 as an extra Gaussian observation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

from . import linear
from .transforms import adstock, hill


# ============================================================================ design
@dataclass
class Design:
    y: np.ndarray            # (G, T) standardised KPI per capita
    mask: np.ndarray         # (G, T) rows in the likelihood
    x: np.ndarray            # (J, G, T) scaled media inputs; J = paid channels + organic
    W: np.ndarray            # (G, T, q) shared-coefficient design (baseline, controls, promo)
    w_names: list
    n_paid: int
    names: list              # media names (paid..., organic)
    y_mean: float
    y_sd: float
    rev_scale: np.ndarray    # (G, T) revenue per unit of standardised y: y_sd * pop * rpk
    spend: np.ndarray        # (M, G, T) paid spend (currency)
    x_scale: np.ndarray      # (J,) median non-zero per-capita input used to scale x
    pop: np.ndarray
    max_lag: int
    roi_window: np.ndarray   # (T,) bool weeks used to define c_m (training weeks)
    baseline_kind: str
    t_train_end: int
    Zc: np.ndarray = None    # (G, T, C) per-geo control columns when controls vary by geo
    c_names: list = None


def trend_basis(T, t_end, every):
    """Linear-spline (hat) basis with knots every `every` weeks over [0, t_end); the first hat is
    dropped (collinear with the geo intercepts). Beyond t_end the trend is held flat."""
    knots = np.arange(0, t_end, every, dtype=float)
    if knots[-1] < t_end - 1:
        knots = np.append(knots, t_end - 1)
    t = np.minimum(np.arange(T, dtype=float), t_end - 1)
    H = np.zeros((T, len(knots)))
    for k, kn in enumerate(knots):
        left = knots[k - 1] if k > 0 else kn
        right = knots[k + 1] if k < len(knots) - 1 else kn
        up = (t >= left) & (t <= kn)
        dn = (t > kn) & (t <= right)
        if kn > left:
            H[up, k] = (t[up] - left) / (kn - left)
        else:
            H[t == kn, k] = 1.0
        if right > kn:
            H[dn, k] = (right - t[dn]) / (right - kn)
    return H[:, 1:], [f"trend_k{k}" for k in range(1, len(knots))]


def fourier_basis(T, period, order):
    t = np.arange(T)
    cols, names = [], []
    for k in range(1, order + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
        names += [f"sin{k}", f"cos{k}"]
    return np.column_stack(cols), names


def build_design(d, cfg, national: bool = False) -> Design:
    mc = cfg["model"]
    train = d.train
    t_end = int(train.sum())
    pop = d.pop.copy()
    kpi, rpk, spend, org, promo, ctrl = d.kpi, d.rpk, d.spend, d.organic, d.promo, d.controls
    if national:   # national totals: one "geo"; per-capita = totals / total population
        pop = np.array([d.pop.sum()])
        kpi = kpi.sum(0, keepdims=True)
        rpk = (d.revenue.sum(0) / d.kpi.sum(0))[None]
        spend = spend.sum(1, keepdims=True)
        org = org.sum(0, keepdims=True)
        promo = (promo * d.pop[:, None]).sum(0, keepdims=True) / d.pop.sum()
        ctrl = (ctrl * d.pop[None, :, None]).sum(1, keepdims=True) / d.pop.sum()
    G, T = kpi.shape
    ypc = kpi / pop[:, None]
    mask = np.broadcast_to(train, (G, T)).copy()
    y_mean, y_sd = ypc[mask].mean(), ypc[mask].std()
    y = (ypc - y_mean) / y_sd
    media = [spend[m] / pop[:, None] for m in range(spend.shape[0])]
    if mc.get("include_organic", True):
        media.append(org / pop[:, None])
    x_scale = np.array([np.median(v[:, :t_end][v[:, :t_end] > 0]) for v in media])
    x = np.stack([v / s for v, s in zip(media, x_scale)])
    # shared design
    if mc["baseline"] == "weekly_fe":
        Bt = np.zeros((T, t_end - 1))
        Bt[np.arange(1, t_end), np.arange(t_end - 1)] = 1.0
        bnames = [f"week{t}" for t in range(1, t_end)]
    else:
        Ht, hn = trend_basis(T, t_end, mc["knot_every_weeks"])
        Ft, fn = fourier_basis(T, mc["period_weeks"], mc["fourier_order"])
        Bt, bnames = np.column_stack([Ht, Ft]), hn + fn
    keep_c = [i for i, n in enumerate(d.control_names) if n in mc.get("controls", d.control_names)]
    ctrl = ctrl[keep_c]
    cnames = [d.control_names[i] for i in keep_c]
    csd = [c[:, :t_end].std() for c in ctrl]
    psd = promo[:, :t_end].std()
    Cz = np.stack([c / s for c, s in zip(ctrl, csd)], -1)          # (G, T, C) standardised controls
    geo_ctrl = mc.get("geo_controls", False)
    W = np.concatenate([np.broadcast_to(Bt, (G, T, Bt.shape[1]))] + ([] if geo_ctrl else [Cz]) +
                       [(promo / psd)[:, :, None]], axis=2)
    names = list(d.channels) + (["Organic"] if mc.get("include_organic", True) else [])
    return Design(y=y, mask=mask, x=x, W=np.ascontiguousarray(W), w_names=bnames + ([] if geo_ctrl else cnames) + ["Promo"],
                  Zc=Cz if geo_ctrl else None, c_names=cnames,
                  n_paid=len(d.channels), names=names, y_mean=y_mean, y_sd=y_sd,
                  rev_scale=y_sd * pop[:, None] * rpk, spend=spend, x_scale=x_scale, pop=pop,
                  max_lag=mc["max_lag"], roi_window=train.copy(), baseline_kind=mc["baseline"], t_train_end=t_end)


# ============================================================================ priors
class Priors:
    def __init__(self, cfg):
        p = cfg["priors"]
        self.roi_mu, self.roi_sd = p["roi"]["mu"], p["roi"]["sigma"]
        self.a_a, self.a_b = p["adstock_alpha"]["a"], p["adstock_alpha"]["b"]
        e = p["ec"]
        self.ec = stats.truncnorm((e["lo"] - e["mu"]) / e["sigma"], (e["hi"] - e["mu"]) / e["sigma"], loc=e["mu"], scale=e["sigma"])
        self.ec_lo, self.ec_hi = e["lo"], e["hi"]
        self.eta_scale = p["eta"]["scale"]
        self.org_scale = p["organic_beta"]["scale"]
        self.tau_sd, self.base_sd, self.ctrl_sd = p["tau_sd"], p["baseline_sd"], p["control_sd"]
        self.sigma_scale = p["sigma"]["scale"]

    # log densities on the *transformed* sampling scale (incl. Jacobians)
    def lp_alpha_logit(self, z):          # alpha = sigmoid(z)
        a = special.expit(z)
        return stats.beta.logpdf(a, self.a_a, self.a_b) + np.log(a) + np.log1p(-a)

    def lp_ec_log(self, le):              # ec = exp(le)
        ec = np.exp(le)
        if ec <= self.ec_lo or ec >= self.ec_hi:
            return -np.inf
        return self.ec.logpdf(ec) + le

    def lp_roi_log(self, lr):             # roi = exp(lr), LogNormal -> Normal on log scale
        return stats.norm.logpdf(lr, self.roi_mu, self.roi_sd)

    def lp_halfnormal_log(self, lx, scale):   # x = exp(lx) ~ HalfNormal(scale)
        return stats.halfnorm.logpdf(np.exp(lx), scale=scale) + lx


# ============================================================================ state
@dataclass
class State:
    alpha: np.ndarray       # (J,)
    ec: np.ndarray          # (J,)
    roi: np.ndarray         # (M,)
    beta_org: float
    eta: np.ndarray         # (J,)
    sigma: np.ndarray       # (G,)
    u: np.ndarray = None    # (G, p) per-geo coefficients [tau, b_0..b_{J-1}]
    s: np.ndarray = None    # (q,)


class Model:
    def __init__(self, design: Design, cfg: dict, calibration: dict | None = None, fixed_eta: float | None = None):
        self.d = design
        self.cfg = cfg
        self.pr = Priors(cfg)
        self.J = design.x.shape[0]
        self.M = design.n_paid
        self.G, self.T = design.y.shape
        self.q = design.W.shape[2]
        self.geo_trend = cfg["model"].get("geo_trend", False)
        self.nc = 0 if design.Zc is None else design.Zc.shape[2]
        self.p = 1 + self.J + (1 if self.geo_trend else 0) + self.nc
        t = np.minimum(np.arange(self.T), design.t_train_end - 1) / max(design.t_train_end - 1, 1)
        self._gt = (t - t[:design.t_train_end].mean())[None, :, None]
        self.cal = calibration        # dict(channel=j, geos=bool (G,), t0, t1, tail, obs, se) or None
        self.fixed_eta = fixed_eta    # national model: no geo hierarchy
        lam_s = np.full(self.q, 1.0 / design.__dict__.get("base_sd", 1.0) ** 2)
        nb = len([n for n in design.w_names if n not in ("Promo",) and not n.startswith(("competitor", "sentiment"))])
        lam_s[:nb] = 1.0 / self.pr.base_sd ** 2
        lam_s[nb:] = 1.0 / self.pr.ctrl_sd ** 2
        self.lam_s, self.mu_s = lam_s, np.zeros(self.q)

    # ---------------------------------------------------------------- pieces
    def curve(self, j, alpha, ec, x=None):
        x = self.d.x[j] if x is None else x
        return hill(adstock(x, alpha, self.d.max_lag), ec)

    def c_factor(self, j, h):
        """beta = roi * c: spend / (revenue per unit effect) over the ROI window."""
        w = self.d.roi_window
        rev_per_unit = (h[:, w] * self.d.rev_scale[:, w]).sum()
        return self.d.spend[j][:, w].sum() / max(rev_per_unit, 1e-12)

    def beta_of(self, j, st, H, cfac):
        return st.roi[j] * cfac[j] if j < self.M else st.beta_org

    def prior_u(self, st, cfac):
        G, J = self.G, self.J
        mu = np.zeros((G, self.p))
        lam = np.zeros((G, self.p))
        lam[:, 0] = 1.0 / self.pr.tau_sd ** 2
        for j in range(J):
            b = self.beta_of(j, st, None, cfac)
            eta = self.fixed_eta if self.fixed_eta is not None else st.eta[j]
            mu[:, 1 + j] = b
            lam[:, 1 + j] = 1.0 / max((eta * b) ** 2, 1e-12)
        k = 1 + self.J
        if self.geo_trend:
            lam[:, k] = 1.0 / self.cfg["priors"].get("geo_trend_sd", 0.5) ** 2
            k += 1
        if self.nc:
            lam[:, k:k + self.nc] = 1.0 / self.pr.ctrl_sd ** 2
        return mu, lam

    def Z_of(self, H):
        cols = [np.ones((self.G, self.T, 1)), np.transpose(H, (1, 2, 0))]
        if self.geo_trend:
            cols.append(np.broadcast_to(self._gt, (self.G, self.T, 1)))
        if self.nc:
            cols.append(self.d.Zc)
        return np.concatenate(cols, axis=2)

    def extra_of(self, H_cf_diff, st):
        """Lift-test observation: obs ~ N(sum_g b[j,g] * D_g, se^2)."""
        if self.cal is None:
            return None
        c = self.cal
        a_u = np.zeros((self.G, self.p))
        a_u[:, 1 + c["channel"]] = H_cf_diff
        return dict(a_u=a_u, a_s=np.zeros(self.q), obs=c["obs"], se=c["se"])

    def lift_coefs(self, j, alpha, ec):
        """D_g: revenue in test geo g (window + tail) lost per unit of b if channel j were zero in the
        window, divided by the spend removed. sum_g b[j,g] D_g = model-implied iROAS of the test."""
        c = self.cal
        x = self.d.x[j]
        x_cf = x.copy()
        x_cf[np.ix_(c["geos"], np.arange(c["t0"], c["t1"]))] = 0.0
        h = self.curve(j, alpha, ec)
        h_cf = self.curve(j, alpha, ec, x_cf)
        tw = np.arange(c["t0"], min(c["t1"] + c["tail"], self.T))
        diff = ((h - h_cf)[:, tw] * self.d.rev_scale[:, tw]).sum(1) * c["geos"]
        removed = self.d.spend[j][np.ix_(c["geos"], np.arange(c["t0"], c["t1"]))].sum()
        return diff / removed

    # ---------------------------------------------------------------- likelihood pieces
    def logml(self, st, H, cfac, D=None):
        mu_u, lam_u = self.prior_u(st, cfac)
        ex = self.extra_of(D, st) if self.cal is not None else None
        post = linear.posterior(self.Z_of(H), self.d.W, self.d.y, self.d.mask, st.sigma, mu_u, lam_u,
                                self.mu_s, self.lam_s, extra=ex)
        return post

    # ---------------------------------------------------------------- sampler
    def init_state(self, rng):
        J, M = self.J, self.M
        return State(alpha=np.clip(rng.uniform(0.2, 0.7, J), 0.05, 0.95), ec=rng.uniform(0.6, 1.6, J),
                     roi=np.exp(rng.normal(0.2, 0.3, M)), beta_org=0.1, eta=np.full(J, 0.2),
                     sigma=np.full(self.G, 0.5))

    def run_chain(self, n_warmup, n_draws, seed, log=None, chain=0):
        rng = np.random.default_rng(seed)
        st = self.init_state(rng)
        J, M = self.J, self.M
        H = np.stack([self.curve(j, st.alpha[j], st.ec[j]) for j in range(J)])
        cfac = np.array([self.c_factor(j, H[j]) for j in range(M)])
        D = self.lift_coefs(self.cal["channel"], st.alpha[self.cal["channel"]], st.ec[self.cal["channel"]]) if self.cal else None
        post = self.logml(st, H, cfac, D)
        cur = post.logml
        scale_ae = np.full((J, 2), 0.15)
        scale_r = np.full(J, 0.15)
        scale_eta = np.full(J, 0.3)
        acc = {k: np.zeros(J) for k in ("ae", "r", "eta")}
        n_tot = n_warmup + n_draws
        keep = dict(alpha=[], ec=[], roi=[], beta_org=[], eta=[], sigma=[], u=[], s=[], logml=[])
        t0 = time.time()
        for it in range(n_tot):
            # 1) adstock + saturation, per channel (collapsed)
            for j in range(J):
                z_old = np.array([special.logit(st.alpha[j]), np.log(st.ec[j])])
                z_new = z_old + scale_ae[j] * rng.standard_normal(2)
                a_new, e_new = special.expit(z_new[0]), np.exp(z_new[1])
                lp_new_prior = self.pr.lp_alpha_logit(z_new[0]) + self.pr.lp_ec_log(z_new[1])
                if not np.isfinite(lp_new_prior):
                    continue
                lp_old_prior = self.pr.lp_alpha_logit(z_old[0]) + self.pr.lp_ec_log(z_old[1])
                Hn = H.copy(); Hn[j] = self.curve(j, a_new, e_new)
                cfn = cfac.copy()
                if j < M:
                    cfn[j] = self.c_factor(j, Hn[j])
                Dn = D
                if self.cal is not None and j == self.cal["channel"]:
                    Dn = self.lift_coefs(j, a_new, e_new)
                stn = State(**{**st.__dict__}); stn.alpha = st.alpha.copy(); stn.ec = st.ec.copy()
                stn.alpha[j], stn.ec[j] = a_new, e_new
                pn = self.logml(stn, Hn, cfn, Dn)
                if np.log(rng.random()) < pn.logml + lp_new_prior - cur - lp_old_prior:
                    st, H, cfac, D, post, cur = stn, Hn, cfn, Dn, pn, pn.logml
                    acc["ae"][j] += 1
            # 2) ROI (paid) / organic beta
            for j in range(J):
                stn = State(**{**st.__dict__}); stn.roi = st.roi.copy()
                if j < M:
                    lo = np.log(st.roi[j]); ln = lo + scale_r[j] * rng.standard_normal()
                    stn.roi[j] = np.exp(ln)
                    dprior = self.pr.lp_roi_log(ln) - self.pr.lp_roi_log(lo)
                else:
                    lo = np.log(st.beta_org); ln = lo + scale_r[j] * rng.standard_normal()
                    stn.beta_org = float(np.exp(ln))
                    dprior = self.pr.lp_halfnormal_log(ln, self.pr.org_scale) - self.pr.lp_halfnormal_log(lo, self.pr.org_scale)
                pn = self.logml(stn, H, cfac, D)
                if np.log(rng.random()) < pn.logml - cur + dprior:
                    st, post, cur = stn, pn, pn.logml
                    acc["r"][j] += 1
            # 3) geo spread
            if self.fixed_eta is None:
                for j in range(J):
                    stn = State(**{**st.__dict__}); stn.eta = st.eta.copy()
                    lo = np.log(st.eta[j]); ln = lo + scale_eta[j] * rng.standard_normal()
                    stn.eta[j] = np.exp(ln)
                    dprior = self.pr.lp_halfnormal_log(ln, self.pr.eta_scale) - self.pr.lp_halfnormal_log(lo, self.pr.eta_scale)
                    pn = self.logml(stn, H, cfac, D)
                    if np.log(rng.random()) < pn.logml - cur + dprior:
                        st, post, cur = stn, pn, pn.logml
                        acc["eta"][j] += 1
            # 4) linear coefficients, exactly
            u, s = linear.sample(post, rng)
            st.u, st.s = u, s
            # 5) residual sd per geo (independent Metropolis, vectorised)
            Z = self.Z_of(H)
            res = self.d.y - np.einsum("gtp,gp->gt", Z, u) - np.einsum("gtq,q->gt", self.d.W, s)
            n_g = self.d.mask.sum(1)
            ss = (res ** 2 * self.d.mask).sum(1)
            lo = np.log(st.sigma); ln = lo + 0.08 * rng.standard_normal(self.G)
            def lp(ls):
                sg = np.exp(ls)
                return -n_g * ls - 0.5 * ss / sg ** 2 + stats.halfnorm.logpdf(sg, scale=self.pr.sigma_scale) + ls
            ok = np.log(rng.random(self.G)) < lp(ln) - lp(lo)
            st.sigma = np.where(ok, np.exp(ln), st.sigma)
            post = self.logml(st, H, cfac, D)
            cur = post.logml
            # adaptation during warmup (target ~0.3 acceptance)
            if it < n_warmup and (it + 1) % 50 == 0:
                for k, sc in (("ae", scale_ae), ("r", scale_r), ("eta", scale_eta)):
                    rate = acc[k] / 50
                    f = np.exp(np.clip(rate - 0.3, -0.5, 0.5) * 2)
                    if sc.ndim == 2:
                        sc *= f[:, None]
                    else:
                        sc *= f
                    acc[k][:] = 0
            if it >= n_warmup:
                for k in ("alpha", "ec", "roi", "eta", "sigma", "u", "s"):
                    keep[k].append(np.array(getattr(st, k), dtype=float).copy())
                keep["beta_org"].append(st.beta_org)
                keep["logml"].append(cur)
            if log and (it + 1) % 250 == 0:
                log(f"    chain {chain}: iter {it + 1}/{n_tot}  {time.time() - t0:.0f}s  logml {cur:.1f}  "
                    f"roi {np.round(st.roi, 2).tolist()}")
        return {k: np.array(v) for k, v in keep.items()}


# ============================================================================ diagnostics
def rhat_ess(draws: np.ndarray):
    """Rank-normalised split R-hat and bulk ESS (Vehtari et al. 2021). draws: (chains, n)."""
    c, n = draws.shape
    half = n // 2
    x = np.concatenate([draws[:, :half], draws[:, half:2 * half]], 0)       # split chains
    r = stats.rankdata(x, method="average").reshape(x.shape)
    z = stats.norm.ppf((r - 0.375) / (x.size + 0.25))
    m, k = z.shape
    means = z.mean(1)
    B = k * means.var(ddof=1)
    Wv = z.var(1, ddof=1).mean()
    var = (k - 1) / k * Wv + B / k
    rhat = np.sqrt(var / Wv) if Wv > 0 else np.nan
    # ESS via autocorrelation (Geyer initial positive sequence) on the rank-normalised draws
    acov = np.array([np.correlate(zz - zz.mean(), zz - zz.mean(), "full")[k - 1:] / k for zz in z])
    rho = 1 - (Wv - acov.mean(0)) / var
    rho[0] = 1.0
    s, t = 0.0, 1
    while t + 1 < k:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
        t += 2
    ess = m * k / (1 + 2 * s) if s > -0.5 else np.nan
    return float(rhat), float(ess)
