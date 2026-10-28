"""Exact Gaussian algebra for the conditionally-linear part of the model.

Given the nonlinear media parameters, the model for geo g is

    y_g = Z_g u_g + W_g s + e_g,     e_g ~ N(0, sigma_g^2 I)

with per-geo coefficients u_g (intercept, media effects) and coefficients s shared by all geos
(trend, seasonality, controls, promo). Priors: u_g ~ N(mu_u_g, diag(1/lam_u_g)), s ~ N(mu_s,
diag(1/lam_s)). The posterior precision is "arrowhead" shaped — G small diagonal blocks plus
one shared block — so it is solved through the Schur complement on the shared block instead of
inverting a (G*p + q)-dimensional matrix. Optionally one extra linear Gaussian observation
(the lift-test likelihood) is added as a rank-one update.

Functions return the posterior (for sampling) and the log marginal likelihood p(y | rest),
which is what the collapsed Metropolis steps on nonlinear parameters use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG2PI = np.log(2 * np.pi)


@dataclass
class Posterior:
    A_chol: np.ndarray     # (G, p, p) Cholesky of per-geo blocks A_g
    Ainv_B: np.ndarray     # (G, p, q) A_g^{-1} B_g
    Ainv_a: np.ndarray     # (G, p)    A_g^{-1} a_g
    S_chol: np.ndarray     # (q, q)    Cholesky of the Schur complement
    m_s: np.ndarray        # (q,)  posterior mean of s
    m_u: np.ndarray        # (G, p) posterior mean of u
    logml: float
    extra: dict | None = None


def posterior(Z, W, y, mask, sigma, mu_u, lam_u, mu_s, lam_s, extra=None, want_logml=True) -> Posterior:
    """Z (G,T,p), W (G,T,q), y (G,T), mask (G,T) bool (rows in the likelihood), sigma (G,),
    mu_u/lam_u (G,p), mu_s/lam_s (q,). extra: dict(a_u (G,p), a_s (q,), obs, se) for one
    additional observation obs ~ N(a_u.u + a_s.s, se^2)."""
    G, T, p = Z.shape
    q = W.shape[2]
    w = mask / sigma[:, None] ** 2                          # (G, T) precision weights
    Zw = Z * w[:, :, None]
    A = np.einsum("gtp,gtr->gpr", Zw, Z) + lam_u[:, :, None] * np.eye(p)[None]
    B = np.einsum("gtp,gtq->gpq", Zw, W)
    a = np.einsum("gtp,gt->gp", Zw, y) + lam_u * mu_u
    C = np.einsum("gtq,gtr->qr", W * w[:, :, None], W) + np.diag(lam_s)
    c = np.einsum("gtq,gt->q", W * w[:, :, None], y) + lam_s * mu_s
    L = np.linalg.cholesky(A)
    Ainv_B = _batched_solve(L, B)
    Ainv_a = _batched_solve(L, a[:, :, None])[:, :, 0]
    S = C - np.einsum("gpq,gpr->qr", B, Ainv_B)
    rs = c - np.einsum("gpq,gp->q", B, Ainv_a)
    S = 0.5 * (S + S.T)
    Ls = np.linalg.cholesky(S)
    m_s = np.linalg.solve(Ls.T, np.linalg.solve(Ls, rs))
    m_u = Ainv_a - np.einsum("gpq,q->gp", Ainv_B, m_s)
    logml = np.nan
    if want_logml:
        n = mask.sum()
        logdetP = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum() + 2 * np.log(np.diag(Ls)).sum()
        logdetPrior = np.log(lam_u).sum() + np.log(lam_s).sum()
        quad_y = (w * y ** 2).sum()
        quad_prior = (lam_u * mu_u ** 2).sum() + (lam_s * mu_s ** 2).sum()
        mPm = (m_u * a).sum() + m_s @ c
        logml = (-0.5 * n * LOG2PI - (mask * np.log(sigma)[:, None]).sum()
                 - 0.5 * quad_y - 0.5 * quad_prior + 0.5 * logdetPrior - 0.5 * logdetP + 0.5 * mPm)
    post = Posterior(L, Ainv_B, Ainv_a, Ls, m_s, m_u, float(logml))
    if extra is not None:
        post = _add_observation(post, B, extra)
    return post


def _batched_solve(L, b):
    """Solve A x = b for each g with A = L L^T. L (G,p,p), b (G,p,k)."""
    y = np.linalg.solve(L, b)
    return np.linalg.solve(np.transpose(L, (0, 2, 1)), y)


def precision_solve(post: Posterior, B, v_u, v_s):
    """x = P^{-1} v for the arrowhead precision P (without the extra observation)."""
    Ainv_vu = _batched_solve(post.A_chol, v_u[:, :, None])[:, :, 0]
    rs = v_s - np.einsum("gpq,gp->q", B, Ainv_vu)
    x_s = np.linalg.solve(post.S_chol.T, np.linalg.solve(post.S_chol, rs))
    x_u = Ainv_vu - np.einsum("gpq,q->gp", post.Ainv_B, x_s)
    return x_u, x_s


def _add_observation(post: Posterior, B, ex):
    """Rank-one update for one extra observation obs ~ N(a.theta, se^2)."""
    a_u, a_s, obs, se = ex["a_u"], ex["a_s"], ex["obs"], ex["se"]
    h_u, h_s = precision_solve(post, B, a_u, a_s)            # P^{-1} a
    aPa = (a_u * h_u).sum() + a_s @ h_s
    pred = (a_u * post.m_u).sum() + a_s @ post.m_s
    v = se ** 2 + aPa                                       # predictive variance of obs
    k = (obs - pred) / v
    post.m_u = post.m_u + k * h_u
    post.m_s = post.m_s + k * h_s
    post.logml += -0.5 * (LOG2PI + np.log(v) + (obs - pred) ** 2 / v)
    post.extra = dict(B=B, a_u=a_u, a_s=a_s, se=se, h_u=h_u, h_s=h_s, v=v)
    return post


def sample(post: Posterior, rng, B=None):
    """One exact draw from the Gaussian posterior (with the rank-one update if present)."""
    G, p, _ = post.A_chol.shape
    q = post.S_chol.shape[0]
    z_s = rng.standard_normal(q)
    s = post.m_s + np.linalg.solve(post.S_chol.T, z_s)
    z_u = rng.standard_normal((G, p))
    dev_u = np.linalg.solve(np.transpose(post.A_chol, (0, 2, 1)), z_u[:, :, None])[:, :, 0]
    # u_g | s ~ N(A^{-1}(a - B s), A^{-1}); written relative to the joint mean
    u = post.m_u - np.einsum("gpq,q->gp", post.Ainv_B, s - post.m_s) + dev_u
    if post.extra is not None:
        e = post.extra
        # conditioning by kriging: draw from the prior-of-update, then correct toward the observation
        # (m_u, m_s already include the update; the draw around them needs the covariance shrink)
        eps = rng.standard_normal() * e["se"]
        resid = (e["a_u"] * (u - post.m_u)).sum() + e["a_s"] @ (s - post.m_s) + eps
        u = u - e["h_u"] * resid / e["v"]
        s = s - e["h_s"] * resid / e["v"]
    return u, s
