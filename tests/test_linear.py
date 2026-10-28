"""The structured (arrowhead) Gaussian algebra must match brute-force dense algebra exactly."""
import numpy as np
from scipy import stats

from project_b.src import linear


def problem(seed=0, G=5, T=30, p=3, q=4, with_extra=False):
    r = np.random.default_rng(seed)
    Z = r.normal(size=(G, T, p)); W = r.normal(size=(G, T, q)); y = r.normal(size=(G, T))
    mask = r.random((G, T)) > 0.2
    sigma = r.uniform(0.5, 2, G)
    mu_u = r.normal(size=(G, p)); lam_u = r.uniform(0.5, 3, (G, p))
    mu_s = r.normal(size=q); lam_s = r.uniform(0.5, 3, q)
    extra = None
    if with_extra:
        extra = dict(a_u=r.normal(size=(G, p)), a_s=r.normal(size=q), obs=0.7, se=0.4)
    return Z, W, y, mask, sigma, mu_u, lam_u, mu_s, lam_s, extra


def dense(Z, W, y, mask, sigma, mu_u, lam_u, mu_s, lam_s, extra):
    G, T, p = Z.shape; q = W.shape[2]
    X = np.zeros((G * T, G * p + q))
    for g in range(G):
        X[g * T:(g + 1) * T, g * p:(g + 1) * p] = Z[g]
        X[g * T:(g + 1) * T, G * p:] = W[g]
    keep = mask.reshape(-1)
    X, yy = X[keep], y.reshape(-1)[keep]
    sd = np.repeat(sigma, T)[keep]
    m0 = np.concatenate([mu_u.reshape(-1), mu_s]); V0 = np.diag(1 / np.concatenate([lam_u.reshape(-1), lam_s]))
    if extra is not None:
        a = np.concatenate([extra["a_u"].reshape(-1), extra["a_s"]])
        X = np.vstack([X, a]); yy = np.append(yy, extra["obs"]); sd = np.append(sd, extra["se"])
    Sy = X @ V0 @ X.T + np.diag(sd ** 2)
    logml = stats.multivariate_normal(X @ m0, Sy).logpdf(yy)
    P = np.linalg.inv(V0) + X.T @ np.diag(1 / sd ** 2) @ X
    Vp = np.linalg.inv(P)
    mp = Vp @ (np.linalg.inv(V0) @ m0 + X.T @ (yy / sd ** 2))
    return mp, Vp, logml


def test_posterior_and_marginal_match_dense():
    for extra in (False, True):
        args = problem(with_extra=extra)
        post = linear.posterior(*args[:9], extra=args[9])
        mp, Vp, logml = dense(*args)
        G, p = args[5].shape
        np.testing.assert_allclose(post.m_u.reshape(-1), mp[:G * p], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(post.m_s, mp[G * p:], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(post.logml, logml, rtol=1e-9)


def test_sampling_covariance_matches_dense():
    for extra in (False, True):
        args = problem(seed=3, G=3, T=20, p=2, q=2, with_extra=extra)
        post = linear.posterior(*args[:9], extra=args[9])
        mp, Vp, _ = dense(*args)
        rng = np.random.default_rng(1)
        draws = np.array([np.concatenate([u.reshape(-1), s]) for u, s in (linear.sample(post, rng) for _ in range(40000))])
        np.testing.assert_allclose(draws.mean(0), mp, atol=0.02)
        np.testing.assert_allclose(np.cov(draws.T), Vp, atol=0.02)
