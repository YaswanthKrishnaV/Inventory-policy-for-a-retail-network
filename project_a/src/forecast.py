"""Probabilistic, coherent 28-day demand forecast.

Model
-----
1. Mean: one global gradient-boosted tree model (Poisson loss) over all item x store series,
   trained on *recovered* demand with censored days masked out of the targets (lags use
   imputed values). Backend: scikit-learn HistGradientBoosting (default) or LightGBM.
2. Spread: demand_{i,t} ~ Poisson(mu_{i,t} * G_item * G_{i,t}), a gamma-Poisson (negative
   binomial) mixture with two multiplicative shocks:
     * G_item ~ Gamma(mean 1, var s_p^2): level error shared by the item's 4 stores and
       persisting over the whole horizon (forecast-level uncertainty, item-wide promotions).
     * G_{i,t} ~ Gamma(mean 1, var s_d^2): independent day-to-day noise.
   Both variances are estimated out-of-sample: the model is refit 28 days before the origin,
   used to forecast the last 28 observed days, and the residuals are matched by moments
   (daily cells by predicted-mean band; 28-day item-network sums by item-volume band).
3. Coherence: S sample paths are drawn per series; the DC path is the sum of its 4 store
   paths, so DC quantiles come from summed paths (quantiles are never added).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import demand_recovery as dr
from .features import CAT_FEATURES, build_feature_arrays, demand_history, gather


# ----------------------------------------------------------------------------- model backends
class GBM:
    def __init__(self, cfg: dict, names: list[str]):
        fc = cfg["forecast"]
        self.backend = fc["backend"]
        self.names = names
        self.cat = [n in CAT_FEATURES for n in names]
        p = fc["params"]
        if self.backend == "lightgbm":
            import lightgbm as lgb
            self.m = lgb.LGBMRegressor(objective="poisson", learning_rate=p["learning_rate"],
                                       n_estimators=p["max_iter"], num_leaves=p["max_leaf_nodes"],
                                       min_child_samples=p["min_samples_leaf"], reg_lambda=p["l2"],
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       random_state=cfg["seed"], verbose=-1)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.m = HistGradientBoostingRegressor(loss="poisson", learning_rate=p["learning_rate"],
                                                   max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
                                                   min_samples_leaf=p["min_samples_leaf"],
                                                   l2_regularization=p["l2"], categorical_features=self.cat,
                                                   early_stopping=False, random_state=cfg["seed"])

    def fit(self, X, y):
        if self.backend == "lightgbm":
            self.m.fit(X, y, categorical_feature=[i for i, c in enumerate(self.cat) if c])
        else:
            self.m.fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self.m.predict(X), 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- fit / predict
def _train_rows(rec, start, origin, raw):
    st = rec.status[:, start:origin]
    if raw:
        ok = ~np.isin(st, (dr.PRELAUNCH, dr.UNLISTED, dr.CLOSED))
    else:
        ok = st == dr.OK
    r, c = np.nonzero(ok)
    return r, c


def fit_mean_model(panel, rec, origin: int, cfg: dict, raw: bool = False, rng=None):
    """Train on target days [origin - train_days, origin) using rec (recovery as of `origin`)."""
    fc = cfg["forecast"]
    start = max(origin - fc["train_days"], 400)
    Y = demand_history(panel, rec, raw=raw)
    F = build_feature_arrays(panel, Y, origin, start, origin, cfg)
    names = list(F.keys())
    r, c = _train_rows(rec, start, origin, raw)
    if fc["subsample"] < 1:
        rng = rng or np.random.default_rng(cfg["seed"])
        keep = rng.random(len(r)) < fc["subsample"]
        r, c = r[keep], c[keep]
    X = gather(F, r, c, names)
    y = (panel.sales[:, start:origin][r, c] if raw else rec.demand[:, start:origin][r, c]).astype(np.float32)
    del F
    model = GBM(cfg, names).fit(X, y)
    return model


def predict_mean(model: GBM, panel, rec, origin: int, horizon: int, cfg: dict, raw: bool = False):
    """Mean forecast for days [origin, origin + horizon). rec must be recovery as of `origin`."""
    Y = demand_history(panel, rec, raw=raw)
    stop = origin + horizon
    F = build_feature_arrays(panel, Y, origin, origin, stop, cfg)
    N = Y.shape[0]
    r = np.repeat(np.arange(N), horizon)
    c = np.tile(np.arange(horizon), N)
    mu = model.predict(gather(F, r, c, model.names)).reshape(N, horizon)
    # not ranged at the origin (dormant) or not listed in the forecast week -> no demand
    mu[rec.status[:, origin - 1] == dr.DORMANT] = 0.0
    P = panel.price
    if cfg["prices"]["use_future_prices"]:
        listed = ~np.isnan(P[:, origin:stop])
    else:
        listed = np.repeat(~np.isnan(P[:, origin - 1:origin]), horizon, 1)
    mu[~listed] = 0.0
    return mu


# ----------------------------------------------------------------------------- dispersion
@dataclass
class Dispersion:
    daily_edges: np.ndarray
    inv_r: np.ndarray          # total NB 1/r per daily-mean band
    item_edges: np.ndarray
    s_p2: np.ndarray           # persistent item-level variance per item-volume band
    diag: dict = field(default_factory=dict)

    def cell_params(self, mu: np.ndarray):
        """(s_p2 per series (N,), s_d2 per cell (N,H)) for a mean forecast mu (N,H)."""
        N = mu.shape[0]
        item_tot = mu.reshape(N // 4, 4, -1).sum((1, 2))
        sp = self.s_p2[np.clip(np.searchsorted(self.item_edges, item_tot, "right") - 1, 0, len(self.s_p2) - 1)]
        sp_series = np.repeat(sp, 4)
        ir = self.inv_r[np.clip(np.searchsorted(self.daily_edges, mu, "right") - 1, 0, len(self.inv_r) - 1)]
        sd = np.maximum((1 + ir) / (1 + sp_series[:, None]) - 1, 1e-4)
        return sp_series.astype(np.float32), sd.astype(np.float32)


def estimate_dispersion(mu: np.ndarray, y: np.ndarray, ok: np.ndarray, cfg: dict) -> Dispersion:
    """Moment-match residuals of a genuine 28-day-ahead forecast.

    mu, y: (N, H); ok: cells with an uncensored observation.
    Daily: Var(y) = mu + mu^2 / r  ->  1/r = sum((y-mu)^2 - mu) / sum(mu^2) within a mu band.
    Item level: over the 28-day network sum Y_k of item k, Var(Y) = sum of daily variances
    (independent part) + s_p^2 * M_k^2  ->  s_p^2 = sum((Y-M)^2 - V_indep) / sum(M^2).
    """
    fc = cfg["forecast"]
    de = np.array(fc["dispersion_bins"], float)
    inv_r = np.zeros(len(de) - 1)
    m, yy = mu[ok], y[ok]
    b = np.clip(np.searchsorted(de, m, "right") - 1, 0, len(inv_r) - 1)
    for k in range(len(inv_r)):
        s = b == k
        if s.sum() < 200 or (m[s] ** 2).sum() == 0:
            inv_r[k] = inv_r[k - 1] if k else 1.0
            continue
        inv_r[k] = max(((yy[s] - m[s]) ** 2 - m[s]).sum() / (m[s] ** 2).sum(), 1e-3)
    # item-level persistent part (only cells observed in all 4 stores enter the sums)
    N, H = mu.shape
    mu4 = np.where(ok, mu, 0).reshape(N // 4, 4, H)
    y4 = np.where(ok, y, 0).reshape(N // 4, 4, H)
    ir_cell = inv_r[np.clip(np.searchsorted(de, mu, "right") - 1, 0, len(inv_r) - 1)]
    v_ind = np.where(ok, mu + mu ** 2 * ir_cell, 0).reshape(N // 4, 4, H).sum((1, 2))
    M, Yk = mu4.sum((1, 2)), y4.sum((1, 2))
    ie = np.array(fc["item_bins"], float)
    s_p2 = np.zeros(len(ie) - 1)
    bi = np.clip(np.searchsorted(ie, M, "right") - 1, 0, len(s_p2) - 1)
    for k in range(len(s_p2)):
        s = (bi == k) & (M > 0)
        if s.sum() < 20:
            s_p2[k] = s_p2[k - 1] if k else 0.05
            continue
        s_p2[k] = max(((Yk[s] - M[s]) ** 2 - v_ind[s]).sum() / (M[s] ** 2).sum(), 0.0)
    # the persistent part cannot exceed the total multiplicative variance of a typical cell
    s_p2 = np.minimum(s_p2, np.median(inv_r))
    diag = dict(bias=float(yy.sum() / max(m.sum(), 1e-9)), n_cells=int(ok.sum()))
    return Dispersion(de, inv_r, ie, s_p2, diag)


# ----------------------------------------------------------------------------- sample paths
def sample_paths(mu: np.ndarray, disp: Dispersion, n_paths: int, rng, chunk: int = 50) -> np.ndarray:
    """(S, N, H) int16 demand paths. Item shock shared across stores and days; daily shock iid."""
    N, H = mu.shape
    sp, sd = disp.cell_params(mu)
    out = np.empty((n_paths, N, H), np.int16)
    sp_item = sp[::4]
    for a in range(0, n_paths, chunk):
        s = min(chunk, n_paths - a)
        gi = rng.gamma(1 / np.maximum(sp_item, 1e-6), np.maximum(sp_item, 1e-6), size=(s, N // 4))
        gi = np.repeat(gi, 4, 1)[:, :, None]
        gd = rng.gamma(1 / sd, sd, size=(s, N, H))
        lam = mu[None] * gi * gd
        out[a:a + s] = np.minimum(rng.poisson(lam), np.iinfo(np.int16).max)
    return out


def to_dc(paths: np.ndarray) -> np.ndarray:
    """Store paths (S, N, H) -> DC paths (S, N/4, H): exact sum over the item's 4 stores."""
    S, N, H = paths.shape
    return paths.reshape(S, N // 4, 4, H).sum(2, dtype=np.int32)


def quantiles(paths: np.ndarray, qs) -> np.ndarray:
    """(Q, N, H) quantiles over the path axis."""
    return np.quantile(paths, qs, axis=0, method="linear").astype(np.float32)


# ----------------------------------------------------------------------------- orchestration
@dataclass
class Forecast:
    origin: int               # first forecast day index (0-based) = number of observed days
    horizon: int
    mu: np.ndarray            # (N, H)
    disp: Dispersion
    raw: bool = False
    timings: dict = field(default_factory=dict)
    adj: dict | None = None

    def paths(self, n_paths: int, seed: int) -> np.ndarray:
        return sample_paths(self.mu, self.disp, n_paths, np.random.default_rng(seed))


def _cache_file(origin, cfg, raw):
    import hashlib, json
    from .config import path
    key = json.dumps([cfg["forecast"], cfg["recovery"], cfg["prices"], cfg["calendar"]["forecast_horizon"],
                      origin, raw, cfg["seed"]], sort_keys=True, default=str)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return path(cfg, "cache_dir") / f"forecast_{origin}_{'raw' if raw else 'rec'}_{h}.pkl"


def run_forecast(panel, origin: int, cfg: dict, raw: bool = False, model=None, log=print,
                 use_cache: bool = True) -> tuple[Forecast, GBM]:
    """Train (unless `model` given) on data before `origin`, calibrate spread, forecast H days.

    origin is a 0-based day index: data with t < origin is observed. For cutoff d_c, origin = c.
    Results are cached in cache/ keyed by origin + forecast/recovery config, so the backtest,
    tuning and final run share fitted models.
    """
    import pickle
    cf = _cache_file(origin, cfg, raw)
    if use_cache and model is None and cf.exists():
        f, m = pickle.loads(cf.read_bytes())
        log(f"  forecast origin d_{origin + 1}: loaded from cache")
        return f, m
    f, m = _run_forecast(panel, origin, cfg, raw, model, log)
    if use_cache:
        cf.write_bytes(pickle.dumps((f, m)))
    return f, m


def _run_forecast(panel, origin, cfg, raw, model, log):
    H = cfg["calendar"]["forecast_horizon"]
    tm = {}
    t0 = time.time()
    rec = dr.recover(panel, origin, cfg)
    if model is None:
        model = fit_mean_model(panel, rec, origin, cfg, raw=raw)
    tm["fit"] = time.time() - t0
    # out-of-sample calibration: model trained H days earlier, forecasting the last H observed days
    t0 = time.time()
    cal_origin = origin - H
    rec_c = dr.recover(panel, cal_origin, cfg)
    model_c = fit_mean_model(panel, rec_c, cal_origin, cfg, raw=raw)
    mu_c = predict_mean(model_c, panel, rec_c, cal_origin, H, cfg, raw=raw)
    y_c = rec.demand[:, cal_origin:origin]
    ok_c = rec.status[:, cal_origin:origin] == dr.OK
    disp = estimate_dispersion(mu_c, y_c, ok_c, cfg)
    tm["calibrate"] = time.time() - t0
    mu = predict_mean(model, panel, rec, origin, H, cfg, raw=raw)
    adj = estimate_adjustments(panel, rec_c, rec, cal_origin, origin, mu_c, cfg, raw=raw)
    mu = apply_adjustments(mu, adj, panel, rec, origin, cfg, raw=raw)
    disp.diag.update(adj["diag"])
    log(f"  forecast origin d_{origin + 1}: fit {tm['fit']:.0f}s, calibrate {tm['calibrate']:.0f}s, "
        f"calib bias (actual/forecast) {disp.diag['bias']:.3f}, dormant resume factor {adj['resume']:.3f}")
    return Forecast(origin, H, mu, disp, raw, tm, adj), model


# ----------------------------------------------------------------------------- post-model adjustments
def _pre_rate(panel, rec, origin, raw):
    """Mean demand over the last 364 days on days the item was sellable (NaN-aware)."""
    Y = demand_history(panel, rec, raw=raw)[:, max(0, origin - 364):origin]
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(Y, 1))


def estimate_adjustments(panel, rec_c, rec, cal_origin, origin, mu_c, cfg, raw=False) -> dict:
    """Two corrections measured on the out-of-sample calibration window [cal_origin, origin).

    * level: actual / forecast by store x dept (lag>=28 features miss recent trend), shrunk.
    * resume: series dormant at cal_origin are forecast at 0 by the model, yet some restart;
      resume = their actual demand in the window / (their pre-dormancy daily rate x H).
    """
    ac = cfg["forecast"]["adjust"]
    H = origin - cal_origin
    y = rec.demand[:, cal_origin:origin]
    ok = rec.status[:, cal_origin:origin] == dr.OK
    dorm_c = rec_c.status[:, cal_origin - 1] == dr.DORMANT
    groups = (panel.meta["store_id"] + "|" + panel.meta["dept_id"]).to_numpy()
    level = np.ones(len(groups), np.float32)
    ratios = {}
    for g in np.unique(groups):
        r = (groups == g) & ~dorm_c
        num, den = y[r][ok[r]].sum(), mu_c[r][ok[r]].sum()
        ratio = num / den if den > 0 else 1.0
        f = float(np.clip(1 + ac["level_shrink"] * (ratio - 1), *ac["level_clip"]))
        level[groups == g] = f
        ratios[g] = round(float(ratio), 3)
    pre = _pre_rate(panel, rec_c, cal_origin, raw)
    den = pre[dorm_c].sum() * H
    resume = float(rec.demand[dorm_c, cal_origin:origin].sum() / den) if den > 0 else 0.0
    resume = float(np.clip(resume, 0, 1))
    return dict(level=level, resume=resume if ac["dormant_resume"] else 0.0,
                diag=dict(level_ratio=ratios, resume_factor=resume))


def apply_adjustments(mu, adj, panel, rec, origin, cfg, raw=False):
    mu = mu * adj["level"][:, None]
    dorm = rec.status[:, origin - 1] == dr.DORMANT
    if adj["resume"] > 0 and dorm.any():
        listed = ~np.isnan(panel.price[:, origin - 1])
        pre = _pre_rate(panel, rec, origin, raw)
        mu[dorm & listed] = (adj["resume"] * pre[dorm & listed])[:, None]
    return mu.astype(np.float32)
