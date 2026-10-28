"""Recover unconstrained demand from censored sales.

Recorded sales = min(demand, stock). A zero (or a low number) can mean four different things,
so every series-day gets a status code:

  OK         0  observed, uncensored -> usable as a demand observation
  PRELAUNCH  1  before the item's first ever sale in this store (price row may exist, item not ranged)
  UNLISTED   2  no sell_prices row that week -> item not on sale (M5's only explicit listing signal)
  CLOSED     3  store-wide near-zero day (Christmas) -> no demand that day
  DISRUPTED  4  store-wide shortfall (store total far below its same-weekday norm) -> censored
  STOCKOUT   5  zero run too long to be chance given the item's local selling rate -> censored
  DORMANT    6  zero run >= dormant_days: not replenished for months; treated as not ranged

"No-demand" statuses (1, 2, 3, 6) are excluded from training and count as zero demand.
"Censored" statuses (4, 5) are excluded from training targets and get an imputed demand.

Stockout detector (zero-run test)
---------------------------------
For each maximal run of k zero-sale days (counted over available days only), estimate the item's
local rate lambda from up to W clean days before and after the run. Under a negative binomial
with mean mu_t = lambda * dow_factor_t and size r, P(zero on day t) = (r / (r + mu_t))^r, so
log P(run) = sum_t r * log(r / (r + mu_t)). Flag the run if P(run) < p_threshold. Repeat once
with flagged days removed from the neighbour windows (they bias lambda down on the first pass).

Only data with t < end is used, so the detector can be run "as of" any cutoff without leakage.
Partial-day stockouts (low but non-zero sales) are not detectable from daily data; see DATA_ISSUES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OK, PRELAUNCH, UNLISTED, CLOSED, DISRUPTED, STOCKOUT, DORMANT = range(7)
STATUS_NAMES = ["ok", "prelaunch", "unlisted", "closed", "disrupted", "stockout", "dormant"]
NO_DEMAND = (PRELAUNCH, UNLISTED, CLOSED, DORMANT)
CENSORED = (DISRUPTED, STOCKOUT)


@dataclass
class Recovery:
    status: np.ndarray    # (N, end) uint8 status code
    demand: np.ndarray    # (N, end) float32: recorded on OK, imputed on censored, 0 on no-demand
    rate: np.ndarray      # (N, end) float32: local rate used for imputation (NaN where not imputed)
    dow_factor: np.ndarray  # (n_store_dept, 7)
    end: int

    @property
    def train_mask(self) -> np.ndarray:
        return self.status == OK

    @property
    def censored_mask(self) -> np.ndarray:
        return np.isin(self.status, CENSORED)


# ----------------------------------------------------------------------------- store level
def store_day_status(sales, meta, cal, end, cfg):
    """Per store-day: CLOSED / DISRUPTED / OK by comparing store total with same-weekday neighbours."""
    rc = cfg["recovery"]
    stores = meta["store_id"].to_numpy()
    out = {}
    for s in np.unique(stores):
        tot = sales[stores == s, :end].sum(0).astype(float)
        ref = np.full((8, end), np.nan)
        for j, lag in enumerate([-28, -21, -14, -7, 7, 14, 21, 28]):
            src = np.arange(end) + lag
            ok = (src >= 0) & (src < end)
            ref[j, ok] = tot[src[ok]]
        expected = np.nanmedian(ref, axis=0)
        st = np.zeros(end, np.uint8)
        st[tot < rc["store_disruption_ratio"] * expected] = DISRUPTED
        st[tot < rc["closed_ratio"] * expected] = CLOSED
        st[cal["is_christmas"].to_numpy()[:end] == 1] = CLOSED
        out[s] = st
    return out


def dow_factors(sales, avail, meta, cal, end):
    """Multiplicative day-of-week factors per (store, dept), from clean days only (mean = 1)."""
    groups = (meta["store_id"] + "|" + meta["dept_id"]).to_numpy()
    dow = cal["dow"].to_numpy()[:end]
    keys = np.unique(groups)
    fac = np.ones((len(keys), 7))
    for g, key in enumerate(keys):
        rows = groups == key
        x = np.where(avail[rows, :end], sales[rows, :end], 0).sum(0).astype(float)
        n = avail[rows, :end].sum(0).astype(float)
        last = slice(max(0, end - 728), end)  # last two years: DOW pattern drifts slowly
        per_day = np.divide(x[last], n[last], out=np.zeros_like(x[last]), where=n[last] > 0)
        f = pd.Series(per_day).groupby(dow[last]).mean().reindex(range(7)).to_numpy()
        fac[g] = f / np.nanmean(f)
    group_idx = pd.Series(np.arange(len(keys)), index=keys).loc[groups].to_numpy()
    return fac, group_idx


# ----------------------------------------------------------------------------- item level
def _runs(mask):
    """Start/end (inclusive) of maximal True runs in a 1-D bool array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def detect_series(x, f, rc, flagged=None):
    """Zero-run test on one compressed series (available days only).

    x: sales on available days; f: DOW factor on those days.
    Returns (flag per day, local lambda per day (nan off-run), run_len per day).
    """
    n = len(x)
    flag = np.zeros(n, bool)
    lam_day = np.full(n, np.nan)
    if n == 0:
        return flag, lam_day
    starts, ends = _runs(x == 0)
    if len(starts) == 0:
        return flag, lam_day
    W, min_n = rc["neighbour_window_days"], rc["min_neighbour_days"]
    lo_r, hi_r = rc["dispersion_clip"]
    thr = np.log(rc["p_threshold"])
    clean = np.ones(n, bool) if flagged is None else ~flagged
    # series-level NB size from clean days (method of moments, de-seasonalised)
    y = (x / f)[clean]
    m, v = y.mean() if len(y) else 0.0, y.var() if len(y) > 1 else 0.0
    r = hi_r if v <= m or m == 0 else np.clip(m * m / (v - m), lo_r, hi_r)
    cs = np.concatenate([[0.0], np.cumsum(np.where(clean, x / f, 0.0))])
    cn = np.concatenate([[0], np.cumsum(clean)])
    b0 = np.maximum(starts - W, 0)
    a1 = np.minimum(ends + 1 + W, n)
    tot = (cs[starts] - cs[b0]) + (cs[a1] - cs[ends + 1])
    cnt = (cn[starts] - cn[b0]) + (cn[a1] - cn[ends + 1])
    lam = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    lengths = ends - starts + 1
    run_id = np.repeat(np.arange(len(starts)), lengths)
    pos = np.concatenate([np.arange(s, e + 1) for s, e in zip(starts, ends)])
    mu = lam[run_id] * f[pos]
    logp = np.bincount(run_id, weights=r * np.log(r / (r + mu)), minlength=len(starts))
    bad = (logp < thr) & (cnt >= min_n)
    flag[pos] = bad[run_id]
    lam_day[pos] = lam[run_id]
    return flag, lam_day


def recover(panel, end: int, cfg: dict) -> Recovery:
    """Run the full status classification + imputation using data with t < end only."""
    rc = cfg["recovery"]
    S = panel.sales[:, :end]
    P = panel.price[:, :end]
    meta, cal = panel.meta, panel.cal
    N = S.shape[0]
    status = np.zeros((N, end), np.uint8)

    # 1) listing / launch
    status[np.isnan(P)] = UNLISTED
    has_sale = S > 0
    first = np.where(has_sale.any(1), has_sale.argmax(1), end)
    pre = np.arange(end)[None, :] < first[:, None]
    status[pre & (status == OK)] = PRELAUNCH

    # 2) store-wide closures / disruptions
    sday = store_day_status(panel.sales, meta, cal, end, cfg)
    stores = meta["store_id"].to_numpy()
    for s, st in sday.items():
        rows = stores == s
        blk = status[rows]
        blk[(blk == OK) & (st[None, :] != OK)] = np.broadcast_to(st, blk.shape)[(blk == OK) & (st[None, :] != OK)]
        status[rows] = blk

    # 3) item-level zero runs on available days
    avail = status == OK
    fac, gidx = dow_factors(S, avail, meta, cal, end)
    dow = cal["dow"].to_numpy()[:end]
    rate = np.full((N, end), np.nan, np.float32)
    dormant_days = rc.get("dormant_days", 56)
    for i in range(N):
        pos = np.flatnonzero(avail[i])
        if len(pos) == 0:
            continue
        x = S[i, pos].astype(float)
        f = fac[gidx[i], dow[pos]]
        flagged = None
        for _ in range(rc["passes"]):
            flagged, lam = detect_series(x, f, rc, flagged)
        # dormant: very long zero runs are "not ranged", whatever the test says
        st_, en_ = _runs(x == 0)
        long_run = np.zeros(len(x), bool)
        for a, b in zip(st_, en_):
            if b - a + 1 >= dormant_days:
                long_run[a:b + 1] = True
        status[i, pos[flagged & ~long_run]] = STOCKOUT
        status[i, pos[long_run]] = DORMANT
        rate[i, pos[flagged & ~long_run]] = (lam * f)[flagged & ~long_run]

    # 4) imputation for store-wide disruptions: local clean rate x DOW factor, at least recorded sales
    clean = status == OK
    W = rc["neighbour_window_days"]
    dis = status == DISRUPTED
    if dis.any():
        xs = np.where(clean, S / fac[gidx][:, dow], 0.0)
        cs = np.concatenate([np.zeros((N, 1)), np.cumsum(xs, 1)], 1)
        cn = np.concatenate([np.zeros((N, 1)), np.cumsum(clean, 1)], 1)
        for t in np.flatnonzero(dis.any(0)):
            a, b = max(0, t - W), min(end, t + W + 1)
            lam = np.divide(cs[:, b] - cs[:, a], cn[:, b] - cn[:, a],
                            out=np.zeros(N), where=(cn[:, b] - cn[:, a]) > 0)
            rows = dis[:, t]
            rate[rows, t] = np.maximum(lam[rows] * fac[gidx[rows], dow[t]], S[rows, t])

    demand = np.where(status == OK, S, 0).astype(np.float32)
    cen = np.isin(status, CENSORED)
    demand[cen] = rate[cen]
    return Recovery(status=status, demand=demand, rate=rate, dow_factor=fac, end=end)


# ----------------------------------------------------------------------------- validation
def injection_test(panel, rec: Recovery, cfg: dict, rng=None) -> pd.DataFrame:
    """Plant synthetic stockouts in clean stretches and measure how many the detector finds.

    For each velocity band and run length k: pick a random series and a random window of k
    consecutive OK days with positive total sales, zero them, rerun the detector on that series,
    and record (a) day-level recall and (b) imputed / true units removed.
    """
    rc = cfg["recovery"]
    ic = rc["injection"]
    rng = rng or np.random.default_rng(ic["seed"])
    end = rec.end
    S = panel.sales[:, :end]
    dow = panel.cal["dow"].to_numpy()[:end]
    groups = (panel.meta["store_id"] + "|" + panel.meta["dept_id"]).to_numpy()
    keys = np.unique(groups)
    gidx = pd.Series(np.arange(len(keys)), index=keys).loc[groups].to_numpy()
    ok = rec.status == OK
    vel = np.divide((S * ok)[:, -365:].sum(1), ok[:, -365:].sum(1), out=np.zeros(S.shape[0]),
                    where=ok[:, -365:].sum(1) > 0)
    bands = [(0, 0.5), (0.5, 2), (2, 5), (5, np.inf)]
    rows = []
    for lo, hi in bands:
        cand = np.flatnonzero((vel >= lo) & (vel < hi))
        for k in ic["lengths"]:
            found, n_days, imp, true = 0, 0, 0.0, 0.0
            tries = 0
            while n_days < ic["n_per_band"] * k and tries < ic["n_per_band"] * 20:
                tries += 1
                i = rng.choice(cand)
                pos = np.flatnonzero(rec.status[i] != DORMANT)
                pos = pos[np.isin(rec.status[i, pos], (OK, STOCKOUT))]  # what the detector sees
                okc = ok[i, pos]
                if len(pos) < 200:
                    continue
                j = rng.integers(60, len(pos) - 60 - k)
                if not okc[j:j + k].all() or S[i, pos[j:j + k]].sum() == 0:
                    continue
                x = S[i, pos].astype(float).copy()
                truth = x[j:j + k].sum()
                x[j:j + k] = 0
                f = rec.dow_factor[gidx[i], dow[pos]]
                flagged = None
                for _ in range(rc["passes"]):
                    flagged, lam = detect_series(x, f, rc, flagged)
                hit = flagged[j:j + k]
                found += hit.sum()
                n_days += k
                if hit.all():
                    imp += (lam * f)[j:j + k].sum()
                    true += truth
            rows.append(dict(band=f"{lo}-{hi}", run_len=k, days=n_days,
                             recall=found / max(n_days, 1),
                             imputed_over_true=imp / true if true else np.nan))
    return pd.DataFrame(rows)


def summarise(panel, rec: Recovery) -> dict:
    """Headline numbers for DATA_ISSUES.md."""
    meta = panel.meta
    st = rec.status
    S = panel.sales[:, :rec.end].astype(float)
    out = {"status_share": {STATUS_NAMES[c]: float((st == c).mean()) for c in range(7)}}
    by = {}
    for key in ["dept_id", "store_id"]:
        g = {}
        for v in np.unique(meta[key]):
            rows = (meta[key] == v).to_numpy()
            last = slice(rec.end - 365, rec.end)
            rec_d = rec.demand[rows, last].sum()
            raw = np.where(np.isin(st[rows, last], NO_DEMAND), 0, S[rows, last]).sum()
            g[v] = dict(stockout_day_share=float((st[rows, last] == STOCKOUT).mean()),
                        hidden_demand_share=float(1 - raw / rec_d) if rec_d else np.nan,
                        dormant_share=float((st[rows, last] == DORMANT).mean()))
        by[key] = g
    out["last_365"] = by
    return out


def null_test(panel, rec: Recovery, cfg: dict, n_series: int = 500, seed: int = 11) -> pd.DataFrame:
    """False-alarm check: simulate stockout-free NB sales from each series' own clean history
    (same availability pattern, DOW factors, local level, dispersion), run the detector, and
    report the share of days flagged. Ideally ~0; anything flagged here is a false positive."""
    rc = cfg["recovery"]
    rng = np.random.default_rng(seed)
    end = rec.end
    S = panel.sales[:, :end]
    dow = panel.cal["dow"].to_numpy()[:end]
    groups = (panel.meta["store_id"] + "|" + panel.meta["dept_id"]).to_numpy()
    keys = np.unique(groups)
    gidx = pd.Series(np.arange(len(keys)), index=keys).loc[groups].to_numpy()
    ok = rec.status == OK
    lo_r, hi_r = rc["dispersion_clip"]
    rows = []
    for i in rng.choice(S.shape[0], n_series, replace=False):
        pos = np.flatnonzero(np.isin(rec.status[i], (OK, STOCKOUT)))
        if len(pos) < 200:
            continue
        f = rec.dow_factor[gidx[i], dow[pos]]
        # level: 28-day rolling mean of clean de-seasonalised sales, forward-filled
        y = np.where(ok[i, pos], S[i, pos] / f, np.nan)
        lvl = pd.Series(y).rolling(56, min_periods=7, center=True).mean().ffill().bfill().to_numpy()
        m, v = np.nanmean(y), np.nanvar(y)
        r = hi_r if v <= m else float(np.clip(m * m / (v - m), lo_r, hi_r))
        mu = np.maximum(lvl * f, 1e-6)
        x = rng.negative_binomial(r, r / (r + mu)).astype(float)
        flagged = None
        for _ in range(rc["passes"]):
            flagged, _ = detect_series(x, f, rc, flagged)
        rows.append(dict(series=int(i), vel=float(np.nanmean(mu)), flagged_share=float(flagged.mean())))
    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df["vel"], [0, 0.5, 2, 5, np.inf], labels=["0-0.5", "0.5-2", "2-5", "5-inf"])
    return df.groupby("band", observed=True)["flagged_share"].agg(["mean", "count"]).reset_index()
