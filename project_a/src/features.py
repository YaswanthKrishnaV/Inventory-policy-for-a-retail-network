"""Feature matrices for the global demand model.

Every feature for target day t uses demand observed at t-28 or earlier ("lag >= 28"), so a
model trained once can forecast any day 1..28 after the forecast origin without recursion,
and the same feature value is valid for training rows and forecast rows.

Features are built as wide (series x days) arrays over a window [start, stop) and then
gathered into a row matrix for the requested (series, day) pairs.

Inputs
------
Y : (N, T) float32 "demand history" — recovered demand (or raw sales for the raw-sales
    benchmark) with NaN on days where the item could not be bought (not launched, not listed,
    store closed, dormant). NaN-aware rolling stats therefore average only over real selling days.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
import pandas as pd

LAG = 28
CAT_FEATURES = ["dept", "store", "dow", "month", "event_type"]


def demand_history(panel, rec, raw: bool = False) -> np.ndarray:
    """Series used for lags. Recovered: imputed on censored days. Raw: recorded sales as-is."""
    from .demand_recovery import NO_DEMAND, PRELAUNCH, UNLISTED, CLOSED
    end = rec.end
    if raw:
        # what a naive analyst would use: recorded sales whenever the item is listed and the store open
        y = panel.sales[:, :end].astype(np.float32)
        y[np.isin(rec.status, (PRELAUNCH, UNLISTED, CLOSED))] = np.nan
    else:
        y = rec.demand.astype(np.float32).copy()
        y[np.isin(rec.status, NO_DEMAND)] = np.nan
    return y


def _shift(a: np.ndarray, k: int, start: int, stop: int) -> np.ndarray:
    """out[:, j] = a[:, start + j - k]; NaN where out of range."""
    n, T = a.shape
    out = np.full((n, stop - start), np.nan, np.float32)
    src = np.arange(start, stop) - k
    ok = (src >= 0) & (src < T)
    out[:, ok] = a[:, src[ok]]
    return out


def _roll(a: np.ndarray, w: int, start: int, stop: int, lag: int = LAG, stat: str = "mean"):
    """NaN-aware rolling mean / std of a over days [t-lag-w+1, t-lag] for t in [start, stop)."""
    n, T = a.shape
    v = np.nan_to_num(a)
    c = (~np.isnan(a)).astype(np.float32)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(v, 1, dtype=np.float64)], 1)
    cn = np.concatenate([np.zeros((n, 1)), np.cumsum(c, 1, dtype=np.float64)], 1)
    t = np.arange(start, stop)
    hi = np.clip(t - lag + 1, 0, T)
    lo = np.clip(t - lag + 1 - w, 0, T)
    s = cs[:, hi] - cs[:, lo]
    k = cn[:, hi] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(k > 0, s / k, np.nan)
        if stat == "mean":
            return mean.astype(np.float32), k.astype(np.float32)
        cs2 = np.concatenate([np.zeros((n, 1)), np.cumsum(v * v, 1, dtype=np.float64)], 1)
        s2 = cs2[:, hi] - cs2[:, lo]
        var = np.where(k > 1, s2 / k - mean ** 2, np.nan)
        return np.sqrt(np.maximum(var, 0)).astype(np.float32), k.astype(np.float32)


def _event_distance(cal: pd.DataFrame, start: int, stop: int, cap: int = 14):
    ev = cal["event_name_1"].notna().to_numpy()
    idx = np.flatnonzero(ev)
    t = np.arange(start, stop)
    nxt = np.searchsorted(idx, t)
    to_next = np.where(nxt < len(idx), idx[np.minimum(nxt, len(idx) - 1)] - t, cap)
    prv = np.searchsorted(idx, t, side="right") - 1
    since = np.where(prv >= 0, t - idx[np.maximum(prv, 0)], cap)
    return np.minimum(to_next, cap), np.minimum(since, cap)


def build_feature_arrays(panel, Y: np.ndarray, origin: int, start: int, stop: int, cfg: dict) -> dict:
    """Wide feature arrays for target days t in [start, stop). Demand data used: Y[:, :origin] only.

    origin = number of observed days (t < origin known). Prices after origin are carried forward
    unless cfg['prices']['use_future_prices'].
    """
    Y = Y[:, :origin]
    N = Y.shape[0]
    F: dict[str, np.ndarray] = {}
    for k in (28, 35, 42, 49, 56):
        F[f"lag{k}"] = _shift(Y, k, start, stop)
    avail = None
    for w in (7, 28, 56, 112, 364):
        F[f"rm{w}"], cnt = _roll(Y, w, start, stop)
        if w == 56:
            avail = cnt / w
    F["avail56"] = avail
    F["rsd28"], _ = _roll(Y, 28, start, stop, stat="std")
    F["dow8"] = np.nanmean(np.stack([_shift(Y, 28 + 7 * j, start, stop) for j in range(8)]), 0)
    F["cv28"] = F["rsd28"] / np.maximum(F["rm28"], 0.1)

    # item level (same item across the 4 stores; rows are item-major, store-minor)
    n_items = N // 4
    item = np.nanmean(F["rm28"].reshape(n_items, 4, -1), 1)
    F["item_rm28"] = np.repeat(item, 4, 0)
    F["store_share"] = F["rm28"] / np.maximum(F["item_rm28"], 1e-3)

    # store x dept level trend: recent vs yearly level (captures CA_2's 2015 step change)
    groups = (panel.meta["store_id"] + "|" + panel.meta["dept_id"]).to_numpy()
    tr = np.zeros_like(F["rm28"])
    for g in np.unique(groups):
        rows = groups == g
        a = np.nansum(F["rm28"][rows], 0)
        b = np.nansum(F["rm364"][rows], 0)
        tr[rows] = (a / np.maximum(b, 1e-3))[None, :]
    F["sd_trend"] = tr

    # age since first sale
    has = np.nan_to_num(Y) > 0
    first = np.where(has.any(1), has.argmax(1), origin)
    F["age"] = (np.arange(start, stop)[None, :] - first[:, None]).astype(np.float32)

    # prices
    P = panel.price
    if not cfg["prices"]["use_future_prices"]:
        P = P.copy()
        last = P[:, origin - 1:origin]
        P[:, origin:] = last
    T = P.shape[1]
    tt = np.minimum(np.arange(start, stop), T - 1)
    price = P[:, tt]
    F["price"] = price
    F["price_m28"] = price / _shift(P, 28, start, stop)
    hist_max = np.fmax.accumulate(np.nan_to_num(P, nan=0.0), axis=1)[:, tt]
    F["price_rel_max"] = price / np.where(hist_max > 0, hist_max, np.nan)
    item_p = np.nanmean(price.reshape(n_items, 4, -1), 1)
    F["price_rel_item"] = price / np.repeat(item_p, 4, 0)

    # calendar (known in advance)
    cal = panel.cal
    days = cal.iloc[start:stop]
    ones = np.ones((N, 1), np.float32)
    F["dow"] = ones * days["dow"].to_numpy(np.float32)[None, :]
    F["month"] = ones * days["month"].to_numpy(np.float32)[None, :]
    F["mday"] = ones * days["date"].dt.day.to_numpy(np.float32)[None, :]
    F["snap"] = ones * days["snap"].to_numpy(np.float32)[None, :]
    et = days["event_type_1"].map({"Cultural": 1, "National": 2, "Religious": 3, "Sporting": 4}).fillna(0)
    F["event_type"] = ones * et.to_numpy(np.float32)[None, :]
    to_ev, since_ev = _event_distance(cal, start, stop)
    F["days_to_event"] = ones * to_ev.astype(np.float32)[None, :]
    F["days_since_event"] = ones * since_ev.astype(np.float32)[None, :]

    # static categoricals
    dept = panel.meta["dept_id"].map({"FOODS_1": 0, "FOODS_2": 1, "FOODS_3": 2}).to_numpy(np.float32)
    store = panel.meta["store_id"].map({"CA_1": 0, "CA_2": 1, "CA_3": 2, "CA_4": 3}).to_numpy(np.float32)
    F["dept"] = np.repeat(dept[:, None], stop - start, 1)
    F["store"] = np.repeat(store[:, None], stop - start, 1)
    return F


def gather(F: dict, rows: np.ndarray, cols: np.ndarray, names: list[str]) -> np.ndarray:
    """Row matrix for (series rows[k], window column cols[k])."""
    X = np.empty((len(rows), len(names)), np.float32)
    for j, n in enumerate(names):
        X[:, j] = F[n][rows, cols]
    return X


FEATURE_NAMES = None  # filled lazily from the first build (dict order)
