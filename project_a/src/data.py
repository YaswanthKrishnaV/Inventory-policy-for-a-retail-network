"""Load M5 and build the CA FOODS panel as wide arrays (series x days).

Why wide arrays: 5.7k series x 1.9k days fits comfortably in memory as int16/float32,
and every downstream step (stockout detection, simulation) is a vectorised scan along time.

Conventions
-----------
* Day index t is 0-based: t = d - 1, so d_1885 -> t = 1884.
* price[i, t] is NaN when there is no sell_prices row for that item/store/week.
  In M5 that means the item was not on sale in that store that week (not listed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, path

N_DAYS_SALES = 1941  # sales_train_evaluation.csv covers d_1 .. d_1941


class Panel:
    """meta: one row per series (aligned with array rows); sales/price: (N, T); cal: one row per day."""

    def __init__(self, meta: pd.DataFrame, sales: np.ndarray, price: np.ndarray, cal: pd.DataFrame):
        self.meta, self.sales, self.price, self.cal = meta, sales, price, cal

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    def idx(self, d: int) -> int:
        """M5 day number -> 0-based column index."""
        return d - 1


def load_calendar(cfg: dict) -> pd.DataFrame:
    cal = pd.read_csv(path(cfg, "data_dir") / "calendar.csv", parse_dates=["date"])
    cal["d"] = cal["d"].str[2:].astype(int)
    state = cfg["scope"]["state"]
    cal["snap"] = cal[f"snap_{state}"].astype(np.int8)
    cal["dow"] = cal["date"].dt.dayofweek.astype(np.int8)  # Monday = 0
    cal["is_event"] = cal["event_name_1"].notna().astype(np.int8)
    cal["is_christmas"] = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)).astype(np.int8)
    keep = ["d", "date", "wm_yr_wk", "dow", "month", "year", "snap", "is_event", "is_christmas",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    return cal[keep].reset_index(drop=True)


def _read_sales(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    sc = cfg["scope"]
    chunks, metas = [], []
    for ch in pd.read_csv(path(cfg, "data_dir") / "sales_train_evaluation.csv", chunksize=4000):
        ch = ch[(ch["state_id"] == sc["state"]) & (ch["cat_id"] == sc["category"])]
        if len(ch):
            metas.append(ch[["id", "item_id", "dept_id", "store_id"]])
            chunks.append(ch[[f"d_{i}" for i in range(1, N_DAYS_SALES + 1)]].to_numpy(np.int16))
    meta = pd.concat(metas, ignore_index=True)
    sales = np.vstack(chunks)
    order = np.lexsort((meta["store_id"].to_numpy(), meta["item_id"].to_numpy()))  # item-major, store-minor
    return meta.iloc[order].reset_index(drop=True), sales[order]


def _read_prices(cfg: dict, meta: pd.DataFrame, cal: pd.DataFrame) -> np.ndarray:
    sc = cfg["scope"]
    parts = []
    for ch in pd.read_csv(path(cfg, "data_dir") / "sell_prices.csv", chunksize=1_000_000):
        parts.append(ch[ch["store_id"].isin(sc["stores"]) & ch["item_id"].str.startswith(sc["category"])])
    pr = pd.concat(parts, ignore_index=True)
    weeks = np.sort(cal["wm_yr_wk"].unique())
    wpos = {w: j for j, w in enumerate(weeks)}
    key = meta["item_id"] + "|" + meta["store_id"]
    rpos = pd.Series(np.arange(len(meta)), index=key.to_numpy())
    pr_key = pr["item_id"] + "|" + pr["store_id"]
    rows = rpos.reindex(pr_key.to_numpy()).to_numpy()
    ok = ~np.isnan(rows)
    weekly = np.full((len(meta), len(weeks)), np.nan, dtype=np.float32)
    weekly[rows[ok].astype(int), pr["wm_yr_wk"].map(wpos).to_numpy()[ok]] = pr["sell_price"].to_numpy()[ok]
    day_week = cal["wm_yr_wk"].iloc[:N_DAYS_SALES].map(wpos).to_numpy()
    return weekly[:, day_week]


def load_panel(cfg: dict | None = None, rebuild: bool = False) -> Panel:
    """Load the CA FOODS panel, caching the arrays in cache/ (first build ~1 min)."""
    cfg = cfg or load_config()
    sc = cfg["scope"]
    cache = path(cfg, "cache_dir") / f"panel_{sc['state']}_{sc['category']}.npz"
    cal = load_calendar(cfg)
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        meta = pd.DataFrame({k: z[f"meta_{k}"] for k in ["id", "item_id", "dept_id", "store_id"]})
        return Panel(meta, z["sales"], z["price"], cal)
    meta, sales = _read_sales(cfg)
    price = _read_prices(cfg, meta, cal)
    np.savez_compressed(cache, sales=sales, price=price,
                        **{f"meta_{k}": meta[k].to_numpy(dtype=str) for k in meta.columns})
    return Panel(meta, sales, price, cal)


if __name__ == "__main__":
    import time
    t0 = time.time()
    p = load_panel(rebuild=True)
    print(f"series={p.n_series} days={p.n_days} built in {time.time() - t0:.1f}s")
    print(p.meta.groupby(["dept_id", "store_id"]).size().unstack())
