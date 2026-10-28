"""Load the Meridian sample data into (geo x week) arrays.

Everything is per capita: the KPI and media spend are divided by geo population so geos of
137k and 994k people sit on one scale. The KPI is then standardised (one mean / sd over the
training rows) so priors are on a unit scale; all ROI arithmetic converts back to revenue.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(path=None, overrides: dict | None = None) -> dict:
    with open(path or ROOT / "config" / "project_b.yaml") as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def out_path(cfg, key) -> pathlib.Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Data:
    geos: list
    dates: pd.DatetimeIndex
    pop: np.ndarray            # (G,)
    kpi: np.ndarray            # (G, T) conversions
    rpk: np.ndarray            # (G, T) revenue per conversion
    spend: np.ndarray          # (M, G, T) paid spend, currency
    impressions: np.ndarray    # (M, G, T)
    organic: np.ndarray        # (G, T) organic impressions
    promo: np.ndarray          # (G, T)
    controls: np.ndarray       # (C, G, T)
    channels: list
    control_names: list
    train: np.ndarray = field(default=None)   # (T,) bool: rows used in the likelihood

    @property
    def G(self): return len(self.geos)

    @property
    def T(self): return len(self.dates)

    @property
    def M(self): return len(self.channels)

    @property
    def revenue(self): return self.kpi * self.rpk

    def week_index(self, date) -> int:
        return int(np.searchsorted(self.dates, pd.Timestamp(date)))


def load(cfg: dict) -> Data:
    dc = cfg["data"]
    df = pd.read_csv(ROOT / cfg["paths"]["data_file"])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["time"] = pd.to_datetime(df["time"])
    df["gid"] = df["geo"].str.replace("Geo", "").astype(int)
    df = df.sort_values(["gid", "time"])
    geos = [f"Geo{g}" for g in sorted(df.gid.unique())]
    dates = pd.DatetimeIndex(sorted(df.time.unique()))
    G, T = len(geos), len(dates)
    if len(df) != G * T:
        raise ValueError(f"expected a full geo x week panel, got {len(df)} rows for {G} x {T}")
    arr = lambda c: df[c].to_numpy(float).reshape(G, T)
    chans = dc["channels"]
    return Data(geos=geos, dates=dates, pop=arr(dc["population"])[:, 0], kpi=arr(dc["kpi"]),
                rpk=arr(dc["revenue_per_kpi"]),
                spend=np.stack([arr(f"{c}_spend") for c in chans]),
                impressions=np.stack([arr(f"{c}_impression") for c in chans]),
                organic=arr(dc["organic"]), promo=arr(dc["promo"]),
                controls=np.stack([arr(c) for c in dc["controls"]]),
                channels=chans, control_names=dc["controls"])


def set_holdout(d: Data, start: str | None, weeks: int) -> tuple[int, int]:
    """Train strictly on weeks before `start`; hold out the next `weeks` weeks. Returns (t0, t1)."""
    t0 = d.week_index(start) if start else d.T - weeks
    if t0 < 52 or t0 >= d.T:
        raise SystemExit(f"--holdout-start {start} leaves {t0} training weeks (need >= 52 and < {d.T})")
    t1 = min(t0 + weeks, d.T)
    d.train = np.arange(d.T) < t0
    return t0, t1
