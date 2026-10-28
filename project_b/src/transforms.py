"""Media transforms shared by fitting, ROI, optimisation and the experiment estimand.

adstock: geometric carry-over over `max_lag` weeks, weights normalised to sum to 1
         (so adstock preserves the average level of spend — Meridian's convention).
hill:    x / (x + ec), i.e. Hill with slope 1 (Meridian default for non-reach-frequency media).
Inputs are spend per capita divided by the channel's median non-zero spend per capita, so
ec = 1 means "half of the maximum effect is reached at typical spend".
"""
from __future__ import annotations

import numpy as np


def adstock(x: np.ndarray, alpha: float, max_lag: int) -> np.ndarray:
    """x (..., T) -> geometric adstock along the last axis, normalised weights."""
    w = alpha ** np.arange(max_lag + 1)
    w = w / w.sum()
    T = x.shape[-1]
    out = np.zeros_like(x, dtype=float)
    for lag, wl in enumerate(w):
        if lag == 0:
            out += wl * x
        else:
            out[..., lag:] += wl * x[..., :T - lag]
    return out


def hill(a: np.ndarray, ec: float) -> np.ndarray:
    return a / (a + ec)


def media_curve(x_scaled: np.ndarray, alpha: float, ec: float, max_lag: int) -> np.ndarray:
    return hill(adstock(x_scaled, alpha, max_lag), ec)
