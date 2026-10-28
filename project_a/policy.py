"""Replenishment policies: `policy(state) -> Orders`.

Two policies share the simulator interface (see simulate.State / simulate.Orders):

BaselinePolicy — the "simple base-stock" comparator
    Stores: order up to mu*(L+R) + z*sigma*sqrt(L+R) with mu, sigma from the last 28 days of
    sales (recorded before the cutoff, own observed sales after), L = 2, R = 1, z = 1.645.
    DC (Mondays): installation base-stock on DC on-hand + on-order with the dept's mean lead
    time + 7-day review. Shortages and capacity are left to the simulator's pro-rata cut.

RecommendedPolicy — forecast-driven, two-echelon, capacity-aware
    Stores (daily): target position = alpha-quantile of demand over the next `store_window`
    days (NB approximation of the forecast mixture), plus a weekend "pre-build" when the DC
    outbound cap would be exceeded later in the week; capped at the 21-day store cap.
    DC (Mondays, per dept): echelon base-stock. Echelon position (DC on hand + on order + all
    store stock in place and in transit) is raised to the beta-quantile of network demand over
    [tomorrow, next order's arrival + store lead], from coherent sample paths with the lead
    time itself drawn from its lognormal (so the long FOODS_3 tail is priced in).
    Allocation when outbound capacity is short: ship where an extra unit saves the most expected
    lost-sale cost (price x P(demand > position)), via a single water-filling threshold.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .simulate import DEPTS, Orders
from .src import forecast as fc


# ============================================================================ baseline
class BaselinePolicy:
    def __init__(self, panel, net, cfg, z: float = 1.645, window: int = 28):
        self.net, self.z, self.w = net, z, window
        self.hist = panel.sales[:, net.start - window:net.start].astype(float)
        self.L, self.R = net.store_lead, 1

    def reset(self, state):
        self.hist = self.hist.copy()

    def __call__(self, st):
        net = self.net
        if st.day > 0:
            self.hist = np.concatenate([self.hist[:, 1:], st.sales[:, st.day:st.day + 1]], 1)
        mu, sd = self.hist.mean(1), self.hist.std(1)
        P = self.L + self.R
        S = mu * P + self.z * sd * np.sqrt(P)
        ip = st.store_on_hand + st.store_inbound.sum(1)
        tr = np.maximum(np.ceil(S - ip), 0)
        po = None
        if st.is_order_day:
            I = net.n_items
            mu_n = mu.reshape(I, 4).sum(1)
            sd_n = np.sqrt((sd ** 2).reshape(I, 4).sum(1))
            Ld = net.lt_mean[net.item_dept] + 7
            S_dc = mu_n * Ld + self.z * sd_n * np.sqrt(Ld)
            po = np.maximum(np.ceil(S_dc - st.dc_on_hand - st.dc_on_order()), 0)
        return Orders(tr, po)


# ============================================================================ recommended
class _Block:
    """Forecast block prepared for the policy: extended means, NB parameters, DC paths."""

    def __init__(self, f: fc.Forecast, ext: int, n_paths: int, seed: int):
        mu = f.mu
        H = mu.shape[1]
        reps = int(np.ceil(ext / 7))
        tail = np.tile(mu[:, H - 7:], (1, reps))[:, :ext]      # extend with the last week's weekday pattern
        self.mu = np.concatenate([mu, tail], 1).astype(np.float64)
        self.origin = f.origin
        sp, sd = f.disp.cell_params(self.mu)
        self.sp, self.sd = sp.astype(np.float64), sd.astype(np.float64)
        self.cmu = np.concatenate([np.zeros((mu.shape[0], 1)), np.cumsum(self.mu, 1)], 1)
        self.cmu2s = np.concatenate([np.zeros((mu.shape[0], 1)), np.cumsum(self.mu ** 2 * self.sd, 1)], 1)
        # coherent DC paths: sum of store paths, generated in chunks and aggregated immediately
        rng = np.random.default_rng(seed)
        S, N, T = n_paths, mu.shape[0], self.mu.shape[1]
        dc = np.empty((S, N // 4, T), np.int32)
        for a in range(0, S, 25):
            s = min(25, S - a)
            gi = rng.gamma(1 / np.maximum(sp[::4], 1e-6), np.maximum(sp[::4], 1e-6), size=(s, N // 4))
            gd = rng.gamma(1 / self.sd, self.sd, size=(s, N, T))
            lam = self.mu[None] * np.repeat(gi, 4, 1)[:, :, None] * gd
            dc[a:a + s] = rng.poisson(lam).reshape(s, N // 4, 4, T).sum(2)
        self.dc_cum = np.concatenate([np.zeros((S, N // 4, 1), np.int32), np.cumsum(dc, 2, dtype=np.int32)], 2)

        self._qt = {}

    def window(self, a, b):
        """Mean and variance of store demand over block days [a, b) (a, b scalars or arrays)."""
        T = self.mu.shape[1]
        a, b = np.clip(a, 0, T), np.clip(b, 0, T)
        M = self.cmu[:, b] - self.cmu[:, a]
        s2 = self.cmu2s[:, b] - self.cmu2s[:, a]
        sp = self.sp if np.ndim(M) == 1 else self.sp[:, None]
        V = M + sp * M ** 2 + (1 + sp) * s2
        return M, V

    def qtable(self, w: int) -> np.ndarray:
        """(Q, N, R) demand quantiles at QGRID for the window [rel+1, rel+1+w), every rel day.
        Built once per (block, window) and reused across days, Monte Carlo runs and knob settings."""
        if w not in self._qt:
            R = self.mu.shape[1] - w
            a = np.arange(R) + 1
            M, V = self.window(a, a + w)
            shape = (len(QGRID),) + M.shape
            self._qt[w] = nb_ppf(np.broadcast_to(QGRID[:, None, None], shape),
                                 np.broadcast_to(M, shape), np.broadcast_to(V, shape)).astype(np.float32)
        return self._qt[w]

    def level(self, w: int, rel: int, q: np.ndarray) -> np.ndarray:
        """Per-series demand quantile at probability q (N,) for the window starting at rel+1."""
        tab = self.qtable(w)
        tab = tab[:, :, min(rel, tab.shape[2] - 1)]                 # (Q, N)
        q = np.clip(q, QGRID[0], QGRID[-1])
        j = np.clip(np.searchsorted(QGRID, q), 1, len(QGRID) - 1)
        g0, g1 = QGRID[j - 1], QGRID[j]
        n = np.arange(tab.shape[1])
        lo, hi = tab[j - 1, n], tab[j, n]
        return lo + (q - g0) / (g1 - g0) * (hi - lo)


def _round_to_budget(x, cap_each, budget):
    """Integer shipments: floor, then give the remaining budget to the largest remainders
    (flooring 5k fractional levels would otherwise strand ~0.5 unit per series of capacity)."""
    out = np.floor(x)
    left = int(budget - out.sum())
    if left > 0:
        frac = np.where(out < cap_each, x - out, -1.0)
        order = np.argsort(-frac)[:left]
        order = order[frac[order] > 0]
        out[order] += 1
    return out


QGRID = np.array([0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
_BLOCKS: dict = {}


def get_block(f, ext, n_paths, seed):
    """Policy-side block for a forecast, cached so knob searches and MC runs share sample paths."""
    key = (id(f), ext, n_paths, seed)
    if key not in _BLOCKS:
        _BLOCKS[key] = (f, _Block(f, ext, n_paths, seed))
    return _BLOCKS[key][1]


def nb_ppf(q, M, V):
    """Quantile of a count with mean M, variance V (negative binomial; Poisson if V <= M)."""
    q = np.broadcast_to(q, M.shape)
    out = np.zeros_like(M)
    pos = M > 1e-9
    od = pos & (V > M * 1.0001)
    n = np.where(od, M ** 2 / np.maximum(V - M, 1e-12), 1.0)
    p = np.where(od, n / (n + M), 0.5)
    out[od] = stats.nbinom.ppf(q[od], n[od], p[od])
    po = pos & ~od
    out[po] = stats.poisson.ppf(q[po], M[po])
    return out


def nb_sf(x, M, V):
    """P(demand > x)."""
    out = np.zeros_like(M)
    pos = M > 1e-9
    od = pos & (V > M * 1.0001)
    n = np.where(od, M ** 2 / np.maximum(V - M, 1e-12), 1.0)
    p = np.where(od, n / (n + M), 0.5)
    out[od] = stats.nbinom.sf(x[od], n[od], p[od])
    po = pos & ~od
    out[po] = stats.poisson.sf(x[po], M[po])
    return out


class RecommendedPolicy:
    def __init__(self, panel, net, cfg, forecasts: list, knobs: dict | None = None, seed: int = 0):
        pc = cfg["policy"]
        k = dict(pc["knobs"])
        k.update(knobs or {})
        self.k = k
        self.net, self.cfg = net, cfg
        self.alpha = np.array([k["alpha_store"][d] for d in DEPTS])[net.series_dept]
        self.beta = np.array([k["beta_dc"][d] for d in DEPTS])[net.item_dept]
        self.blocks = [get_block(f, pc["extend_days"], pc["n_paths_dc"], seed + j) for j, f in enumerate(forecasts)]
        self.block_start = [f.origin - net.start for f in forecasts]   # test-day index where each block starts
        self.price = net.price_store
        self.lt_rng = np.random.default_rng(seed + 99)

    def reset(self, state):
        pass

    def _block(self, day):
        j = max(i for i, s in enumerate(self.block_start) if s <= day)
        return self.blocks[j], day - self.block_start[j]

    # ------------------------------------------------------------------ store side
    def _store_targets(self, st):
        b, rel = self._block(st.day)
        # IP after today's order must cover demand on days day+1 .. day+w
        T = b.level(self.k["store_window"], rel, self.alpha)
        return T, b, rel

    def _prebuild(self, st, b, rel):
        """Extra units to ship today because the DC cannot ship enough on upcoming peak days."""
        net = self.net
        C = net.dc_out_cap * self.k["cap_eta"]
        w = self.k["store_window"]
        look = self.k["lookahead_days"]
        # expected unconstrained shipment on day d+j ~ network demand that day's order must cover (day d+j+w)
        need = np.array([b.mu[:, min(rel + j + w, b.mu.shape[1] - 1)].sum() for j in range(look + 1)])
        carry = np.zeros(look + 2)
        for j in range(look, 0, -1):
            carry[j - 1] = max(0.0, need[j] + carry[j] - C)
        extra = carry[0]
        if extra <= 0:
            return np.zeros(net.n_series)
        # spread the pull-forward over items in proportion to their demand on the over-cap days
        over = [j for j in range(1, look + 1) if need[j] + carry[j] > C]
        days = [min(rel + j + w, b.mu.shape[1] - 1) for j in over]
        weight = b.mu[:, days].sum(1)
        return extra * weight / max(weight.sum(), 1e-9)

    def _allocate(self, st, req, ip, b, rel, extra):
        """Cap total shipments at the DC outbound limit and at DC stock, by marginal value."""
        net = self.net
        I = net.n_items
        # DC stock per item: pro-rata over the item's stores
        need = req.reshape(I, 4).sum(1)
        f = np.where(need > st.dc_on_hand, st.dc_on_hand / np.maximum(need, 1), 1.0)
        req = np.floor(req * np.repeat(f, 4))
        C = net.dc_out_cap
        if req.sum() <= C:
            return req
        # water-filling: ship to series i up to level x_i(lam) where price_i * P(D > x) >= lam
        w = self.k["store_window"]
        p = np.maximum(self.price[:, st.day], 1e-6)
        lo, hi = 0.0, float(p.max())
        for _ in range(30):
            lam = (lo + hi) / 2
            ship = np.clip(b.level(w, rel, 1 - lam / p) + extra - ip, 0, req)
            if ship.sum() > C:
                lo = lam
            else:
                hi = lam
        x = np.clip(b.level(w, rel, 1 - hi / p) + extra - ip, 0, req)
        return _round_to_budget(x, req, C)

    # ------------------------------------------------------------------ DC side
    def _dc_order(self, st):
        net = self.net
        b, rel = self._block(st.day)
        S = b.dc_cum.shape[0]
        T = b.dc_cum.shape[2] - 1
        store_pos = (st.store_on_hand + st.store_inbound.sum(1)).reshape(net.n_items, 4).sum(1)
        ep = st.dc_on_hand + st.dc_on_order() + store_pos
        po = np.zeros(net.n_items)
        for k in range(3):
            items = net.item_dept == k
            # next order is placed in 7 days; its lead time is random (believed mean, CV from config)
            m_next = st.believed_lt_mean(k, st.day + 7)
            m_now = st.believed_lt_mean(k, st.day)
            m = max(m_next, m_now)
            mu_ln = np.log(m) - net.lt_sigma ** 2 / 2
            L = np.maximum(1, np.round(self.lt_rng.lognormal(mu_ln, net.lt_sigma, size=S)))
            end = np.minimum(rel + 1 + 7 + L.astype(int) + net.store_lead, T)
            a = min(rel + 1, T)
            cum = b.dc_cum[:, items]
            dem = cum[np.arange(S), :, end] - cum[:, :, a]          # (S, n_items_dept)
            target = np.quantile(dem, self.beta[items][0], axis=0)
            po[items] = np.maximum(np.ceil(target - ep[items]), 0)
        return po

    # ------------------------------------------------------------------ main
    def __call__(self, st):
        T, b, rel = self._store_targets(st)
        extra = self._prebuild(st, b, rel) if self.k["prebuild"] else np.zeros(self.net.n_series)
        T = np.minimum(T + extra, self.net.store_cap)
        ip = st.store_on_hand + st.store_inbound.sum(1)
        req = np.maximum(np.ceil(T - ip), 0)
        tr = self._allocate(st, req, ip, b, rel, extra)
        po = self._dc_order(st) if st.is_order_day else None
        return Orders(tr, po)


# ============================================================================ factory
def build_forecasts(panel, net, cfg, log=print, model=None):
    """Forecast blocks for the replay: one at the cutoff, then every `refit_every` days.

    The model is trained once on data <= cutoff. Later blocks re-forecast with the same model
    using demand observed up to that day (no retraining on test data)."""
    from .src import demand_recovery as dr
    f0, model = fc.run_forecast(panel, net.start, cfg, model=model, log=log)
    blocks = [f0]
    step = cfg["calendar"]["refit_every"]
    H = cfg["calendar"]["forecast_horizon"]
    o = net.start + step
    while step and o < net.start + net.days:
        rec = dr.recover(panel, o, cfg)
        mu = fc.predict_mean(model, panel, rec, o, H, cfg)
        if f0.adj is not None:
            mu = fc.apply_adjustments(mu, f0.adj, panel, rec, o, cfg)
        blocks.append(fc.Forecast(o, H, mu, f0.disp, adj=f0.adj))
        o += step
    return blocks, model


def make_policy_factory(name, panel, net, cfg, forecasts=None, knobs=None, log=print):
    if name == "baseline":
        return (lambda: BaselinePolicy(panel, net, cfg)), {}
    if forecasts is None:
        forecasts, _ = build_forecasts(panel, net, cfg, log=log)
    shared = {}

    def make():
        # blocks (incl. DC sample paths) are built once and reused across Monte Carlo runs
        if "p" not in shared:
            shared["p"] = RecommendedPolicy(panel, net, cfg, forecasts, knobs, seed=cfg["seed"])
        pol = shared["p"]
        pol.lt_rng = np.random.default_rng(cfg["seed"] + 99)
        return pol
    return make, {"forecasts": forecasts}
