"""Simulator invariants on a small synthetic network (no M5 data needed).

    pytest -q tests
"""
import numpy as np
import pandas as pd
import pytest

from project_a import simulate as sm
from project_a.src.config import load_config


def make_net(n_items=6, days=21, cap_days=21, out_cap=1e9):
    cfg = load_config()
    N = n_items * 4
    meta = pd.DataFrame({"id": [f"s{i}" for i in range(N)],
                         "item_id": np.repeat([f"FOODS_{1 + (k % 3)}_{k:03d}" for k in range(n_items)], 4),
                         "dept_id": np.repeat([f"FOODS_{1 + (k % 3)}" for k in range(n_items)], 4),
                         "store_id": np.tile(["CA_1", "CA_2", "CA_3", "CA_4"], n_items)})
    dept = meta["dept_id"].map({d: k for k, d in enumerate(sm.DEPTS)}).to_numpy()
    avg = np.full(N, 5.0)
    return sm.Network(cfg=cfg, meta=meta, start=0, days=days, item_dept=dept[::4], series_dept=dept,
                      series_store=np.tile(np.arange(4), n_items), avg_daily=avg, store_cap=np.floor(cap_days * avg),
                      dc_out_cap=out_cap, price_store=np.full((N, days), 2.0, np.float32),
                      price_dc=np.full((n_items, days), 2.0, np.float32), dow=np.arange(days) % 7,
                      lt_mean=np.array([5.0, 7.0, 10.0]), lt_sigma=float(np.sqrt(np.log(1.16))), store_lead=2,
                      h_daily=0.25 / 365, lost_rate=0.35, po_fixed=50.0, transfer_cost=0.05, hold_in_transit=False), cfg


class Greedy:
    """Asks for far more than allowed, every day, to exercise every rule."""
    def __call__(self, st):
        return sm.Orders(np.full(st.net.n_series, 1e6), np.full(st.net.n_items, 500.0))


class Recorder:
    def __init__(self, inner):
        self.inner, self.log = inner, []

    def __call__(self, st):
        o = self.inner(st)
        self.log.append((st.day, st.store_on_hand.copy(), st.store_inbound.copy(), st.dc_on_hand.copy()))
        return o


def test_conservation_and_rules():
    net, cfg = make_net(out_cap=40)
    rng = np.random.default_rng(0)
    dem = rng.poisson(5, size=(net.n_series, net.days)).astype(float)
    pol = Recorder(Greedy())
    res = sm.simulate(pol, net, dem, rng)
    # stock never negative, store position never above cap
    for day, s_oh, s_in, dc in pol.log:
        assert (s_oh >= 0).all() and (dc >= 0).all()
        assert (s_oh + s_in.sum(1) <= net.store_cap + 1e-9).all()
    # outbound cap respected every day
    assert max(res["daily"]["outbound"]) <= net.dc_out_cap
    # units: sold + lost = demand
    assert res["units_lost"] == pytest.approx(dem.sum() - np.sum(res["daily"]["sales"]))
    # cost components add up
    parts = sum(v for k, v in res.items() if k.startswith("cost_"))
    assert res["total_cost"] == pytest.approx(parts)
    # POs only on Mondays (dow 0) and one per dept per Monday
    mondays = int((net.dow == 0).sum())
    assert res["n_pos"] == mondays * 3
    assert res["cost_po"] == pytest.approx(50 * mondays * 3)


def test_deterministic_given_seed():
    net, _ = make_net()
    dem = np.random.default_rng(1).poisson(5, size=(net.n_series, net.days)).astype(float)
    a = sm.simulate(Greedy(), net, dem, np.random.default_rng(7))
    b = sm.simulate(Greedy(), net, dem, np.random.default_rng(7))
    assert a["total_cost"] == b["total_cost"] and a["fill_rate"] == b["fill_rate"]


def test_perfect_supply_gives_full_fill():
    """With ample stock, no caps and a policy that tops stores up, every unit sells."""
    net, cfg = make_net(cap_days=1000)
    net.dc_out_cap = 1e9
    dem = np.full((net.n_series, net.days), 3.0)

    class TopUp:
        def __call__(self, st):
            ip = st.store_on_hand + st.store_inbound.sum(1)
            return sm.Orders(np.maximum(30 - ip, 0), np.full(st.net.n_items, 1000.0) if st.is_order_day else None)

    cfg["capacity"]["initial_stock_days"] = 7
    res = sm.simulate(TopUp(), net, dem, np.random.default_rng(0))
    assert res["fill_rate"] == pytest.approx(1.0)


def test_lognormal_parameterisation():
    """mean of the lead-time draw equals the configured mean (sigma^2 = ln(1+CV^2))."""
    sig = np.sqrt(np.log(1 + 0.4 ** 2))
    x = np.random.default_rng(0).lognormal(np.log(10) - sig ** 2 / 2, sig, 400_000)
    assert x.mean() == pytest.approx(10, rel=0.01)
    assert x.std() / x.mean() == pytest.approx(0.4, rel=0.02)
