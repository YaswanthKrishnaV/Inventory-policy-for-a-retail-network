"""Replay simulator: runs a replenishment policy through the test window and prices the result.

    python -m project_a.simulate --cutoff 1885 --policy recommended --n-sims 100

Network: 1 DC -> 4 stores, every CA FOODS item. Each test day, in this order:

1. Receive: supplier POs whose lead time ends today arrive at the DC; DC->store transfers shipped
   two days ago arrive at stores.
2. Demand realises at stores. sales = min(on hand, demand); the rest is lost (no backorders).
3. The policy is called with the end-of-day state (it sees its own sales, not true demand) and
   returns DC->store transfers for today and, on Mondays, one PO per department.
4. The simulator enforces the rules itself (never trusts the policy): POs only on Mondays;
   store on-hand + inbound + transfer <= 21 x avg daily sales; transfers <= DC on-hand;
   total DC outbound <= 1.25 x avg daily network sales (last 365 training days). Any cut is
   pro-rata and is logged as a violation.
5. Ship: transfers arrive at day+2. Each PO gets one lead-time draw L ~ LogNormal(mean_dept,
   CV 0.4) (sigma^2 = ln(1+CV^2)), arriving round(L) days later (>= 1). Orders may cross.
6. Costs: holding 25%/yr x price on end-of-day on-hand (DC and stores; in-transit optional),
   lost sales 35% x price, $50 per department PO, $0.05 per unit shipped DC->store.

Demand on censored test days (the important subtlety)
-----------------------------------------------------
Recorded test-period sales are themselves capped by Walmart's own stock. If we replayed them as
demand, a policy would be rewarded for stocking out exactly when Walmart did. So demand is:
* recorded sales on uncensored days (status OK in a recovery run on all data up to the window end);
* on stockout / disrupted days, a negative-binomial draw around the imputed rate
  (fresh draw in every Monte Carlo run, so imputation uncertainty widens the cost range);
* zero on not-listed / dormant days (the item could not be bought).
`--demand-mode recorded` and `synthetic` (forecast sample paths) are sensitivity checks.

Monte Carlo runs vary the lead-time draws and the censored-day demand draws.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .src import demand_recovery as dr
from .src.config import load_config, path
from .src.data import load_panel

DEPTS = ["FOODS_1", "FOODS_2", "FOODS_3"]


# ----------------------------------------------------------------------------- static network
@dataclass
class Network:
    cfg: dict
    meta: pd.DataFrame
    start: int                     # 0-based index of first test day (= cutoff)
    days: int                      # test length D
    item_dept: np.ndarray          # (n_items,) dept index 0..2
    series_dept: np.ndarray        # (N,)
    series_store: np.ndarray       # (N,) store index 0..3
    avg_daily: np.ndarray          # (N,) average daily sales (store cap / initial stock basis)
    store_cap: np.ndarray          # (N,) max on-hand + inbound
    dc_out_cap: float              # max units/day leaving the DC
    price_store: np.ndarray        # (N, D)
    price_dc: np.ndarray           # (n_items, D) holding price at DC: mean store price
    dow: np.ndarray                # (D,) Monday = 0
    lt_mean: np.ndarray            # (3,) supplier lead-time mean by dept
    lt_sigma: float
    store_lead: int
    h_daily: float
    lost_rate: float
    po_fixed: float
    transfer_cost: float
    hold_in_transit: bool

    @property
    def n_series(self):
        return len(self.series_dept)

    @property
    def n_items(self):
        return len(self.item_dept)


def build_network(panel, cutoff: int, cfg: dict, rec=None) -> Network:
    """Everything the rules need, computed from data with d <= cutoff (prices from the test window
    are used for costs only)."""
    cc, cap = cfg["calendar"], cfg["capacity"]
    D = min(cc["test_days"], cc["last_data_day"] - cutoff)
    start = cutoff
    rec = rec if rec is not None else dr.recover(panel, cutoff, cfg)
    W = cap["avg_sales_window_days"]
    win = slice(cutoff - W, cutoff)
    st = rec.status[:, win]
    sellable = ~np.isin(st, (dr.PRELAUNCH, dr.UNLISTED, dr.CLOSED))
    base = panel.sales[:, win].astype(float) if cap["avg_sales_basis"] == "recorded" else rec.demand[:, win].astype(float)
    avg = np.divide((base * sellable).sum(1), sellable.sum(1), out=np.zeros(panel.n_series), where=sellable.sum(1) > 0)
    net_avg = panel.sales[:, cutoff - cap["dc_outbound_window_days"]:cutoff].sum(0).mean()
    P = panel.price[:, start:start + D].copy()
    last = panel.price[:, :start + D]
    # fill gaps with the last known price (holding/lost costs need a price even for unlisted weeks)
    filled = pd.DataFrame(last.T).ffill().bfill().to_numpy().T[:, start:start + D]
    P = np.where(np.isnan(P), filled, P)
    P = np.nan_to_num(P, nan=float(np.nanmedian(panel.price)))
    n_items = panel.n_series // 4
    dept_map = {d: k for k, d in enumerate(DEPTS)}
    sd = panel.meta["dept_id"].map(dept_map).to_numpy()
    lt = cfg["lead_times"]
    return Network(
        cfg=cfg, meta=panel.meta, start=start, days=D,
        item_dept=sd[::4], series_dept=sd,
        series_store=panel.meta["store_id"].map({"CA_1": 0, "CA_2": 1, "CA_3": 2, "CA_4": 3}).to_numpy(),
        avg_daily=avg, store_cap=np.floor(cap["store_cap_days"] * avg), dc_out_cap=float(np.floor(cap["dc_outbound_multiple"] * net_avg)),
        price_store=P.astype(np.float32), price_dc=P.reshape(n_items, 4, D).mean(1).astype(np.float32),
        dow=panel.cal["dow"].to_numpy()[start:start + D], lt_mean=np.array([lt["supplier_mean_days"][d] for d in DEPTS], float),
        lt_sigma=float(np.sqrt(np.log(1 + lt["supplier_cv"] ** 2))), store_lead=lt["dc_to_store_days"],
        h_daily=cfg["costs"]["holding_rate_annual"] / 365.0, lost_rate=cfg["costs"]["lost_sale_rate"],
        po_fixed=cfg["costs"]["po_fixed"], transfer_cost=cfg["costs"]["transfer_per_unit"],
        hold_in_transit=cfg["costs"]["hold_in_transit"])


# ----------------------------------------------------------------------------- demand
def test_demand_frame(panel, cutoff: int, cfg: dict, full=None):
    """(rate, base, censored) for the test window, from a recovery run on data to the window end."""
    D = min(cfg["calendar"]["test_days"], cfg["calendar"]["last_data_day"] - cutoff)
    end = cutoff + D
    full = full if full is not None else dr.recover(panel, end, cfg)
    st = full.status[:, cutoff:end]
    base = panel.sales[:, cutoff:end].astype(np.float32)
    base[np.isin(st, dr.NO_DEMAND)] = 0
    cen = np.isin(st, dr.CENSORED)
    rate = np.where(cen, full.rate[:, cutoff:end], 0).astype(np.float32)
    return base, rate, cen


def draw_demand(base, rate, cen, cfg, rng, mode="imputed", synthetic=None):
    if mode == "recorded":
        out = base.copy()
        return out
    if mode == "synthetic":
        return synthetic.astype(np.float32)
    ir = cfg["simulation"]["censored_inv_r"]
    lam = rate[cen] * rng.gamma(1 / ir, ir, size=cen.sum())
    out = base.copy()
    out[cen] = np.maximum(rng.poisson(lam), base[cen])   # demand >= what was actually sold
    return out


# ----------------------------------------------------------------------------- dynamic state
@dataclass
class PO:
    placed: int
    arrive: int
    dept: int
    qty: np.ndarray                # (n_items,) zeros outside dept


@dataclass
class State:
    """What the policy sees at the end of test day `day` (after that day's sales)."""
    net: Network
    day: int
    store_on_hand: np.ndarray      # (N,)
    store_inbound: np.ndarray      # (N, store_lead): column k arrives at the start of day+1+k
    dc_on_hand: np.ndarray         # (n_items,)
    pipeline: list                 # open POs
    sales: np.ndarray              # (N, D) own observed sales so far (columns > day are 0)
    context: dict = field(default_factory=dict)   # forecasts, lead-time warnings, ...

    @property
    def t(self):
        return self.net.start + self.day

    @property
    def is_order_day(self):
        return self.net.dow[self.day] == 0

    def dc_on_order(self) -> np.ndarray:
        q = np.zeros(self.net.n_items)
        for p in self.pipeline:
            q += p.qty
        return q

    def believed_lt_mean(self, dept: int, order_day: int) -> float:
        """Lead-time mean the policy believes for an order placed on `order_day` (warnings only)."""
        w = self.context.get("lt_warning")
        m = self.net.lt_mean[dept]
        if w and dept == w["dept_idx"] and self.day >= w["known_from"] and w["from_day"] <= order_day < w["to_day"]:
            return m * w["multiplier"]
        return m


@dataclass
class Orders:
    transfers: np.ndarray          # (N,) units DC -> store shipped today
    po: np.ndarray | None = None   # (n_items,) units ordered from suppliers today


# ----------------------------------------------------------------------------- simulate
def simulate(policy, net: Network, demand: np.ndarray, rng, shock: dict | None = None,
             context: dict | None = None, record_daily: bool = True) -> dict:
    cfg = net.cfg
    N, I, D, L = net.n_series, net.n_items, net.days, net.store_lead
    init = cfg["capacity"]["initial_stock_days"]
    s_oh = np.round(init * net.avg_daily)
    dc_oh = np.round(init * net.avg_daily.reshape(I, 4).sum(1))
    st = State(net, 0, s_oh, np.zeros((N, L)), dc_oh, [], np.zeros((N, D), np.float32), dict(context or {}))
    if shock:
        st.context["lt_warning"] = None
        if shock.get("warn_days") is not None:
            st.context["lt_warning"] = dict(dept_idx=DEPTS.index(shock["dept"]), from_day=shock["from_day"],
                                            to_day=shock["to_day"], multiplier=shock["multiplier"],
                                            known_from=shock["from_day"] - shock["warn_days"])
    if hasattr(policy, "reset"):
        policy.reset(st)
    lost_s = np.zeros(N)
    dem_s = demand.sum(1)
    cost = dict(holding_store=0.0, holding_dc=0.0, lost=0.0, po=0.0, transfer=0.0)
    viol = dict(store_cap=0, dc_stock=0, dc_capacity=0, po_weekday=0)
    daily = {k: np.zeros(D) for k in ("demand", "sales", "lost", "outbound", "store_inv_value", "dc_inv_value")}
    n_pos = 0
    for d in range(D):
        st.day = d
        # 1) receive
        arrived = [p for p in st.pipeline if p.arrive <= d]
        for p in arrived:
            st.dc_on_hand += p.qty
        st.pipeline = [p for p in st.pipeline if p.arrive > d]
        st.store_on_hand += st.store_inbound[:, 0]
        st.store_inbound[:, :-1] = st.store_inbound[:, 1:]
        st.store_inbound[:, -1] = 0
        # 2) demand
        dem = demand[:, d]
        sold = np.minimum(st.store_on_hand, dem)
        lost = dem - sold
        st.store_on_hand -= sold
        st.sales[:, d] = sold
        lost_s += lost
        cost["lost"] += float((lost * net.price_store[:, d]).sum() * net.lost_rate)
        # 3) policy
        o = policy(st)
        tr = np.maximum(np.round(np.nan_to_num(o.transfers)), 0)
        # 4) enforce rules
        room = np.maximum(net.store_cap - st.store_on_hand - st.store_inbound.sum(1), 0)
        if (tr > room).any():
            viol["store_cap"] += int((tr > room).sum())
            tr = np.minimum(tr, room)
        need = tr.reshape(I, 4).sum(1)
        short = need > st.dc_on_hand
        if short.any():
            viol["dc_stock"] += int(short.sum())
            f = np.where(short, st.dc_on_hand / np.maximum(need, 1), 1.0)
            tr = np.floor(tr * np.repeat(f, 4))
        if tr.sum() > net.dc_out_cap:
            viol["dc_capacity"] += 1
            tr = np.floor(tr * net.dc_out_cap / tr.sum())
        # 5) ship
        st.dc_on_hand -= tr.reshape(I, 4).sum(1)
        st.store_inbound[:, L - 1] += tr
        cost["transfer"] += float(tr.sum() * net.transfer_cost)
        if o.po is not None and np.nansum(o.po) > 0:
            po = np.maximum(np.round(np.nan_to_num(o.po)), 0)
            if not st.is_order_day:
                viol["po_weekday"] += 1
            else:
                for k in range(3):
                    q = np.where(net.item_dept == k, po, 0)
                    if q.sum() <= 0:
                        continue
                    m = net.lt_mean[k]
                    if shock and DEPTS.index(shock["dept"]) == k and shock["from_day"] <= d < shock["to_day"]:
                        m *= shock["multiplier"]
                    mu_ln = np.log(m) - net.lt_sigma ** 2 / 2
                    lt = max(1, int(round(rng.lognormal(mu_ln, net.lt_sigma))))
                    st.pipeline.append(PO(d, d + lt, k, q))
                    cost["po"] += net.po_fixed
                    n_pos += 1
        # 6) holding
        s_val = float((st.store_on_hand * net.price_store[:, d]).sum())
        if net.hold_in_transit:
            s_val += float((st.store_inbound.sum(1) * net.price_store[:, d]).sum())
        dc_val = float((st.dc_on_hand * net.price_dc[:, d]).sum())
        cost["holding_store"] += s_val * net.h_daily
        cost["holding_dc"] += dc_val * net.h_daily
        if record_daily:
            daily["demand"][d] = dem.sum(); daily["sales"][d] = sold.sum(); daily["lost"][d] = lost.sum()
            daily["outbound"][d] = tr.sum(); daily["store_inv_value"][d] = s_val; daily["dc_inv_value"][d] = dc_val
    total = sum(cost.values())
    fill = 1 - lost_s.sum() / max(dem_s.sum(), 1)
    by_dept = {DEPTS[k]: float(1 - lost_s[net.series_dept == k].sum() / max(dem_s[net.series_dept == k].sum(), 1)) for k in range(3)}
    by_store = {f"CA_{k + 1}": float(1 - lost_s[net.series_store == k].sum() / max(dem_s[net.series_store == k].sum(), 1)) for k in range(4)}
    has = dem_s > 0
    item_fill = float(np.mean(1 - lost_s[has] / dem_s[has]))
    end_val = float((st.store_on_hand * net.price_store[:, -1]).sum() + (st.dc_on_hand * net.price_dc[:, -1]).sum()
                    + (st.store_inbound.sum(1) * net.price_store[:, -1]).sum()
                    + sum((p.qty * net.price_dc[:, -1]).sum() for p in st.pipeline))
    return dict(total_cost=total, **{f"cost_{k}": v for k, v in cost.items()}, fill_rate=float(fill),
                fill_by_dept=by_dept, fill_by_store=by_store, fill_item_avg=item_fill,
                units_demand=float(dem_s.sum()), units_lost=float(lost_s.sum()), n_pos=n_pos,
                ending_inventory_value=end_val, violations=viol,
                daily={k: v.tolist() for k, v in daily.items()} if record_daily else None,
                lost_by_series=lost_s, demand_by_series=dem_s)


def monte_carlo(make_policy, net, panel, cfg, n_sims: int, seed: int, shock=None, demand_mode=None,
                context=None, frame=None, log=print, synthetic_paths=None) -> tuple[pd.DataFrame, list]:
    """Repeat the replay over lead-time and censored-demand draws. Returns per-run metrics."""
    mode = demand_mode or cfg["simulation"]["demand_mode"]
    base, rate, cen = frame if frame is not None else test_demand_frame(panel, net.start, cfg)
    rows, runs = [], []
    for k in range(n_sims):
        rng = np.random.default_rng(seed + k)
        syn = synthetic_paths[k % len(synthetic_paths)] if mode == "synthetic" else None
        dem = draw_demand(base, rate, cen, cfg, rng, mode, syn)
        res = simulate(make_policy(), net, dem, rng, shock=shock, context=context, record_daily=(k == 0))
        flat = {k2: v for k2, v in res.items() if not isinstance(v, (dict, np.ndarray, list)) and v is not None}
        flat.update({f"fill_{k2}": v for k2, v in res["fill_by_dept"].items()})
        flat.update({f"viol_{k2}": v for k2, v in res["violations"].items()})
        rows.append(flat)
        runs.append(res if k == 0 else None)
    return pd.DataFrame(rows), runs


def summarise_runs(df: pd.DataFrame, target: float) -> dict:
    out = {}
    for c in ["total_cost", "cost_lost", "cost_holding_store", "cost_holding_dc", "cost_po", "cost_transfer",
              "fill_rate", "fill_item_avg", "ending_inventory_value", "fill_FOODS_1", "fill_FOODS_2", "fill_FOODS_3"]:
        out[c] = dict(mean=float(df[c].mean()), p05=float(df[c].quantile(0.05)), p95=float(df[c].quantile(0.95)))
    out["p_fill_ge_target"] = float((df["fill_rate"] >= target).mean())
    out["violations"] = {c: float(df[c].mean()) for c in df.columns if c.startswith("viol_")}
    out["n_sims"] = int(len(df))
    return out


def main(argv=None):
    from .policy import make_policy_factory
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default="recommended", choices=["recommended", "baseline"])
    ap.add_argument("--n-sims", type=int, default=None)
    ap.add_argument("--demand-mode", default=None, choices=["imputed", "recorded", "synthetic"])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cutoff = a.cutoff or cfg["calendar"]["default_cutoff"]
    panel = load_panel(cfg)
    net = build_network(panel, cutoff, cfg)
    make, ctx = make_policy_factory(a.policy, panel, net, cfg)
    t0 = time.time()
    df, runs = monte_carlo(make, net, panel, cfg, a.n_sims or cfg["simulation"]["n_sims"], cfg["simulation"]["seed"],
                           demand_mode=a.demand_mode, context=ctx)
    s = summarise_runs(df, cfg["fill_rate"]["target"])
    print(json.dumps(s, indent=1))
    print(f"{len(df)} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
