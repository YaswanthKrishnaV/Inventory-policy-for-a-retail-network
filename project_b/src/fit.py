"""Run several chains in parallel and summarise convergence."""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from .model import rhat_ess

_MODEL = None


def _chain(args):
    k, warm, draws, seed = args
    return _MODEL.run_chain(warm, draws, seed + 1000 * k, log=None, chain=k)


def fit(model, cfg, chains=None, warmup=None, draws=None, seed=None, log=print):
    sc = cfg["sampler"]
    chains = chains or sc["chains"]
    warmup = warmup or sc["warmup"]
    draws = draws or sc["draws"]
    seed = seed if seed is not None else sc["seed"]
    global _MODEL
    _MODEL = model
    procs = min(chains, os.cpu_count() or 1)
    with mp.get_context("fork").Pool(procs) as pool:
        outs = pool.map(_chain, [(k, warmup, draws, seed) for k in range(chains)])
    post = {k: np.stack([o[k] for o in outs]) for k in outs[0]}     # (chains, draws, ...)
    diag = diagnostics(post, model)
    if log:
        worst = max(diag.items(), key=lambda kv: kv[1]["rhat"])
        log(f"  fit: {chains} chains x {draws} draws; worst R-hat {worst[1]['rhat']:.3f} ({worst[0]}), "
            f"min ESS {min(v['ess'] for v in diag.values()):.0f}")
    return post, diag


def diagnostics(post, model):
    out = {}
    names = model.d.names
    for j, n in enumerate(names):
        for key in ("alpha", "ec", "eta"):
            r, e = rhat_ess(post[key][:, :, j])
            out[f"{key}[{n}]"] = dict(rhat=r, ess=e)
    for m in range(model.M):
        r, e = rhat_ess(post["roi"][:, :, m])
        out[f"roi[{names[m]}]"] = dict(rhat=r, ess=e)
    if model.J > model.M:
        r, e = rhat_ess(post["beta_org"])
        out["beta_org"] = dict(rhat=r, ess=e)
    r, e = rhat_ess(np.log(post["sigma"]).mean(2))
    out["mean log sigma"] = dict(rhat=r, ess=e)
    return out


def flat(post):
    """(chains, draws, ...) -> (chains*draws, ...)"""
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items()}
