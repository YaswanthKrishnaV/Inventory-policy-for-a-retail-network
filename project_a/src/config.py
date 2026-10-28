"""Config loading. Every tunable assumption lives in config/project_a.yaml."""
from __future__ import annotations

import copy
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "project_a.yaml"


def load_config(path: str | pathlib.Path | None = None, overrides: dict | None = None) -> dict:
    """Load YAML config; `overrides` is a nested dict merged on top (used for live what-ifs)."""
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def path(cfg: dict, key: str) -> pathlib.Path:
    """Resolve a configured path relative to the repo root and make sure it exists."""
    p = ROOT / cfg["paths"][key]
    if key != "data_dir":
        p.mkdir(parents=True, exist_ok=True)
    return p
