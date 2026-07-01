"""Learned recipe cache (Phase 13).

`autofusion learn` measures the best recipe per task-category (via the optimizer)
and writes it here; `CategoryRouter` reads it so routing is data-driven, not
hand-set. Cache lives next to the config: `.autofusion/recipes.json`.

Shape: {category: {"recipe": str, "score": float, "cost": float, "benchmark": str, "n": int}}
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config

CACHE_DIR = ".autofusion"
CACHE_FILE = "recipes.json"


def cache_path(config: Config) -> Path:
    base = config.path.parent if config.path else Path.cwd()
    return base / CACHE_DIR / CACHE_FILE


def load_recipes(config: Config) -> dict:
    p = cache_path(config)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_recipes(config: Config, data: dict) -> Path:
    p = cache_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))
    return p
