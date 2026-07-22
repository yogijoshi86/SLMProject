"""Configuration loading with dotted-key CLI overrides and reproducible seeding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute + dotted-path access (cfg.model.name / cfg['model']['name'])."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str) -> Any:
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(raw: str) -> Any:
    """Best-effort YAML scalar coercion for CLI override values."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML config, applying ``a.b=value`` overrides from the CLI."""
    with open(path, "r", encoding="utf-8") as handle:
        cfg = Config(yaml.safe_load(handle))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key, raw = override.split("=", 1)
        cfg.set_path(key.strip(), _coerce(raw.strip()))

    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (best-effort — torch/numpy optional at import)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
