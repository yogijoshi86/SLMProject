"""Shared CLI argument parsing for scripts."""

from __future__ import annotations

import argparse

from guardrail_audit.utils import Config, load_config, set_seed


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a config value, e.g. --set extraction.batch_size=4",
    )
    return parser


def resolve(args) -> Config:
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    return cfg
