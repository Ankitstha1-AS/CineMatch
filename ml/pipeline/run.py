"""Offline training pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML training configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the CineMatch training pipeline.")
    parser.add_argument("--config", type=Path, default=Path("ml/config.yaml"))
    return parser.parse_args()


def main() -> int:
    """Validate configuration and report the current pipeline status."""

    args = parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset", {}).get("name", "unknown")

    print(f"Loaded config from {args.config}")
    print(f"  dataset: {dataset}")
    print(f"  seed: {config.get('seed')}")
    print("Training and evaluation are implemented in Phases 2 and 3 (see AGENTS.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
