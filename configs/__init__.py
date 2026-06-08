"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML config file with nested attribute access."""
    config = _load_yaml(Path(path))
    return _to_namespace(config)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_path = config.pop("base", None)
    if base_path is None:
        return config

    base = _load_yaml(Path(base_path))
    return _deep_merge(base, config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value
