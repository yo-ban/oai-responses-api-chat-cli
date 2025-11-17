# Purpose: Manage application settings stored in YAML.
# Why: Centralized helpers keep the CLI free of configuration plumbing.
"""Utility helpers for loading and persisting chat CLI configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

APP_HOME = Path.home() / ".responses-cli"
CONFIG_DIR = APP_HOME / "config"
DATA_DIR = APP_HOME / "data"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"
_DEFAULT_CONFIG: Dict[str, Any] = {
    "developer_prompt": (
        "You are an assistant that provides concise, accurate responses while citing assumptions.\n"
        "Keep answers in Japanese unless the user explicitly requests another language.\n"
    ),
    "base_url": None,
    "api_key": None,
    "model": "gpt-5",
    "max_output_tokens": None,
    "pdf_render_dpi": 200,
    "reasoning": {
        "effort": "medium",
        "summary": None,
    },
    "profiles": {},
}


def ensure_config(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Create the configuration file with defaults when it is missing."""
    if path == DEFAULT_CONFIG_PATH:
        APP_HOME.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    save_config(_DEFAULT_CONFIG, path)
    return True


def load_config(path: Path = DEFAULT_CONFIG_PATH, *, ensure: bool = True) -> Dict[str, Any]:
    """Load configuration from disk, merging in default keys."""
    if ensure:
        ensure_config(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    merged = _merge_defaults(data)
    return merged


def save_config(config: Dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist configuration to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _merge_defaults(config)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            sanitized,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )


def load_prompt(path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Return the developer prompt text."""
    return load_config(path)["developer_prompt"]


def save_prompt(prompt_text: str, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Update the developer prompt within the configuration file."""
    config = load_config(path)
    config["developer_prompt"] = prompt_text.rstrip() + "\n"
    save_config(config, path)


def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults into a config dict without mutating the input."""
    merged = dict(_DEFAULT_CONFIG)
    merged.update({k: v for k, v in config.items() if k not in {"reasoning", "profiles"}})
    reasoning = dict(_DEFAULT_CONFIG["reasoning"])
    reasoning.update(config.get("reasoning") or {})
    merged["reasoning"] = reasoning
    profiles = dict(_DEFAULT_CONFIG["profiles"])
    profiles.update(config.get("profiles") or {})
    merged["profiles"] = profiles
    return merged
