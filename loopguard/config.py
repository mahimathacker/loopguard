"""Small dependency-free config helpers for LoopGuard.

LoopGuard intentionally supports a tiny config shape before pulling in a full YAML
dependency. The parser handles the simple ``loopguard.yml`` files used by local scripts
and CI: top-level keys, one nested mapping, and list items.
"""

from __future__ import annotations

import json
from argparse import ArgumentTypeError
from pathlib import Path


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise ArgumentTypeError("must be >= 1")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise ArgumentTypeError("must be > 0")
    return parsed


def parse_scalar(value: str):
    """Parse the tiny scalar set used by LoopGuard config files."""
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


def parse_simple_yaml(text: str) -> dict:
    """Parse the small loopguard.yml shape without adding a YAML dependency."""
    data: dict = {}
    current_path: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("- "):
            if not current_path:
                raise ValueError("list item without a parent key")
            item = parse_scalar(stripped[2:])
            if len(current_path) == 1:
                key = current_path[0]
                if not isinstance(data.get(key), list):
                    data[key] = []
                data[key].append(item)
                continue
            parent_key, key = current_path[:2]
            parent = data.get(parent_key)
            if not isinstance(parent, dict):
                raise ValueError(f"cannot add list under {parent_key}")
            if not isinstance(parent.get(key), list):
                parent[key] = []
            parent[key].append(item)
            continue

        if ":" not in stripped:
            raise ValueError(f"invalid config line: {raw}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            data[key] = {} if value == "" else parse_scalar(value)
            current_path = [key]
            continue

        if not current_path:
            raise ValueError(f"nested key without a parent: {raw}")
        parent_key = current_path[0]
        if not isinstance(data.get(parent_key), dict):
            raise ValueError(f"cannot add nested key under {parent_key}")
        data[parent_key][key] = {} if value == "" else parse_scalar(value)
        current_path = [parent_key, key]
    return data


def load_config(path: str | None) -> dict:
    """Load JSON or small YAML config."""
    if path is None:
        return {}
    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = parse_simple_yaml(text)
    if not isinstance(loaded, dict):
        raise ValueError("config must be a mapping")
    return loaded


def section(config: dict, name: str) -> dict:
    """Return a named config section, falling back to top-level keys."""
    value = config.get(name, config)
    if not isinstance(value, dict):
        raise ValueError(f"{name} config must be a mapping")
    return value


def config_bool(config: dict, key: str, default: bool) -> bool:
    raw = config.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{key} must be true or false")


def config_int(config: dict, key: str, default: int) -> int:
    raw = config.get(key, default)
    return positive_int(str(raw))


def config_float(config: dict, key: str, default: float) -> float:
    raw = config.get(key, default)
    return positive_float(str(raw))


def config_budget(config: dict, key: str) -> int | None:
    budgets = config.get("budgets", {})
    raw = config.get(key)
    if raw is None and isinstance(budgets, dict):
        raw = budgets.get(key)
    if raw is None:
        return None
    return positive_int(str(raw))
