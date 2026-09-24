"""Shared helpers for Markdown front matter and safe paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    """Return YAML front matter and body from a Markdown file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: missing opening front-matter delimiter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            raw = "".join(lines[1:index])
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path}: front matter must be a mapping")
            return data, "".join(lines[index + 1 :])
    raise ValueError(f"{path}: missing closing front-matter delimiter")


def confined_path(root: Path, relative: str) -> Path:
    """Resolve a user path and reject traversal outside root."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative}") from exc
    return candidate
