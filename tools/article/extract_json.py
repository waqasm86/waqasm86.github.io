#!/usr/bin/env python3
"""Select benchmark JSON values and emit a Markdown table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def lookup(document: Any, path: str) -> Any:
    value = document
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(path)
    return value


def render_table(document: Any, fields: list[str]) -> str:
    rows: list[tuple[str, str]] = []
    for field in fields:
        if "=" in field:
            label, path = field.split("=", 1)
        else:
            label = path = field
        value = lookup(document, path)
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        rows.append((label.replace("|", "\\|"), rendered.replace("|", "\\|")))
    body = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return f"| Metric | Value |\n|---|---:|\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--field", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        result = render_table(document, args.field)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
        else:
            sys.stdout.write(result)
    except (OSError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        print(f"extract_json: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
