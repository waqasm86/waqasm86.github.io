#!/usr/bin/env python3
"""Validate required Chirpy metadata and post filenames."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

from tools.article.common import load_front_matter

REQUIRED = ("title", "description", "date", "categories", "tags", "toc", "math", "mermaid")
POST_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")


def validate_post(path: Path) -> list[str]:
    errors: list[str] = []
    if not POST_NAME.fullmatch(path.name):
        errors.append("filename must be YYYY-MM-DD-lowercase-slug.md")
    try:
        data, _ = load_front_matter(path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    for field in REQUIRED:
        if field not in data or data[field] in (None, "", []):
            errors.append(f"missing required field: {field}")
    if data.get("toc") is not True:
        errors.append("toc must be true")
    for field in ("math", "mermaid"):
        if not isinstance(data.get(field), bool):
            errors.append(f"{field} must be boolean")
    for field in ("categories", "tags"):
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} must be a YAML list")
    if "date" in data and not isinstance(data["date"], (date, datetime, str)):
        errors.append("date has an unsupported type")
    image = data.get("image")
    if image is not None and not (
        isinstance(image, str)
        or (isinstance(image, dict) and image.get("path") and image.get("alt"))
    ):
        errors.append("image must be a path or a mapping with path and alt")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(Path("_posts").glob("*.md"))
    failures = 0
    for path in paths:
        errors = validate_post(path)
        for error in errors:
            print(f"{path}: {error}", file=sys.stderr)
        failures += len(errors)
    if failures:
        print(f"front matter validation failed with {failures} error(s)", file=sys.stderr)
        return 1
    print(f"validated {len(paths)} post(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
