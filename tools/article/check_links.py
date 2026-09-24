#!/usr/bin/env python3
"""Check local Markdown links, site routes, and referenced assets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.article.common import load_front_matter

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")


def site_routes(root: Path) -> set[str]:
    routes = {"/", "/feed.xml", "/sitemap.xml", "/robots.txt"}
    for path in (root / "_tabs").glob("*.md"):
        routes.add(f"/{path.stem}/")
    for path in (root / "_posts").glob("*.md"):
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", path.stem)
        routes.add(f"/posts/{slug}/")
    return routes


def referenced_targets(path: Path) -> list[str]:
    data, body = load_front_matter(path)
    targets = [match.group(1) for match in LINK.finditer(body)]
    image = data.get("image")
    if isinstance(image, str):
        targets.append(image)
    elif isinstance(image, dict) and image.get("path"):
        targets.append(str(image["path"]))
    return targets


def check(root: Path, paths: list[Path]) -> list[str]:
    root = root.resolve()
    routes = site_routes(root)
    errors: list[str] = []
    for path in paths:
        try:
            targets = referenced_targets(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        for raw in targets:
            if raw.startswith(("http://", "https://", "mailto:", "#", "{{", "{%")):
                continue
            clean = unquote(urlsplit(raw).path)
            if not clean:
                continue
            if clean.startswith("/"):
                candidate = root / clean.lstrip("/")
                if clean.endswith("/") and clean in routes:
                    continue
                if candidate.is_file():
                    continue
            else:
                candidate = (path.parent / clean).resolve()
                if candidate.is_file():
                    continue
            errors.append(f"{path}: missing local target: {raw}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((args.root / "_posts").glob("*.md")) + sorted(
        (args.root / "_tabs").glob("*.md")
    )
    errors = check(args.root, paths)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"local link validation failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"checked local links in {len(paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
