#!/usr/bin/env python3
"""Verify required generated pages and internal HTML asset/link targets."""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

REQUIRED_PAGES = (
    "index.html",
    "projects/index.html",
    "research/index.html",
    "series/index.html",
    "about/index.html",
    "posts/why-i-built-kaggle-vllm/index.html",
    "feed.xml",
    "sitemap.xml",
    "robots.txt",
)


class Targets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in {"href", "src"}:
                self.values.append(value)


def resolve(site: Path, page: Path, target: str) -> Path | None:
    if target.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "#")):
        return None
    path = unquote(urlsplit(target).path)
    if not path:
        return None
    candidate = site / path.lstrip("/") if path.startswith("/") else page.parent / path
    if path.endswith("/"):
        candidate /= "index.html"
    return candidate.resolve()


def check(site: Path) -> list[str]:
    site = site.resolve()
    errors = [f"missing required output: {name}" for name in REQUIRED_PAGES if not (site / name).is_file()]
    for page in site.rglob("*.html"):
        parser = Targets()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in parser.values:
            candidate = resolve(site, page, target)
            if candidate is None:
                continue
            try:
                candidate.relative_to(site)
            except ValueError:
                errors.append(f"{page}: internal target escapes site: {target}")
                continue
            if not candidate.is_file():
                errors.append(f"{page}: missing generated target: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=Path("_site"))
    args = parser.parse_args()
    errors = check(args.site)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"generated site validation failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"generated site validated: {args.site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
