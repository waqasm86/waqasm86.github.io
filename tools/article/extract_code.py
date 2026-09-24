#!/usr/bin/env python3
"""Extract committed source lines as an attributed Markdown code fence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.article.common import confined_path


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_range(value: str, total: int) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, TypeError) as exc:
        raise ValueError("line range must use positive START:END integers") from exc
    if start < 1 or end < start or end > total:
        raise ValueError(f"line range {value} is outside 1:{total}")
    return start, end


def extract(
    repo: Path,
    file_name: str,
    line_range: str,
    language: str,
    source_url: str,
) -> str:
    repo = repo.resolve()
    if git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError(f"not a Git worktree: {repo}")

    source = confined_path(repo, file_name)
    if not source.is_file():
        raise ValueError(f"source file does not exist: {file_name}")
    relative = source.relative_to(repo).as_posix()
    git(repo, "ls-files", "--error-unmatch", relative)
    if git(repo, "status", "--porcelain", "--", relative):
        raise ValueError(f"refusing to extract uncommitted content: {relative}")

    commit = git(repo, "rev-parse", "HEAD")
    if "<sha>" in source_url:
        source_url = source_url.replace("<sha>", commit)
    if commit not in source_url:
        raise ValueError("source URL must contain <sha> or the checked-out commit SHA")

    lines = source.read_text(encoding="utf-8").splitlines()
    start, end = parse_range(line_range, len(lines))
    selected = "\n".join(lines[start - 1 : end])
    return (
        f"<!-- source: {relative}; commit: {commit}; lines: {start}:{end} -->\n"
        f"Source: [`{relative}` lines {start}–{end}]({source_url}#L{start}-L{end}) "
        f"at `{commit}`.\n\n"
        f"```{language}\n{selected}\n```\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--file", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = extract(args.repo, args.file, args.lines, args.language, args.source_url)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
        else:
            sys.stdout.write(result)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"extract_code: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
