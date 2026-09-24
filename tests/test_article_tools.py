from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.article.extract_code import extract
from tools.article.extract_json import render_table
from tools.article.validate_frontmatter import validate_post


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    run("git", "init", "-b", "main", cwd=repo)
    run("git", "config", "user.name", "Test", cwd=repo)
    run("git", "config", "user.email", "test@example.invalid", cwd=repo)
    source = repo / "example.py"
    source.write_text("one = 1\ntwo = 2\nthree = 3\n", encoding="utf-8")
    run("git", "add", "example.py", cwd=repo)
    run("git", "commit", "-m", "fixture", cwd=repo)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, commit


def test_extract_code_records_commit_and_range(tmp_path: Path) -> None:
    repo, commit = git_repo(tmp_path)
    output = extract(
        repo,
        "example.py",
        "2:3",
        "python",
        "https://github.com/example/source/blob/<sha>/example.py",
    )
    assert commit in output
    assert "lines: 2:3" in output
    assert "two = 2\nthree = 3" in output


def test_extract_code_refuses_dirty_source(tmp_path: Path) -> None:
    repo, _ = git_repo(tmp_path)
    (repo / "example.py").write_text("changed = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted"):
        extract(repo, "example.py", "1:1", "python", "https://example.invalid/<sha>/example.py")


def test_extract_json_table() -> None:
    document = {"measurements": {"throughput": 12.5, "samples": [1, 2]}}
    table = render_table(document, ["Output throughput=measurements.throughput"])
    assert "| Output throughput | 12.5 |" in table


def test_validate_frontmatter_reports_missing_fields(tmp_path: Path) -> None:
    post = tmp_path / "2026-01-01-test.md"
    post.write_text("---\ntitle: Test\n---\nBody\n", encoding="utf-8")
    errors = validate_post(post)
    assert "missing required field: tags" in errors


def test_example_json_is_valid() -> None:
    json.loads('{"status": "executed"}')
