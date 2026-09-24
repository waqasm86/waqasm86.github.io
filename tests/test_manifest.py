from __future__ import annotations

import subprocess
from pathlib import Path

from tools.article.render_manifest import render


def command(*args: str, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_manifest_renders_code_and_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    command("git", "init", "-b", "main", cwd=repo)
    command("git", "config", "user.name", "Test", cwd=repo)
    command("git", "config", "user.email", "test@example.invalid", cwd=repo)
    command("git", "remote", "add", "origin", "https://github.com/example/source.git", cwd=repo)
    (repo / "example.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")
    (repo / "result.json").write_text('{"status":"executed"}\n', encoding="utf-8")
    command("git", "add", ".", cwd=repo)
    command("git", "commit", "-m", "fixture", cwd=repo)
    commit = command("git", "rev-parse", "HEAD", cwd=repo)

    manifest = tmp_path / "manifest.yml"
    manifest.write_text(
        f"""title: Example
slug: example
source:
  repository: example/source
  ref: {commit}
code:
  - file: example.py
    lines: "1:2"
    language: python
evidence:
  - result.json
categories: [Testing]
tags: [evidence]
""",
        encoding="utf-8",
    )
    paths = render(manifest, repo, tmp_path / "output")
    assert any(path.name == "code-01.md" for path in paths)
    assert (tmp_path / "output" / "example" / "evidence" / "result.json").is_file()
    assert (tmp_path / "output" / "example" / "metadata.json").is_file()


def test_manifest_accepts_relative_repository_path(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    command("git", "init", "-b", "main", cwd=repo)
    command("git", "config", "user.name", "Test", cwd=repo)
    command("git", "config", "user.email", "test@example.invalid", cwd=repo)
    command("git", "remote", "add", "origin", "https://github.com/example/source.git", cwd=repo)
    (repo / "example.py").write_text("print('relative')\n", encoding="utf-8")
    command("git", "add", ".", cwd=repo)
    command("git", "commit", "-m", "fixture", cwd=repo)
    commit = command("git", "rev-parse", "HEAD", cwd=repo)
    manifest = tmp_path / "manifest.yml"
    manifest.write_text(
        f"""title: Example
slug: relative
source: {{repository: example/source, ref: {commit}}}
code: [{{file: example.py, lines: "1:1", language: python}}]
categories: [Testing]
tags: [evidence]
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    paths = render(Path("manifest.yml"), Path("source"), Path("output"))
    assert any(path.name == "code-01.md" for path in paths)
