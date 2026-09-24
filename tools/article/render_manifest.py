#!/usr/bin/env python3
"""Render reviewable code assets from a declarative publication manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from tools.article.common import confined_path
from tools.article.extract_code import extract, git


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a YAML mapping")
    for field in ("title", "slug", "source", "code", "categories", "tags"):
        if field not in data:
            raise ValueError(f"manifest missing required field: {field}")
    source = data["source"]
    if not isinstance(source, dict) or not source.get("repository") or not source.get("ref"):
        raise ValueError("source requires repository and immutable ref")
    if not isinstance(data["code"], list):
        raise ValueError("code must be a list")
    return data


def render(manifest_path: Path, repo: Path, output_dir: Path) -> list[Path]:
    manifest = load_manifest(manifest_path)
    expected_ref = str(manifest["source"]["ref"])
    actual_commit = git(repo, "rev-parse", "HEAD")
    expected_commit = (
        expected_ref.lower()
        if re.fullmatch(r"[0-9a-fA-F]{40}", expected_ref)
        else git(repo, "rev-parse", f"{expected_ref}^{{commit}}")
    )
    if expected_commit != actual_commit:
        raise ValueError(f"source checkout {actual_commit} does not match manifest ref {expected_ref}")
    repository = str(manifest["source"]["repository"])
    origin = git(repo, "remote", "get-url", "origin")
    normalized_origin = origin.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    if not normalized_origin.lower().endswith(f"github.com/{repository}".lower()):
        raise ValueError(f"source origin {origin} does not match manifest repository {repository}")

    slug = str(manifest["slug"])
    target = output_dir / slug
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, item in enumerate(manifest["code"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"code item {index} must be a mapping")
        file_name = str(item["file"])
        source_url = f"https://github.com/{repository}/blob/<sha>/{file_name}"
        content = extract(
            repo,
            file_name,
            str(item["lines"]),
            str(item["language"]),
            source_url,
        )
        output = target / f"code-{index:02d}.md"
        output.write_text(content, encoding="utf-8")
        written.append(output)

    evidence_metadata: list[dict[str, str]] = []
    for evidence_name in manifest.get("evidence", []):
        source = confined_path(repo, str(evidence_name))
        if not source.is_file():
            raise ValueError(f"evidence file does not exist: {evidence_name}")
        relative = source.relative_to(repo).as_posix()
        git(repo, "ls-files", "--error-unmatch", relative)
        if git(repo, "status", "--porcelain", "--", relative):
            raise ValueError(f"refusing uncommitted evidence: {relative}")
        destination = target / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        evidence_metadata.append({"file": relative, "sha256": sha256})
        written.append(destination)

    metadata = {
        "schema_version": 1,
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_repository": repository,
        "source_commit": actual_commit,
        "generated_files": [path.relative_to(target).as_posix() for path in written],
        "evidence": evidence_metadata,
        "review_required": True,
    }
    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(metadata_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/import"))
    args = parser.parse_args()
    try:
        paths = render(args.manifest, args.repo, args.output_dir)
    except (KeyError, OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"render_manifest: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
