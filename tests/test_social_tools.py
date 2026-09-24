from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from tools.social.buffer_publish import BufferClient, BufferError, plan
from tools.social.generate_linkedin import generate


class Response:
    def __init__(self, document: dict[str, Any]) -> None:
        self.payload = json.dumps(document).encode("utf-8")

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_buffer_create_post_uses_bearer_and_variables() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, timeout: int) -> Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response({"data": {"createPost": {"post": {"id": "post-1", "text": "hello"}}}})

    result = BufferClient("secret", opener=opener).create_post("hello", "channel-1")
    assert result["post"]["id"] == "post-1"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["variables"]["input"]["channelId"] == "channel-1"
    assert captured["timeout"] == 30


def test_buffer_mutation_error_is_actionable() -> None:
    def opener(request: Any, timeout: int) -> Response:
        return Response({"data": {"createPost": {"message": "channel disconnected"}}})

    with pytest.raises(BufferError, match="channel disconnected"):
        BufferClient("secret", opener=opener).create_post("hello", "channel-1")


def test_dry_run_plan_contains_no_secret() -> None:
    document = plan("queue", "hello", "channel-1", None)
    assert document["dry_run"] is True
    assert "api_key" not in document


def test_linkedin_generator_uses_method_when_no_result(tmp_path: Path) -> None:
    post = tmp_path / "2026-01-01-evidence-note.md"
    post.write_text(
        """---
title: "Evidence Note"
description: "A bounded compatibility experiment."
source_repository: "https://github.com/example/source"
---

## Problem

How can the runtime identity remain traceable?

## Experiment

Inspect the environment, pin the source revision, verify the artifact checksum, stage the runtime, and preserve the resulting manifest before any performance comparison.
""",
        encoding="utf-8",
    )
    draft = generate(post, "https://example.github.io")
    assert "does not introduce a new quantitative result" in draft
    assert "https://example.github.io/posts/evidence-note/" in draft
    assert "https://github.com/example/source" in draft
    assert 150 <= len(draft.split()) <= 350
