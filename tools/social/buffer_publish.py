#!/usr/bin/env python3
"""Create an optional Buffer queue item or draft; dry-run unless --execute is set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

BUFFER_ENDPOINT = "https://api.buffer.com"


class BufferError(RuntimeError):
    """Actionable Buffer API failure."""


class BufferClient:
    """Small GraphQL client whose HTTP opener can be replaced in tests."""

    def __init__(
        self,
        api_key: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
        endpoint: str = BUFFER_ENDPOINT,
    ) -> None:
        if not api_key:
            raise ValueError("BUFFER_API_KEY is required for live requests")
        self.api_key = api_key
        self.opener = opener
        self.endpoint = endpoint

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "waqasm86.github.io-publisher/1.0",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=30) as response:
                document = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BufferError(f"Buffer HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BufferError(f"Buffer request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise BufferError("Buffer returned invalid JSON") from exc
        if document.get("errors"):
            raise BufferError(f"Buffer GraphQL error: {document['errors']}")
        return document

    def create_post(self, text: str, channel_id: str, mode: str = "addToQueue") -> dict[str, Any]:
        query = """
        mutation CreatePost($input: CreatePostInput!) {
          createPost(input: $input) {
            ... on PostActionSuccess { post { id text dueAt } }
            ... on MutationError { message }
          }
        }
        """
        document = self.graphql(
            query,
            {"input": {"text": text, "channelId": channel_id, "schedulingType": "automatic", "mode": mode}},
        )
        result = document.get("data", {}).get("createPost", {})
        if result.get("message"):
            raise BufferError(f"Buffer rejected post: {result['message']}")
        return result

    def create_idea(self, text: str, organization_id: str, title: str) -> dict[str, Any]:
        query = """
        mutation CreateIdea($input: CreateIdeaInput!) {
          createIdea(input: $input) {
            ... on Idea { id content { title text } }
            ... on MutationError { message }
          }
        }
        """
        document = self.graphql(
            query,
            {"input": {"organizationId": organization_id, "content": {"title": title, "text": text}}},
        )
        result = document.get("data", {}).get("createIdea", {})
        if result.get("message"):
            raise BufferError(f"Buffer rejected idea: {result['message']}")
        return result


def plan(mode: str, text: str, channel_id: str | None, organization_id: str | None) -> dict[str, Any]:
    return {
        "dry_run": True,
        "endpoint": BUFFER_ENDPOINT,
        "operation": "createIdea" if mode == "draft" else "createPost",
        "mode": mode,
        "channel_id": channel_id,
        "organization_id": organization_id,
        "characters": len(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path, help="UTF-8 text file generated for LinkedIn")
    parser.add_argument("--mode", choices=("draft", "queue", "now", "next"), default="draft")
    parser.add_argument("--title", default="Technical article draft")
    parser.add_argument("--execute", action="store_true", help="Make the live API request")
    args = parser.parse_args()
    try:
        text = args.draft.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("draft is empty")
        channel_id = os.environ.get("BUFFER_CHANNEL_ID")
        organization_id = os.environ.get("BUFFER_ORGANIZATION_ID")
        if not args.execute:
            print(json.dumps(plan(args.mode, text, channel_id, organization_id), indent=2))
            return 0

        client = BufferClient(os.environ.get("BUFFER_API_KEY", ""))
        if args.mode == "draft":
            if not organization_id:
                raise ValueError("BUFFER_ORGANIZATION_ID is required for Buffer Ideas")
            result = client.create_idea(text, organization_id, args.title)
        else:
            if not channel_id:
                raise ValueError("BUFFER_CHANNEL_ID is required for queued posts")
            mode = {"queue": "addToQueue", "now": "shareNow", "next": "shareNext"}[args.mode]
            result = client.create_post(text, channel_id, mode)
        print(json.dumps(result, indent=2))
    except (BufferError, OSError, ValueError) as exc:
        print(f"buffer_publish: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
