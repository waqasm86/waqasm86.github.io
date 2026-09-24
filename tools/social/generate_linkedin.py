#!/usr/bin/env python3
"""Generate an evidence-bounded LinkedIn draft from a published article."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tools.article.common import load_front_matter


def plain_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`>#]", "", text)
    return " ".join(text.split())


def section(body: str, name: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(name)}\s*$\n(.*?)(?=^##\s|\Z)",
        body,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return plain_text(match.group(1)) if match else ""


def slug_for(path: Path) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", path.stem)


def generate(path: Path, site_url: str) -> str:
    data, body = load_front_matter(path)
    title = str(data["title"])
    problem = section(body, "Problem")
    experiment = section(body, "Experiment")
    result = str(data.get("verified_result", "")).strip()
    repository = str(data.get("source_repository", "https://github.com/waqasm86"))
    article_url = f"{site_url.rstrip('/')}/posts/{slug_for(path)}/"

    hook = problem.split("?", 1)[0].strip() + "?" if "?" in problem else title
    method = experiment[:520].rsplit(" ", 1)[0] if len(experiment) > 520 else experiment
    if result:
        result_paragraph = f"Verified result: {result}"
    else:
        result_paragraph = (
            "This article does not introduce a new quantitative result. It documents the "
            "method, compatibility boundary, and evidence required before publishing one."
        )

    draft = f"""{hook}

I published a technical note on {title}. The focus is the engineering boundary between a managed GPU environment and a native LLM inference runtime: source identity, Python and CUDA compatibility, immutable artifacts, activation, and evidence capture.

What was tested or inspected: {method or data.get('description', '')}

{result_paragraph}

Why it matters: infrastructure comparisons are useful only when the runtime, workload, topology, and raw evidence remain traceable. The note separates smoke-test evidence from performance claims and records what the experiment cannot conclude.

The publication path is intentionally reviewable: code excerpts point to immutable source, machine-readable artifacts remain available, and interpretation stays separate from the measurements that support it.

Read the article: {article_url}

Source and evidence: {repository}
"""
    return draft.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("article", type=Path)
    parser.add_argument("--site-url", default="https://waqasm86.github.io")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/social"))
    args = parser.parse_args()
    try:
        draft = generate(args.article, args.site_url)
        words = len(draft.split())
        if not 150 <= words <= 350:
            raise ValueError(f"generated draft has {words} words; expected 150–350")
        output = args.output_root / slug_for(args.article) / "linkedin.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(draft, encoding="utf-8")
    except (KeyError, OSError, ValueError) as exc:
        print(f"generate_linkedin: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
