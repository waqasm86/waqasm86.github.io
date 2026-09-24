from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.article.check_links import check
from tools.render.benchmark_chart import load_records, render_chart
from tools.render.social_card import render


def test_local_link_checker_accepts_asset_and_route(tmp_path: Path) -> None:
    (tmp_path / "_posts").mkdir()
    (tmp_path / "_tabs").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "ok.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "_tabs" / "projects.md").write_text("---\ntitle: Projects\n---\n", encoding="utf-8")
    post = tmp_path / "_posts" / "2026-01-01-test.md"
    post.write_text(
        "---\ntitle: Test\n---\n[asset](/assets/ok.txt) [projects](/projects/)\n",
        encoding="utf-8",
    )
    assert check(tmp_path, [post]) == []


def test_local_link_checker_reports_missing_asset(tmp_path: Path) -> None:
    (tmp_path / "_posts").mkdir()
    (tmp_path / "_tabs").mkdir()
    post = tmp_path / "_posts" / "2026-01-01-test.md"
    post.write_text("---\ntitle: Test\n---\n![missing](/assets/no.png)\n", encoding="utf-8")
    assert "missing local target" in check(tmp_path, [post])[0]


def test_benchmark_chart_renders_existing_csv(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    source.write_text("concurrency,throughput\n1,10\n2,18\n", encoding="utf-8")
    output = tmp_path / "chart.png"
    records = load_records(source)
    render_chart(records, "concurrency", ["throughput"], output, "Test", "Concurrency", "Tokens/s")
    assert output.read_bytes().startswith(b"\x89PNG")


def test_social_card_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "card.png"
    render("A reproducible systems experiment", "Engineering Notes", output)
    with Image.open(output) as image:
        assert image.size == (1200, 630)
