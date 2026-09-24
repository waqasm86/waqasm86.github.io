#!/usr/bin/env python3
"""Render deterministic benchmark charts from existing CSV or JSON evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def lookup(document: Any, path: str) -> Any:
    value = document
    if not path:
        return value
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def load_records(path: Path, records_path: str = "") -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        records = lookup(json.loads(path.read_text(encoding="utf-8")), records_path)
    else:
        raise ValueError("input must be .csv or .json")
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError("selected evidence must be a non-empty list of objects")
    return records


def render_chart(
    records: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    output: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    try:
        x_values = [float(row[x_key]) for row in records]
        series = {key: [float(row[key]) for row in records] for key in y_keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or non-numeric chart value: {exc}") from exc

    plt.rcParams.update({"font.size": 11, "figure.dpi": 100, "savefig.dpi": 150})
    figure, axis = plt.subplots(figsize=(9, 5.25), constrained_layout=True)
    for key in y_keys:
        axis.plot(x_values, series[key], marker="o", linewidth=2, label=key)
    axis.set(title=title, xlabel=x_label, ylabel=y_label)
    axis.grid(True, alpha=0.25)
    if len(y_keys) > 1:
        axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"Creator": "waqasm86.github.io benchmark_chart.py"})
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--records", default="", help="Dotted path to the JSON record list")
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--x-label", required=True)
    parser.add_argument("--y-label", required=True)
    args = parser.parse_args()
    try:
        records = load_records(args.input, args.records)
        render_chart(records, args.x, args.y, args.output, args.title, args.x_label, args.y_label)
    except (OSError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        print(f"benchmark_chart: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
