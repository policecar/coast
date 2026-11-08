#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
#   "msgspec",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Convert a `coast-eval-report/1` report.json to `coast-eval-report/2`.

Usage:
    scripts/eval_report_upgrade.py <report.json>...

Each input is written as `report.v2.json` beside it (`<stem>.v2.json` for another name); the
input is never changed, and an existing output is refused rather than overwritten. /2 differs
from /1 in two ways:
- `protocol.extra_lead` is gone. A /1 report with extra_lead 0 converts by dropping it; any
  other value means steps were dropped beyond the settle lead, which /2 cannot express, so it
  is refused.
- `separation_chance` and `domination_chance` are per-seed measures that a /1 report written
  before 5f0e17d8 lacks. They need the codes, which a report does not hold, so its conversion
  lacks them too; `eval_plot.py` and the Markdown then say so instead of drawing a chance
  level. A /1 report written after 5f0e17d8 holds them, and they are copied unchanged.
A /2 input is refused: there is nothing to convert, and converting it again would only make
a second copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import msgspec

from coast_eval import report

OLD = "coast-eval-report/1"


def upgrade(data: dict[str, Any]) -> report.Report:
    """The /2 report for the decoded JSON of a /1 report; raise if it cannot be converted."""
    schema = data.get("schema")
    if schema == report.SCHEMA:
        raise ValueError(f"already {report.SCHEMA}: nothing to convert")
    if schema != OLD:
        raise ValueError(f"not a {OLD} report (schema {schema!r})")
    protocol = dict(data["protocol"])
    extra = protocol.pop("extra_lead", 0)
    if extra != 0:
        raise ValueError(
            f"extra_lead {extra}: its codes dropped steps beyond the settle lead, which "
            f"{report.SCHEMA} cannot express"
        )
    new = {**data, "schema": report.SCHEMA, "protocol": protocol}
    # through JSON, so seed keys decode as the report reads them ("0" -> 0), strictly
    return msgspec.json.decode(json.dumps(new), type=report.Report)


def output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.v2.json")


def convert(path: Path) -> Path:
    """Write the /2 report of the /1 report at `path` beside it; return where."""
    out = output_path(path)
    if out.exists():
        raise ValueError(f"{out}: exists; remove it first to convert again")
    r = upgrade(json.loads(path.read_text()))
    out.write_bytes(report.to_json(r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reports", type=Path, nargs="+")
    a = ap.parse_args()
    for path in a.reports:
        print(f"{path} -> {convert(path)}")


if __name__ == "__main__":
    main()
