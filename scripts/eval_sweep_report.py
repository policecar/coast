#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Summarise a hyperparameter sweep: one table of variants per training length.

Usage:
    scripts/eval_sweep_report.py <out>

`<out>` is the output directory of a sweep (`design/eval_sweep_plan.md` D4): it holds
`sweep.json` and a `<variant>/<length>/report.json` per done point. Writes
`<out>/sweep_report.json` (`coast-eval-sweep-report/1`) and `<out>/sweep_report.md`, and
nothing else. Per length: the median over seeds of each measure per variant, the change
against the `base` variant paired by seed with the reading rule (a measure moved only if
every seed moved the same way), the collapse guards (code size far from base's, never-wins
up), and a ranking by τ-a and by the MNIST-C worst AUC.
Points not done, or done without a report, are listed. The logic is
`coast_eval.sweep_report`; run it again as more points finish.

Draw the curves of a few variants over length with
`scripts/eval_plot.py --curves --by variant <out>/<variant>/*/report.json ...`, and one
measure over the grid with `scripts/eval_plot.py --heatmap <measure> <out>/sweep_report.json`
(add `--fix AXIS=VALUE` until two or three axes are left).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from coast_eval import sweep_report


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path, help="the sweep's output directory (holds sweep.json)")
    a = ap.parse_args(argv)
    for path in sweep_report.write(a.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
