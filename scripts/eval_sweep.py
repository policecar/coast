#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
#   "numpy",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Run a hyperparameter sweep of the frozen evaluation through coastd and write its reports.

Usage:
    scripts/eval_sweep.py --spec eval/sweeps/stage1_ngm25_lr_inhib.json --ref main <sha> \\
        --daemon host-a --daemon host-b --dataset-base <NAS>/datasets --data-root data \\
        --out tmp/sweep

The spec (coast-eval-sweep-spec/1, design/eval_sweep_plan.md D1) names the template (relative
to the repository root), the axes and named variants, the seeds, the training lengths, the
tuning conditions and the number of tuning images. Every variant runs at the one code state
--ref names (LABEL is printed only; SHA must be a full commit SHA). Seed s runs on daemon
s mod N, the daemons sorted by hostname and port, as in eval_ab.py; each (variant, seed)
chain trains to the first length, continues to each further length, and runs its tuning
trials off each. The tuning trials present train images held out of training
(coast_eval.protocol.tuning_split), never test images.

Output in --out: sweep.json (the index: spec, variants, points and their status),
<variant>/<length>/report.{json,md} per point (one state, all seeds), ledger.json and the
inputs. An interrupted sweep is resumed by running the same command again: the ledger
re-attaches its jobs and a point whose report exists is skipped. The same command with
--cancel cancels every active job of the sweep. A run dir the checks refuse is named by its
key; the same command with --redo KEY submits it afresh (a training key also redoes the later
lengths of its chain).

Every input file is checked against its SHA256 pin under --data-root before anything runs,
and each daemon hashes every file a run reads (coastd hash_inputs) before the first job and
after the last.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coast_eval import datasets, protocol, report, runner, sweep
from coast_eval.daemon import parse_daemon_spec

REPO = Path(__file__).parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--ref", nargs=2, required=True, metavar=("LABEL", "SHA"))
    ap.add_argument(
        "--daemon",
        action="append",
        required=True,
        metavar="HOST[:PORT][=LOCAL_RUNS_ROOT]",
        help="repeatable: a coastd to run on",
    )
    ap.add_argument(
        "--dataset-base", type=Path, required=True, help="the daemons' dataset_base on this host"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cancel",
        action="store_true",
        help="cancel every active job of this sweep (give the same arguments as the run); exit",
    )
    ap.add_argument(
        "--redo",
        action="append",
        default=[],
        metavar="VARIANT/LENGTH/SEED/CONDITION",
        help="repeatable: submit this job afresh although one is adopted",
    )
    ap.add_argument("--timeout-s", type=float, default=48 * 3600, help="the whole sweep")
    a = ap.parse_args()
    try:
        specs = [parse_daemon_spec(d) for d in a.daemon]
        raw = a.spec.read_bytes()
        spec = sweep.load_spec(raw)
        report.check_conditions(spec.conditions)  # before any run, not after all of them
        pins = sweep.input_pins(spec.conditions)
    except ValueError as e:
        ap.error(str(e))
    datasets.verify_pins(a.data_root, pins)
    template = Path(spec.template)
    local = protocol.write_tuning_inputs(a.out / "inputs", a.data_root, spec.tuning_images)
    daemons = runner.canonical([runner.connect(s) for s in specs])
    placed = runner.assign(spec.seeds, len(daemons))
    for i, d in enumerate(daemons):
        print(f"{d.name}: seeds {[s for s in spec.seeds if placed[s] == i]}")
    label, sha = a.ref
    files = [Path(f) for f in (local.train_manifest, local.tuning_manifest, local.images,
                               local.labels, local.zeros_images)]  # fmt: skip
    name = f"sweep-tuning-{spec.tuning_images}"
    shipped = protocol.TuningInputs(
        local.tuning_indices, *runner.ship_files(files, a.dataset_base, name)
    )
    expected = pins | {f"eval/{name}/{f.name}": datasets.sha256_file(f) for f in files}
    sw = sweep.make_sweep(spec, json.loads(raw), ref=sha, inputs=shipped,
                          template_path=template if template.is_absolute()
                          else REPO / template)  # fmt: skip
    print(f"{label} ({sha}): {len(sw.variants)} variants x {len(spec.seeds)} seeds x "
          f"{len(spec.lengths)} lengths")  # fmt: skip
    if a.cancel:
        gone = runner.cancel_evaluation(daemons, sweep.fingerprint(daemons, sw, expected))
        for d, job_id, key in gone:
            print(f"cancelled {d.name}: {job_id} ({key})")
        print(f"{len(gone)} active job(s) of this sweep cancelled")
        return
    pixels = {c: sweep.tuning_pixels(a.data_root, local, c) for c in spec.conditions}
    doc = sweep.run_sweep(daemons, sw, out=a.out, pixels=pixels, expected_inputs=expected,
                          poll_s=5.0, timeout_s=a.timeout_s, redo=a.redo)  # fmt: skip
    done = sum(p.status == "done" for p in doc.points)
    print(f"wrote {a.out}/sweep.json: {done} of {len(doc.points)} points done")


if __name__ == "__main__":
    main()
