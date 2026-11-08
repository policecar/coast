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
"""Run the milestone-1 frozen protocol for one or two code states and write the report.

Usage:
    scripts/eval_ab.py --ref before <sha> --ref after <sha> --ports-file <data-root>/coastd.ports \\
        --dataset-base data --data-root data --out tmp/ab --report-dir tmp/ab
    scripts/eval_ab.py --ref before <sha> --ref after <sha> --daemon host-a --daemon host-b \\
        --dataset-base <NAS>/datasets --data-root data --out tmp/ab --report-dir tmp/ab
    scripts/eval_ab.py --discover
    scripts/eval_ab.py --binary before <coast> <sha> --binary after <coast> <sha> \\
        --data-root data --out tmp/ab --report-dir tmp/ab

--ref runs through coastd (eval/runner.py), the normal path: on the one local daemon of
--ports-file, or on every --daemon HOST[:PORT][=LOCAL_RUNS_ROOT] (port default 51900;
=LOCAL_RUNS_ROOT when this host mounts the daemon's runs_root elsewhere). Seed s runs on daemon
s mod N, for both states, the daemons sorted by hostname and port, whatever order they are
named in. Seeds should spread evenly over the residues mod N, as consecutive seeds do (seeds 0
2 4 on two daemons all land on one; correct, only slower); the placement is printed before
anything is submitted. --dataset-base is where this host ships the inputs, which every daemon
must then report with the expected hashes. An interrupted --ref run is resumed by running the
same command again: <out>/ledger.json names the run, and its jobs are re-attached, not
submitted again (remove the ledger to start afresh). The same command with --cancel cancels
every active job of the evaluation, of any run, ledger or not. A run dir the checks refuse
(a crash can leave a frozen trial's observe files incomplete) is named by its key; the same
command with --redo KEY submits that job afresh and continues the evaluation. --discover
lists the daemons that answer a UDP probe on the local network (not through a VPN), with
their --daemon arguments, and runs nothing.

--binary spawns a prebuilt coast here, for a state that is not a git ref coastd can build
(#243's "before" is 7ec9e0b4 plus one hunk). Its run dirs are laid out like coastd's
(config.json, coast_runtime.json, final_state.h5) under <out>/<label>/seed<k>/<condition>/,
with `data` and `inputs` linked in; a run whose manifest exists, whose config is unchanged and
whose inputs record (inputs.sha256.json: the SHA256 of the coast binary, of the state it loads
and of the ./inputs files its config names, and the directory its `data` link resolves to) is
unchanged is not run again. A binary must record the run's top-level `frozen` in its manifest
(every build since it replaced the per-group flag); an older one is refused after its first run.

A frozen trial's settled code averages the steps after the settle lead, the hop count from
the dataset to the group (`protocol.settle_lead`).

Each state runs the pixel template (eval/configs/mnist_ngm25_frozen_eval.json) unless
--template LABEL PATH names another, e.g. a retina template; the report records each state's
template file name.

Every input file is checked against its SHA256 pin before any run: under --data-root, which
supplies the pixel references, and for --ref under --dataset-base, where the inputs are
shipped. For --ref the daemon itself hashes every pinned and shipped file (coastd
`hash_inputs`) before the first job and after the last, each run's `data` link must point to
the daemon's dataset_base, and each run dir must hold the config submitted for its job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from coast_eval import datasets, protocol, report, runner, runs
from coast_eval.daemon import Client, DaemonSpec, discover, parse_daemon_spec

TEMPLATE = Path(__file__).parents[1] / "eval" / "configs" / "mnist_ngm25_frozen_eval.json"
TRAIN_IMAGES = 60_000


def spawn(
    binary: Path,
    run: runs.RunDir,
    job: protocol.Job,
    links: dict[str, Path],
    load_state: Path | None,
) -> runs.RunDir:
    """Run one job in its own run dir with the coast at `binary`; raise if coast fails. A
    finished run is reused only if its config text and its inputs record (the step count, the
    SHA256 of the binary, the loaded state and the ./inputs files, and the data directory) are
    unchanged: another --steps redoes the training run, a retrained state its frozen runs, and
    a rebuilt coast and another --data-root every run."""
    text = job.config_text
    run.root.mkdir(parents=True, exist_ok=True)
    for name, target in links.items():  # re-pointed if an earlier run left another target
        runs.link(run, name, target)
    record = runs.inputs_record(run, text, load_state, binary, job.steps)
    if runs.reusable(run, text, record):
        return run
    run.manifest.unlink(missing_ok=True)
    run.config.write_text(text)
    run.inputs_record.write_bytes(record)
    cmd = [
        str(binary.resolve()),  # coast runs in the run dir: a relative path would not resolve
        "-c",
        "config.json",
        "-n",
        str(job.steps),
        "--save-state",
        "final_state.h5",
        "--emit-run-manifest",
        "coast_runtime.json",
    ]
    if load_state is not None:
        cmd += ["--load-state", str(load_state.resolve())]
    with (run.root / "coast.log").open("w") as log:
        subprocess.run(cmd, cwd=run.root, stdout=log, stderr=subprocess.STDOUT, check=True)
    return run


def run_binary(
    a: argparse.Namespace,
    label: str,
    binary: Path,
    sha: str,
    template: dict[str, Any],
    written: protocol.Inputs,
) -> dict[tuple[int, str], runs.RunDir]:
    """Train every seed, then run every condition frozen from it, with the coast at `binary`."""
    files = (written.train_manifest, written.test_manifest, written.zeros_images)
    inputs = protocol.Inputs(written.test_indices, *(f"./inputs/{Path(f).name}" for f in files))
    links = {"data": a.data_root, "inputs": a.out / "inputs"}

    trained: dict[int, runs.RunDir] = {}

    def go(job: protocol.Job) -> runs.RunDir:
        """Train from scratch, or run frozen from this seed's trained state at step a.steps."""
        base = 0 if job.condition == "train" else a.steps
        state = trained[job.seed].final_state if base else None
        name = a.out / label / f"seed{job.seed}" / job.condition
        run = spawn(binary, runs.RunDir(name), job, links, state)
        m = runs.read_manifest(run)
        runs.check_provenance(m, git_sha=sha, base_step=base, steps=job.steps,
                              frozen=job.condition != "train")  # fmt: skip
        return run

    train = protocol.train_jobs(
        template,
        inputs,
        data_ref="./data",
        seeds=a.seeds,
        steps=a.steps,
        change_interval=a.steps_per_image,
    )
    with ThreadPoolExecutor(a.jobs) as pool:
        trained.update(zip(a.seeds, pool.map(go, train), strict=True))
        frozen = [
            job
            for s in a.seeds
            for job in protocol.frozen_jobs(
                template,
                inputs,
                data_ref="./data",
                seed=s,
                conditions=a.conditions,
                base_step=a.steps,
            )
        ]
        done = pool.map(go, frozen)
        out = {(j.seed, j.condition): run for j, run in zip(frozen, done, strict=True)}
    return out | {(s, "train"): run for s, run in trained.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", nargs=2, action="append", metavar=("LABEL", "SHA"), default=[])
    ap.add_argument(
        "--binary", nargs=3, action="append", metavar=("LABEL", "COAST", "SHA"), default=[]
    )
    ap.add_argument(
        "--template",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        default=[],
        help=f"the template of state LABEL (default: {TEMPLATE.name})",
    )
    ap.add_argument("--ports-file", type=Path, help="--ref only: <coastd data root>/coastd.ports")
    ap.add_argument(
        "--redo",
        action="append",
        default=[],
        metavar="LABEL/SEED/CONDITION",
        help="--ref only, repeatable: submit this job afresh although one is adopted (a run "
        "the checks refused names its key); a training key redoes its frozen trials too",
    )
    ap.add_argument(
        "--daemon",
        action="append",
        default=[],
        metavar="HOST[:PORT][=LOCAL_RUNS_ROOT]",
        help="--ref only, repeatable: a coastd to run on, instead of --ports-file",
    )
    ap.add_argument("--discover", action="store_true", help="list the daemons that answer; exit")
    ap.add_argument(
        "--cancel",
        action="store_true",
        help="--ref only: cancel every active job of this evaluation (any run, no ledger "
        "needed: give the same arguments as the run); exit",
    )
    ap.add_argument(
        "--dataset-base", type=Path, help="--ref only: the daemons' dataset_base on this host"
    )
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path, help="scratch: inputs and --binary runs")
    ap.add_argument("--report-dir", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument(
        "--steps-per-image",
        type=int,
        default=1,
        help="training only; 1 is the regime of the shipped MNIST configs (#243 ran 5)",
    )
    ap.add_argument("--test-images", type=int, default=1000, help="`protocol.evaluation_sample`")
    ap.add_argument("--train-images", type=int, default=TRAIN_IMAGES, help="train indices 0..N-1")
    ap.add_argument("--conditions", nargs="+", default=list(protocol.CONDITIONS))
    ap.add_argument("--projection-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--jobs", type=int, default=4, help="--binary only: coast processes at once")
    ap.add_argument(
        "--timeout-s", type=float, default=24 * 3600, help="--ref only: the whole run, all states"
    )
    a = ap.parse_args()
    if a.discover:
        found = discover()
        for d in found:
            print(f"{d.hostname:20} --daemon {d.spec:24} proto v{d.proto_version}")
        if not found:
            print("no coastd answered (UDP 51901 on the local network)")
        return
    missing = [f"--{n.replace('_', '-')}" for n in ("data_root", "out", "report_dir")
               if getattr(a, n) is None]  # fmt: skip
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")
    if not 1 <= len(a.ref) + len(a.binary) <= 2 or (a.ref and a.binary):
        ap.error("give one or two states, all --ref or all --binary")
    labels = [state[0] for state in a.ref + a.binary]
    given = dict(a.template)
    if len(set(labels)) != len(labels) or len(given) != len(a.template):
        ap.error("state labels and --template labels must be unique")
    if not given.keys() <= set(labels):
        ap.error(f"--template for unknown states {sorted(given.keys() - set(labels))}")
    template_paths = {label: Path(given.get(label, TEMPLATE)) for label in labels}
    if a.ref and ((a.ports_file is None) == (not a.daemon) or a.dataset_base is None):
        ap.error("--ref needs --dataset-base and either --ports-file or --daemon")
    if (a.ports_file or a.daemon or a.cancel) and not a.ref:
        ap.error("--ports-file, --daemon and --cancel are --ref only")
    try:
        specs = [parse_daemon_spec(d) for d in a.daemon]
    except ValueError as e:
        ap.error(str(e))
    try:
        report.check_conditions(a.conditions)  # before any run, not after all of them
    except ValueError as e:
        ap.error(str(e))
    datasets.verify_pins(a.data_root, datasets.PINS)  # the pixel references
    templates = {label: json.loads(path.read_text()) for label, path in template_paths.items()}
    inputs = protocol.write_inputs(
        a.out / "inputs",
        a.data_root,
        range(a.train_images),
        protocol.evaluation_sample(a.test_images),
    )
    p = report.Protocol(
        train_steps=a.steps,
        train_steps_per_image=a.steps_per_image,
        train_images=a.train_images,
        test_indices=list(inputs.test_indices),
        conditions=a.conditions,
        projection_seeds=a.projection_seeds,
        frac=0.3,
    )
    pixels = {c: report.condition_pixels(a.data_root, c, p.test_indices) for c in a.conditions}
    states: list[report.State] = []
    for label, binary, sha in a.binary:
        dirs = run_binary(a, label, Path(binary), sha, templates[label], inputs)
        states.append(
            report.measure_state(
                label,
                dirs,
                p,
                pixels,
                template=template_paths[label].name,
                binary_sha256=datasets.sha256_file(Path(binary)),
            )
        )
    if a.ref:
        if a.ports_file:
            specs = [DaemonSpec("127.0.0.1", Client.from_ports_file(a.ports_file).port)]
        daemons = runner.canonical([runner.connect(spec) for spec in specs])
        placed = runner.assign(a.seeds, len(daemons))
        for i, d in enumerate(daemons):
            print(f"{d.name}: seeds {[s for s in a.seeds if placed[s] == i]}")
        name = f"m1-{a.train_images}-{a.test_images}"
        shipped = runner.ship_inputs(inputs, a.dataset_base, name)
        # what every run may read, relative to dataset_base: the pins and the shipped files
        files = (inputs.train_manifest, inputs.test_manifest, inputs.zeros_images)
        expected = dict(datasets.PINS) | {
            f"eval/{name}/{Path(f).name}": datasets.sha256_file(Path(f)) for f in files
        }
        ref_states = [runner.State(label, sha, templates[label]) for label, sha in a.ref]
        if a.cancel:
            fp = runner.fingerprint(daemons, ref_states, shipped, expected, a.seeds, a.steps,
                                    a.steps_per_image, a.conditions)  # fmt: skip
            gone = runner.cancel_evaluation(daemons, fp)
            for d, job_id, key in gone:
                print(f"cancelled {d.name}: {job_id} ({key})")
            print(f"{len(gone)} active job(s) of this evaluation cancelled")
            return
        dirs = runner.run_milestone1(
            daemons,
            ref_states,
            shipped,
            expected_inputs=expected,
            seeds=a.seeds,
            steps=a.steps,
            change_interval=a.steps_per_image,
            conditions=a.conditions,
            poll_s=5.0,
            timeout_s=a.timeout_s,
            ledger=a.out / "ledger.json",
            redo=a.redo,
        )
        for label, _ in a.ref:
            states.append(
                report.measure_state(
                    label, dirs[label], p, pixels, template=template_paths[label].name
                )
            )
    r = report.build(
        p, report.pixel_measures(pixels, np.asarray(p.test_indices, dtype=np.int64)), states
    )
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "report.json").write_bytes(report.to_json(r))
    (a.report_dir / "report.md").write_text(report.to_markdown(r))
    print(f"wrote {a.report_dir}/report.json and report.md")


if __name__ == "__main__":
    main()
