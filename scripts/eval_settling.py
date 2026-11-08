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
"""Measure how the frozen-trial code settles within a presentation (issue #82).

Usage:
    scripts/eval_settling.py --binary <coast> <sha> --data-root data --out tmp/settling \\
        --report-dir tmp/settling [--template eval/configs/<template>.json]

When this was measured, the milestone-1 readout presented each test image for 15 steps,
dropped the first 2 and averaged the last 13 before the relative winner rule. This script
measures what that choice does: it presents each image for 30 steps and keeps every step.
(The lead has since become the hop count, 1 for pixels, decided from this curve; the
definitions below stay as registered.)

Runs (each spawned with the coast at --binary, run dirs laid out as by eval_ab.py --binary):
- train: per seed, --template (default the pixel template,
  eval/configs/mnist_ngm25_frozen_eval.json) through `protocol.train_jobs`, --steps steps at
  --steps-per-image, train indices 0..--train-images-1, --save-state.
- settle30: per seed, one frozen clean trial loaded from that state over test indices
  0..--test-images-1 at change_interval 30, observing `population` and `input` on every
  step. Its config is `protocol.frozen_jobs`'s clean job with the change interval, the step
  count and the observe range rescaled from 15 to 30 steps per image (the library has no
  parameter for it; everything else is the library's config, unchanged).
- protocol15: per seed, the library's clean frozen trial unchanged (15 steps per image),
  read by `observe.read_codes` at the reference's lead (lead 2 on pixels, as the report
  read it when this was measured).

Pre-registered definitions (written before the first real run; measures may be added
after it, never removed or redefined). Per presentation p of settle30 and step
k = 1..30, k = 1 being the first step on which `input/index` shows the image. No
shift is applied for the buffered hop: on the pixel template the group's output at step
k answers the input of step k - 1, so k = 1 shows the response to the previous image.
- amplitude_k: `population/out` at step k (float32, one value per unit).
- code_k: the relative-rule winner set (frac 0.3) of amplitude_k alone.
- ref: the protocol's code when this was measured, the winner set of the mean of steps 3..15
  (lead 2, window 13 of 15). ref30: the winner set of the mean of steps 3..30.
- avg_k (k >= 3): the winner set of the mean of steps 3..k, the code the protocol would
  give if the presentation were cut at step k. avg_15 = ref, avg_30 = ref30.
- Means are taken in float64 and rounded to float32, as `observe.presentation_means` does.
- Jaccard(a, b) = |a & b| / |a | b|; two empty codes score 1.0. Cosine is over all units;
  a zero norm on either side scores 0.0.
Measures, each the mean over the presentations, per seed and as the mean over seeds:
- jaccard_step_ref[k] = Jaccard(code_k, ref), k = 1..30.
- exact_step_ref[k] = fraction of presentations with code_k == ref, k = 1..30.
- jaccard_avg_ref[k] = Jaccard(avg_k, ref), k = 3..30.
- cosine_step_ref_last[k] = cosine(amplitude_k, amplitude_15), k = 1..30.
- code_size_step[k] = |code_k|, k = 1..30; code_size_ref, code_size_full (ref30).
- jaccard_ref_full = Jaccard(ref, ref30).
- tau_a[v]: Kendall tau-a of pixel-cosine vs code-cosine distances over the clean images
  (`metrics.similarity_preservation`) for the codes v = step1, step2, step3 (code_k with
  its amplitude_k), avg5 (3..5), avg10 (3..10), ref (3..15), ref30 (3..30), and
  protocol15 (the codes of the protocol15 run).
- jaccard_ref_protocol15 = Jaccard(ref, the protocol15 run's code): ref is read inside a
  30-step trial, whose previous image ran 30 steps, not 15; this says whether that
  carry-over changes the code the protocol reads.

Added for other templates (#276; the pixel definitions above are unchanged). The steps above
are those of the pixel template, one hop from the dataset to the group. For a template of h
hops (`protocol.settle_lead`), every reference window starts at h + 2 instead of 3: ref is
steps h+2..15, ref30 h+2..30, avg_k is defined for k >= h+2, the tau-a step codes are step1
.. step(h+2), avg5 and avg10 start at h+2, and protocol15 is read at lead h+1. Starting one
step after the first expected settled step h+1 compares that step with a code that does
not contain it. A retina template (2 hops) gets steps 4..15.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import msgspec

from coast_eval import datasets, metrics, observe, protocol, report, runs, settling

TEMPLATE = Path(__file__).parents[1] / "eval" / "configs" / "mnist_ngm25_frozen_eval.json"
STEPS_PER_IMAGE = 30  # frozen, this measurement only
FRAC = 0.3
CLEAN_PINS = (datasets.TRAIN_IMAGES, datasets.TRAIN_LABELS, datasets.TEST_IMAGES,
              datasets.TEST_LABELS)  # fmt: skip


class Windows(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The reference windows of one template, in steps 1..30 of a presentation."""

    ref: tuple[int, int]
    full: tuple[int, int]
    tau_steps: tuple[int, ...]
    tau_windows: dict[str, tuple[int, int]]

    @property
    def lead(self) -> int:
        """The lead protocol15 is read at: the steps before the reference starts."""
        return self.ref[0] - 1


def windows(cfg: dict[str, Any]) -> Windows:
    """The reference windows for a template of h hops: each starts at h + 2, one step after
    the first expected settled step, so that step is compared with a code that does not
    contain it. h = 1 (pixels) gives the registered 3..15."""
    first = protocol.settle_lead(cfg) + 2
    ref = (first, protocol.FROZEN_STEPS_PER_IMAGE)
    full = (first, STEPS_PER_IMAGE)
    return Windows(
        ref=ref,
        full=full,
        tau_steps=tuple(range(1, first + 1)),
        tau_windows={"avg5": (first, 5), "avg10": (first, 10), "ref": ref, "ref30": full},
    )


class SeedResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    curve: settling.Curve
    tau_a: dict[str, float]
    jaccard_ref_protocol15: float
    code_size_protocol15: float


def spawn(
    binary: Path, run: runs.RunDir, job: protocol.Job, links: dict[str, Path], state: Path | None
) -> runs.RunDir:
    """Run one job in its own run dir with the coast at `binary`, as `eval_ab.py --binary`
    does (a finished run with the same config and inputs record is reused); raise if coast
    fails."""
    run.root.mkdir(parents=True, exist_ok=True)
    for name, target in links.items():
        runs.link(run, name, target)
    record = runs.inputs_record(run, job.config_text, state, binary, job.steps)
    if runs.reusable(run, job.config_text, record):
        return run
    run.manifest.unlink(missing_ok=True)
    run.config.write_text(job.config_text)
    run.inputs_record.write_bytes(record)
    cmd = [
        str(binary.resolve()),
        "-c",
        "config.json",
        "-n",
        str(job.steps),
        "--save-state",
        "final_state.h5",
        "--emit-run-manifest",
        "coast_runtime.json",
    ]
    if state is not None:
        cmd += ["--load-state", str(state.resolve())]
    with (run.root / "coast.log").open("w") as log:
        subprocess.run(cmd, cwd=run.root, stdout=log, stderr=subprocess.STDOUT, check=True)
    return run


def rescaled(job: protocol.Job, n_images: int, base_step: int) -> protocol.Job:
    """`job` (a `frozen_jobs` clean job) at STEPS_PER_IMAGE: the change interval, the step
    count and the observe range, each checked to hold the library's value before it changes."""
    cfg: dict[str, Any] = json.loads(job.config_text)
    ds = cfg["datasets"]["mnist"]["default"]
    old = protocol.FROZEN_STEPS_PER_IMAGE
    window = [[base_step, base_step + n_images * old - 1]]
    groups = cfg["observe"]["groups"]
    if ds["change_interval"] != old or job.steps != n_images * old:
        raise ValueError(f"frozen job is not {n_images} images at {old} steps each")
    if sorted(groups) != ["input", "population"] or any(
        g["ranges"] != window for g in groups.values()
    ):
        raise ValueError(f"frozen job observes {groups}, expected {window} for both groups")
    steps = n_images * STEPS_PER_IMAGE
    ds["change_interval"] = STEPS_PER_IMAGE
    for g in groups.values():
        g["ranges"] = [[base_step, base_step + steps - 1]]
    text = json.dumps(cfg, indent=2, sort_keys=True)
    return protocol.Job(seed=job.seed, condition=job.condition, steps=steps, config_text=text)


def measure_seed(
    settle: runs.RunDir,
    proto: runs.RunDir,
    pixels: report.Floats,
    test_ids: list[int],
    w: Windows,
) -> SeedResult:
    """The pre-registered measures of one seed's settle30 and protocol15 runs."""
    m = runs.read_manifest(settle)
    d = runs.observe_dir(settle, m, ("population", "input"))
    out = observe.read_site(d / "population.observe.h5", "out")
    index = observe.read_site(d / "input.observe.h5", "index")
    ids, amps = settling.presentation_steps(
        out, index, steps_per_image=STEPS_PER_IMAGE, min_segment=2
    )
    protocol.check_aligned(ids, test_ids)
    c = settling.curve(amps, frac=FRAC, ref_first=w.ref[0], ref_last=w.ref[1])
    codes = settling.variants(amps, frac=FRAC, steps=w.tau_steps, windows=w.tau_windows)

    pm = runs.read_manifest(proto)
    # ref's lead: the protocol's when this was measured, the hop count + 1 (the hop count
    # alone since; see protocol.settle_lead)
    lead = protocol.settle_lead(json.loads(proto.config.read_text())) + 1
    if lead != w.lead:
        raise ValueError(f"{proto.config}: lead {lead}, the definitions assume {w.lead}")
    p15 = observe.read_codes(
        runs.observe_dir(proto, pm, ("population", "input")),
        window=protocol.FROZEN_STEPS_PER_IMAGE,
        lead=lead,
        min_segment=2,
        frac=FRAC,
    )
    protocol.check_aligned(p15.stim_ids, test_ids)
    codes["protocol15"] = (p15.amps, p15.masks)
    tau = {k: metrics.similarity_preservation(pixels, a, w) for k, (a, w) in codes.items()}
    ref_masks = codes["ref"][1]
    return SeedResult(
        curve=c,
        tau_a=tau,
        jaccard_ref_protocol15=float(settling.jaccard(ref_masks, p15.masks).mean()),
        code_size_protocol15=float(p15.masks.sum(axis=1).mean()),
    )


def mean_over_seeds(per_seed: list[SeedResult]) -> SeedResult:
    """Element-wise mean of every measure over the seeds."""

    def avg(xs: list[float]) -> float:
        return statistics.fmean(xs)

    def avg_lists(xs: list[list[float]]) -> list[float]:
        return [statistics.fmean(col) for col in zip(*xs, strict=True)]

    cs = [r.curve for r in per_seed]
    curve = settling.Curve(
        jaccard_step_ref=avg_lists([c.jaccard_step_ref for c in cs]),
        exact_step_ref=avg_lists([c.exact_step_ref for c in cs]),
        cosine_step_ref_last=avg_lists([c.cosine_step_ref_last for c in cs]),
        code_size_step=avg_lists([c.code_size_step for c in cs]),
        jaccard_avg_ref=avg_lists([c.jaccard_avg_ref for c in cs]),
        jaccard_ref_full=avg([c.jaccard_ref_full for c in cs]),
        code_size_ref=avg([c.code_size_ref for c in cs]),
        code_size_full=avg([c.code_size_full for c in cs]),
    )
    keys = per_seed[0].tau_a.keys()
    if any(r.tau_a.keys() != keys for r in per_seed):
        raise ValueError("the seeds measured different tau-a variants")
    return SeedResult(
        curve=curve,
        tau_a={k: avg([r.tau_a[k] for r in per_seed]) for k in keys},
        jaccard_ref_protocol15=avg([r.jaccard_ref_protocol15 for r in per_seed]),
        code_size_protocol15=avg([r.code_size_protocol15 for r in per_seed]),
    )


def to_markdown(r: dict[str, Any]) -> str:
    p = r["protocol"]
    mean = r["mean"]
    seeds = sorted(r["seeds"], key=int)
    k_all = list(range(1, STEPS_PER_IMAGE + 1))
    ref, full = p["ref_steps"], p["ref30_steps"]
    lines = [
        "# Frozen-trial settling curve",
        "",
        f"coast `{r['coast_git_sha']}`, template `{p['template']}`. Training: "
        f"{p['train_steps']} steps, {p['train_steps_per_image']} per image, over train indices "
        f"0..{p['train_images'] - 1}, seeds {', '.join(seeds)}. Frozen clean trial: test indices "
        f"0..{p['test_images'] - 1}, {p['frozen_steps_per_image']} steps per image, relative rule "
        f"frac {p['frac']}. `ref` = mean of steps {ref[0]}..{ref[1]} (the reference code), "
        f"`ref30` = mean of steps {full[0]}..{full[1]}; step k = 1 is the first step the dataset "
        "presents the image. Definitions: the docstring of `scripts/eval_settling.py`. Each "
        "value is the mean over the presentations; `mean` is the mean over seeds.",
    ]

    def table(title: str, key: str, ks: list[int], offset: int) -> None:
        head = ["k", "mean", *(f"seed {s}" for s in seeds)]
        lines.extend(["", f"## {title}", "", "| " + " | ".join(head) + " |",
                      "|---:|" + "---:|" * (len(head) - 1)])  # fmt: skip
        for k in ks:
            vals = [mean["curve"][key][k - offset]]
            vals += [r["seeds"][s]["curve"][key][k - offset] for s in seeds]
            lines.append(f"| {k} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    table("Jaccard(code_k, ref)", "jaccard_step_ref", k_all, 1)
    table("Fraction of presentations with code_k == ref", "exact_step_ref", k_all, 1)
    table(f"Jaccard(avg_k, ref), avg_k = mean of steps {ref[0]}..k", "jaccard_avg_ref",
          list(range(ref[0], STEPS_PER_IMAGE + 1)), ref[0])  # fmt: skip
    table(f"cosine(amplitude_k, amplitude_{ref[1]})", "cosine_step_ref_last", k_all, 1)
    table("Code size |code_k|", "code_size_step", k_all, 1)

    head = ["measure", "mean", *(f"seed {s}" for s in seeds)]
    scalars = [
        ("Jaccard(ref, ref30)", ("curve", "jaccard_ref_full")),
        ("Jaccard(ref, protocol15)", ("jaccard_ref_protocol15",)),
        ("code size ref", ("curve", "code_size_ref")),
        ("code size ref30", ("curve", "code_size_full")),
        ("code size protocol15", ("code_size_protocol15",)),
    ]

    def get(x: Any, path: tuple[str, ...]) -> float:
        for key in path:
            x = x[key]
        return float(x)

    lines.extend(["", "## Reference codes", "", "| " + " | ".join(head) + " |",
                  "|---|" + "---:|" * (len(head) - 1)])  # fmt: skip
    for name, path in scalars:
        vals = [get(mean, path), *(get(r["seeds"][s], path) for s in seeds)]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines.extend(["", "## Kendall tau-a (pixel vs code distances, clean images)", "",
                  "| code | mean | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                  "|---|" + "---:|" * (len(seeds) + 1)])  # fmt: skip
    for v in [*(f"step{k}" for k in p["tau_steps"]), *p["tau_windows"], "protocol15"]:
        vals = [mean["tau_a"][v], *(r["seeds"][s]["tau_a"][v] for s in seeds)]
        lines.append(f"| {v} | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    lines.extend(["", "## What the curve shows", "", r["summary"]])
    return "\n".join([*lines, ""])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--binary", nargs=2, required=True, metavar=("COAST", "SHA"))
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="scratch: inputs and run dirs")
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=TEMPLATE, help="default: pixels")
    ap.add_argument("--summary", type=Path, help="Markdown paragraph for 'What the curve shows'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--steps-per-image", type=int, default=1, help="training only")
    ap.add_argument("--train-images", type=int, default=60_000, help="train indices 0..N-1")
    ap.add_argument("--test-images", type=int, default=200, help="test indices 0..N-1")
    ap.add_argument("--jobs", type=int, default=3, help="coast processes at once")
    a = ap.parse_args()
    binary, sha = Path(a.binary[0]), a.binary[1]
    datasets.verify_pins(a.data_root, {k: datasets.PINS[k] for k in CLEAN_PINS})
    template: dict[str, Any] = json.loads(a.template.read_text())
    w = windows(template)
    written = protocol.write_inputs(
        a.out / "inputs", a.data_root, range(a.train_images), range(a.test_images)
    )
    files = (written.train_manifest, written.test_manifest, written.zeros_images)
    inputs = protocol.Inputs(written.test_indices, *(f"./inputs/{Path(f).name}" for f in files))
    links = {"data": a.data_root, "inputs": a.out / "inputs"}
    test_ids = list(inputs.test_indices)

    def go(name: str, job: protocol.Job, state: Path | None, base: int) -> runs.RunDir:
        run = spawn(binary, runs.RunDir(a.out / f"seed{job.seed}" / name), job, links, state)
        m = runs.read_manifest(run)
        runs.check_provenance(m, git_sha=sha, base_step=base, steps=job.steps,
                              frozen=job.condition != "train")  # fmt: skip
        print(f"seed {job.seed} {name}: {m.wall_clock_seconds:.0f} s", flush=True)
        return run

    def go_train(job: protocol.Job) -> runs.RunDir:
        return go("train", job, None, 0)

    def go_frozen(named: tuple[str, protocol.Job]) -> runs.RunDir:
        name, job = named
        return go(name, job, trained[job.seed].final_state, a.steps)

    train = protocol.train_jobs(template, inputs, data_ref="./data", seeds=a.seeds,
                                steps=a.steps, change_interval=a.steps_per_image)  # fmt: skip
    with ThreadPoolExecutor(a.jobs) as pool:
        trained = dict(zip(a.seeds, pool.map(go_train, train), strict=True))
        frozen: list[tuple[str, protocol.Job]] = []
        for s in a.seeds:
            (clean,) = protocol.frozen_jobs(template, inputs, data_ref="./data", seed=s,
                                            conditions=["clean"], base_step=a.steps)  # fmt: skip
            frozen += [("protocol15", clean), ("settle30", rescaled(clean, len(test_ids), a.steps))]
        done = list(pool.map(go_frozen, frozen))
    by = {(job.seed, name): run for (name, job), run in zip(frozen, done, strict=True)}
    pixels = report.condition_pixels(a.data_root, "clean", test_ids)
    per_seed = {
        s: measure_seed(by[s, "settle30"], by[s, "protocol15"], pixels, test_ids, w)
        for s in a.seeds
    }
    r: dict[str, object] = {
        "schema": "coast-eval-settling/1",
        "coast_git_sha": sha,
        "binary_sha256": datasets.sha256_file(binary),
        "protocol": {
            "template": a.template.name,
            "train_steps": a.steps,
            "train_steps_per_image": a.steps_per_image,
            "train_images": a.train_images,
            "test_images": a.test_images,
            "frozen_steps_per_image": STEPS_PER_IMAGE,
            "frac": FRAC,
            "ref_steps": list(w.ref),
            "ref30_steps": list(w.full),
            "tau_windows": {k: list(v) for k, v in w.tau_windows.items()},
            "tau_steps": list(w.tau_steps),
        },
        "seeds": {str(s): v for s, v in per_seed.items()},
        "mean": mean_over_seeds(list(per_seed.values())),
        "summary": a.summary.read_text().strip() if a.summary else "(not written yet)",
    }
    a.report_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(msgspec.to_builtins(r), sort_keys=True, indent=1) + "\n"
    (a.report_dir / "report.json").write_text(text)
    (a.report_dir / "report.md").write_text(to_markdown(json.loads(text)))
    print(f"wrote {a.report_dir}/report.json and report.md")


if __name__ == "__main__":
    main()
