# Hyperparameter sweep tool for the eval toolkit — plan

Status: in work on `feat/eval-sweep`, 2026-11-08. Owner decisions taken: ngm25 CPU; stage-1
grid primary_lr × secondary_lr (ratio) × inhibition exponent × common_lr (96 variants × 2 seeds × 500k); tuning images
held out of the 60k train set; two PRs (driver, then aggregator).

Supersedes `.context/teuxdeux/261107_eval-sweep-tool.md`. Reuses the #284 runner (coastd
placement, ledger/resume, `hash_inputs`) and the #285 schema (top-level `frozen`, `run_id`).

## D1 — A sweep is variants × seeds × lengths over one base template

`sweep spec` (JSON, `coast-eval-sweep-spec/1`):

```json
{"schema": "coast-eval-sweep-spec/1",
 "template": "eval/configs/mnist_ngm25_frozen_eval.json",
 "axes": {"/ngm25/neuron_groups/L4-mnist/primary_learning_rate": [0.01, 0.03, 0.1, 0.3],
          "/ngm25/neuron_groups/L4-mnist/common_learning_rate":  [1e-4, 1e-3],
          "/ngm25/neuron_groups/L4-mnist/secondary_learning_rate":
              {"ratio_of": "/ngm25/neuron_groups/L4-mnist/primary_learning_rate",
               "factors": [0.001, 0.01, 0.1]},
          "/ngm25/inhibitions/ab_pinned/strength":              [1, 5, 10, 20]},
 "variants": {"custom_name": {"/ptr": value}},
 "seeds": [0, 1],
 "lengths": [250000, 500000],
 "conditions": ["clean", "...all 15 MNIST-C...", "zeros"],
 "tuning_images": 1000}
```

Stage-1 grid (jk's input + owner, 2026-11-08): primary × secondary-as-ratio-of-primary ×
max_softmax exponent × common = 4 × 3 × 4 × 2 = 96 variants. primary 0.003 was dropped as
too low; common gets the default and 10× (the June "density lever").

- A **derived axis** (`{"ratio_of": <pointer of another axis>, "factors": [...]}`) takes, per grid
  point, the referenced axis's value times each factor. It exists because `secondary ≤ primary`
  must hold at every primary (the leaf hierarchy inverts otherwise), which an absolute list
  cannot guarantee. In the variant name a derived value appears as its factor:
  `secondary_learning_rate=x0.01`.
- `axes` expands to the full grid; a variant's name is `k=v` pairs joined by `,` with the pointer's
  last segment as `k` (`primary_learning_rate=0.1,common_learning_rate=1e-4,strength=10`).
  `variants` adds named override sets; the default (no overrides) is always included as `base`.
- Overrides are applied by JSON pointer to the template after the protocol has written its own
  keys; a pointer that does not exist in the template is an error (coast only warns on unknown
  keys, so a typo would otherwise run the default silently — `json_loader.cpp:37-48`).
- `conditions` must be from `protocol.CONDITIONS` and pass `report.check_conditions`: a report
  refuses a partial MNIST-C set, so stage 1 runs all 17 conditions (owner's decision, 2026-11-08;
  reports stay comparable with the full A/B reports).

## D2 — Lengths are chained training jobs, not snapshots

coastd's `load_state_from` accepts only a done job's `final_state.h5`. A (variant, seed) chain
is therefore: train job to `lengths[0]`, then a continuation job per further length
(`load_state_from` previous, `steps_target` = next length, same `run_id` so the sequencer
continues — #285), then frozen trials off each done training job. The length of a point is the
manifest's `steps_completed` of its training job.

## D3 — Tuning split: held-out train images, never the eval pool

- `protocol.tuning_split(n)` draws `n` indices from the 60k train set with a fixed seed
  (`TUNING_SEED`), disjoint from the training manifest, which excludes them.
- Frozen trials for tuning present these images (`clean` from `train-images`, corruptions from
  MNIST-C's **train** split — to be verified in the Zenodo zip; `fetch_mnist_c.py` and
  `datasets.PINS` are extended for the needed files, pinned by SHA-256).
- The tool refuses `test` indices as tuning images. Scoring on the eval pool (`< SAMPLE_POOL`)
  or the retina band (9000–9999) is impossible by construction.
- Stage 2 (confirm) reruns the winners on the eval pool via plain `eval_ab.py`; the sweep tool
  does not do that.

## D4 — Output the existing tools read

```
<out>/sweep.json                      coast-eval-sweep/1 (below)
<out>/<variant>/<length>/report.json  coast-eval-report/1, ONE state (label = variant), all seeds
<out>/<variant>/<length>/report.md
<out>/ledger.json                     the #284 runner's ledger (resume)
```

`sweep.json`:
```json
{"schema": "coast-eval-sweep/1", "spec": {...the spec as submitted...},
 "template_sha256": "…", "ref": "<sha>", "daemons": ["host:port", ...],
 "variants": {"<name>": {"/ptr": value, ...}},
 "points": [{"variant": "<name>", "length": 250000, "report": "<variant>/250000/report.json",
             "seeds": [0, 1], "status": "done|pending|failed"}]}
```

Reports are the unchanged `report.build` with one state; the 2-state cap stays. The aggregator
(PR 2, `scripts/eval_sweep_report.py`) reads `sweep.json` + the reports and writes
`<out>/sweep_report.{json,md}`: variant × measure table per length, median over seeds, the
per-seed sign-agreement rule against `base` as "moved", `code_size` / never-wins as collapse
guards; `eval_plot.py` gains grouping by variant.

## D5 — Parallelism is coastd's

Submit everything; a `--slots 1` daemon per machine keeps one coast process per host. Placement
by seed mod N as in the runner. No scheduler in the tool.
Follow-ups (continuations, tuning trials) carry a higher coastd priority than first training jobs, so points finish chain by chain.

## D6 — Resumable

`runs.reusable` per run dir, the runner's ledger per job, and a point whose `report.json`
exists is skipped. Running the same command again continues.

## Budget (stage 1)

ngm25 CPU ≈ 2000 steps/s: 500k train ≈ 4 min; 17 conditions × 1000 images × 15 steps ≈ 255k
frozen steps ≈ 2 min + launch overheads, per length. ≈ 9 min per (variant, seed) chain ⇒ 192 chains
≈ 28 h on one slot, ≈ 14 h on two daemons.
