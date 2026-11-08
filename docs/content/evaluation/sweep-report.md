+++
title = "Sweep Report"
weight = 30
+++

# Sweep Report

A hyperparameter sweep runs one template under many override sets (variants) at several training lengths, and writes one [frozen-evaluation]({{% relref "frozen-evaluation" %}}) report per variant and length. The sweep report reads all of them and says, per length, which variants moved which measure against the template's own settings, which ones did so by changing code size rather than code content, and how they rank. It does not run anything. The plan behind the sweep is `design/eval_sweep_plan.md`, the code is `eval/src/coast_eval/sweep_report.py`, and its driver is `scripts/eval_sweep_report.py`.

## Input and output

The input is the sweep's output directory:

| File | Content |
|---|---|
| `sweep.json` | `coast-eval-sweep/1`: the spec, the build (`ref`), the template's SHA256, every variant's overrides, and every point (variant, length) with its status |
| `<variant>/<length>/report.json` | an unchanged frozen-evaluation report with one state, labelled with the variant's name, holding all training seeds |

```bash
scripts/eval_sweep_report.py <out>     # writes <out>/sweep_report.json and <out>/sweep_report.md
```

`sweep_report.json` (`coast-eval-sweep-report/1`) holds everything the Markdown shows, and the variants and grid axes the heatmap needs. Running it again as more points finish rewrites both files.

## What it reports

Per training length, one row per variant, with the median over training seeds of each measure, as the frozen-evaluation report takes it:

- **τ-a** and the per-corruption **AUCs** (`model/<corruption>`).
- **MNIST-C mean and worst** (`model/mnistc_mean`, `model/mnistc_worst`), as the reports give them, and as a reference the random projection's MNIST-C mean (`random/mnistc_mean`): the median over seeds and projection seeds of `random/<q>/mnistc_mean`, as a report's controls table takes it. It has no Δ.
- **Separation** with its chance level, **domination** with its chance level, **never-wins** and **code size**.

### Against base

Every sweep has a `base` variant: the template without overrides. For each other variant, each training seed's value minus `base`'s value for the same seed gives one difference per seed. The row shows the median of these differences (Δ) and the reading rule of the frozen evaluation over them: a measure **moved** (↑ or ↓) only if every seed moved the same strict way. The pairing matters: seeds can differ from each other more than a variant differs from `base`, so the difference of the two medians can point another way than the seeds do. A variant whose seeds ran on another host or toolchain than `base`'s is refused, as a two-state report refuses it.

### Collapse guards

A variant can raise τ-a or an AUC by changing how many units win instead of what they code. Two guards flag such rows. They do not drop or re-rank them:

- **`code_size`**: the variant's median code size is outside ½ to 2 times `base`'s.
- **`never_wins`**: never-wins rose in every seed, so more of the population went unused.

### Ranking

Per length, the variants are ranked by each primary measure that is present: τ-a, and MNIST-C worst when the sweep ran the corruptions. Ties are broken by name, and guarded variants are marked. With both measures, the report also names the variants that no other variant beats on both (at least as high on each, higher on one). That set needs no weighting of one measure against the other.

### What is missing

A point that is not done, or is done but has no report, is listed under "Not in this report" with its status. It is never left out silently. If `base` is missing at a length, that length's rows have no Δ and no guards. A report that does not match its point (another label, length or seed set), or a sweep whose reports differ in protocol beyond the training length, is refused with the point named.

## Plots

`scripts/eval_plot.py` draws from the same files:

```bash
# curves over training length, one per variant (at most four per figure)
scripts/eval_plot.py --curves --by variant <out>/base/*/report.json <out>/<variant>/*/report.json \
    --out-dir <dir>

# one measure over the grid: rows the first axis, columns the second, a panel per value of a
# third, a row of panels per length
scripts/eval_plot.py --heatmap kendall_tau_a <out>/sweep_report.json --out-dir <dir>

# stage 1 has four axes: fix one to draw the other three
scripts/eval_plot.py --heatmap model/mnistc_worst --fix common_learning_rate=0.0001 \
    <out>/sweep_report.json --out-dir <dir>
```

`--curves` groups by template unless `--by variant` is given. All points of a sweep run one template, so they would collide at each length.

The heatmap's measure is any key of the sweep report's `measures`. It draws two or three axes. `--fix AXIS=VALUE` (repeatable) keeps only the grid variants with that value on that axis and drops the axis. AXIS is the pointer or its last key, and VALUE is written as the driver names the variants (`0.0001`, `x0.01` for a derived axis, whose factors are floats, so `x1.0`) or as JSON of the same value by the driver's rule, `sweep.same_value` (`1e-4`, `x1`). The file name carries the fixed values (`heatmap_model_mnistc_worst_common_learning_rate=0.0001.png`), so slices do not overwrite each other. A derived axis (a value given as a ratio of another axis) is drawn by its factor.

A grid point whose every value is the template's is not run a second time. The driver lists its grid name in `sweep.json`'s `aliases` as `base`, and the heatmap shows `base`'s values in that cell. Named variants, and `base` when no alias places it, are not on the grid; the caption lists them.
