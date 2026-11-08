+++
title = "Golden Protocol"
weight = 20
+++

# Golden Protocol

A-tier integration tests work by producing a JSON **blob** that fingerprints the simulation state after N steps, then comparing that blob against a directory of retained golden blobs via `tools/coast_compare`. This page documents the protocol — the blob format, the retention policy, the multi-version match semantics, and the comparison logic. For *operating* the protocol (capturing, updating, reading failures), see [`tests/how_to_run.md`](https://github.com/) and [`tests/how_to_write.md`](https://github.com/).

## Blob format — JSON with binary sidecars

Each test writes a JSON file containing whatever fields the test author chose to fingerprint. The shape is free-form; leaves are scalars or 1D numeric arrays. Large arrays are externalised as binary sidecar files referenced from the JSON.

An excerpt from `tests/integration/mnist_seed0_baseline/goldens/20261030_ceb97686.json`:

```json
{
  "buffer_0": {
    "max":  1.0,
    "mean": 0.1938535079769325,
    "min":  1.3611524082079995e-05,
    "size": 784,
    "sum":  151.98115025391508,
    "values": { "binary": "20261030_ceb97686.buffer_0.bin", "dtype": "f64", "shape": [784] }
  },
  "buffer_1": {
    "max":  0.5103124380111694,
    "mean": 0.11541243743151426,
    ...
    "values": [ 0.103118, 0.0423155, 0.118504, ... ]
  }
}
```

Two conventions visible in that excerpt:

- **Summary statistics inline.** `min`/`max`/`mean`/`sum`/`size` are kept as scalar JSON fields. They diff cleanly in `git diff` and provide a coarse top-level discriminator before any per-element comparison kicks in.
- **Bulk arrays as sidecars.** `buffer_0/values` is 784 doubles; it lives in `20261030_ceb97686.buffer_0.bin` (raw little-endian f64) referenced by `{"binary": ..., "dtype": ..., "shape": ...}`. `coast_compare` dereferences these as it walks. Smaller arrays (`buffer_1/values`, 50 elements) stay inline because they remain reviewable.

### Why JSON and not HDF5

HDF5 was the obvious candidate — COAST already uses it for the logger and for state snapshots. It was rejected because HDF5 is **not thread-safe** and the simulation routes all HDF5 writes through a serializing `hdf5::executor`. Plumbing test writes through that executor would couple the test infrastructure to executor lifecycle: the test would need to start, drain, and stop the executor around each capture. JSON has none of that and brings two further wins:

- `git diff` on a golden file shows the actual numeric change a reviewer needs to see.
- The size pressure of "this has to be reviewable as text" creates a design discipline pushing test authors toward summary statistics rather than dumping full state.

The binary sidecar escape hatch handles the case where summarization genuinely isn't possible — but using it carries the diff-readability cost as a deliberate choice, not a default.

## Retention — N=5, β-naming

```
tests/integration/<test>/goldens/
    <YYYYMMDD>_<sha>.json           ← oldest
    <YYYYMMDD>_<sha>.buffer_0.bin
    ...                             (up to 5 goldens)
    <YYYYMMDD>_<sha>.json           ← newest
    <YYYYMMDD>_<sha>.buffer_0.bin
```

**Filename convention** (called β-naming): `YYYYMMDD_<commit-short-sha>.json`. Chronological string-sort gives chronological order for free; no rename churn when the oldest file is rotated out.

**Each golden owns its sidecars.** `scripts/update_golden.sh` stores a captured sidecar `buffer_0.bin` as `<golden>.buffer_0.bin` and rewrites the golden's `"binary"` reference to that name; the comparator resolves the reference relative to the golden's directory. It refuses to store a golden whose references do not resolve to its own sidecars — staging the capture first, so a refusal writes nothing, not even over a same-stamp golden's sidecars — and after rotation it deletes every `.bin` no remaining golden references. Goldens captured before this rule share one bare `buffer_0.bin`, which every capture overwrote — so only the newest of them still carries its own fingerprint. They keep resolving until they rotate out, and the shared file goes with the last of them.

**Retention count: 5.** Why this number:

- A repository churn-y enough to fill 5 goldens in a few months will have at least one developer (or AI agent) reverting an earlier fix; with 5 retained versions, "this PR matches the second-oldest golden" is visible — that's a fingerprint of "this PR re-introduced bug X".
- Larger N inflates the repo without commensurate signal: the failure-localisation story doesn't improve once you have ~3 historical points.
- The N=5 oldest rotation is automatic; the developer never edits the goldens directory by hand.

## Multi-version match semantics

A test **passes** if its new blob matches **any** of the retained goldens within tolerance. Failure means "no retained golden matched"; the report names the closest of the bunch.

This shape has two consequences worth being explicit about:

1. **Intentional changes that happen to match a previous golden** revert silently. If the new behavior happens to coincide with a historical golden, that golden was retained for a reason and the match is meaningful — usually it means the change reverted to a previous state. The test passes, and the diff against the *closest* retained version is the diagnostic to look at.
2. **Genuine drift inside the tolerance** never escalates. If every reasonable code path stays inside `rel=1e-6, abs=1e-9` of *some* historical golden forever, the suite stays green forever. This is the design — tolerance-based comparison is by definition a noise filter. The fix when this becomes a problem is to tighten tolerances per field, not to redesign the matching.

## Tolerance

Default `tolerance` is `rel = 1e-6, abs_floor = 1e-9`. Comparison:

```
|a - b|  <  max( rel * max(|a|, |b|),  abs_floor )
```

The `abs_floor` term matters for fields whose magnitudes legitimately span many orders: a `min` value near zero would otherwise produce huge relative deltas from float noise alone.

**Per-field overrides** live in `tests/integration/<test>/tolerances.json`:

```json
{
    "default":   { "rel": 1e-6, "abs_floor": 1e-9 },
    "overrides": { "buffer_2/spike_counts": { "rel": 1e-4 } }
}
```

The discipline is **add overrides only after a real rerun trips the default**. Pre-emptive overrides defeat the tripwire — a field whose tolerance was widened in advance no longer tells you anything when it would have caught a regression.

## Comparison logic — strict for arrays

Any single element of an array exceeding tolerance fails the **whole field**. The flat report surfaces the worst-offending element (highest Δ) first.

Loose comparison modes — "X% of elements within tolerance", "mean delta below threshold" — were rejected during Design. They turn the tripwire into a noise filter: a real bug that shifts every element by exactly the wrong amount averages to zero. The point of A-tier is to *catch* things, not to *smooth over* them.

## Aggregate mode — for CPU vs GPU parity

Strict comparison is the default and is unchanged. But the *ngm3-gpu-port* cycle added a **second, opt-in mode** — `aggregate` — for one specific situation: comparing a GPU entity against its CPU equivalent. The GPU ngm3 uses a stateless hash PRNG, not the CPU's `mt19937` stream, so the two **diverge per-element by design** (see [Decisions]({{% relref "/architecture/decisions" %}}), ADR-5). Strict comparison cannot apply; bit-equivalence is structurally impossible.

Aggregate mode reduces each numeric field to **(mean, max, std)** and compares those three statistics independently, with their own tolerances from a new `aggregate` block in `tolerances.json`:

```json
{
    "default":   { "rel": 1e-6, "abs_floor": 1e-9 },
    "aggregate": {
        "base":   { "rel": 1e-4 },
        "mean":   { "rel": 1e-5 },
        "stddev": { "rel": 1e-3 }
    }
}
```

This is **not** the "loose comparison" that was rejected for strict mode. It is a deliberately different *contract*: where strict mode asserts "this exact state reproduces," aggregate mode asserts "this state is statistically indistinguishable from the reference distribution." The tolerances were pinned by a CPU-vs-CPU-different-seed reference run that establishes the noise floor — wide enough to ignore RNG-stream divergence, tight enough to still catch a real structural regression.

### The four ngm3 A-tier tests

One test per `io_buffer` variant, all on the same MNIST→ngm3 pipeline:

| Test | Pipeline | io_buffer variant | Mode |
|------|----------|-------------------|------|
| `mnist_ngm3_philox` | CPU → CPU | `io_buffer_cpu` | **strict** (shared baseline) |
| `mnist_ngm3_cpu_to_gpu` | CPU mnist → GPU ngm3 | `io_buffer_cpu_to_gpu` | aggregate |
| `mnist_ngm3_gpu_to_cpu` | GPU mnist → CPU ngm3 | `io_buffer_gpu_to_cpu` | aggregate |
| `mnist_ngm3_heterogeneous` | GPU mnist → GPU ngm3 | `io_buffer_gpu` | aggregate |

The CPU `mnist_ngm3_philox` run is the strict reference the three heterogeneous tests are measured against. The test harness selects the mode via a compile definition (`COAST_TEST_COMPARE_MODE_AGGREGATE`) set by the `coast_add_integration_test` CMake helper.

**Fingerprinting a GPU-only buffer.** A pure `io_buffer_gpu` (as in `mnist_ngm3_heterogeneous`) has no CPU-visible read span. Rather than alter production code, the test harness performs an **on-demand GPU readback** (staging `CopyBufferToBuffer` + map) at the post-run idle boundary, so the buffer fingerprints identically to a CPU-visible variant. This is test-side only — production `io_buffer_gpu` stays GPU-resident. (This resolved the one conditional gap flagged during Demonstrate.)

## The `coast_compare` report

The report is severity-sorted, highest relative delta first:

```
No retained golden matched; closest was <YYYYMMDD>_<sha>.json
(considered 3 retained golden(s))

FAIL    buffer_1/values[3]   golden=0.827   new=0.806   Δ=2.57%   tol=rel=1e-06 abs=1e-09
FAIL    buffer_1/max         golden=0.827   new=0.806   Δ=2.57%   tol=rel=1e-06 abs=1e-09
FAIL    buffer_2/mean        golden=0.131   new=0.129   Δ=1.52%   tol=rel=1e-06 abs=1e-09
```

Three exit codes:

| Code | Meaning |
|------|---------|
| `0` | Matched at least one retained golden within tolerance |
| `1` | No retained golden matched within tolerance (numerical drift) |
| `2` | Structural error — missing field, dtype/shape mismatch, NaN, missing sidecar. The contract changed, not the values. |

A `2` is almost always a sign that the test's blob shape was intentionally changed; the response is to capture and commit a new golden, not to look for a numerical bug. A `1` is the regression signal proper.

`--verbose` prints PASS entries below the FAIL list for diagnosing wide divergence patterns.

## What a fingerprint cannot tell you

The fingerprint is a **localizer**, not a coverage proof. A regression in a dimension the fingerprint does not record will not trip the test. The mitigation is to include enough independent fingerprint axes (per-buffer summary + sample values + sidecar full arrays for buffers that matter) that a single hot path is unlikely to be invisible — but this is a judgement, not a guarantee. The Discover-stage idea of validating fingerprint coverage by injecting synthetic bugs was reframed in IDD §11: fingerprints localize when a known regression has occurred; their job is not to discover unknown regressions in dimensions nobody has fingerprinted.

The honest version: an A-tier failure tells you *something has changed*; an A-tier pass tells you *the fingerprinted dimensions did not change*. The space between those is the test author's responsibility.
