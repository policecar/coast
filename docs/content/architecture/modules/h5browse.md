+++
title = "h5browse"
weight = 67
+++

# h5browse — HDF5 Log-File Browser (domain layer)

The `h5browse` module is the GUI-free domain layer of the **HDF5 Browser**
(Analysis menu): an offline, read-only triage tool that opens one
COAST-written HDF5 log file at a time — observe or buffer-log
family — and answers *"is this run's data healthy or garbage?"*. Files may
be multi-GB and corrupt by design (crashed runs); the module stays
memory-bounded, responsive, and unkillable by its input.

The module contains **zero ImGui includes**. The rendering panel lives in
`gui_tools/h5_browser_window` and consumes this module exclusively through
snapshots and commands.

## Directory Layout

```
include/h5browse/
├── scalar_type.h        # The 10 writer dtypes + compile-time dispatch (HDF5-free)
├── scan_stats.h         # Merge-associative count/NaN/Inf/min/max/mean/M2 accumulator
├── base_histogram.h     # Fixed-bin histogram, power-of-two range doubling, exact rebin/merge
├── row_filter.h         # AND-only structured filter (IR + provenance strings)
├── stats_kernel.h       # Hot per-chunk loops (own TU; extern template over 10 dtypes)
├── h5b_error.h          # error/result<T> value channel + H5Ewalk triage harvesting
├── file_model.h         # Family detection, semantic tree, dataset descriptors
├── chunk_cache.h        # Byte-budgeted LRU row-block cache (512 MB)
├── dataset_reader.h     # Shape-matched executor reads; depth-2 read pipeline
├── pairwise.h           # Pairwise request types + TBB tile kernel
└── scan_engine.h        # Scan thread, two-lane queues, snapshots, log ring

src/h5browse/            # matching implementations
```

## Execution model

All libhdf5 access goes through the process-wide single-threaded
`hdf5::executor` (libhdf5 is built non-threadsafe). The engine owns one
**scan thread** that submits ≥4 MB whole-chunk read jobs and does all
accumulation itself; read jobs interleave fairly with concurrent sim
logging in the executor's FIFO queue.

```
GUI thread  ──commands──▶  scan thread  ──chunk read jobs──▶  hdf5::executor
GUI thread  ◀─snapshots──  (latest_value mailboxes, latest-state-wins)
```

**Two-lane servicing**: analysis operations (open, stats scan, histograms,
pairwise, series, row-index) are strictly serialized on the scan thread and
cancellable per batch; table-view **block fetches** are serviced *between
batches* of any running analysis, so the table stays live during long
scans.

**Errors are values** (`h5b::result<T>`, consistent with the `step_result`
doctrine): every libhdf5 failure is caught inside the executor job and
surfaced as a log-ring entry plus sticky per-dataset state. One rotten
dataset never hides the healthy rest; only file-level failures reject an
open, with `H5Ewalk`-harvested triage text ("file is truncated…", "file is
currently being written…").

## Measured production rules (from the design-verification stage)

1. Read-job memspaces must match the file-selection shape — a flat 1-D
   memspace is **15.9×** slower (`dataset_reader.cpp`).
2. Accumulation must be multi-lane: the 16-lane shifted-sums kernel is
   **8×** faster end-to-end than scalar Welford and numerically equivalent
   (~1e-11); it lives in its own TU, with a manual throughput benchmark
   (`bench_h5b_kernel_throughput`, not a ctest: a throughput floor would make
   a test's verdict depend on the machine's load) because identical source
   measured 5.0 vs 3.5 GB/s in different TU compositions.
3. The per-dataset chunk cache is disabled for uncompressed scan opens
   (27× RSS reduction, +40% speed on many-dataset files) and kept for
   compressed datasets (−8% if disabled).
4. Depth-2 read pipelining (+60% over serial; depth 4 adds nothing).

Benchmark results against the G1–G7 acceptance targets:
`benchmarks/perf_h5browse.md`.

## Key structures

- **`file_model`** — immutable per-open description: family, semantic tree
  (observe `__`-flattened names un-flattened; buffer ids labeled from
  topology), one `dataset_desc` per scannable
  dataset (native dtype, shape, chunk, rows, `step_index_path`).
- **`base_histogram`** — adaptive ranges are always `[0, 2^k)` or
  `[-2^k, 2^k)` with exact bin-pair transitions, so *any* two adaptive
  histograms merge exactly (thread-local pairwise merges rely on this);
  `rebin(m)` re-aggregates base bins — interactive rebin never rescans.
- **`row_filter`** — AND-combined column conditions + half-open row range;
  doubles as an IR and serializes into provenance strings shown in the GUI
  and the log.
- **Snapshots** — `file/stats/histogram/pairwise/row_index/series`
  snapshots published through `latest_value<shared_ptr<const T>>`
  mailboxes (unconditional latest-state-wins; publication throttled during
  scans). Note: `demand_published` is *not* used here — its contract needs
  a producer that keeps re-offering, which an event-driven engine does not.
- **`chunk_cache`** — LRU over `shared_ptr<const row_block>`; an evicted
  block survives the GUI frame drawing it.

## Dtype policy

Data keeps its stored type end-to-end (reads, cache, table formatting,
streaming kernels widen per element to `double` internally). The only
conversion point is the pairwise gather into a dense k×width **f32**
matrix. Conversion artifacts would be indistinguishable from data
corruption in an inspection tool.
