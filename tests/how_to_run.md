# How to run the tests

## Build

Tests are off by default. Use a separate build dir.

```bash
cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-tests -j
```

## Run

```bash
cd build-tests && ctest --output-on-failure
```

The full suite is ~70 s (the three A-tier mnist tests dominate at ~22 s each). Everything else combined is under 100 ms.

### Useful filters

```bash
ctest -R 'unit\.'              # just unit tests (sub-second)
ctest -R 'integration\.'       # just A-tier (in-process sim runs)
ctest -R 'integration_b'       # just B-tier (subprocess CLI tests)
ctest -E 'integration\.'       # everything except A-tier (good under sanitizers)
ctest -R 'unit.cli.parse'      # one specific test
ctest -V                       # verbose: per-case output
ctest -j8                      # parallel (each ctest target runs serially internally)
```

### Working directory

`ctest` runs every test with `WORKING_DIRECTORY=<repo root>` so `configs/mnist.json` and the `./data/...` paths in those configs resolve. **Running a test binary directly will likely fail** because the child process's CWD is wherever you invoked it from — `cd <repo root>` before running them by hand.

## Capture or refresh goldens

After an intentional change that legitimately shifts simulation output:

```bash
cmake --build build-tests --target coast_update_goldens               # all A-tier tests
cmake --build build-tests --target coast_update_goldens_mnist_seed0_baseline   # just one
git add tests/integration/<test>/goldens/                             # commit the new baseline
```

Goldens rotate at N=5: the oldest is dropped automatically.

## Reading a `coast_compare` failure

Failure report format:

```
No retained golden matched; closest was 20260522_4ce7adf.json
(considered 1 retained golden(s))

FAIL    buffer_1/values[3]   golden=0.827   new=0.806   Δ=2.57%   tol=rel=1e-06 abs=1e-09
FAIL    buffer_1/max         golden=0.827   new=0.806   Δ=2.57%   tol=rel=1e-06 abs=1e-09
```

- `STRUCT  <path>  <reason>` — structural mismatch (missing field, dtype/shape mismatch, NaN, missing sidecar). Almost always means the contract changed; re-baseline.
- `FAIL    <path>  golden=X  new=Y  Δ=%  tol=…` — numerical drift past tolerance. Highest Δ first.
- Exit code: `0` = matched, `1` = no tolerance match, `2` = structural error.

To loosen tolerance for a specific field, drop a `tests/integration/<test>/tolerances.json`:

```json
{"default": {"rel": 1e-6, "abs_floor": 1e-9},
 "overrides": {"buffer_2/spike_counts": {"rel": 1e-4}}}
```

Add overrides **only after a real rerun trips the default**, not pre-emptively.

## Sanitizers

See [`SANITIZERS.md`](../SANITIZERS.md) at the repo root for full details. TL;DR:

```bash
cmake -B build-asan  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCOAST_SANITIZER=asan
cmake --build build-asan -j
cd build-asan && ASAN_OPTIONS="detect_leaks=1" ctest --output-on-failure -E 'integration\.'
```

**Important:** sanitizer builds use `Release`, not `RelWithDebInfo`, and skip A-tier tests (they trip on FP drift from instrumentation, not real bugs).

## Static analysis (clang-tidy)

```bash
cmake --build build --target coast_tidy        # reports only; does not gate the build
```

Baseline at the time of writing: 66 warnings, captured in `.clang-tidy-baseline.txt`.
