# How to write a new test

The test tree has three tiers. Pick the one that matches what you're testing.

| Tier | Where it lives | When to use |
|------|----------------|-------------|
| **Unit** | `tests/unit/<module>/` | A pure function or a small stateful class you can exercise directly (cli::parse, spsc_queue, json_loader, etc.) |
| **A-tier integration** | `tests/integration/<name>/` | A full simulation run; fingerprint observable state; compare against retained goldens. The regression tripwire. |
| **B-tier integration** | `tests/integration_b/` | The user-facing `coast` binary's CLI surface, exit codes, file side-effects |

All tests use **doctest** (vendored at `tests/third_party/doctest/doctest.h`).

---

## Adding a unit test

1. Pick a subdirectory under `tests/unit/` matching the module you're testing. If it doesn't exist, create one and add `add_subdirectory(<name>)` to `tests/unit/CMakeLists.txt`.

2. In `tests/unit/<module>/CMakeLists.txt`:

   ```cmake
   add_executable(test_<name> test_<name>.cpp)
   target_link_libraries(test_<name> PRIVATE doctest::doctest coast_lib)
   add_test(NAME unit.<module>.<name> COMMAND test_<name>)
   ```

3. The test file:

   ```cpp
   #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
   #include "doctest.h"

   #include "the_header_under_test.h"

   TEST_CASE("descriptive name: a specific behavior") {
       CHECK(condition);
       CHECK_THROWS_AS(expr, std::runtime_error);
   }
   ```

4. Reconfigure and run:

   ```bash
   cmake --build build-tests -j
   cd build-tests && ctest -R 'unit\.<module>\.<name>' --output-on-failure
   ```

**Need a scratch directory?** Use the pattern in `tests/unit/test_support/test_blob_writer.cpp` — a small RAII struct that creates a unique dir under `$TMPDIR` and wipes it on destruction.

**Need a fixture file?** Drop it under `tests/unit/<module>/fixtures/` and pass the directory via `target_compile_definitions(test_<name> PRIVATE COAST_<NAME>_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")`. See `tests/unit/json_cfg/` for an example.

---

## Adding an A-tier integration test (full sim + fingerprint)

The harness is shaped so adding a test is a CMake-line + a 10-line file.

1. In `tests/integration/CMakeLists.txt`, register the test:

   ```cmake
   coast_add_integration_test(
       NAME        <name>
       CONFIG      configs/<your_config>.json
       STEPS       100000
       SEED_OFFSET 0
   )
   ```

2. Create `tests/integration/<name>/main.cpp`:

   ```cpp
   #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
   #include "doctest.h"

   #include <filesystem>

   #include "integration_helpers.h"

   TEST_CASE(COAST_TEST_NAME) {
       coast::test::run_io_buffer_fingerprint_test(
           COAST_TEST_CONFIG,
           COAST_TEST_SEED_OFFSET,
           COAST_TEST_STEPS,
           std::filesystem::path{ COAST_TEST_BLOB_PATH },
           std::filesystem::path{ COAST_TEST_GOLDENS_DIR });
   }
   ```

3. Create the goldens directory: `mkdir -p tests/integration/<name>/goldens && touch tests/integration/<name>/goldens/.gitkeep`.

4. Capture the initial baseline:

   ```bash
   cmake --build build-tests -j
   cmake --build build-tests --target coast_update_goldens_<name>
   ```

5. Inspect `tests/integration/<name>/goldens/<date>_<sha>.json` (and any `.bin` sidecars) to sanity-check the captured state, then commit them.

6. Re-run the test to confirm it matches its own freshly-captured golden:

   ```bash
   cd build-tests && ctest -R 'integration\.<name>' --output-on-failure
   ```

### Need a fingerprint other than io_buffer state?

`run_io_buffer_fingerprint_test()` is one shape. For something else, write the body inline in `main.cpp` and call `coast::test::open_blob()` / `blob_writer::scalar` / `blob_writer::array` directly (see `integration_helpers.cpp` for the canonical pattern). Then `coast::test::compare()` against your goldens dir as usual.

### Per-test tolerance

When a real rerun trips the default `rel=1e-6 abs_floor=1e-9` (most likely on a different CPU/compiler), add `tests/integration/<name>/tolerances.json`:

```json
{"default": {"rel": 1e-6, "abs_floor": 1e-9},
 "overrides": {"buffer_2/spike_counts": {"rel": 1e-4}}}
```

Pass its path as the third argument to `coast::test::compare()`.

---

## Adding a B-tier subprocess test

B-tier shares **one executable** (`test_b_tier`) hosting all cases. Add a new `TEST_CASE` to `tests/integration_b/test_b_tier.cpp`:

```cpp
TEST_CASE("b_tier.<name>: <what it checks>") {
    scratch_dir sd{"<name>"};
    auto r = coast::test::run(COAST_BIN, {
        "--headless", "-n", "500",
        "--<your-flag>", (sd.path / "thing.out").string(),
        "configs/mnist.json"
    });
    CHECK(r.exit_code == 0);
    CHECK(contains(r.stdout_text, "expected message"));
    CHECK(std::filesystem::file_size(sd.path / "thing.out") > 0u);
}
```

- `coast::test::run(...)` is defined in `tests/support/subprocess.h` (POSIX fork+exec+poll). Returns `{exit_code, stdout_text, stderr_text}`.
- `scratch_dir` and `contains` helpers are at the top of `test_b_tier.cpp`.
- **Never write into the source tree**: always anchor side-effect paths to `sd.path`. If you need to override a config field (e.g. `snapshot_dir`), copy the config to scratch and mutate it (see `checkpoint_creates_files` for the pattern).

No CMake changes needed — the test compiles into the existing target.

---

## Style + reusable plumbing

- **Naming**: `TEST_CASE` strings follow `<tier>.<subject>: <behavior>`, e.g. `"unit.cli.parse: missing value throws parse_error"`. Keep the prefix consistent with the ctest target name.
- **One assertion topic per `TEST_CASE`**. If you want to share setup, use doctest `SUBCASE`.
- **CHECK vs REQUIRE**: `REQUIRE` aborts the test on failure (use for preconditions whose violation makes downstream checks meaningless: "file exists", "exit code 0"). `CHECK` lets the test continue (use for value comparisons).
- **Approximate floats**: `doctest::Approx(x)` for default-tolerance comparisons.
- **`unistd.h`** is fine to include for `::getpid()` etc. — both Linux and macOS support it.

## Decision tree

> "I added a new public function in `sim_core/`."
- Pure / no IO → **unit test**.
- Uses HDF5 or the threading controller → consider whether the **A-tier** fingerprint already covers it indirectly. If yes, do nothing. If no, write a focused integration helper.

> "I added a new CLI flag."
- Always a **B-tier** test for argv parsing + exit code + observable effect.
- If the flag changes simulation behavior, also adjust an A-tier fingerprint.

> "I changed simulation math."
- The existing A-tier mnist tests should trip if the change is real. Re-baseline only after confirming the change is intended.
- If the new code path isn't exercised by mnist, add a new A-tier test against a config that does exercise it.
