# COAST review questions: tests & parameter space

Read `questions-common.md` first: it says how to answer and how to report.

## 1. Adversarial questions

**Parameter space**
- [ ] Which values pass `validate()` (or the loader) but give inverted,
      degenerate, constant, NaN/inf or useless output? Look at boundaries,
      near-equal pairs, extremes, tiny and huge sizes.
- [ ] Where is a value checked in one type and used in another (`double` →
      `float`, `int` → `uint32`/`uint64`, `int` arithmetic on sizes or seeds)?
      Sign extension, overflow, inf after narrowing?
- [ ] Do the model's **statistics** (not its trajectory) change significantly
      across backends, compilers or types? Then the **model** is brittle;
      report it as such, don't engineer the test around it.

**Tests**
- [ ] For each new or changed test, name a plausible wrong implementation that
      still passes. If one exists, the test is incomplete.
- [ ] Can the test pass vacuously (empty loop, trivially true comparison, guard
      that guards nothing, parameter value where the term under test vanishes,
      e.g. k = 0 or 1)?
- [ ] Exception safety in tests: captured references, global sinks, state
      restored on a failing `REQUIRE`? Cases parameterised through mutable
      globals (forcing them to run in sequence) need a stated reason.
- [ ] Does a test depend on ulp-level agreement across CPU/GPU, compilers or
      types? Then the **test** is brittle: compare within a stated tolerance,
      or compare statistics instead of trajectories.

**Claims** (PR description, commit messages, write-ups)
- [ ] Which claim ("tests pass", "byte-identical", "no behaviour change", a
      speed-up) is not backed by output from the build being merged, with its
      test set and build options stated and every exclusion listed?
      (`GPU_BACKEND=none` reports a green 106/106 without compiling the GPU
      tests; see CLAUDE.md.)
- [ ] Does a conclusion go beyond what was measured (one run, one seed, one
      config, platform, compiler or backend), or use a metric that doesn't
      measure the property claimed?

## 2. COAST-specific checks

- [ ] **`validate()` vs config editor**: can the editor produce a value that
      `validate()` refuses, or clamp a value that `validate()` accepts?
- [ ] **GUI/headless parity**: both paths accept and refuse the same configs
      and files.
- [ ] **Determinism and goldens**: does the change alter simulation output? If
      so, are goldens re-baselined intentionally, never as a side effect?
- [ ] **Docs** (`docs/content/`): updated for every changed module, data flow
      or config key; `hugo` builds without warnings.
