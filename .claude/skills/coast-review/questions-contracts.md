# COAST review questions: contracts & callers

Read `questions-common.md` first: it says how to answer and how to report.

## 1. Adversarial questions

**Callers and producers**
- [ ] For every check: who produces the value that reaches it? Can a caller
      change, merge or drop the value before it gets there? (PR #154:
      `init_io` checked "exactly one input", but `sim_builder`, which keys
      input maps by channel name and composes node-set deltas and offsets, had
      already merged inputs that shared a name.)
- [ ] Which existing caller is newly broken by a stricter check (configs, GUI
      flows, node sets/deltas, tests, CLI offsets)?

**Reused code**
- [ ] For every reused helper: does its documented contract fit this caller?
      (PR #154: `read_rng` is lenient for formats that predate the rng group,
      which is wrong for a format that always had it.)
- [ ] **"Same pattern as sibling X" is not evidence.** If the sibling has the
      defect, report it as an inherited defect, and open an issue if it is out
      of scope.

**Threads**
- [ ] Which thread runs this code, and which thread owns every piece of state
      it touches (hook registration, probe producer/consumer, `io_buffer`
      swap)? Sim thread and GUI thread share state only through the paths
      CLAUDE.md names ("Threading Model").

## 2. COAST-specific checks

- [ ] **Checkpoints** (`include/logging/state_checks.h`): a field the format
      always had and that is missing or wrong-sized is a corrupt checkpoint
      (throw). Only fields added after checkpoints existed may be missing
      (keep the constructed value). Is the `config_record` complete? Is it
      compared by value?
- [ ] **Seeds**: do the value the RNG is seeded with, `effective_seed()`,
      `node_seeds` in the manifest and the checkpoint record agree for every
      reachable seed + offset (including negative offsets and overflow)?
- [ ] **Failure halts**: a failed load or check shows in the system's state
      and stops the sim (sticky halt); it doesn't let it keep stepping.
- [ ] **Structural validation**: the full shape (input widths included) is
      checked before anything is assigned, and the same way in every model. A
      claim like "every mismatch is refused" holds in all of them.
- [ ] **Explicit context**: new state is passed explicitly, not through
      `thread_local`, globals or a pointer set up on the side.
- [ ] **Hot path**: does the change touch per-step code (`process()`,
      dendrite loops, GPU kernels)? Measure steps/s before and after,
      interleaved, on a config that runs the changed path (`configs/mnist.json`
      is ngm2 CPU only). Report the numbers. Flag a finding only if speed drops
      significantly beyond run-to-run noise, or in a way the change doesn't
      explain.
