# Auditor brief — tests (coast-audit)

You audit one area of COAST's **tests** and draft GitHub issues for defective tests.
The deliverable is a list of issue drafts, or the line `nothing to file`. Both are
complete results. Drafting an issue for a test that is fine is the failure this brief
exists to prevent. So is skimming.

A test is defective when either:
- it **passes on a broken implementation**; or
- it **fails or flakes on a correct one**.

Read CLAUDE.md first (test tiers, goldens, ctest usage).

## Phase 1 — Search (always run every step, on every file of the area)

1. **Inventory.** List every test case (`TEST_CASE` / `SUBCASE`) with its line, and the claim it makes, from its name and comment.
2. **Can it fail?**
   - First, the case's own target. If the name or comment says what the case guards (a bug, a fix, an edge case), re-introduce exactly that failure: revert the fix, or inject the value it is about. Then check whether the case fails. Do this before naming any other wrong implementation.
   - Then, for every assertion, name one plausible wrong implementation it would catch: an off-by-one, a swapped argument, a missing call, a wrong sign. If you cannot name one, find out why. For example, it is true by construction, compares a value with itself, or the input makes the checked quantity degenerate (`pow(1, x) == 1`, an empty set, a single element).
3. **Does it test the claim?** For each case, trace the call path from the test to the code its name or comment claims to verify. Flag a case that reaches that code only through a helper, a hand-written fixture, a direct call that bypasses the claimed path, or the reader side of a writer/reader pair.
4. **Setup checked?** For every setup call (`[[nodiscard]]` values, status codes, file writes, subprocesses, `git`), check that a failure stops the test with a clear message. Otherwise a failed setup passes the test or turns into a confusing later failure.
5. **Environment.** List everything the test depends on outside itself:
   - fixed ports, fixed paths, `/tmp` names, globs over repository directories (which see untracked files);
   - wall-clock timeouts and sleeps;
   - test order, unseeded randomness, toolchain-specific floating point.

   For each, write down what happens under: two ctest runs at once; a loaded machine; a working tree with extra or stale untracked files.
6. **Guards through production.** List every guard in the code under test: a check that halts, rejects or throws. For each, find a test that triggers it through the entry point that runs it in production (e.g. `process()`, `load_state`, the CLI). A test that calls the checker directly does not count. Then check that a positive case shows the guard does not fire on valid input.
7. **Copies.** Find near-identical cases or fixtures. Check whether the copies still agree.

Write down what each step found, including "nothing", in a working list.

## Phase 2 — Admit (apply to every item of the working list)

Keep an item only if all of these hold:
- **Verified:** you read the test and the code it exercises, and can cite file:line. A guess is not admitted. If you can run the test outside the checkout, do it.
- **Failure scenario:** either a concrete wrong implementation that still passes, or a concrete correct setup that fails.
- **Reach** is stated, as one of: every run | concurrent or loaded runs | a dirty working tree | only a hypothetical change nobody would make.

Label each admitted item:
- `bug`: a defective test (see the definition at the top) whose reach is not "only a hypothetical change".
- `cleanup required`: structural test debt with a named consequence. For example, duplicated fixtures that have drifted, or a test that cannot tell which of several things failed.
- Everything else (style, redundant asserts that still can fail, naming) goes into **one** issue labelled `check cleanup`.

Merge items with one root cause into one issue (list every site).

## Phase 3 — Report

Before writing, re-open every file:line you will cite (`sed -n '<N>p' <file>` in the checkout), and confirm the line shows what your sentence says. Fix the reference, or drop the sentence. Number lines one file at a time (`sed -n` or `cat -n <one file>`): `cat -n` over several files numbers them continuously, and its numbers are wrong for every file after the first.

First line: `nothing to file`, or `<b> bug, <c> cleanup required, <k> check cleanup`.

Then, for each issue:

```
### <title: the defect, not the fix>
Label: <bug | cleanup required | check cleanup>
Reach: <every run | concurrent/loaded runs | dirty working tree | hypothetical only>

## What happens
<the test and the code path, with file:line; the wrong implementation that passes or the correct setup that fails>

## Fix
<smallest fix; size S/M/L>
```

Then a section `Not examined`, listing what you did not read or could not settle.
