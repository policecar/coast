# Auditor brief (coast-audit)

You audit one area of COAST and draft GitHub issues for it. The deliverable is
a list of issue drafts, or the line `nothing to file`. Both are complete results.
Drafting an issue for something that is not there is the failure this brief exists
to prevent. So is reading "be careful" as "do not look".

Read CLAUDE.md first (threading model, error channel, run-length rules, style).

## Phase 1 — Search (always run every step, on every file of the area)

1. **Inventory.** List the area's files with line counts and one line saying what each does.
2. **Lifetime and threads.** For every thread, hook, callback, cache or registration in the area, write down:
   - who creates it, who removes it, and what can still run or still point at it after removal or destruction;
   - member declaration order versus the moment a thread or callback starts;
   - `static`/process-lifetime state keyed by pointers that can die.
3. **Error channel.** For every function on the per-step path (`process()`, `swap_io()`, hooks), list each operation that can throw, and the frame that catches it (or "none"). Also list every ignored return value: I/O, HDF5, `fread`, `[[nodiscard]]`.
4. **Untrusted input.** For every value that comes from a file, a config or the CLI, write down what happens when it is:
   - too short or truncated;
   - too large;
   - of the wrong shape or size;
   - negative, or out of range for its type (JSON numbers wrap silently into narrow integers);
   - NaN or inf;
   - duplicated.

   Check sizes read from a file before they size a buffer, index an array or bound a loop.
5. **Load / restore.** For every function that restores state:
   - is everything validated before anything is assigned?
   - What happens on a missing field, and on a field of the wrong size?
6. **Copies.** Find logic that exists twice or more (CPU/GPU, per dataset, per model, per hook point). For each copy, check whether the copies still agree: defaults, ranges, edge cases.
7. **Claims.** For each comment that states behaviour ("never", "always", "independent", "mirrors X", "called from Y"), check it against the code.
8. **Randomness.** List every random generator in the area: where it is seeded, and from which value. Flag every pair of generators fed the same seed value, and every comment calling generators independent that this list contradicts.
9. **Dead code.** For each function, member or parameter that looks unused, grep the whole checkout (`src include tests tools coastd`) before calling it unused.

Write down what each step found, including "nothing", in a working list.

## Phase 2 — Admit (apply to every item of the working list)

Keep an item only if all of these hold:
- **Verified:** you read the code that shows it, and can cite file:line. A guess ("likely", "might") is not admitted. Settle it by reading further, or drop it.
- **Failure scenario:** concrete input or sequence → wrong result, crash or UB. For a cleanup: the concrete maintenance failure (which future change goes wrong, and where).
- **Reach** is stated, as one of: normal use | a hand-written config | a crafted or corrupt file | only by misuse of an internal API.
  - "Hand-written config" means a value a user would plausibly write: a typo, a sign error, an option combination.
  - A value reachable only by absurd magnitudes (e.g. more than 2^31 steps) counts as misuse only.
  - This plausibility bound applies to configs only. A crafted or corrupt file may hold any value, so file input gets no plausibility discount.

Label each admitted item:
- `bug`: the code can produce a wrong result, a crash or UB through a reach other than "misuse of an internal API". Silently accepting corrupt or mismatched input and carrying on as if it were valid (no error, no warning) is a wrong result.
- `cleanup required`: structural (duplication that has drifted or will, an unenforced boundary, a function doing several jobs), with a named maintenance consequence.
- Everything else (dead code, stale comments, style, misuse-only hazards) goes into **one** issue labelled `check cleanup`.

Merge items with one root cause into one issue (list every site). Do not split one defect across issues.

## Phase 3 — Report

Before writing, re-open every file:line you will cite (`sed -n '<N>p' <file>` in the checkout), and confirm the line shows what your sentence says. Fix the reference, or drop the sentence.

First line: `nothing to file`, or `<b> bug, <c> cleanup required, <k> check cleanup`.

Then, for each issue:

```
### <title: the defect, not the fix>
Label: <bug | cleanup required | check cleanup>
Reach: <normal use | hand-written config | crafted/corrupt file | misuse only>

## What happens
<mechanism with file:line, then the concrete failure scenario>

## Fix
<smallest fix; size S/M/L>
```

Then a section `Not examined`, listing what you did not read or could not settle.
