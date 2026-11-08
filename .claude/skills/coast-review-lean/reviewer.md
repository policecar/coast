# COAST lean review — reviewer brief (v7)

You review one change range. Work **read-only** in the checkout you were given:
read, `git diff`/`git log`/`grep`; do not edit, build, or create files inside it.
You may compile a small probe program *outside* the checkout if it settles a
question (`ran` beats `plausible`).

Two phases, kept apart: **search wide, report narrow.**

## Phase 1 — search (do all of it, every time)

Expect to find defects; competent changes still have them. Answer each question
for the code the range adds or changes. "None found" is an answer only after
you looked. When a question yields a *set* of instances (four enum tables,
five callers, three load paths, every `switch` over a type), list all of them
and answer per instance; the defect is usually in the one you would skip.

1. **Checks and their producers.** For every check the range adds (a size, a
   count, "exactly one", a range, a type), find the code that *produces* the
   checked value. Does the producer merge, collapse, default or coerce before
   the check sees it? (A map keyed by name collapses duplicate names; an enum
   table with a `null` row maps `null` to a sentinel.)
2. **Accepted but degenerate.** For every parameter `validate()` or a loader
   accepts: which accepted values give a wrong, silent, NaN, inverted, empty or
   all-zero result? Two sub-steps, both required:
   - *single values*: 0, tiny (1e-200), huge, negative, the boundary of a cast;
   - *relations*: read the formula or algorithm and write down every relation
     between two parameters it silently assumes (`a < b`, `a != b`, `a + b <= 1`,
     `width*height == size`). For each, ask what the output is when the relation
     is exactly met (`a == b`) and when it is inverted (`a > b`). Does
     `validate()` enforce the relation? If not, that is a candidate row.
   Anything `validate()` accepts is **reachable**: a hand-written JSON config is
   this system's primary user interface, not an exotic input.
3. **Recorded vs used.** Every value the range records (seed, config record,
   manifest, log message): is it the same value the code actually uses?
4. **Persistence.** For every save/load the range touches: a missing, short or
   extra group or field — does the load throw, warn, or silently continue with
   defaults? Does a resumed run equal an uninterrupted one?
5. **New throws.** For every exception the range newly raises: who catches it,
   and on which thread? Then the step reviewers skip: **walk every stack frame
   between the throw site and that catch** and list each resource those frames
   acquire by hand rather than by RAII (`hid_t` from `H5*open/create`, `FILE*`,
   `new`, a lock, a partially applied state). Each one is leaked or left
   inconsistent by the new throw. "A catch exists" is not the answer to this
   question.
6. **Two paths, one contract.** Where two code paths claim to accept the same
   input (startup vs GUI load, headless vs GUI, CPU vs GPU): does the range
   make them disagree?
7. **Description vs range.** Does the range contain what the description
   claims (behaviour, tests, numbers)? A claim with no code behind it is a
   finding.
8. **Tests.** For each test the range adds: name one wrong implementation that
   would still pass it. For every numeric tolerance, compute what it actually
   permits at the *smallest* expected magnitude in the data (doctest's
   `Approx(x).epsilon(e)` allows `e·(1+|x|)`, so for x ≈ 1e-5 it allows ~e,
   not e·x). A tolerance that lets a kernel write zeros pass is a row. Report
   test gaps only if the wrong implementation matters for a shipped config.
9. **Subtraction.** What in the range can be deleted or merged without changing
   tested behaviour? Dead code the range touches, duplicated sequences, a
   wrapper around nothing.
10. **Toolchains.** The project builds with gcc/libstdc++, Homebrew clang and
   AppleClang/libc++ (CLAUDE.md). For each new construct that compilers treat
   differently, say which: `T(expr)` as a macro argument (clang reads
   `T(std::numeric_limits<float>::infinity())` as a declaration — most vexing
   parse; braces fix it), a POSIX macro applied to an rvalue (`WIFEXITED(f())`
   takes an address on Apple's libc), `<version>`/`std::format`/ranges
   availability, `-Werror` differences. A construct that fails to compile on
   one supported toolchain is a row with guarantee "build break".

## Phase 2 — report (narrow)

**Verdict, first line:** `merge as is` / `merge after <n> fixes` / `do not merge`.
- `do not merge`: a `fix here` row with guarantee wrong result, crash/UB or
  corruption that a shipped config, a GUI action or a documented workflow
  reaches.
- `merge after <n>`: every other actionable row, **including `owner's call`
  rows whose guarantee is crash/UB or wrong result and whose fix is ≤ 10
  lines**. The owner decides, but the verdict names the decision. A PR with a
  known one-line-fix UB path is not "merge as is".
- `merge as is`: Phase 1 done and no actionable row survives. Say so plainly;
  do not pad.

A row is **actionable** (`fix here` / `owner's call`) only if every column can
be filled honestly:

- **guarantee** broken: wrong sim result / crash or UB / data loss or corrupt
  file / silent divergence between paths that claim to agree / false claim in
  the description or docs. "Inconsistent with module X" is not one.
- **reach**: name the concrete thing that gets there: a value `validate()` or
  the loader accepts (a hand-written JSON config **is** reach), a GUI control,
  a shipped config, a CLI flag, a file this repo's own writer produces, a test
  fixture. Not reach: a file only a foreign tool or an attacker produces, a
  caller that does not exist yet → `noted, no action`.
- **fix est.**: lines added/removed and what the fix deletes. A fix larger than
  the defect's reach deserves is `owner's call`, never a silent `fix here`.
  **Exception that overrides weak reach:** a fix that removes at least as many
  lines as it adds (dead code, a manual close replaced by an existing RAII
  guard, a duplicated sequence folded into an existing helper) is `fix here`
  even if the defect is reached only by a malformed file. Shrinking code is
  never growth; the reach test exists to stop growth.
- **test?**: yes if a test could fail before and pass after the fix.
  Defensive code for states nothing produces is "no" → `noted, no action`.
- **origin**: the range introduced the defect or made it worse (more
  reachable, harder to detect). Check at the base commit. A pre-existing
  defect whose reach the range does not change is `out of range`, even in
  lines the range touched, unless its fix removes at least as many lines as
  it adds.

Confidence per row: `read` / `ran` / `plausible`. Rank actionable rows by
guarantee severity, then one line each for `noted, no action`, then one line
per Phase-1 question you could not finish, then `out of range` items as short
issue drafts (title, evidence, reach) for the owner to approve. A finding that
matches an open issue from the list you were given is a reference (`see #N`,
plus anything new), not a new row or draft — unless the range introduced it
or made it worse (**origin** above): then it is a normal row with `see #N`
in the defect column. Do **not** report style, naming, comment wording,
precedent, or hypothetical future callers.

End with the **subtraction line** (Phase 1, item 9): one concrete deletion, or
"looked, found none".

Target: under 600 words plus the table. Length is a cost.

## Table

| # | file:line | defect | guarantee | reach | fix est. | test? | conf. | disposition |

Dispositions: `fix here` / `owner's call` / `noted, no action` / `out of range`.
