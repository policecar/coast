# COAST lean sweep — sweeper brief (v4)

You inspect one commit range ("the window") of this repository for code that
can be deleted or merged **without changing the behaviour the test suite
defines**. Work read-only in the checkout: read, `git log`/`git diff`/`grep`;
do not edit, build or create files inside it. Probes go outside it. History
after HEAD does not exist for you.

Two phases, kept apart: **search everywhere the window points, report only what
passes the admission test.** "Nothing to do" is a complete, correct answer when
the search is done and nothing passed.

## Phase 1 — locate (git)

1. `git log --stat <window>`. Rank files by (a) lines added and (b) number of
   commits touching them. The hotspot list is the top five by those numbers
   **plus every non-test source file that a `fix`, `harden`, `review` or
   `follow-up` commit in the window touched, however small the change**: a
   six-line fix to a file is the strongest signal that the file carries
   residue (a dead copy that had to be patched too, a check duplicated from
   elsewhere). List all of them with their numbers.
2. For each hotspot, read the *current* file(s), not only the diff. Residue
   from fix rounds lives in the file as a whole.

## Phase 2 — search (do all of it, for every hotspot)

Answer each question per hotspot. "None" is an answer only after you looked.
When a question yields a set (every function the window added, every test case
in a file), list all members and answer per member.

1. **Dead.** For every function, member, enum variant, message/action type,
   config key or file the window added or touched: grep for uses outside its
   own definition. Zero uses → candidate. A value written and never read →
   candidate. Then the reverse direction, which needs the window's *deletions*:
   `git diff <window> | grep '^-'` for every output the window stopped
   producing — a dataset or attribute name, a struct field, a file, a log key —
   and grep the whole tree for code that still **reads** it. A reader left
   behind is dead even though the window never touched its file.
2. **Copies.** The same sequence of statements at two or more sites; comments
   saying "mirrors", "must match", "same as in", "keep in sync" → candidate
   for one shared helper. Name every site.
3. **Reimplementations.** Code in the hotspot that does what a helper already
   in the tree does (spawn + timeout + reap, write-all on a socket, a hash, a
   JSON field list, a validation walk). Grep the tree for the helper first.
4. **Wrappers.** A function whose body is one call with the same arguments →
   candidate.
5. **Comments vs code.** A comment block longer than the code under it, a
   rationale stated twice, a comment describing code the window changed.
   Comment-only findings are **not rows**: list them one line each under
   "comments" after the table. They do not count in the verdict.
6. **Tests.** For every test case in the hotspot's test files:
   - *Can it fail?* Name one wrong implementation that would fail it. If
     every value it checks holds by construction (`max >= mean`, `count > 0`
     after inserting, a property the test's own comment says cannot be
     violated), it is tautological → candidate for deletion.
   - *Does it test what its name says?* Follow the assertion back to the
     value it reads: is that the object the test names, or whatever a loop
     left last / a fixture defaulted to?
   - *Is its setup checked?* A `[[nodiscard]]` result or a fixture step whose
     failure would make the test vacuous and that nothing asserts → candidate.
   - *Merge.* Two cases that drive the same path with the same oracle and
     differ only in a constant → candidate for one parametrised case.

## Phase 3 — admission

A row is admitted only if every column can be filled honestly:

- **what**: file:line, and what is deleted or merged into what.
- **−/+**: lines removed / lines added. A row must be **net ≤ −5**, or remove a
  copy (two sites → one) at net ≤ 0. A row that adds lines is never a row.
  Smaller net-negative items (an unused include, a one-line implied CHECK, a
  −3 dead member) are real but go **one line each under "small items"**, not
  in the table and not in the count.
- **covered by**: the test(s) that pin the behaviour that remains, or "none
  pins it" (then say what would show a mistake).
- **check**: the command that proves nothing changed (which ctest label or
  binary comparison).

**Not a row:** removing a feature, a config knob or a validation the user can
reach (that is an owner's decision — one line under "owner's call"); renames;
formatting; "could be simpler" without a concrete deletion.

## Report

First line: `nothing to do` or `<n> reductions, about −<lines> net` — counting
code and test rows only, never comment items.

| # | hotspot | file:line | what | −/+ | covered by | check | confidence |

Confidence: `read` / `ran`. Then one line per small item, one line per
comment item, one line per owner's-call item, one line per Phase-2 question
you did not finish. Under 500 words plus the table. Length is
a cost.
