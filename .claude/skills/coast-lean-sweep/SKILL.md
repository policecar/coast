---
name: coast-lean-sweep
description: Periodic leanness sweep of COAST — inspects recent git history for expansion hotspots (fix rounds, review follow-ups) and proposes deletions or merges that keep test-defined behaviour; when tests are the target, flags tautological tests, tests that do not test what they claim, unchecked setup, and cases that could be merged. Use when asked to make the code leaner, to find dead or duplicated code, or to run a periodic cleanup pass over a commit range or time span.
argument-hint: [commit range or --since=<date>; default: last 30 days]
---

# COAST lean sweep

**Window:** $ARGUMENTS (default `--since=30.days`, first-parent history of the current branch)

A sweep has one deliverable: a short list of **reductions** — concrete deletions
or merges, each net negative, each with the test that pins the behaviour that
stays — or the line `nothing to do`. Both are complete results. A sweep that
manufactures findings to fill a table is the failure this skill exists to
prevent; so is a sweep that reads "be lean" as "do not look".

## 1. Scope (orchestrator)

- Resolve the window to `base..head` and state it. Everything after `head` is
  out of scope for the sweepers.
- Do not pass your own hotspot guesses or conclusions to the sweepers.
- **Split large windows.** Above ~30 commits or ~5k added lines one sweeper
  covers part of the surface and says so under "not finished" (tuning round
  2: a 97-commit window left core.cpp and builder.cpp unswept). Partition by
  directory (e.g. `coastd/src`, `coastd/include`, `tests/unit/coastd`) and run
  one sweeper per part with the same window; merge the tables.

## 2. Sweeper brief (when delegating)

Give each sweeper `sweeper.md` **verbatim**, the window, and a read-only
checkout. Two sweepers at most, for independent agreement; items both reach
are usually real. Different models count as independent.

## 3. Merging and verifying (orchestrator)

- Keep a row only if it passes `sweeper.md` Phase 3 as written: net negative,
  behaviour pinned by a named test or explicitly unpinned, a check command.
- **Feature, knob or validation removal is never a reduction row.** List it
  under "owner's call", one line each, and stop there.
- Before proposing a row as done, apply it in a scratch build and run the
  check it names, with the build options stated (CLAUDE.md, "Totals depend on
  the build options"). A row whose check fails is dropped, not weakened.
- For a test row (tautology, mis-targeted assertion, unchecked setup), state
  the wrong implementation that the surviving tests still catch, or that the
  deleted assertion never caught anything.

## 4. Result

First line: `nothing to do` or `<n> reductions, about −<lines> net`. Then the
table from `sweeper.md`, then owner's-call lines, then unfinished questions.
Target under 500 words plus the table.

## 5. Version stamp

Start the first line of everything posted from this skill (issue, PR, comment)
with the version, after any prefix your own instructions require, e.g.
`jk's agent (coast-lean-sweep v4):`. The next tuning pass finds live uses with
`gh search` on that stamp. Bump the version whenever this file or `sweeper.md`
changes what a sweeper does.

*Origin:* main grew 4k → 116k lines in ten months at 4.7 lines added per line
deleted; the coastd fix rounds alone left four copies of one comparator and a
reimplemented subprocess runner that two later commits had to remove by hand.
**Version:** v4 (`sweeper.md` v4), tuned with `skill-evolve` over four rounds
and 46 agent runs against eight historical cleanups replayed at the commit
before each cleanup.
Known limits: a single sweeper on a 90-commit window covers part of the surface
and its recall varies by several items between runs (hence the split rule);
residue outside the window's own expansion is out of reach by design; sonnet
finds the large copies but not the cross-module dead reader — use opus for the
sweep, sonnet at most as the second opinion; sweepers build nothing, so the
scratch-build check in §3 is the only execution step.

Before changing this skill, read its tuning log: `coast-lean-sweep/LOG.md` on
the orphan branch `skills-lab` (`git worktree add ../cimc-coast-skills-lab
skills-lab`). It also lists verified reductions the history never made
(candidates for a live sweep).
