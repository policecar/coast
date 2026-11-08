---
name: coast-review-lean
description: Lean review of a COAST change (PR, branch, commit range). Answers one question first — merge as is, or not — and admits a finding only with a reachable failure and a stated fix cost. Use when a review is requested and the goal is a correct, small codebase rather than a long findings list. Also triages open issues (real vs. theoretical, urgency) with the same admission test, see §8. Alternative to coast-review; both may be run on the same range.
argument-hint: [PR number, branch, commit range, or issue numbers]
---

# COAST lean review

**Target:** $ARGUMENTS

A review has one deliverable: a **verdict** — `merge as is`, `merge after <n> fixes`,
or `do not merge` — and the smallest list of findings that justifies it.
"Merge as is" is the expected outcome for a competent change. A review that
finds nothing wrong and says so in three lines is a complete, successful review.

The codebase grows when reviews produce findings that are theoretically true
but never reachable, and when each finding is fixed by adding code. Every
finding you report will probably be fixed. So the burden of proof is on the
finding, not on the code: an unproven finding is `noted, no action`.

That filter applies to *reporting*, not to *searching*. Round 1 of the
experiments behind this skill (see "Origin and limits") showed that a brief with the filter
alone found nothing at all: reviewers read "merge as is is expected" as
"do not look". `reviewer.md` therefore has two phases — a fixed search list
that is always run in full, and the admission test that decides what becomes
a row. Never shorten the search to shorten the report.

## 1. Scope (orchestrator)

- Name the exact range (`base..head`). Read the whole range, tests and docs included.
- List decisions the owner has already made (PR text, earlier review threads).
  Reviewers do not re-argue them; they may report an unseen consequence.
- Defects in code the range did not touch are **not findings** of this review.
  They are `out of range` (§5).
- **Existing issues.** Once per review, list the open issues whose text names a
  file the range changes (`gh issue list --state open --json number,title,body`,
  match the changed paths), and scan the titles of the rest. Rule out
  same-file-different-function matches by function or line. For each match:
  the range **fixes it** (fully or partly) → the PR says `Fixes #N` / "partly
  addresses #N" and the issue gets a note; the range **makes it worse or newly
  reachable** → a normal finding; **unaffected** → drop it. Pass the list to
  the reviewers so they do not report a known issue as new.

## 2. Reviewer brief (when delegating)

Give each reviewer `reviewer.md` **verbatim**, the range, the owner's decisions,
the list of matching open issues (§1), and a read-only checkout to read from. Do not pass your own conclusions. Two
reviewers at most; a second reviewer is for independent agreement, not for a
second list.

## 3. Admission test for a finding (orchestrator and reviewers)

A finding is **actionable** only if all four hold. Otherwise it is
`noted, no action` and gets one line, not a paragraph.

1. **Guarantee.** It names the guarantee broken: a wrong simulation result, a
   crash or UB, data loss or a corrupt file, a silent divergence between two
   paths that claim to agree, or a false claim in the PR/docs.
2. **Reach.** A concrete path reaches it from something that exists: a shipped
   config, a documented CLI flag, a GUI action, a test fixture, a file the
   writer in this repo produces. "A crafted file could" or "a future caller
   might" is not reach — note it, no action.
3. **Cost.** The reviewer estimates the fix in lines and names what it would
   delete. A finding whose fix is larger than the defect's reach warrants is
   `owner's call`, with the estimate, never a silent fix. **Exception:** a fix
   that removes at least as many lines as it adds (dead code, a manual close
   replaced by an existing RAII guard, a duplicate folded into an existing
   helper) is `fix here` even on weak reach. Shrinking code is never growth.
4. **Test.** The defect can be shown by a test that fails before and passes
   after the fix. No such test → no fix on this branch. (Defensive checks for
   states nothing produces fail this test by construction.)

Precedent is not a defect: "module X does it differently" is a finding only
if the difference breaks a guarantee from item 1.

**Origin.** A defect in code the range touched counts towards the verdict only
if the range **introduced it or made it worse** (more reachable, harder to
detect). A pre-existing defect whose reach the range does not change is
`out of range`, even in touched lines: a PR does not inherit every old bug in
the code it moves. Check at the base commit (run or read there). Exception, as
in item 3: a fix that removes at least as many lines as it adds may still be
`fix here`.

## 4. Verdict and budget

- State the verdict first, then the table. Verdict rules:
  `do not merge` needs a `fix here` finding with guarantee = wrong result /
  crash / corruption reached from a shipped config, GUI action or documented
  workflow. `merge after <n>` covers every other actionable finding, including
  `owner's call` rows whose guarantee is crash/UB or wrong result and whose fix
  is ≤ 10 lines: the owner decides, but the verdict names the decision.
  `merge as is` only when the search list is done and no actionable row remains.
- **Fix budget:** the sum of estimated fix lines for `fix here` findings should
  stay under ~20% of the range's own added lines. If it does not, the
  findings point at the design; say so and recommend a redesign discussion or
  a follow-up, not a longer patch series.
- **Subtraction:** every review names at least one thing the range could
  delete or merge without changing tested behaviour, or states that it looked
  and found none. Fixes that delete code are preferred over fixes that add.

## 5. Result table

| # | file:line | defect (one sentence) | guarantee | reach | fix est. (lines, +/−) | test? | disposition |

Dispositions: `fix here` / `owner's call` / `noted, no action` / `out of range`.
`out of range` items, and findings that duplicate an existing issue whose
reach the range does not change, go below the table: a duplicate as a
reference (`see #N`, plus anything new), the rest as short issue drafts
(title, evidence with commit, reach, class as in §8) for the owner to approve
before anything is filed. They never count towards the verdict. A known
defect the range introduced or made worse (origin, §3) stays a row, with
`see #N` in the defect column, and counts towards the verdict. Then one line
per search question **not finished**. Target length of the whole report:
under 600 words plus the table. Length is a cost, not a merit.

**Posted text has no tables.** The table is for the report to the owner. A
review, PR comment, issue or issue comment posted on GitHub gives each
actionable row as a numbered item instead: a bold one-line title with
`file:line` and the disposition, then a short list of the row's other columns
(defect, reach, fix estimate; guarantee and test where they are not obvious
from the defect).
`noted, no action` items stay one bullet each. Wide tables with sentence-long
cells wrap badly and do not scale with the number of findings (review of
#292/#293).

## 6. After fixing

- A fix commit is reviewed with the same admission test. Each fix commit cites
  the finding number and the test that fails without it.
- Re-run the suite with its build options stated (CLAUDE.md, "Totals depend on
  the build options").

## 7. Version stamp

Start the first line of everything posted from this skill (review, PR comment,
issue, issue comment) with the version, after any prefix your own
instructions require, e.g. `jk's agent (coast-review-lean v7):`. The next
tuning pass finds live uses with `gh search` on that stamp and scores them
against what happened afterwards (fixes, reverts, issues closed as invalid).
Bump the version whenever this file or `reviewer.md` changes what a reviewer
does.

## 8. Issue triage (target: issue numbers)

The question per issue is: does it describe a **real** defect, or a
**theoretical**, constructed one? The admission test of §3 decides, applied to
the issue's claims instead of a range.

1. **Check against current `main`, not the issue's commit.** Issues go stale:
   later merges may have fixed part of one. Do this **before** reading the
   code at the issue's line references: run `git log <issue commit>..main`
   on the cited files first. Name the commit you checked.
2. **Split bundled issues into sub-claims** and classify each. One issue often
   mixes a real, reachable bug with several theoretical ones; the real one
   sets the urgency and is worth splitting out.
3. **Reproduce before classifying.** If a claim can be run in minutes, run it:
   a headless `-DCOAST_WITH_GUI=OFF -DCOAST_GPU_BACKEND=none` build (~20 s),
   a modified copy of a shipped config, a probe program, `perf` for a
   performance claim, `ulimit -f` with `trap '' XFSZ` for "full disk". Reading
   alone misjudges both ways: a run showed one claim to be worse than the
   issue said, and a check of the callers showed another to be unreachable.
4. **Reach includes the caller.** Confirm that the triggering call happens on
   the claimed path: a GUI control that is disabled in that state, or an API
   only a code path that never meets the condition calls, means no reach.
   Check also whether the GUI's own control limits (slider ranges) reach a
   degenerate value; that is a GUI action, the strongest kind of reach.
5. **The proposed fix passes the test too.** Flag fixes broader than the
   defect: a structural change for "all N owners" when only one is affected,
   a unit test per entity family for a state nothing produces, a general
   validation framework for inputs only absurd values break. Prefer fixes
   that delete code.
6. **Classes:** `real` (guarantee + reach + test hold) / `real, contained`
   (reachable, but an existing mechanism prevents the wrong result, or the
   defect is in docs only) / `theoretical` (no reach) / `design question`
   (no stated guarantee broken; the owner decides) / `not shown` (could not
   reproduce; name the experiment that would decide it).
7. **Urgency** (labels `urgency: high|medium|low`): **high** = reachable in
   normal use (shipped config, GUI action) with a silently wrong result or
   data loss; **medium** = real and reachable, but needs a specific workflow
   or environment (resume, full disk, unthrottled run); **low** =
   contained, theoretical, a design question, or not shown.
8. **Output:** one table row per sub-claim (claim, still holds?, guarantee,
   reach, class, ran/read), then per issue a short comment for the
   collaborator: class, evidence with commit, fix scope, urgency. Post
   comments and set labels only after the owner asks.

**Cleanup issues** (`cleanup required`) break no guarantee by definition, so
items 6–7 are replaced by:

- **real**: the duplication or dead code exists (diffed or grepped, not taken
  from the issue) **and** removing it shrinks the code without changing
  tested behaviour, **or** history shows it already caused drift (a fix
  applied to both copies, a site that missed a new type).
- **theoretical**: it only "could drift", with no evidence, or existing tests
  already guard the risk.
- **growth**: the proposed change adds structure without removing more than it
  adds (splitting a function into new files, renaming keys with aliases, a
  validation framework for absurd inputs) → owner's call.
- Bug claims inside a cleanup issue get the normal defect test; run them if
  they are runnable.
- "Unused" means unused through **every** way code gets called, not just
  compiled callers: the embedded console, scripts, tests, CMake targets.
- Dead code with a bug in it is **deleted, not fixed**: a fix adds code to
  something nobody calls.
- Labels: `urgency: low` unless a real bug is hidden inside; add `quick win`
  for a small change that removes more code than it adds.

## Origin and limits

**Version:** v7 (`reviewer.md` v7; v7 keeps a known defect the range made
worse as a counted row, §5). §1 "existing issues", the origin rule in
§3, and §8 are drafts from live sessions, not tuned.

*Origin:* main grew from 4k to 116k lines in ten months; of the last 400
commits 151 were `fix` and 16 `harden`, 12 `refactor`. Review findings were a
major driver. This skill trades some recall for precision and conciseness, on
purpose. `reviewer.md` was tuned with `skill-evolve` over 6 versions and 53
reviewer runs: tuned on #154/#155/#160 replayed at the commit the real review
saw, held out #71/#87/#90/#151/#164, control #162.

Known limits:
- **Large ranges.** On 8k-line ranges one reviewer covers part of the surface;
  suggest a split (§1).
- **Read-only reach.** Findings that need the binary to run are out of reach
  of a read-only reviewer; run the probe yourself when a claim depends on it.
- **Variance.** One item moves per run; two reviewers for agreement.

Before changing this skill, read its tuning log: `coast-review-lean/LOG.md` on
the orphan branch `skills-lab` (`git worktree add ../cimc-coast-skills-lab
skills-lab`). It says why each sentence of `reviewer.md` is there; raw reports
are next to it under `experiments/`.
