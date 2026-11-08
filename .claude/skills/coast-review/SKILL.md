---
name: coast-review
description: Checklist for reviewing a COAST change (a PR, a branch, or a fix commit) — how to brief reviewer agents, the adversarial questions to answer, COAST-specific checks, and how to verify and merge findings. Use whenever a review of code in this repository is requested, including a review of fixes made in response to an earlier review.
argument-hint: [PR number, branch, or commit range]
---

# COAST review checklist (draft)

**Target:** $ARGUMENTS

A review answers one question: *for which inputs, states or call paths is this
change wrong?* "It reads well" or "it matches the neighbouring code" is not an
answer. Every finding is a concrete counterexample: input/state → wrong result.

**The deliverable is the findings table (section 4) plus the list of what was
not checked. This checklist is never the output: a ticked list is not a
review.**

This file is for the orchestrator. The questions reviewers answer are next
to this file, one file per lens: `questions-contracts.md` and
`questions-tests.md`. `questions-common.md` says how to answer and how to
report, for both. Pass them to reviewers verbatim.

---

## 1. Scope

- [ ] Name the exact range under review (`base..head`) and read **all** of it,
      including tests, docs, configs and GUI code, not just the core module.
- [ ] Read earlier review comments on the PR. A fix round is reviewed against
      them: is each point fully, partially, or not addressed?
- [ ] List the decisions the owner has already made. Reviewers don't re-argue
      them, but they report consequences the owner may not have seen.
- [ ] **Fix commits are reviewed too**, with the same checklist, before they are
      pushed. Bugs are easily introduced while fixing review findings.
- [ ] Is the range one concern? If it holds several, suggest a split.

## 2. Reviewer agents (when delegating)

- [ ] **Different lenses, not identical briefs.** Same-brief reviewers read the
      code the same way and miss the same things. Default split:
  - *Contracts & callers* (`questions-contracts.md`): every check traced to
    the code that produces its value; reused helpers; threads; state and
    checkpoint paths; hot path.
  - *Tests & parameter space* (`questions-tests.md`): can each test fail?
    Which configs pass validation but are degenerate? Do the PR's own claims
    hold?
  - *Both* (`questions-common.md`, shared questions): sources of truth and
    layering. Both reviewers answer them independently, so their answers
    show how well the reviewers agree.
- [ ] Give each reviewer the code and the facts, **not your conclusions**. Leave
      out claims like "tests pass" or "mutations were killed" unless you want
      them checked.
- [ ] Give each reviewer `questions-common.md` plus its lens file
      **verbatim**, not a list of topics. A topic list ("check save/load")
      produced no counterexamples on PR #154. A single reviewer gets all
      three files.
- [ ] The report format is in `questions-common.md`; don't restate it in the
      brief.
- [ ] **One git worktree per reviewer**, checked out at the head under review,
      named in its brief. Reviewers may instrument code and change build
      options; separate build directories alone still share one source tree.
      A fresh worktree needs `git submodule update --init -- 3rd_party` (add
      `docs/themes/hugo-theme-relearn` only if the docs are built; with
      submodule URLs pointing at local clones, add
      `-c protocol.file.allow=always`) and a `data` symlink to the main
      checkout's datasets. Afterwards check each worktree is clean, then
      `git worktree remove --force` (needed because of the submodules).

## 3. Verify before relaying

- [ ] Check every reviewer finding against the code at `file:line` before
      passing it on. Reviewer line numbers may be diff positions.
- [ ] Trace "can throw / can be reached" claims to real callers and threads.
- [ ] Correct a reviewer claim that is wrong, explicitly. Don't just drop it.
- [ ] **Mutation spot-check** (unit tests run in under a second): break the code
      the test covers, rebuild the one target, expect a failure, restore. Do it
      in a worktree no running reviewer uses. Report
      which mutations were tried. Mandatory for every test that is the PR's
      evidence that a fix works (a test added for a review finding); optional
      for the rest.

## 4. Merge the findings

- [ ] **Every finding goes into the result table with a disposition**: fix
      here / owner's call (policy) / no action (with reason) / follow-up issue.
      Nothing is dropped silently, and nothing is merged into another finding
      until it disappears. (PR #154: a reviewer's "policy choice" about sigma
      order was folded into another item and lost.)
- [ ] Policy questions go to the owner with a recommendation; neither the
      reviewer nor the orchestrator decides them.
- [ ] **Shared questions: report agreement.** For each shared question, say
      whether the reviewers found the same defects, different ones, or one
      found a defect the other answered "none found". A finding both reported
      is one row in the table, marked as found by both; don't merge it
      silently.
- [ ] Out-of-scope defects become issues, not silent extra work on the branch.
- [ ] The table is followed by what was **not** checked, merged from the
      reviewers' lists and your own.

## 5. After fixing

- [ ] Re-run the full suite with the options stated (see CLAUDE.md, "Totals
      depend on the build options"); GPU tests outside the sandbox.
- [ ] Reply under each review comment: what changed, which commit, which test
      covers it.

---

*Origin:* PR #154 (retina node). Three model reviews missed findings that a
collaborator's review then made: precedent taken as proof, checks read without
tracing their callers, tests judged by their names, a reviewer finding dropped
while merging, and a fix commit pushed without review. This checklist is a
first draft, to be developed in its PR.
