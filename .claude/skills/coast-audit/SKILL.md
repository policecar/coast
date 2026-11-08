---
name: coast-audit
description: Quality audit of one part of COAST (source, tests or documentation; directories or files) that produces GitHub issue drafts — one per defect labelled `bug` or `cleanup required`, plus one `check cleanup` summary of minor items — and creates them only after the owner approves. Use when asked to review the quality or cohesion of a module, directory, test suite or set of docs pages and turn the result into issues, or to "audit X and file issues".
argument-hint: <paths…> (directories or files, relative to the repo root)
---

# COAST area audit → issue drafts

**Area:** $ARGUMENTS

A complete result is one of:
- a set of issue drafts, each verified against the code;
- the line `nothing to file`.

An audit that files issues for things it did not verify is the failure this skill
exists to prevent. So is an audit that misses defects because it only skimmed.

## 1. Scope (orchestrator)

- Resolve the area to a file list, count its lines, and state the HEAD commit.
- **Pick the brief by what the files are.** Split a mixed area by type and run each part with its own brief.

  | files | brief | what `bug` means there |
  |---|---|---|
  | C++/WGSL source (`src/`, `include/`, `coastd/`, `tools/`) | `auditor.md` | wrong result, crash or UB |
  | tests (`tests/`) | `auditor_tests.md` | a test that passes on a broken implementation, or fails/flakes on a correct one |
  | docs (`docs/content/`, `SPEC.md`, `README.md`, `design/`) | `auditor_docs.md` | a statement about the code that is false at HEAD |

  Do not run the source brief on tests or docs. Measured: on docs it audits the code behind the pages and puts every doc error into one `check cleanup`; on tests it finds no more than having no brief at all.
- **Split above ~2.5k lines.** One auditor covers 300–2.5k lines well. Partition by
  subdirectory or by file group, and run one auditor per part. Run up to two
  auditors per part for independent agreement; findings both reach are usually real.
- **Use opus for the auditor.** In tuning, sonnet invented line numbers in 0–4 cited refs per report, while opus cited none. Sonnet is fine as the second, independent auditor.
- Do not pass your own hypotheses or prior findings to the auditors. Give each one
  its brief **verbatim**, the area, and a read-only checkout (or the instruction
  to edit nothing).

## 2. Verify (orchestrator, before anything leaves the session)

For each drafted `bug` and `cleanup required` issue:
- Re-read the cited file:line yourself. If the mechanism is not there, drop the draft.
- If you have not re-read it, the body must say so ("auditor-reported, not
  re-checked"); it must not read as verified.
- Merge drafts with one root cause, both across auditors and across parts.
- Check the label rules in Phase 2 of the brief used. "Misuse only" never makes a `bug`.

## 3. Duplicates

Run `gh issue list --state all --search "<key terms>"` for each draft (search on the
function name and the file). Then, per draft:
- an **open** issue already covers it → drop the draft, and mention the issue number in the summary;
- a **closed** issue covers it → keep the draft only if the code shows that the defect is back, and link the old issue.

## 4. Present, then create only on approval

Show the owner one table (label, title, reach, verified yes/no). **Create nothing
until the owner says so**, because issues notify people and are hard to retract.
On approval:
- Write each body to a file and run `gh issue create --label <label> --title … --body-file …`.
  This runs outside the sandbox: network access is needed.
- Then edit the `check cleanup` issue to link the numbers of the `bug` /
  `cleanup required` issues it may overlap.
- Pin line references to the HEAD commit ("line refs at `<sha>`").

## 5. Version stamp

Start the first line of every issue body and comment this skill creates with
the brief and its version, after any prefix your own instructions require,
e.g. `jk's agent (coast-audit, auditor_tests v3):`. The next tuning pass finds
live uses with `gh search` on that stamp; issues later closed as invalid or
duplicate are the precision signal. Bump a brief's version whenever it changes
what an auditor does.

*Origin and limits.* **Versions:** `auditor.md` v4, `auditor_tests.md` v3, `auditor_docs.md` v3 (v3: a normative clause loses to the code only through a recorded decision; untuned). Modes: `auditor_docs.md` and `auditor_tests.md` were tuned the same way as `auditor.md` (68 runs).
- **docs, held-out:** 3/4 found, all labelled as bugs; the baseline found 2/4, none labelled as bugs. Control 0/0.
- **tests, held-out:** recall equal to the baseline, but right labels 3/4 vs 0/4. Control 0/0 vs the baseline's 2 bugs + 1 cleanup. Its bugs come with mutation probes.
- **Tests mode is provisional:** small truth set, partial convergence.

`auditor.md` v4 was tuned with `skill-evolve` over 46 agent runs. The test set was 7 historical cases, replayed at the commit before each defect was reported: #52, #96, #149/#150/#159/#166, #189, #191, and the 2026-10-31 review. Results against a no-skill baseline:
- **Held-out:** 8/8 bug-class truth found, vs the baseline's 5/8.
- **Clean control area (retina):** 0 `bug` and 0 `cleanup required`. The baseline filed 1 bug on both tiers plus 3 cleanups on sonnet.

Known limits:
- **Strict cleanup bar.** `cleanup required` needs drift: duplication that has drifted, or a named way it will. Identical copy-paste goes to `check cleanup`. Change Phase 2 if you want it otherwise.
- **Variance.** One finding moves per run, so use two auditors when it matters.
- **Callers outside the area.** A defect that shows up only there (e.g. a checkpoint taken while stepping) is found reliably only by opus.
- **Orchestrator steps unexercised.** §§1–4 were not exercised by the experiments; only live use (§5) validates them.
- **Spec right, code wrong is untested.** The docs-mode truth set has no such case; its one `SPEC.md` case (#185) is a stale clause.
- **Large SPEC-style documents** are not fully covered in docs mode: sonnet skipped per-line citation checks, opus accepted line drift.
- **Tests mode with mutation probes is expensive:** opus runs took 12–15 min and about 200k tokens.

Before changing this skill, read its tuning log: `coast-audit/LOG.md` on the orphan branch `skills-lab` (`git worktree add ../cimc-coast-skills-lab skills-lab`; Part 2 covers the modes). It says why each sentence of the briefs is there.
