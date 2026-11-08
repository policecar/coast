# COAST review: how to answer, and shared questions

Every reviewer gets this file plus the questions file for its lens
(`questions-contracts.md` or `questions-tests.md`). A reviewer given both
lens files answers both.

Every reviewer answers the shared questions at the end of this file, whatever
its lens. Answer them on your own reading; the answers are compared across
reviewers.

Answer each question for every relevant change in the range. An answer is a
concrete counterexample (input/state → wrong result), or "none found" with how
you checked. "It matches the neighbouring code" is not an answer.

Work only in your own git worktree (the brief names it), and build there in a
build directory named after your lens (`build-review-contracts/`,
`build-review-tests/`). Never edit or build in the main checkout or another
reviewer's worktree. Reviewers run in parallel and may instrument code or
change build options; a shared source tree or build directory mixes their
edits and builds. Restore anything you instrumented before you report, and say
what you changed.

Report every finding as: `file:line`, one-sentence defect, concrete failure
scenario, confidence (confirmed by reading / by running / plausible), ranked by
severity. Report a finding you consider "policy" anyway, flagged as policy.
End with what you did **not** check.

## Shared questions (every reviewer)

**Sources of truth**
- [ ] Where is the same fact stored more than once (validation vs GUI widget
      ranges, docs vs code, code comments, string tables, hand-written field
      lists, commit message)? Do all copies agree after this change?

**Layering**
- [ ] `coast_lib` never depends on `coast_gui`; entity modules (`io`,
      `hd_ngm*`, `retina`, …) do not include `json_cfg`. Run the headless
      build in your own worktree and build directory to prove it (see CLAUDE.md). ngm2 /
      ngm25 / ngm3 do not include each other's **core** headers (neuron
      group, neuron, dendrite, …). Shared tool or helper headers are fine. If
      one model needs part of another's core, is that part lifted into a
      common helper instead?
