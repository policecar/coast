# Auditor brief — documentation (coast-audit)

You audit one area of COAST's **documentation** (Hugo pages under `docs/content`,
`SPEC.md`, `README.md`, `design/` notes) and draft GitHub issues for statements that
are wrong. The deliverable is a list of issue drafts, or the line `nothing to file`.
Both are complete results. Drafting an issue for a statement that is true is the
failure this brief exists to prevent. So is skimming.

Which side is the reference depends on the document. You may read the whole checkout.
- **Descriptive pages** (`docs/content`, `README.md`): the code at HEAD is the
  reference. A page is wrong where it disagrees with the code.
- **Normative documents** (`SPEC.md`, and any page with its own authority section):
  apply that document's authority rules first. Label a clause wrong only if a
  recorded decision superseded it, and cite the decision: a commit, issue or PR, a
  dated maintainer ruling, or a stated contract in docs or a header comment. Code
  that merely behaves differently is not enough. If no such decision exists, draft
  the discrepancy against the code, or as an open authority question for the
  document's open-questions list. Do not propose rewriting the clause.
- **`design/` notes** are plans: a gap between a note and the code may be unfinished
  work. Report it only where the note says the work is done.

## Phase 1 — Search (always run every step, on every page of the area)

1. **Inventory.** For every page, list its concrete claims, each with its line:
   - file paths and file:line citations;
   - function, type, config-key and CLI-option names;
   - numeric limits and defaults;
   - behaviour ("X happens on thread Y", "A before B", "uses Z");
   - commands;
   - links to other pages or files.
2. **Names exist.** For each named file, function, type, key or option, grep the checkout. For each file:line citation, open the line and check that it shows what is claimed.
3. **Links resolve.** For each link or file reference, check that the target exists and is tracked (`git -C <checkout> ls-files <path>`). Untracked local files do not count.
4. **Behaviour matches.** For each behavioural claim, find the code that implements it and compare: which thread, which order, which path, which granularity.
5. **Numbers match.** For each numeric limit or default, compare it with the constant in the code.
6. **Removed things.** For each feature, option or enum value the page names, check that the code still has it.
7. **Contradictions.** For each claim the area states twice (two pages, or SPEC vs docs vs header comments), check that the statements agree.
8. **Commands work.** For each build or run command, check that its options exist in the CLI parser, CMake options or presets.

Write down what each step found, including "nothing", in a working list.

## Phase 2 — Admit (apply to every item of the working list)

Keep an item only if all of these hold:
- **Verified:** you read the page line and the code that contradicts it, and can cite both. A guess is not admitted.
- **Failure scenario:** what a reader who follows the page does wrong, or what they wrongly believe about the code.
- **Reach** is stated, as one of: anyone reading the page | only someone cross-checking several pages | nobody (purely cosmetic).

Label each admitted item:
- `bug`: a false statement about the code, at reach "anyone reading the page". This covers a wrong path, a wrong behaviour, a wrong limit, a removed feature, and a dangling reference.
- `cleanup required`: structural, with a named consequence. For example, two sources that contradict each other, or a module with no page.
- Everything else (wording, typos, history kept as if current, cosmetic drift) goes into **one** issue labelled `check cleanup`.

Merge items with one root cause into one issue (list every site).

## Phase 3 — Report

Before writing, re-open every file:line you will cite, on both the page and the code side, and confirm it shows what your sentence says. Fix the reference, or drop the sentence. Number lines one file at a time (`sed -n` or `cat -n <one file>`): `cat -n` over several files numbers them continuously, and its numbers are wrong for every file after the first.

First line: `nothing to file`, or `<b> bug, <c> cleanup required, <k> check cleanup`.

Then, for each issue:

```
### <title: the wrong statement, not the fix>
Label: <bug | cleanup required | check cleanup>
Reach: <anyone reading | cross-checking only | cosmetic>

## What happens
<page:line says …; the code at file:line does …; what a reader gets wrong>

## Fix
<the corrected statement, or the smallest change; size S/M/L>
```

Then a section `Not examined`, listing what you did not read or could not settle.
