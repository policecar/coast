---
name: skill-evolve
description: Develop or tune a skill (a brief that agents follow) empirically — build a ground-truth test set from history, run agents with and without the candidate skill, score against the truth, change one thing per round, hold out cases, log everything, stop when it converges. Use when asked to create, improve or compare a skill, prompt or reviewer brief whose quality can only be judged by what agents produce with it. Origin: coast-review-lean (its LOG.md on branch skills-lab is the full worked example).
argument-hint: [skill name or goal, and where the historical cases live]
---

# skill-evolve: develop a skill the way you'd develop code

A skill is a program whose interpreter is a language model. You would not
ship code without tests; do not ship a skill without a test set. The only
evidence that a skill works is the output agents produce with it, scored
against something you fixed **before** you looked at that output.

**Target:** $ARGUMENTS

## 0. Decide what "works" means, in numbers, before writing a word

Write down, in the log you are about to create:
- the **outcome** the skill must change (for a review skill: real defects
  found, false findings avoided, report length, lines a fix would add);
- the **failure the current state produces** that motivates the skill, with a
  measurement (e.g. "151 `fix` commits of 400; 4.7 lines added per line
  deleted"). If you cannot measure the motivating failure, you cannot tell
  whether the skill fixed it;
- the **scores** you will compute per agent run (3–6 numbers: recall against
  truth, noise count by class, length, a domain cost such as fix lines);
- the **stop rule**: what convergence looks like (e.g. two consecutive rounds
  where the change moves only the item it targeted and nothing else).

## 1. Build the test set from history, with ground truth fixed in advance

- Find cases where the outcome is already known: past PRs with review threads,
  past tickets with root causes, past decisions with outcomes. `gh`, `git
  log`, issue trackers. Prefer cases with a **before** state you can
  reconstruct (the commit a review saw) and an **after** state that tells
  you what was right (what got merged, what was reverted).
- For each case write a **ground-truth table**: id, finding, class (real /
  theoretical / cosmetic / policy), what the fix cost. Classify it yourself,
  with reasons. This table is frozen once agents start running.
- **Split**: 3–4 cases to tune on, at least as many held out, sampled across
  time and size, plus **one control** where the right answer is "nothing to
  do". The control catches a skill that manufactures output.
- **Reconstruct the before state honestly.** Use a checkout at the historical
  commit. Give agents only what existed then: the *first* commit message, not
  the final PR description (it was edited after the review and leaks the
  answers — this contaminated round 1 of coast-review-lean). Hide future
  history too: a worktree shares the object store, so `git log --all` shows
  the fix; use a shallow clone or state in the brief that history after HEAD
  is off limits.

## 2. Build the harness before the skill

- A **brief generator**: one script that, given a case and a skill version,
  writes the agent prompt (goal, inputs, read-only rules, where to write the
  report). Agents read the skill file verbatim; you never paraphrase it into
  the prompt, or you are testing your paraphrase.
- **Read-only inputs, isolated outputs.** One worktree per case, agents may
  read and run `git diff` but not edit or build inside it; probes go in their
  own directory. Several agents can share a read-only worktree.
- **Report capture**: each agent writes to `results/<round>/<case>_<skill>_<model>.md`.
  Word count is a free metric; take it.
- A **baseline**: the existing skill, or no skill, run through the same
  harness on the same cases. Without it you cannot tell a skill effect from a
  model effect.

## 3. Write v1 small, then run the loop

Write the skill as short as you can get away with. You will add to it only
what a measured failure demands; a skill that starts long never learns which
sentences matter.

Each round:
1. **Run** all tuning cases × {baseline, candidate} × {two model tiers}, in
   parallel. Two tiers (e.g. sonnet and opus) from the start: a failure that
   appears on both tiers is the skill's; one that appears on one tier is the
   model's. Add a small tier (haiku) and the largest tier once, late, to
   learn the operating range, not every round.
2. **Score** every report against the frozen truth: hits by id, misses,
   noise by class, length, verdict. Put it in the log as a table.
3. **Diagnose** each miss and each noise item: which sentence of the skill
   caused it, or which sentence is missing. Read the agents' own words; they
   usually say why ("hand-written JSON only", "a catch exists").
4. **Change the smallest thing** that addresses the diagnosed cause. One or
   two edits per round, each tied to a named failure. Write the mechanical
   step the agent should perform, not the goal ("walk every frame between the
   throw and the catch", not "check for leaks"). Agents execute questions
   literally: "try tiny and huge" finds tiny and huge; "list the relations
   the formula assumes" finds the inverted pair.
5. **Re-run** and check two things: the targeted failure moved, and nothing
   else moved. If other things moved, the edit was too broad.

## 4. Lessons that transferred (from coast-review-lean, 53 runs)

- **Separate search from judgement.** A brief that only says what *not* to
  report yields nothing at all; agents read "be concise" as "don't look".
  Make the search a fixed list that always runs in full, and the filter a
  separate admission test applied afterwards.
- **Define the domain's terms in the domain's words.** "Reach" meant nothing
  until it said that a hand-written config is this system's user interface.
- **Rules need their exceptions stated.** "Fix cost must match reach" blocked
  a fix that deleted lines. The exception ("shrinking code is never growth")
  had to be written down.
- **Verdict logic must be explicit**, or a known one-line UB fix gets "merge
  as is" because the row was filed as the owner's call.
- **Sets need per-instance enumeration.** Small models check three of four
  instances and stop; "list all N and answer each" fixes it.
- **Expect one item of variance per run.** Re-running the same brief moves one
  finding. Two agents for agreement is worth it; more add items, not
  certainty. Items both agents reach independently are usually real.
- **Model-specific skills were not warranted.** Every failure fixed appeared on
  both tiers in the same round and vanished on both the next. Tier choice
  became a cost decision, not a skill decision.
- **Held-out cases find the gaps tuning hides.** Tuning on three PRs about
  configs, persistence and parameters produced a skill blind to toolchain
  breaks; the held-out set exposed it on the first run.
- **A confirmation run on the cases that exposed a gap proves the wording,
  not generalisation.** Say so in the log. The next live case is the test.

## 5. Log, always — on the `skills-lab` branch, not next to the skill

Tuning material lives on the orphan branch `skills-lab`, never on a code
branch: raw reports carry hundreds of `file:line` references frozen at old
commits, and every repo search (including the skill's own agents') would hit
them. Work in a worktree: `git worktree add ../<repo>-skills-lab skills-lab`
(add `--orphan -b skills-lab` if the branch does not exist yet).

- `skills-lab:<skill>/LOG.md`: method, frozen ground truth, one section per
  round (table of scores, observations, the change for the next round), a
  final state with the held-out numbers and known limits. The log is what
  lets the next person (or you, in a month) change the skill without
  re-deriving why each sentence is there.
- `skills-lab:<skill>/experiments/<round>/`: the raw agent reports.
- The harness (brief generator, worktrees, scratch probes) stays in a scratch
  directory; commit to `skills-lab` only what the log cites.

On the code branch the skill carries only what agents and users need: the
briefs, a **version** (bump it whenever a brief changes what agents do), a
**version stamp** rule (the first line of every post the skill makes names
`<skill> v<n>`), and an "Origin and limits" section with the held-out
numbers, the known limits and a pointer to the log on `skills-lab`.

## 6. Stop rule and release

Stop when: the tuning cases are stable across two rounds, the held-out set
shows recall comparable to the baseline on in-reach truth, the control case
yields "nothing to do", and the last change moved only its target. Release
with the limits written down (what kind of finding the skill cannot reach,
what case sizes break it). Then let live use accumulate before the next pass:
`gh search issues "<skill> v<n>"` (and `gh search prs`) finds every post the
skill made; what happened to it afterwards (fixed, reverted, closed as
invalid) is the outcome to score.

## Anti-patterns

- Judging a skill by reading its text. Text that reads well produced zero
  findings in round 1.
- Scoring against what the agents found instead of against truth fixed
  beforehand. You will approve of whatever they said.
- Briefing agents with your conclusions or with post-hoc descriptions.
- Changing five sentences per round. You learn nothing about which one acted.
- Tuning until the tuning set is perfect. Hold out, or you are memorising.
- Skipping the control. A skill that always finds something is a skill that
  manufactures work — the exact failure this process exists to prevent.

*Second worked example:* `coast-lean-sweep` (its LOG.md on `skills-lab`) — 4 versions, 46
runs, 8 replayed cleanups. Two lessons it added: check a replayed truth
against the HEAD it is replayed at (one truth item was already fixed there),
and decide what the *control* means before scoring it (a cleanup commit itself
leaves residue; the control measured noise class, not zero rows).
