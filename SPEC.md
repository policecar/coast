# COAST Specification

Normative specification of COAST's **model core** (the HD-NGM neuron-group model and its implementations) and **headless core** (`coast_lib`: CLI, configuration, simulation core, datasets, persistence, observation/logging, determinism, test protocol). The GUI, `coastd` and `campaign` are out of scope.

Every requirement is a numbered clause (`M.x.y` for the model, `S.x.y` for the system) so conformance can be assessed clause by clause. Keywords MUST / SHOULD / MAY follow RFC 2119.

## Authority

1. **Author intent wins on mechanisms.** For the model, the model author's specification ("the author's spec"; kept with the author's materials, not in this repository) — specifically its §3, the hierarchical-dendrite model — is normative for *what the model does*. §1 of that document (the earlier basic neuron group) is used only where §3 is silent; §2 (partial input processing) is future work.
2. **The code wins on numbers and functions.** Where the general mechanism is unchanged and only specific numbers or functions differ, the code is authoritative: the clause states the as-built rule, is tagged `[R]`, and keeps the author's original for traceability. Where the mechanism itself changed, or the implementations give no single answer, the `[A]` clause stands and the question goes to the author as a GitHub issue. Implementations MUST NOT disagree with each other in what they compute (M.0.6); where they do, it is a defect, and ngm2 is only the provisional target until the author rules.
3. Where the author's spec is silent, clauses are derived from the project's stated contracts — `CLAUDE.md`, the `docs/` site, contract comments in headers, and test assertions — never from incidental implementation behaviour.
4. Contradictions *inside* the model sources are listed, with the reading each clause adopted, in **M.9**. These are open questions for the author/maintainers, not settled law. In the system part the clause itself is the adopted reading (S.0.3).

Because the author's spec is not in this repository, every clause that relies on it quotes the relevant text or formula, so this file can be read on its own. It cites the author's spec and does not replace it.

## Provenance tags

| Tag | Source | What conformance can establish |
|---|---|---|
| `[A]` | the author's spec §3, quoted | Whether the implementations do what the author specified |
| `[A§1]` | the author's spec §1, where §3 is silent | Same, with lower authority |
| `[M]` | Maintainer ruling; its date and wording are in the git history of the clause | Normative; same as `[A]` |
| `[R]` | Ratified: author's mechanism (§3), code's numbers/functions (Authority 2) | Whether the implementations do what the code-of-record does, consistently |
| `[D]` | Derived model contract: docs, derived spec, code comments | Consistency across implementations and internal correctness |
| `[FUTURE]` | Described by the author, not yet required | Nothing — informative |
| `[C]` | `CLAUDE.md` | Stated project contract |
| `[Doc]` | `docs/content/` page | Stated project contract |
| `[H]` | Contract comment / `@decision` in a header, source, CMake file or script | Stated project contract |
| `[T]` | Evidenced only by a test assertion | Contract as tested |

Citations are `path:line` against `main` at `be8694bb`; treat line numbers as approximate as the code moves.

---

## M. Model

Normative specification of the HD-NGM neuron-group model as implemented in COAST. Authority as stated at the top of this file: the author's spec wins on mechanisms, the code wins where only numbers or functions differ (`[R]` clauses), and the three implementations must agree with each other (M.0.6).

---

### M.0 Scope, notation, conformance terms

**M.0.1 Normative keywords.** MUST / MUST NOT / SHOULD / SHOULD NOT / MAY are used as in RFC 2119. A clause with no keyword is a definition.

**M.0.2 Provenance tags.** Every clause carries exactly one tag:

| Tag | Meaning | How conformance is judged |
|---|---|---|
| `[A]` | Stated in the author's spec, §3 (author-normative). Cites the section and quotes the formula. | Each implementation either CONFORMS or DIVERGES against the clause text. |
| `[A§1]` | Taken from the superseded §1 of the author's spec because §3 says nothing on the point. | Same as `[A]`. |
| `[D]` | Not in the author's spec. Comes from the derived spec, the docs, design docs or the code, and is the intended contract. | Phrased as an invariant the implementations must share. Checked only for cross-implementation consistency and internal correctness. |
| `[R]` | Ratified: The author's mechanism (§3) with the code's numbers and functions (Authority 2). Keeps the author's original for traceability. | Each implementation CONFORMS or DIVERGES against the as-built rule (M.7.i). |
| `[M]` | Maintainer ruling; its date and wording are in the git history of the clause. | Same as `[A]`. |
| `[FUTURE]` | Described by the author but explicitly not yet required. | Not checked. Recorded for completeness. |

**M.0.3 Implementations covered.**

| Id | Code | What it is |
|---|---|---|
| **ngm2** | `include/hd_ngm2/`, `src/hd_ngm2/` | Reference implementation. Object graph group → neuron → dendrite. Each dendrite owns a per-dendrite SoA of synapses that **grows** on branching. CPU only (TBB). Uses a sequential `mt19937_simd` RNG per group, per neuron and per dendrite. |
| **ngm25** | `include/hd_ngm25/`, `src/hd_ngm25/` | "Flat-sparse" re-expression that targets ngm2 semantics (`design/ngm2_5_flat_sparse.md` §1). One group-owned CSR synapse arena, sized to the realised structure and re-packed at branch events. Draws come from a stateless PCG hash keyed by structural identity. Has a CPU entity and a GPU entity (`hd_ngm25_gpu.wgsl`). Also has a penalty-policy research seam. |
| **ngm3-CPU** | `include/hd_ngm3/hd_ngm3_neuron_group.h`, `src/hd_ngm3/hd_ngm3_neuron_group.cpp` | Flat, dense, pre-allocated SoA per bank, shaped `[neuron × dendrite × rep × K]` with `rep = 2^L`. The full branched tree exists from step 0, and branching rewrites segment ids in place. TBB. Master RNG plus one RNG per neuron (`mt19937_simd`). |
| **ngm3-GPU** | `include/hd_ngm3/hd_ngm3_gpu_neuron_group.h`, `src/hd_ngm3/hd_ngm3_gpu_neuron_group.cpp`, `src/hd_ngm3/hd_ngm3_gpu.wgsl` | WebGPU port of ngm3-CPU with the same layout. All draws come from a stateless PCG hash keyed by (coordinate, step, seed, salt). Host-side initialisation is shared with the CPU path. |

Unless a clause names an implementation, it applies to all four (ngm25-GPU counts under ngm25).

**M.0.4 Notation.** Plain Unicode math.
- N is the set of neurons in a group, |N| ≥ 2 (M.1.1). n.D is the set of dendrites of neuron n. d.DS is the set of segments of dendrite d. ds.S is the set of synapses of segment ds.
- i ∈ [0,1]^K is the input vector seen by a dendrite. K is its width. iₖ is component k.
- Iˢ = Σₖ iₖ (element-wise sum). Iₘₓ = maxₖ iₖ.
- τ is the permanence threshold and L = d.max_branch_level.
- the author's spec uses the symbol β for two different quantities (see M.9.2). This spec splits it: **β_sm** is the softmax factor and **β_sec** is the secondary learning rate.
- ϖ_c is the common learning rate. μ is the mismatch smoothing factor. Θ is the adaptation-history threshold. T_b is the branching interval. ρ_b is the branching fraction (SPEC: "5% of K").
- clamp(x,a,b) = min(max(x,a),b).
- sigm and gate are defined in M.4.1–M.4.2.
- A *step* is one call of the group's `process()` on one input vector.

**M.0.5 Out of scope.** These are not model semantics and are specified elsewhere: IO buffers, the observe/probe channels, GUI, persistence file format (persistence semantics are in M.7.17–M.7.19) and performance.

**M.0.6 Implementation equivalence** `[M]`. ngm2, ngm25 and ngm3 (each on CPU and GPU) MUST compute the same model. They MAY compute it differently — data layout, parallelisation, floating-point evaluation order, PRNG algorithm and stream mapping are free — but not compute something different. Precisely:
  - **(a) Deterministic parts.** Given the same state, input and random draws, every implementation MUST produce the same response, learning update, branching decision and persisted state, up to floating-point evaluation-order error.
  - **(b) Stochastic parts.** Every random quantity (noise, low-input penalty, winner draw, branch jitter, initial permanences) MUST be drawn from the same distribution, at the same granularity — one draw per the same model entity per step — and with the same seed semantics. A change of granularity that changes the distribution of a computed quantity is a divergence even if each individual draw is correct.
  - **(c) Configuration domain.** For any configuration, an implementation MUST either compute the same thing as the others or reject the configuration with an error. Silently computing something else (dropping an input, capping a parameter, outputting noise) is a divergence.
  - **(d) Disagreement is a defect,** never a tolerated variant. The target behaviour is the `[A]`/`[R]` clause; where that clause is open, the author's ruling sets the target, and until then ngm2 is the provisional target.
  - **(e) Verification.** Equivalence SHOULD be checked by a differential test that runs the implementations from the same initial state with the same draws (or with randomness disabled for the test) and compares them step by step, plus a distributional comparison for (b). `integration.impl_equivalence` (the three CPU implementations) and `integration.impl_equivalence_gpu` (adding ngm25-GPU and ngm3-GPU) do the step-by-step part.

---

### M.1 Structure

**M.1.1** `[M]` A neuron group consists of |N| ≥ 2 neurons. A configuration with fewer MUST be rejected with an error at construction, never silently adjusted: the neuron count is the group's output width, so a changed count would make downstream buffers, observe widths and saved state disagree with the configuration. In ngm2, |N| is the total over the group's neuron sets. *(The group's mechanism is competition between its neurons — with one neuron the same neuron always wins. Consistent with the author's spec, whose neuron selection factor ηₙ = 𝜛 sigm(|N|(1 − rₙ)/(|N| − 1))·gate(n.a) is undefined for |N| = 1; §3 Structure: "the model describes a group of |N| neurons".)*

**M.1.2** `[A]` Each neuron has |n.D| ≥ 1 main dendrites; a small number (e.g. 3–5) is typical. *(§3 Structure: "a small number (e.g., 3 to 5) |n.D| main dendrites")*

**M.1.3** `[A]` Each dendrite is a binary tree of dendritic segments. At creation it has exactly one segment, with ds.id = 1: "|d.DS| = 1 … The first segment has ds.id = 1". *(§3 Structure)*

**M.1.4** `[A]` Segment ids are heap-indexed. The children of segment c have ids 2c and 2c+1. The parent of c (for c > 1) has id ⌊c/2⌋. The depth of c is ⌊log₂ c⌋, so the root has depth 0. *(§3 Response, Remark: "the next two possible segments … have either the ID cs.id * 2 or cs.id * 2 + 1 … the preceding segment … has always the ID cs.id / 2")*

**M.1.5** `[A]` The depth of every dendrite is bounded by L = d.max_branch_level, the maximum number of sequential bifurcations. A dendrite MUST NOT have more than 2^L leaf segments, nor more than 2^(L+1) − 1 segments in total. Equivalently, every segment id lies in [1, 2^(L+1) − 1]. *(§3 Structure: "the number of 'dendritic endpoints' … is limited to 2^d.max_branch_level and the number of all dendritic segments … is limited to 2^(d.max_branch_level + 1) − 1")*

**M.1.6** `[A]` Every synapse s carries three values: a permanence s.p ∈ [0,1], a mismatch s.mm and an adaptation history s.ah. s.p MUST lie in [0,1] after every operation. *(§3 Structure: "Each synapse has a permanence value s.p ∈ [0,1], a mismatch value s.mm, and an adaptation history value s.ah")*

**M.1.7** `[A]` Every synapse is bound to exactly one input index k ∈ [0,K). The binding never changes, including when the synapse is moved or cloned. *(§3 Structure: "has a synapse for each input"; §3 Branching moves synapses between segments without remapping them)*

**M.1.8** `[A]` At creation, the root segment of each dendrite holds exactly K synapses, one per input, and no other segment exists. *(§3 Structure: "The initial dendritic segment (ds.id = 1) of each dendrite has a synapse for each input i ∈ [0,1]ᵏ, i.e., |ds.S| = k")*

**M.1.9** `[A]` A synapse is *connected* if and only if s.p > τ, using a strict inequality. τ ∈ [0,1] is a hyperparameter with default 0.3. *(§3 Structure; §3 Response: "if sₖ.p > τ")* The comparison is made at the permanences' precision, float: p > float(τ). The default shows why this matters: 0.3 is not exactly a float, ~7 % of the initial permanences equal 0.3f, and 0.3f > 0.3 holds in double.

**M.1.10** `[A]` A *leaf segment* is a segment that has no child segments. A *dendritic segment path* (DSP) is the chain of segments from the root to one leaf. Each DSP is uniquely identified by its leaf id, and its members are the leaf id followed by repeated ⌊·/2⌋ up to 1. *(§3 Response: "A dendritic segment path (DSP) is a chain of dendritic segments that leads from the initial ('root') segment to a leaf segment")*

**M.1.11** `[A]` Every segment has either zero or exactly two children. As a consequence of M.1.8 and M.6.5, every DSP contains exactly one synapse per input index k, at all times. This is a testable structural invariant. *(Follows from §3 Branching: an ambiguous synapse is removed from its segment and added to **both** subsequent segments)*

**M.1.12** `[A]` Inputs are restricted to i ∈ [0,1]^K. Behaviour for inputs outside [0,1] is undefined. *(§3 Structure: "receives … input i ∈ [0,1]ᵏ")*

**M.1.13** `[A]` Without bank routing (M.7.1), every dendrite of every neuron receives the same input vector i. *(§3 Structure, §1 Structure)*

**M.1.14** `[A]` The group output is a vector r ∈ [0,1]^|N|, with one entry per neuron. *(§3 Response: "The response r ∈ [0,1]^|N| of the neuron group")*

---

### M.2 Initialization

**M.2.1** `[R]` Each root-segment synapse permanence MUST be drawn independently as s.p = clamp(X/100, 0, 1), where X ~ Pois(⌊100·τ⌋), so the mean is ≈ τ. As built in every implementation, ⌊100·τ⌋ is computed as int(100·τ) in float; on the 0.01 grid of τ this equals ⌊100·τ⌋ except at τ = 0.53 and 0.59, where it gives 52 and 58 (open: the exact floor would need a different computation in all five). *(ngm2 `hd_ngm2_dendrite.cpp:123-130`; ngm3 `hd_ngm3_neuron_group.cpp:128-137`; ngm25 `hd_ngm25_arena.cpp:234-247`)* *Author's original (§3):* X ~ Pois(50·τ), mean τ/2 — same mechanism (Poisson-distributed permanences around a τ-relative mean), different parameter.

**M.2.2** `[A]` At creation, s.mm = 0 and s.ah = 0 for every synapse. *(§3 Structure: "The mismatch values sₖ.mm and adaptation history values sₖ.ah are zero-initialized")*

**M.2.3** `[D]` An implementation's initial state MUST be a pure function of (configuration, `random_seed`, `seed_offset`). Two constructions with equal inputs MUST produce bit-identical initial state. *(ngm2 `hd_ngm2_dendrite.cpp:113,146-150`; ngm3 `hd_ngm3_neuron_group.cpp:54,94-100,140-146`; ngm25 `hd_ngm25_arena.cpp:234-247`)*

**M.2.4** `[D]` The group's effective seed is `random_seed + seed_offset`, modulo 2³² (a seed is a `uint32`, S.3.24). Every implementation MUST expose this value as `effective_seed()` for the run manifest, and it MUST be the value the group's RNGs are seeded with. A `load_state` that restores the group's random state from a checkpoint restores its seed with it: from then on `effective_seed()` and every later checkpoint name the checkpoint's seed, not the configured one (which M.7.8 reports as not applied). ngm2 records it as `effective_seed` from format version 5 on and falls back to its configuration record's entry for older checkpoints; a checkpoint with neither keeps the configured seed. *(ngm2 `hd_ngm2_neuron_group.cpp:40,46`; ngm3 `hd_ngm3_neuron_group.cpp:54,59`; ngm25 `hd_ngm25_neuron_group.cpp:43`)*

**M.2.5** `[D]` For equal configuration and seed, ngm3-GPU MUST produce initial bank state (permanences, segment ids, mm, ah) byte-identical to ngm3-CPU. It does this by replaying the CPU RNG sequence: a master seed, then |N| per-neuron pre-draws, then the Poisson draws. *(`hd_ngm3_gpu_neuron_group.cpp:129-181,412-416`; `hd_ngm3_neuron_group.cpp:85-87,128-137`)*

**M.2.6** `[D]` In ngm3, the initial value of every replica row r ∈ [0,R) of a dendrite equals the value of row 0, so every dendrite starts with all synapses on the root. *(`hd_ngm3_neuron_group.cpp:124-137`)*

---

### M.3 Response

**M.3.1** `[R]` A connected synapse (s.p > τ, see M.1.9) contributes its input iₖ and, when the input is low, a stochastic negative penalty, exactly as in M.7.31. An unconnected synapse contributes 0. *Author's original (§3):* deterministic sₖ.a = 2·iₖ − Iₘₓ — same mechanism (connected synapses whose input is low pull the segment's activity down), different function. *(Ratified as a function choice under Authority 2; the stochastic draw is the one judgment call in that classification.)*

**M.3.2** `[A]` In M.3.1, Iₘₓ is the maximum value of the dendrite's input vector, and Iˢ is its element-wise sum. Both are computed once per step from the same input i. *(§3 Response: "Iₘₓ the maximum input value"; "Iˢ the element-wise sum over the input")* Where a bank concatenates several channels, the statistics are computed **per channel** in every implementation.

**M.3.3** `[R]` Segment activity is the accumulation of its synapses' contributions (M.3.1), floored at 0 immediately after each penalised synapse (M.7.31). *Author's original (§3):* unclamped sum Σ sₖ.a, which may be negative — same mechanism, different function.

**M.3.4** `[R]` DSP activity is dsp.a = clamp(Σ_{ds ∈ dsp} ds.a · a / Iˢ, 0, 1), where a is the noise attenuation of M.7.32. *Author's original (§3):* clamp((Σ ds.a / Iˢ + 1) / 2, 0, 1) — same mechanism (path activity = normalised, clamped sum over the path), different normalisation plus an attenuation factor.

**M.3.5** `[A]` Dendrite activity is the maximum over all of its DSPs: d.a = max_{dsp ∈ d} dsp.a. *(§3 Response: "The activity d.a of a dendrite d is the maximum activity of all possible dendritic segment paths")*

**M.3.6** `[A]` Neuron activity is the maximum over its dendrites: n.a = max_{d ∈ n.D} d.a. The aggregation MUST be max. The author's spec notes min and average as alternatives it observed; that note is informative only. *(§3 Response: "The activity n.a of a neuron n is the maximum value of all dendrite activities d.a")* M.7.2 builds the final neuron activity from this n.a when apical dendrites or noise are present.

**M.3.7** `[R]` The group response is r = inhibit(n.a_final), computed by the group's configured inhibitor (default max_softmax; M.7.11–M.7.14). Each rₙ lies in [0,1]; Σ rₙ need not be 1. This inhibitor is the model's local inhibition. *Author's original (§3):* softmax rₙ = exp(β·n.a) / Σ exp(β·m.a) with β = 10, Σ rₙ = 1 — same mechanism (local competition among the group's neurons), different function.

**M.3.8** `[R]` The group output written for downstream consumers is r from M.3.7. *(§3 Response; r is the inhibitor's output, not a softmax)*

**M.3.9** `[A]` On every step, the response (M.3.1–M.3.7) is computed before learning (M.4–M.6). Learning on step t MUST use step t's input i, its DSP activities, n.a and r. *(Structure of §3: Learning is defined in terms of the response quantities)*

**M.3.10** `[D]` Degenerate input: if Iˢ is zero or not finite, every d.a MUST be 0. No NaN or Inf may reach the output or any persisted state. *(ngm2 `hd_ngm2_dendrite.cpp:209-214`; ngm3 `hd_ngm3_neuron_group.cpp:432,438,480`; ngm25 `hd_ngm25_response.cpp:136-137`)*

**M.3.11** `[D]` Every value of the group output MUST be finite and lie in [0,1]. *(clamps in ngm2 `hd_ngm2_neuron.cpp:116-120`; `random_index.cpp:112`; max_softmax has no clamp: its output lies in [0, max(v)], so it stays in [0,1] because its input does)*

---

### M.4 Learning

**M.4.1** `[A]` The normalized sigmoid is sigm(x,s,t) = 1 / (1 + e^−(x/step + start)), where step = 1 − s^0.1 and start = −t/step. The defaults are s = 0.5 and t = 0.5. *(§3 Learning: "sigm(x,s,t) := 1 / ( 1 + e^-( x / step + start) ), step := 1 - s^0.1, start := -t / step"; identical to §2 code)*

**M.4.2** `[R]` The learning-window gate has the form min(1 − sigm(x, ·), sigm(x, ·)), combined with the running-maximum bootstrap term as in M.7.34. Its two sigmoid parameter pairs are tracked, not configured (M.4.2a). *Author's original (§3):* gate(x,s,t) = min(1 − sigm(x,s,t), sigm(x,s,1−t)) with s = 0.6, t = 0.66 — same mechanism (a band-pass learning window on neuron activity), different parameters plus a bootstrap term.

**M.4.2a** `[R]` Each group keeps two learning windows, one gating the step's winner and one gating every other neuron. A window's steepnesses are fixed (lower edge 0.57, upper edge 0.78); its transition points ("edges") are running quantiles of its population's combined activity before inhibition. Once per learning step (M.4.15), after adaptation, every edge q becomes clamp(q + 0.001·(τ − p), 0, 1), where τ = 0.19 for a lower and 0.94 for an upper edge and p is the share of the population's activities below q this step: for the winner's edges the winner's activity (p ∈ {0, 1}), for the others' edges every non-winner's activity, or every neuron's when there is no winner (then the winner's edges do not move). A step adapts with the edges the previous learning step left; both windows start at 0.33 / 0.66. The winner's edges follow the drawn winner even on a step where it does not adapt (ngm3 skips a neuron whose bank maxima are degenerate; ngm2 and ngm25 always adapt the winner). The edges are learning state, saved under `activity_window_tracking/` in every implementation's checkpoint (format version 2): a format-2 checkpoint without them, or with a field missing or outside [0, 1], is corrupt; a format-1 checkpoint (written before tracking) starts the trackers over at 0.33 / 0.66, whatever they held before the load. All arithmetic is f32. Single source: `include/ngm/window_tracking.h` (the GPU kernels receive its constants and update step as a generated WGSL prelude). *(Experiments on MNIST, CIFAR-10 and dSprites, 500k steps each: the quantiles and steepnesses reproduce a window fitted to MNIST winner activity at 500k steps; a separate non-winner window keeps the non-winners' learning factor at 0.61–0.75 where a window shared with the winner left it at 0.16–0.70.)*

**M.4.3** `[A]` Learning is local and unsupervised. It happens on every step, in every segment of every neuron, subject only to M.4.13 and M.4.15. *(§3 Learning; §1 Learning: "Learning is strictly local … and fully unsupervised")*

**M.4.4** `[A]` The winning neuron n* is drawn at random from the near-maximal neurons of the post-inhibition output. Draw win_act ~ U(mx·stochastic_win_thres, mx), where mx = max(out), then choose n* uniformly at random among the neurons with out + FLT_EPSILON ≥ win_act. There is one such draw per group per learning step, with the same distribution in every implementation (M.0.6 b). No neuron is favoured by its index, so ties need no rule. *(Replaces §3's deterministic winner, "the neuron with the highest activity n.a", measured before the softmax.)* Checked by the M.4.4 case of `integration.impl_equivalence{,_gpu}` (χ² against uniform over tied neurons).

**M.4.5** `[R]` The winner learns with the winner weight of M.7.34; every other neuron's weight is additionally scaled by the common learning rate ϖ_c, default **0.0001**. *Author's original (§3):* ϖ = 1.0 for the winner and 0.02 otherwise — same mechanism, different value.

**M.4.6** `[R]` The neuron selection factor is the as-built neuron weight of M.7.34: winner sigm(1 − outₙ, weight_filter); every other neuron sigm(1 − outₙ/Σout, weight_filter)·ϖ_c; both multiplied by the learning-window envelope on n.a_final. *Author's original (§3):* ηₙ = ϖₙ·sigm(|N|(1 − rₙ)/(|N| − 1))·gate(n.a) if rₙ > 1/|N|, else ϖₙ·sigm(|N|·rₙ)·gate(n.a) — same mechanism (neurons learn more when the response is flat and less when one neuron dominates, windowed by activity), different function. Whether each neuron receives exactly one factor per step is M.4.14.

**M.4.7** `[A]` The gate argument is n.a, the activity before the softmax, not rₙ. *(§3 η: "based on the neuron activity before the neuron group softmax (hence use of n.a instead of rₙ)")*

**M.4.8** `[R]` Within each bank, the winning path is the first leaf whose activity is ≥ the bank maximum − ε (M.7.33). *Author's original (§3):* the single DSP whose activity equals n.a, argmax over all dendrites of n — same mechanism (the path that produced the activity learns most), different tie and bank handling. On ties across dendrites, several paths receive the primary rate.

**M.4.9** `[R]` Segment weights follow M.7.33: the winning path gets synapse_weight·α (α default 0.01); every other leaf gets act·synapse_weight·β_sec / bank max (β_sec default **0.0001**); the weights are max-folded towards the root. *Author's original (§3):* ζⱼ = α on dsp*, else β_sec·(max_{dsp ∋ dsⱼ} dsp.a)/n.a with β_sec = 0.0002 — same mechanism (primary rate on the winning path, activity-scaled secondary rate elsewhere), different value and normaliser.

**M.4.10** `[R]` Input iₖ is *high* if iₖ > h, where h = avg/2 + FLT_EPSILON and avg is the mean of the input; otherwise it is *low* (M.7.33). *Author's original (§3):* threshold 1/K — same mechanism (high inputs drive permanence up, low inputs down), different threshold. This also resolves M.9.5.

**M.4.11** `[R]` The update weight is θ = clamp(w_seg·((i − h)/(1 − h) if i > h else (h − i)/h)·a_adapt, 0, 1), where w_seg is the segment weight of M.4.9 (already including the neuron factor) and a_adapt is the adapt-side noise attenuation (M.7.32, M.7.33). *Author's original (§3):* θₖ = ((K·iₖ − 1)/(K − 1))·ζⱼ·ηₙ if high, (1 − K·iₖ)·ζⱼ·ηₙ if low — same mechanism (distance from the high/low threshold, times the selection factors), different function.

**M.4.12** `[R]` Permanence update: p ← clamp(p·(1 − θ) + [i > h]·θ, 0, 1) — exponential smoothing towards 1 for high inputs and 0 for low inputs. *Author's original (§3):* identical without the clamp.

**M.4.13** `[A]` An implementation MAY skip the permanence update (M.4.12) and the adaptation-history update (M.5.1) for a segment whose ζⱼ·ηₙ is below an implementation-defined ε. It MUST NOT skip the mismatch update (M.5.2) on that basis. *(§3: "If ζⱼ ηₙ is close to zero for a given dendritic segment, the update might be skipped entirely to improve performance"; scope adopted per M.9.13)*

**M.4.14** `[A]` Within a step, every synapse is updated at most once by M.4.12; a neuron is never adapted twice because it qualifies both as the winner and as a common neuron. Each neuron is adapted exactly once per learning step: the winner with the winner weight of M.7.34, every other neuron with the common weight. *(Follows from §3: there is one ηₙ per neuron.)*

**M.4.15** `[M]` *Zero group output.* If the group output sums to 0 — equivalently, every output is 0, since every output lies in [0,1] (M.3.11) — the step is not a learning step. It MUST compute the response, combine and inhibition and write the output exactly as any other step, and then change nothing in the learning state: no winner draw, no adaptation (permanence, mm, ah), no branch-counter increment, no branching and no running-max update. This is the cut a frozen step makes (M.7.6, M.7.7), decided per step from the output; stateless step keys advance as on any step.

---

### M.5 Mismatch and adaptation history

**M.5.1** `[A]` On each update, the adaptation history accumulates the update weight: ah₍ₜ₎ = ah₍ₜ₋₁₎ + θₖ, with θₖ from M.4.11. *(§3: "dsⱼ.sₖ.ah₍ₜ₎ = dsⱼ.sₖ.ah₍ₜ₋₁₎ + θₖ")*

**M.5.2** `[R]` When the update runs (M.5.3), mm ← mm·(1 − μ) + μ·m·act_ratio, where m = 1 − iₖ/Iₘₓ if s.p > τ and m = iₖ/Iₘₓ otherwise, and act_ratio = segment activity / bank maximum. μ defaults to 0.001. *(ngm2 `hd_ngm2_dendrite.cpp:455-460`; ngm3 `hd_ngm3_neuron_group.cpp:709-719`)* *Author's original (§3):* target (1 − iₖ)/Iₘₓ (connected) or iₖ/Iₘₓ (not), weighted by (max_{dsp ∋ dsⱼ} dsp.a)/n.a — same mechanism (smoothed disagreement between connectivity and input, weighted by how much the segment contributed), different function.

**M.5.3** `[R]` The mismatch update runs only for synapses whose segment's act_ratio ≥ `mismatch_act_thres` (default 0.8). *Author's original (§3):* unconditional update — same mechanism, with the activity weighting turned into a gate.

**M.5.4** `[R]` The connectivity test "s.p > τ" in M.5.2 uses the permanence **after** this step's M.4.12 update. *Author's original (§3):* unspecified (M.9.6) — an ordering detail, so the code decides.

**M.5.5** `[A]` Behavioural property, testable in isolation. Take a connected synapse (p > τ) with iₖ = 0, constant Iₘₓ > 0 and ρⱼ = 1. Its mm rises monotonically towards 1/Iₘₓ. For a disconnected synapse with iₖ = Iₘₓ under the same conditions, mm rises towards 1. *(§3: "the mismatch value is high when the synapse should be connected but it isn't, or … should not be connected but it is")*

**M.5.6** `[D]` mm and ah MUST remain finite and ≥ 0 at all times. *(ngm25 arena invariant 5, `hd_ngm25_arena.h:123-124`)*

**M.5.7** `[D]` If Iₘₓ = 0, the mismatch update for that step MUST be skipped or made inert (contributing 0). It MUST NOT produce NaN or Inf. *(ngm3 `hd_ngm3_neuron_group.cpp:633`; ngm25 `hd_ngm25_response.cpp:110`; M.9.9)*

---

### M.6 Branching

**M.6.1** `[A]` Every T_b learning steps (default 5000), every dendrite of every neuron MUST be checked for ambiguous synapses. A "learning step" is a non-frozen group step whose output does not sum to 0 (M.4.15; adopted reading, M.9.10). *(§3 Branching: "In periodic intervals (e.g., after 5000 learning steps) the existing dendritic segments of each neuron should be checked")* Each neuron's branch counter therefore advances by exactly 1 per learning step (M.4.14). Within a step, branching happens after the neuron's own adaptation, so the response that triggered learning is unaffected. Because each neuron adapts once per step, it makes no difference whether a neuron branches right after its own adaptation or after all neurons have adapted.

**M.6.2** `[R]` A synapse is ambiguous when ah ≥ Θ, mm ≥ mm_thres and its segment can still branch (2·ds.id + 1 ≤ 2^(L+1) − 1), with mm_thres from M.6.3 and Θ default 2.0 (M.7.35). *Author's original (§3):* strict s.mm > mm_μ + mm_σ ∧ s.ah > Θ, with the self-contradictory depth guard of M.9.1 — same mechanism (synapses that adapted a lot and still disagree are ambiguous), different comparisons; the structural depth guard is the reading the code uses.

**M.6.3** `[R]` mm_thres = mm_μ + var(mm)·`min_mismatch_deviation` + 1/K, where mm_μ is the mean and var the variance (no square root) of mm over the dendrite's synapses (M.7.35). *Author's original (§3):* mm_μ + mm_σ (standard deviation) — same mechanism (above-average mismatch), different statistic.

**M.6.4** `[R]` A branch event fires in dendrite d if and only if |AS| ≥ ρ_b·K, with ρ_b = `min_mismatch_percentage`, default **0.002**. *Author's original (§3):* |AS| > 0.05·K — same mechanism, different threshold.

**M.6.5** `[A]` In a branch event, every s ∈ AS is removed from its segment (call its id id_s) and one copy is added to each of the segments 2·id_s and 2·id_s + 1. *(§3: "ds.S \ {s} … ds.S ∪ {s}, ds.id = id_s·2 … ds.S ∪ {s}, ds.id = id_s·2 + 1")*

**M.6.6** `[A]` If the child segments do not exist, the branch event MUST create them. Both children are created together (see M.1.11). *(§3: "If these segments are not yet existing (and max_branch_level is not reached yet) then the segments are created")*

**M.6.7** `[R]` Each synapse produced by M.6.5 gets s.p ← clamp(p_parent + U, 0, 1), with U ~ U(−0.1, 0.1) drawn independently for each child (M.7.35). Draws for apical and proximal cells with the same index MUST be independent. *Author's original (§3):* s.p ← s.p − Pois(33)/100·s.mm — same mechanism (clones start from a perturbed copy of the parent's permanence), different perturbation.

**M.6.8** `[A]` Each synapse produced by M.6.5 gets s.mm = 0 and s.ah = 0. *(§3: "s.mm = 0, s.ah = 0")*

**M.6.9** `[A]` A branch event MUST NOT modify synapses outside AS. *(§3: only "s ∈ AS" are moved)*

**M.6.10** `[A]` AS, mm_μ and mm_σ are evaluated on the state before the event. Synapses created by an event are not re-evaluated within the same event. *(§3: AS is defined as a set, then acted on)*

**M.6.11** `[A]` A branch event MUST NOT create a segment with id > 2^(L+1) − 1. *(§3 Structure bound, M.1.5)*

**M.6.12** `[D]` Branching is irreversible. No implementation removes or merges segments. *(no implementation has a segment-removal path: ngm2 `hd_ngm2_dendrite.cpp`, ngm3 `hd_ngm3_neuron_group.cpp`, ngm25 `hd_ngm25_branch.cpp`)*

**M.6.13** `[A]` After any branch event, M.1.11 still holds: every DSP contains exactly one synapse per input index. *(Consequence of M.6.5)*

---

### M.7 Extensions not in the author's spec

#### M.7.a Apical / proximal banks

**M.7.1** `[D]` Every dendrite has a type in {apical, proximal}. Each neuron MUST have ≥ 1 proximal dendrite, and MAY have 0 apical dendrites. Each routed input channel feeds exactly one bank. Routing a channel to the apical bank while apical_cnt = 0 MUST be a construction error. *(ngm2 `hd_ngm2_cfg.h:20-24`; ngm3 `hd_ngm3_neuron_group.cpp:61-62,185-191`; ngm25 `hd_ngm25_neuron_group.cpp:42-43,144-155`)*

**M.7.2** `[D]` Final neuron activity is n.a_final = clamp(A·P + ξ, 0, 1). P = clamp(max over proximal dendrites of d.a, 0, 1). A = clamp(max over apical dendrites of d.a, 0, 1), or A = 1.0 if the neuron has no apical dendrite. ξ is the baseline noise from M.7.3. With no apical dendrites and ξ ≡ 0, n.a_final equals M.3.6's n.a. *(ngm2 `hd_ngm2_neuron.cpp:93-120`; ngm3 `hd_ngm3_neuron_group.cpp:497-506`; WGSL `hd_ngm3_gpu.wgsl:289-300`; ngm25 `hd_ngm25_response.h:100-110`)*

**M.7.3** `[D]` The baseline noise ξ ~ U(lo, hi) is drawn per neuron per step. The default is [lo, hi] = [0.01, 0.05]. **Neuron baseline noise encodes the resting / non-activation state**; 0 is not the resting state. Setting [0,0] is therefore a behaviour-changing setting, not a neutral one. Configurations that want a clean or frozen probe SHOULD zero the *input* noise and keep the neuron baseline noise at its default. *(maintainer note on model semantics; ngm2 `hd_ngm2_cfg.h:66-69`; M.9.17)*

**M.7.4** `[D]` Both banks of a neuron learn with the same neuron-level weight (η-equivalent). Each bank computes its ζ normalisation (M.4.9) and mismatch ratio (M.5.2) against its **own** bank maximum. *(ngm2 `hd_ngm2_neuron.cpp:171-173`; ngm3 `hd_ngm3_neuron_group.cpp:583-586,617`)*

**M.7.5** `[D]` Bank inputs are concatenated in the declared order, in every implementation: ngm3 and ngm25 in the order of their routes, ngm2 in the order of each dendrite's `input_channels`, which must name exactly the channels mapped to that dendrite, each once, and at least one (otherwise `init_io` throws; a dendrite fed by a single channel may leave it empty). Two channel mappings onto the same channel are refused too. The order decides which input each permanence meets, and because a segment's sum is clamped at 0 after every penalised synapse (M.3.3), it changes the response even with uniform permanences. It MUST NOT depend on wiring: channel ids, the order of channel mappings, or the order in which buffers are handed over. Checkpoints older than format version 3 were written in ngm2's wiring order, which they do not record, so ngm2 refuses one if any dendrite has more than one input channel, before assigning anything; single-channel checkpoints load. Every checkpoint from format version 4 on records this order as each channel's name and width (ngm2 per dendrite, ngm25 and ngm3 per bank); ngm2 records the group-level channel name a dendrite's channel resolves to, not the dendrite's own local channel key, since a channel_mappings hop can rename it again on the way there without the dendrite's own declaration changing. A load whose record differs in name, order or width is refused before anything is assigned. The record does not cover which underlying source feeds a name: it does not conflict with the wiring-independence rule above, so swapping the sources of two equal-width channels under otherwise-unchanged names still loads. Older checkpoints have none and load unchecked, with one notice. *(ngm3 `hd_ngm3_neuron_group.cpp` `init_io`; ngm2 `hd_ngm2_dendrite.cpp` `init_input`; the record in `logging/config_record.h`; checked by the M.7.5 cases of `integration.impl_equivalence{,_gpu}`)*

#### M.7.b Frozen (inference-only) mode

**M.7.6** `[D]` `frozen` is a property of the run, not of a group: the configuration's top-level `frozen` (default false) applies to every group, and a per-group `frozen` key MUST be refused. When `frozen = true`, a step MUST compute the response, combine and inhibition exactly as a non-frozen step would, and write the output. It MUST then return without the winner draw, adaptation, branching, the branch-counter increment, or the running-max update. *(ngm2 `hd_ngm2_neuron_group.cpp:199-201`, `hd_ngm2_neuron.cpp:130-133`; ngm3 `hd_ngm3_neuron_group.cpp:511-514,528-530`; ngm3-GPU `hd_ngm3_gpu_neuron_group.cpp:361-366,560-566`, WGSL `:305`; ngm25 `hd_ngm25_neuron_group.cpp:234-248`)*

**M.7.7** `[D]` Across any number of frozen steps, the following MUST stay bit-unchanged: permanences, mm, ah, segment structure, the running max, the branch counter and the learning tunables. Only noise and penalty draws (or the stateless step key) advance. *(docs `architecture/modules/hd_ngm3.md` "Frozen Inference Mode")*

**M.7.8** `[M]` *Checkpoint versus configuration.* On `load_state` the configuration wins: every runtime tunable (learning rates, inhibition strength, thresholds, weight filter, …) takes its value from the configuration the group was built with, never from the checkpoint, whatever the value of `frozen`. A checkpoint carries the learned state (M.7.17) plus a record of the configuration it was created with. When a checkpoint is loaded under a configuration whose model settings differ from that record, the load MUST succeed and MUST emit a notification naming each differing setting with its recorded and its current value. In detail:
  - **What is compared.** Each neuron group records its own resolved parameters (after defaults), and exactly those are compared. Run-level settings (dataset paths, step counts, output paths, `frozen`, `run_id`) are not part of a group's parameters and are not compared. Structural differences still refuse the load (M.7.18).
  - **Random state is state, not a setting.** The checkpoint's RNG state (or effective seed and step key) and random-derived state — `random_index`'s matrix M (M.7.14) — win over the configuration, so a resume stays exact (M.7.17). A differing seed in the configuration is reported as not applied. `ri_reach` and `ri_strength` are settings: the configuration's values apply to the saved M.
  - **Every difference is reported,** deliberate ones included (e.g. other learning rates for a continued run), as one notification listing each setting with its recorded and current value. A checkpoint without a configuration record loads with a single notice that none was found.

  Checked by the M.7.8 cases of `integration.impl_equivalence{,_gpu}`.

#### M.7.c Future work

**M.7.9** `[FUTURE]` Partial input processing. Inputs carry timestamps t ∈ ℕ^K (or one timestamp per simultaneous subset), and older inputs are discounted through a sigmoid "integration window". The purpose is to avoid deadlock in recurrent networks. Not required. *(the author's spec §2)*

**M.7.10** `[FUTURE]` Continuous activation, i.e. leaky integration of neuron activity: state ← state + (drive − state)/τ_act, plus an activity-adaptive τ. This is on the author roadmap ("Time is not yet properly modeled … continuous activation is a planned addition"). It is **not implemented**. Not required.

#### M.7.d Inhibition

**M.7.11** `[D]` Each group owns exactly one inhibitor, chosen by `inhibition_cfg`, which names an entry in `cfg_store.inhibitions`. An empty name MUST synthesise `max_softmax` with β = `local_inhibition_strength` (default 10). A non-empty unknown name MUST be a construction error. *(`src/inhibition/inhibition.cpp:23-44`)*

**M.7.12** `[D]` The inhibitor maps the vector of final neuron activities (M.7.2) to the group output. Every implementation feeds the post-inhibition output, not r, to winner selection and adaptation. *(ngm2 `hd_ngm2_neuron_group.cpp:179,204-236`; ngm3 `hd_ngm3_neuron_group.cpp:523,534-570`)*

**M.7.13** `[D]` `max_softmax(β)` takes raw activities v. β MUST be finite and ≥ 0; anything else is a construction error. If m = max(v) (reduced from 0) is not a normal float, the output is unchanged. Otherwise vᵢ ← m·exp(−β·(m − vᵢ)), the max-rescaled softmax: the leader keeps its activity, nothing exceeds it, and β = 0 sets every entry to m. CPU and GPU MUST implement the same formula. *(`src/inhibition/max_softmax.cpp`, `src/inhibition/gpu_max_softmax.cpp`)*

**M.7.14** `[D]` `random_index` (variant `threshold_linear`) works as follows:
  - M is an |N|×|N| matrix with entries U[0,1) drawn on first use, with a zero diagonal. Its seed is `ri_seed` composed with the owning group's offsets like every entity seed (S.3.24), or, with `ri_seed_from_time`, a seed drawn at construction (no offsets). The group MUST report that seed as `io_entity::inhibition_seed()` (S.7.43). [M]
  - Tᵢⱼ = ri_strength if Mᵢⱼ < ri_reach, else 0. T has a zero diagonal.
  - out = clamp(v − T·v, 0, 1), computed from v alone so that the in-place call is alias-safe.

  ri_reach and ri_strength MUST be in [0,1]; anything else is a construction error. M MUST be persisted in state. The GPU inhibitor factory MUST reject this kind. *(`src/inhibition/random_index.cpp:29-34,53-112`; `src/inhibition/gpu_inhibition.cpp:17-21`)*

#### M.7.e Determinism, RNG, persistence

**M.7.15** `[D]` For a fixed configuration, seed and input sequence, each CPU implementation MUST produce a bit-identical trajectory (outputs and state) run-to-run, independent of the TBB thread count. Each neuron owns its RNG streams or uses stateless keys. Group-level draws (winner threshold, ngm3 branch jitter) are serial. *(ngm2 per-neuron/per-dendrite `fast_rng_t` `hd_ngm2_neuron.cpp:34,48-52`, `hd_ngm2_dendrite.cpp:113`; ngm3 `hd_ngm3_neuron_group.cpp:94-100`; ngm25 `hd_ngm25_prng.h:33-48`, `hd_ngm25_response.h:113-130`)*

**M.7.16** `[D]` RNG derivation per implementation. The derivation is part of each implementation's reproducibility contract and MUST NOT change without re-baselining its goldens. In every implementation a different `seed_offset` MUST give fresh streams, not the same streams relabelled: offset 1 MUST NOT equal offset 0 shifted by one neuron or dendrite.
  - **ngm2.** Group: `s_g = random_seed_g + off`. Neuron j (index in its group): `s_j = stream_seed(random_seed_n, j, s_g)`. Dendrite k of neuron j: `stream_seed(rnd_seed_d, k, s_j)`. `stream_seed` chains its three inputs (seeds zero-extended) through the splitmix64 finaliser and keeps the upper 32 bits. *(`hd_ngm2_tools.h` `stream_seed`; `hd_ngm2_neuron_group.cpp:46,68`; `hd_ngm2_neuron.cpp:34,48`; `hd_ngm2_dendrite.cpp:113`)*
  - **ngm3-CPU.** A master RNG is seeded with `random_seed + off`. It seeds the per-neuron RNGs and then draws the init, winner and jitter values.
  - **ngm3-GPU.** Stateless PCG keyed by (coordinate, step, seed, salt).
  - **ngm25.** Stateless PCG keyed by (seed, step, dendrite, segment, input_idx). Initialisation uses one `mt19937(seed)` drawn in arena-slot order.

  *(ngm2 `hd_ngm2_neuron_group.cpp:35,51-54`, `hd_ngm2_neuron.cpp:32,40-43`, `hd_ngm2_dendrite.cpp:92`; ngm3 `hd_ngm3_neuron_group.cpp:49,85-87`; WGSL `hd_ngm3_gpu.wgsl:227,252,299,370,594`; `hd_ngm25_prng.h:33-48`, `hd_ngm25_arena.h:157-168`)*

**M.7.17** `[D]` Save at step t, load into a freshly constructed group with the same configuration, then continue. The continuation MUST be bit-identical to an uninterrupted run. To achieve this, the following MUST be persisted: all RNG state (or the effective seed and step key), the branch counters, the running max, the full synapse state and the inhibitor state. The checkpoint also records the configuration it was created with (M.7.8). *(ngm2 `hd_ngm2_neuron.cpp:219-224`, `hd_ngm2_dendrite.cpp:688-689`; ngm3 `hd_ngm3_neuron_group.cpp:1020-1036`; ngm25 `hd_ngm25_neuron_group.cpp:457,502-508`)*

**M.7.18** `[D]` If a checkpoint's structural sizes (neuron count, dendrite counts and their apical/proximal split, tree depth, input widths) do not match the group, `load_state` MUST NOT partially apply the checkpoint. It MUST either leave the group unchanged (ngm2, ngm3, ngm25 — ngm25 reads and checks the whole checkpoint, arena consistency included, before it assigns anything) or throw. The inhibitor's part counts as part of the group's: a stateful inhibitor (random_index) checks its matrix against the group's neuron count and loads all or nothing, and ngm3 and ngm25 load it before they commit their own state. ngm2 checks every neuron's dendrite count and every dendrite's type (apical/proximal), tree depth and own shape (its synapse arrays against its tree depth and input width) in a read-only pre-pass over the whole checkpoint, before any neuron is loaded — not only once the load loop reaches the offending dendrite. A dendrite records its type and depth from checkpoint format version 5 on; a matching segment-id range is not evidence of an equal depth. An older ngm2 checkpoint is checked for depth against its configuration record (`dendrite[<cfg>].max_branch_level`), refused on a difference; its dendrite types are recorded nowhere, so it loads unchecked for them, with one notice per checkpoint. *(ngm2 `hd_ngm2_neuron_group.cpp:597,620`, `hd_ngm2_dendrite.cpp:769`; ngm3 `hd_ngm3_neuron_group.cpp:1043-1051`; ngm25 `hd_ngm25_host.cpp` `stage_checkpoint`)*

**M.7.19** `[D]` The running max mₙ ← max(0.99999·mₙ, n.a_final) is part of the persisted learning state. It is updated once per learning step: not on frozen steps (M.7.6) and not on zero-output steps (M.4.15). *(ngm2 `hd_ngm2_neuron.cpp:130-133`; ngm3 `hd_ngm3_neuron_group.cpp:511-514`; WGSL `:305-309`; ngm25 `hd_ngm25_neuron_group.cpp:234-236`)*

#### M.7.f CPU/GPU equivalence

**M.7.20** `[D]` ngm3-GPU MUST compute the same deterministic arithmetic as ngm3-CPU: stats, attenuation, segment-weight fold, θ, permanence, ah, mm, the branch predicate and the split write. The only intended difference is the PRNG: a stateless PCG instead of mt19937. The deterministic arithmetic MUST match (M.0.6 a). Statistical-aggregate parity — per-field (mean, max, std) within tolerance, plus determinism and no NaN/Inf — is only the acceptance test for the stochastic part (M.0.6 b), and it does not license a systematic shift. *(docs `architecture/decisions.md:78-90` ADR-5/MH3; `key-abstractions/heterogeneous-execution.md:121`; WGSL header `hd_ngm3_gpu.wgsl:19`)*

**M.7.21** `[D]` ngm3-GPU MUST be deterministic for a given (seed, step). The GPU step key MUST advance on every step, including frozen steps. The branch counter advances only on learning steps (M.6.1), and the GPU branch gate MUST fire on exactly the same steps as the CPU gate. *(`hd_ngm3_gpu_neuron_group.cpp:361-366`; WGSL `hd_ngm3_gpu.wgsl:615`)*

**M.7.22** `[D]` ngm25-CPU and ngm25-GPU MUST draw bit-identical random numbers, because both use the same structural-key hash. Their numerics are expected to be close, but not bit-exact. *(`design/ngm2_5_gpu.md:256-279`; `hd_ngm25_prng.h:21-27`)*

#### M.7.g ngm3 flat layout invariants

**M.7.23** `[D]` Each bank b stores permanences, segment ids, ah and mm with shape [|N| × D_b × R × K_b], where R = 2^L. rep_act has shape [|N| × D_b × R]. L MUST be in [1,4], since u8 segment ids give at most 32 segment slots. The CPU stores segment ids as u8; the GPU stores them as u32 and narrows them on readback. *(`hd_ngm3_neuron_group.h:30-34,241`; `hd_ngm3_neuron_group.cpp:64-70`; docs `hd_ngm3.md` "GPU Storage")*

**M.7.24** `[D]` *Run alignment.* For every (n, d, k) and every replica row r ∈ [0,R), seg_id[n,d,r,k] is an ancestor-or-self of virtual leaf 2^L + r. Equivalently, the rows form maximal runs. A run of segment id s at depth ℓ has length 2^(L−ℓ) and starts at row (s − 2^ℓ)·2^(L−ℓ). *(`hd_ngm3_neuron_group.cpp:813-843` split write; `:818` split_size = 2^(L − bit_width(s)))*

**M.7.25** `[D]` *Replica coherence.* All rows of one run hold bit-identical (p, mm, ah) at all times, on both CPU and GPU. A run is one logical synapse in the sense of M.1.6. *(adapt applies the same θ per segment id and input, `hd_ngm3_neuron_group.cpp:686-720`; split writes the whole run, `:826-837`)*

**M.7.26** `[D]` Branching statistics (mm_μ, mm_σ, |AS|) MUST count each run once, by skipping row r when seg_id[r] = seg_id[r−1]. *(`hd_ngm3_neuron_group.cpp:753-807`; WGSL `:632-679`)*

**M.7.27** `[D]` rep_act[n,d,r] is the activity of the DSP that ends at virtual leaf 2^L + r. Rows that share a leaf (an unsplit subtree) denote the same DSP. *(`hd_ngm3_neuron_group.cpp:451-481`)*

**M.7.28** `[D]` After `init_io`, ngm3 MUST NOT allocate or resize bank storage. Branching rewrites ids and values in place. *(`hd_ngm3_neuron_group.h:26-46`)*

#### M.7.h ngm25 arena invariants

**M.7.29** `[D]` After construction and after every branch event, the arena MUST satisfy `check_arena_consistency` invariants 1–5:
  1. CSR offsets are monotone, and all SoA columns have equal length.
  2. The directory is neuron-major, with apical dendrites before proximal ones.
  3. Synapses are in input-major order, clones are adjacent, and every input_idx lies within its type's range.
  4. Per (dendrite, input), the segment-id set is an antichain cut covering the root. This is M.1.11 restated for the arena.
  5. Segment ids are in [1, 2^(L+1) − 1], p ∈ [0,1], and mm and ah are finite and ≥ 0.

  *(`hd_ngm25_arena.h:112-127`)*

**M.7.30** `[D]` ngm25 depth L MUST be in [1,6]. Its permanence threshold MUST be in [0.01, 1). *(`hd_ngm25_neuron_group.cpp:44-45,58`)*

#### M.7.i As-built shared mechanisms (normative)

Under Authority point 2, these clauses are the normative detail behind the `[R]` clauses of M.2–M.6: they state the as-built response and learning rule that all implementations share. All implementations MUST agree on them (M.0.6).

**M.7.31** `[D]` *Stochastic low-input penalty.* For each connected synapse, add iₖ to the accumulator. Then draw u ~ U(0, I_max/2); if u > iₖ, subtract ((p − τ)/(1 − τ))·(1 − iₖ/Iˢ). The accumulator is floored at 0 immediately after each penalised synapse (the per-synapse floor). For ngm25, `penalty_policy` MAY select `expected_value` or `none`; its default MUST be `stochastic`. *(ngm2 `hd_ngm2_dendrite.cpp:273-282`; ngm3 `hd_ngm3_neuron_group.cpp:467-479`; WGSL `:200-211,231,255`; ngm25 `hd_ngm25_response.cpp:147-163`, `hd_ngm25_prng.h:82-116`)*

**M.7.32** `[D]` *Noise attenuation.* x is 1 − Hoyer sparsity of an input channel, x = 1 − (√n − L1/L2)/(√n − 1) clamped to [0,1], with x = 0 when the channel's sum is not a normal float and when n = 1 (no shape evidence; at x = 0 the gate passes the channel at ≥ 0.99 whatever avg is). The response attenuation is a = 1 − sigm((x − 0.125)/0.84, 0.25, 0.5)·(1 − sigm(avg, 0.5, 0.5)), with x and avg the minimum over the channels whose sum is a normal float and that have n ≥ 2 entries; with no such channel a = 1. Every leaf is set to clamp(leaf·a/Iˢ, 0, 1). The adapt-side attenuation judges each channel on its own: a_adapt = 1 − sigm((x + 0.25)/1.405, 0.5, 0.5)·(1 − sigm(avg, 0.5, 0.5)). Both windows are fixed constants shared by every implementation (`ngm2::response_noise_window`, `ngm2::learning_noise_window`; the ngm3 GPU kernels receive them as generated WGSL constants). The windows are final. *(`include/hd_ngm2/hd_ngm2_tools.h`; ngm2 `hd_ngm2_dendrite.cpp` get_response/adapt_synapses; ngm25 `hd_ngm25_response.cpp` build_step_ctx; ngm3 `hd_ngm3_neuron_group.cpp` build_bank_ctx; `hd_ngm3_gpu.wgsl` stats_main, bank_atten, adapt_atten)*

**M.7.33** `[D]` *As-built learning weight.* The high threshold is h = avg/2 + FLT_EPSILON. θ = clamp(w_seg·((i − h)/(1 − h) if i > h else (h − i)/h)·a_adapt, 0, 1), and p ← clamp(p(1 − θ) + [i > h]·θ, 0, 1). The leaf weights are:
  - the first leaf whose activity is ≥ bank max − ε gets synapse_weight·primary_lr;
  - every other leaf gets act·synapse_weight·secondary_lr/bank max.

  The weights and activities are then max-folded towards the root. *(ngm2 `hd_ngm2_dendrite.cpp:370-385,414-445`; ngm3 `:648-704`; WGSL `:497-559`)*

**M.7.34** `[D]` *As-built neuron weight.* win_act ~ U(mx·stochastic_win_thres, mx), where mx = max(out). The winner is chosen as in M.4.4. The winner weight is sigm(1 − out, weight_filter). Other neurons get sigm(1 − out/Σout, weight_filter)·common_lr. Both are multiplied by the envelope min(max(1 − sigm(m_run), sigm(n.a_final, w₁)), 1 − sigm(n.a_final, w₂)). *(ngm2 `hd_ngm2_neuron_group.cpp:204-236`, `hd_ngm2_neuron.cpp:163-166`; ngm3 `:534-581`; WGSL `:339-397,465-480`; ngm25 `hd_ngm25_adapt.cpp:166-219`)*

**M.7.35** `[D]` *As-built mismatch gate and branch rule.*
  - mm is updated only when act_ratio = seg_act/bank_max ≥ mismatch_act_thres.
  - The branch threshold is mm_thres = mm_μ + var(mm)·min_mismatch_deviation + 1/K, where var is a variance with no square root.
  - A synapse qualifies when ah ≥ Θ, mm ≥ mm_thres and 2·seg + 1 ≤ 2^(L+1) − 1.
  - The event fires when count ≥ K·min_mismatch_percentage.
  - Each child gets p = clamp(p_parent + U(−0.1, 0.1), 0, 1), with an independent draw per child.

  *(ngm2 `hd_ngm2_dendrite.cpp:455-461,481-506,557-564`; ngm3 `:709-719,787-843`; WGSL `:550-559,593-715`; ngm25 `hd_ngm25_branch.cpp:15-70`, `hd_ngm25_arena.cpp:64-70`)*

---

### M.8 Parameters

Config keys are JSON field names. ngm2 splits parameters across the dendrite (`d.`), neuron (`n.`) and group (`g.`) config levels. In ngm3 and ngm25, all parameters are flat per group.

| Symbol | the author's spec name / default | ngm2 key | ngm25 key | ngm3 key (CPU+GPU) | Code default | Clauses |
|---|---|---|---|---|---|---|
| \|N\| | \|N\| / — | via `g.neurons[].neuron_count` | `neuron_cnt` | `neuron_cnt` | ngm2 0 (config-defined); ngm25/3: 16 | M.1.1 |
| \|n.D\| | \|n.D\| / 3–5 | `n.dendrites[].dendrite_count` | `apical_cnt` + `proximal_cnt` | `apical_cnt` + `proximal_cnt` | ngm25/3: 0 + 4 | M.1.2, M.7.1 |
| L | d.max_branch_level / configurable | `d.max_branch_level` | `max_branch_level` | `max_branch_level` | 3 (ngm3 range [1,4], ngm25 range [1,6], ngm2 range [0,4]) | M.1.5, M.6.2, M.7.23, M.7.30 |
| τ | τ / 0.3 | `d.permanence_threshold` | `permanence_threshold` | `permanence_threshold` | 0.3 | M.1.9, M.2.1, M.3.1, M.5.2 |
| — | init Pois(50τ)/100 | (hard-coded Pois(int(100τ))) | (hard-coded) | (hard-coded) | λ = ⌊100τ⌋ = 30 | M.2.1 |
| β_sm | β (softmax) / 10 (5–15) | — (no softmax; nearest is `g.default_local_inhibition_strength`) | `local_inhibition_strength` | `local_inhibition_strength` | 10.0 (β of the default max_softmax) | M.3.7, M.7.13 |
| ϖ_c | 𝜛 common learning rate / 0.02 | `g.default_common_learning_rate` | `common_learning_rate` | `common_learning_rate` | 0.0001 | M.4.5, M.7.34 |
| α | α primary learning rate / 0.01 | `d.default_primary_learning_rate` | `primary_learning_rate` | `primary_learning_rate` | 0.01 | M.4.9, M.7.33 |
| β_sec | β secondary learning rate / 0.0002 | `d.default_secondary_learning_rate` | `secondary_learning_rate` | `secondary_learning_rate` | 0.0001 | M.4.9, M.7.33 |
| μ | μ mismatch smoothing / 0.001 | `d.default_mismatch_smoothing` | `mismatch_smoothing` | `mismatch_smoothing` | 0.001 | M.5.2 |
| Θ | Θ adaptation history threshold / 2.0 | `d.default_accumulated_theta_thres` | `accumulated_theta_thres` | `accumulated_theta_thres` | 2.0 | M.6.2, M.7.35 |
| T_b | branching interval / 5000 | `n.default_branch_interval` | `branch_interval` | `branch_interval` | 5000 (ngm3/ngm25 reject 0) | M.6.1 |
| ρ_b | branching threshold / 5% of K | `d.default_min_mismatch_percentage` | `min_mismatch_percentage` | `min_mismatch_percentage` | 0.002 | M.6.4, M.7.35 |
| — (σ multiplier, implicitly 1) | mm_μ + mm_σ | `d.default_min_mismatch_deviation` | `min_mismatch_deviation` | `min_mismatch_deviation` | 1.0 (multiplies a variance) | M.6.2, M.7.35 |
| — | (none; SPEC mm ungated) | `d.default_mismatch_act_thres` | `mismatch_act_thres` | `mismatch_act_thres` | 0.8 | M.5.3, M.7.35 |
| — | Pois(33) branch reduction | (hard-coded U(−0.1, 0.1)) | (hard-coded) | (hard-coded) | ±0.1 | M.6.7, M.7.35 |
| (s,t) of sigm | sigm defaults 0.5, 0.5 | — | — | — | 0.5, 0.5 | M.4.1 |
| (s,t) of gate | gate defaults 0.6, 0.66 | tracked (`ngm::window_tracking`) | tracked | tracked | steepness 0.57 / 0.78, edges start 0.33 / 0.66 | M.4.2, M.4.2a, M.7.34 |
| — | (none) | `g.default_weight_filter` | `weight_filter` | `weight_filter` | {0.5, 0.33} | M.7.34 |
| — | (none) | `g.default_stochastic_win_thres` | `stochastic_win_thres` | `stochastic_win_thres` | 0.8 | M.7.34 |
| [lo,hi] | (none) | `n.default_neuron_noise_range` | `neuron_noise_range` | `neuron_noise_range` | [0.01, 0.05] | M.7.3 |
| — | (none) | top-level `frozen` (the run's) | top-level `frozen` | top-level `frozen` | false | M.7.6–M.7.8 |
| seed | (none) | `g.random_seed`, `n.random_seed`, `d.rnd_seed` | `random_seed` | `random_seed` | 0 | M.2.3–M.2.5, M.7.16 |
| — | (none) | `g.inhibition_cfg` | `inhibition_cfg` | `inhibition_cfg` | "" → max_softmax | M.7.11–M.7.14 |
| — | (none) | — | `penalty_policy` | — | `stochastic` | M.7.31 |
| — | (none) | `d.type`, channel mappings | `inputs[].bank` | `inputs[].bank` | proximal | M.7.1, M.7.5 |
| — | (none; inhibitor-level) | `inhibitions.<name>.{type, strength, ri_reach, ri_strength, ri_seed, ri_seed_from_time}` | same | same | max_softmax, 10.0, 0.1, 0.5, 0, false | M.7.13, M.7.14 |

Sources: `include/hd_ngm2/hd_ngm2_cfg.h:27-95`, `include/hd_ngm25/hd_ngm25_cfg.h:48-90`, `include/hd_ngm3/hd_ngm3_cfg.h:32-73`, `include/inhibition/inhibition_cfg.h:33-46`, `include/json_cfg/{ngm2,ngm25,ngm3}_json.h`.

---

### M.9 Spec-internal defects in the author's spec (and in the `[D]` sources)

Each entry gives the two readings, the reading this spec adopts, and why. Entries about numbers or functions (e.g. M.9.2, M.9.5, M.9.6, M.9.7) are superseded by the `[R]` clauses under Authority 2.

**M.9.1 Depth guard in AS contradicts the structure bound.**
- **The defect.** §3 Structure allows L bifurcations: up to 2^L leaves and 2^(L+1) − 1 segments. §3 Branching, however, guards AS with `ds.id·2 + 1 < 2^L − 1`.
- **Reading (a), literal.** With L = 3 this allows ids 1 and 2 to branch but not id 3. The tree would be asymmetric and could never reach depth L; in general it stops about two levels short.
- **Reading (b), structural.** A child id must be a valid segment id: 2·id + 1 ≤ 2^(L+1) − 1, i.e. depth(ds) < L.
- **Adopted: (b)** (M.6.2). Reading (a) is internally inconsistent with the Structure paragraph and breaks the left/right symmetry of the heap. All four implementations use (b).

**M.9.2 β is used for two different quantities.** It denotes the softmax factor (10) and the secondary learning rate (0.0002); the Hyperparameter Summary lists both under β. **Adopted:** two symbols, β_sm and β_sec (M.0.4). The two are unrelated, so no reading is lost.

**M.9.3 α and γ change meaning between §1 and §3.** In §1, θ = α·(…)^γ·ζη, where α is a global step size and γ an exponent. §3 drops both from θ and reuses α as the primary learning rate inside ζ.
- Reading (a): §1's α and γ still apply.
- Reading (b): §3's θ is complete as written.
- **Adopted: (b)** (M.4.11). §3 supersedes §1, and in §3 α already enters through ζ.

**M.9.4 Notation collisions.**
- §3 uses k for the input width ("i ∈ [0,1]ᵏ", "|ds.S| = k") and K for the same quantity ("1/K", "0.05 K").
- k is also the synapse and input index.
- In "maxⱼ(dspⱼ.a)", j indexes both the segment and the paths that contain it.

**Adopted:** K is the width, k the index, and the explicit form max_{dsp ∋ dsⱼ} dsp.a (M.4.9, M.5.2).

**M.9.5 The high/low threshold 1/K assumes a softmaxed input** ("Given that the input is (typically) softmaxed"). For non-normalised inputs such as raw image pixels, Σi ≫ 1, so 1/K classifies almost every non-zero input as "high".
- Reading (a), literal: threshold 1/K.
- Reading (b), normalised: threshold Iˢ/K = mean(i).
- **Adopted: (a)** (M.4.10). It is the literal text, and the author's spec states the softmax precondition. The precondition gap is flagged here. All implementations use a third variant, avg/2.

**M.9.6 When the mismatch connectivity test is evaluated.** The test `dsⱼ.sₖ.p > τ` has no time index, while the ah and mm updates are indexed (t).
- Reading (a): p₍ₜ₋₁₎, the connection state that produced this step's response.
- Reading (b): p₍ₜ₎, after M.4.12.
- **Adopted: (a)** (M.5.4). The stated semantics, "the synapse is connected … and the axon is not firing despite the segment contributing", describe the state that generated the contribution. All implementations use (b).

**M.9.7 The Pois(33) reduction is underspecified.**
- (i) It is not stated whether the two clones share one draw or draw independently.
- (ii) p − Pois(33)/100·mm can fall below 0. mm can exceed 1, because (1 − iₖ)/Iₘₓ > 1 when Iₘₓ < 1; and Pois(33)/100 ≈ 0.33 ± 0.06.
- (iii) The text says the reduction depends on "the old mismatch value", but the formulas list s.p = s.p − … before s.mm = 0, which leaves the ordering unclear.

**Adopted** (M.6.7):
- an independent draw per resulting synapse, because the clones are then separate synapses;
- clamp to [0,1], to preserve the M.1.6 invariant;
- mm_old taken before the reset, as the text says.

**M.9.8 The population for mm_μ and mm_σ is not stated.**
- Reading (a): population standard deviation over every logical synapse of the dendrite.
- Reading (b): sample standard deviation, or per-segment statistics.
- **Adopted: (a)** (M.6.3). AS is defined per dendrite, and the author's spec gives no hint of per-segment statistics.

**M.9.9 Degenerate inputs.** §1 clamps Iₘₓ to at least 1/k; §3 does not. With Iₘₓ = 0, M.5.2 divides by zero, and with Iˢ = 0, M.3.4 divides by zero.
- Reading (a): import §1's floor, Iₘₓ ← max(Iₘₓ, 1/K).
- Reading (b): keep §3 as written and define the degenerate cases separately.
- **Adopted: (b).** §3 is explicit, and the floor is a §1 artifact of a different activity formula. The degenerate cases are covered by the `[D]` clauses M.3.10 and M.5.7.

**M.9.10 What counts as a "learning step" for branching.**
- Reading (a): non-frozen group steps.
- Reading (b): per-neuron adaptation calls. ngm2 and ngm25 counted these, so the winner advanced 2 per step until M.4.14 was confirmed; with one adaptation per neuron per step, (a) and (b) coincide.
- Reading (c): steps on which the neuron's ζη > ε.
- **Adopted: (a)**, excluding zero-output steps, which are not learning steps (M.6.1, M.4.15). The author's spec speaks of periodic checks of "each neuron" at a fixed interval, and (a) is the only reading under which that interval is the same for every neuron.

**M.9.11 Ties are unspecified** for dsp* (M.4.8). **Adopted:** the lowest leaf id in the lowest dendrite index. This is deterministic and matches the as-built first-index convention. The winner neuron needs no tie rule, because M.4.4 draws uniformly among the qualifying neurons.

**M.9.12 ζ's denominator when extensions are active.** In M.4.9 and M.5.2, n.a is the max over DSPs, so ρⱼ ≤ 1. With apical modulation or baseline noise (M.7.2), n.a_final ≠ max dsp.a, and ρⱼ can exceed 1 or be distorted.
- **Adopted:** use M.3.6's n.a (the pre-extension drive) in ζ and ρ, and M.7.4's per-bank maxima when there are banks. The implementations already normalise by the bank maximum.

**M.9.13 Scope of "the update might be skipped".** The author's spec does not say whether the skip covers ah and mm as well as p.
- **Adopted:** the skip may cover p and ah, both of which are functions of θ ≈ 0. It must not cover mm, which does not depend on θ (M.4.13).

**M.9.14 L = 0.** The author's spec calls max_branch_level "configurable" and gives no default or lower bound. L = 0 (no branching) is consistent with the Structure paragraph. ngm3 and ngm25 reject L = 0 and ngm2 accepts it. **Adopted:** L = 0 is a valid model configuration, and an implementation MAY reject it if its own bounds say so (M.7.23, M.7.30).

**M.9.15 The softmax and the separate inhibition.** §3 says the softmax "simulate[s] a stronger local inhibition"; no other inhibition is described. **Adopted:** in the author's spec's terms the softmax *is* the inhibition (M.3.7). The swappable inhibitors in M.7.11–M.7.14 are `[D]` extensions that sit where the softmax should be. This is recorded as a divergence, not resolved.

**M.9.16 Initial dendrite count.** "e.g., 5" (§3 intro) and "e.g., 3 to 5" (§3 Structure) are both illustrative. Not normative beyond |n.D| ≥ 1 (M.1.2).

**M.9.17 `[D]`-source conflict over neuron noise [0,0].**
- The code comments say to set `default_neuron_noise_range = [0,0]` "for a deterministic, learning-frozen evaluation pass" (`hd_ngm2_cfg.h:66-69`, `hd_ngm3_cfg.h:55-58`).
- The author-level note says [0,0] "is not a neutral 'clean' setting — it removes the substrate's resting representation" (maintainer note on model semantics).

**Adopted:** the author-level semantic (M.7.3). [0,0] is permitted but behaviour-changing.

---

## S. Headless system

Normative specification of the COAST headless core (`coast_lib` plus the `coast` executable's headless path). Model numerics are out of scope and referred to as "the model, see §M".

---

### S.0 Scope, conformance terms, build configurations

**S.0.1 Scope.** In scope: `cli/`, `app/` (`build_env`, `run_headless`), `sim_core/`, `io/`, `json_cfg/`, `run_config/`, `observe/`, `logging/`, `hdf5/`, `core/` (incl. `app_config`, `path_resolver`, `step_error`, `require`, `demand_published`, `core/gpu` seam), `probe/` (recording side only), `tools/` (`sim_builder`, `dataset_loader`, `coast_compare`), `diag/` (only where it defines CLI or headless-output contracts), state save/load/checkpoint/resume, the golden-test protocol, and the build boundary. Out of scope: all `gui_*`, `probe/observers/`, `sim_console`, `console`, `h5browse`, `coastd`, `campaign`, and model math (hd_ngm2/25/3, inhibition: §M).

**S.0.2 Conformance terms.** MUST / MUST NOT / SHALL / SHOULD / MAY are used per RFC 2119. Every clause is one testable requirement.

**S.0.3 Provenance tags.** `[C]` = repo-root `CLAUDE.md`; `[Doc]` = a page under `docs/content/` (or `SANITIZERS.md` / `docs/macos_validation.md`, cited by name); `[H]` = a contract stated in a code comment or `@decision` annotation — header, source file, CMake file or script (cited `path:line`); `[T]` = evidenced only by a test assertion (test name cited). Where sources conflict, the adopted reading is the clause.

**S.0.4 Build configurations covered.**

| Id | Configuration | Notes |
|---|---|---|
| B-full | `COAST_WITH_GUI=ON` (default) | only the `coast_lib` part and the headless path are specified here |
| B-headless | `COAST_WITH_GUI=OFF` (`cmake --preset headless`) | headless-only binary |
| G-webgpu / G-null / G-none | `COAST_GPU_BACKEND=webgpu` (default) / `null` / `none` | orthogonal to B-* |
| T | `BUILD_TESTING=ON` | test suite; off by default |
| SAN | `COAST_SANITIZER=asan\|ubsan\|tsan` | out-of-tree, one per mode |

---

### S.1 Build & module boundary

- **S.1.1** `coast_lib` MUST link only TBB and HDF5 (plus wgpu-native under G-webgpu) and MUST NOT link SDL2, OpenGL, ImGui, ImPlot or ImNodeFlow. [Doc] architecture/_index.md:29-32; [C] CLAUDE.md:258-283
- **S.1.2** The link dependency MUST point only `coast_gui → coast_lib`; no `coast_lib` target or TU may depend on `coast_gui`. [C] CLAUDE.md:290-291; [Doc] architecture/_index.md:34
- **S.1.3** A `coast_lib` TU that includes an ImGui-bearing header MUST fail to compile in every configuration (the imgui/implot/ImNodeFlow/SDL2 include roots belong to `coast_gui` only). [C] CLAUDE.md:293-295; [Doc] architecture/_index.md:39-41
- **S.1.4** A `coast_lib` TU that references a symbol defined only in `coast_gui` MUST fail to link in B-headless and in any unit test linking bare `coast_lib`. [C] CLAUDE.md:296-300; [Doc] architecture/_index.md:42-47
- **S.1.5** B-headless MUST configure and build on a machine with no SDL2 and no OpenGL installed; with `-DCMAKE_DISABLE_FIND_PACKAGE_SDL2=ON -DCMAKE_DISABLE_FIND_PACKAGE_OpenGL=ON` CMake MUST report both as unused. [C] CLAUDE.md:50-56, 69-76
- **S.1.6** In B-headless, `coast_gui` MUST NOT be created and ImGui/ImPlot/ImNodeFlow MUST NOT be compiled. [C] CLAUDE.md:50-54; [Doc] architecture/_index.md:54-59
- **S.1.7** In B-headless the embedded console MUST be forced off regardless of `COAST_WITH_CONSOLE`, without overwriting the user's cached `COAST_WITH_CONSOLE` value. [C] CLAUDE.md:53-54; [H] CMakeLists.txt:128-135
- **S.1.8** The B-headless binary's `otool -L` / `ldd` output MUST NOT list SDL2 or OpenGL. [C] CLAUDE.md:73-74
- **S.1.9** `COAST_GPU_BACKEND` MUST accept exactly `none`, `webgpu`, `null` (default `webgpu`), and any other value MUST be a CMake configure-time fatal error. [H] CMakeLists.txt:104-109; [Doc] core_gpu.md:142
- **S.1.10** For any `COAST_GPU_BACKEND ≠ none`, `COAST_WITH_GPU` MUST be defined PUBLIC on `coast_lib` and exactly one backend TU defining `make_gpu_device()` MUST be compiled. [Doc] core_gpu.md:37, 142
- **S.1.11** Under G-none no GPU TU and no backend TU MUST be compiled, and the binary MUST build and run CPU-only with no GPU dependency. [Doc] core_gpu.md:15, 150
- **S.1.12** Under G-null no wgpu dependency MUST be linked, while the agnostic `gpu_device` seam MUST still be exercised. [Doc] core_gpu.md:150
- **S.1.13** `<webgpu/webgpu.h>` MUST be included only by the WebGPU backend TU(s) under `src/core/gpu/backends/webgpu/`; no agnostic header (`gpu_device.h`, `io_buffer.h`, `gpu_io_entity.h`) may name a `WGPU*` type. [Doc] core_gpu.md:11-13, 37, 109-111
- **S.1.14** Under G-webgpu, wgpu-native v29.0.0.0 MUST be fetched as a per-platform prebuilt with a pinned SHA256 that is hard-verified at configure time. [Doc] core_gpu.md:146; architecture/decisions.md:24
- **S.1.15** The `coast` executable MUST link `coast_lib` (and `coast_gui` when present) whole-archive, so static-initializer registrars are not dropped; test executables MUST NOT require whole-archive linking. [C] CLAUDE.md:308-310; [H] CMakeLists.txt:980-1008; [Doc] testing/decisions.md:20
- **S.1.16** `BUILD_TESTING` MUST default to OFF so the default `build/` tree contains no tests. [Doc] testing/decisions.md:96-104; [H] CMakeLists.txt:12
- **S.1.17** The CMake minimum MUST be 3.24. [C] CLAUDE.md:117-118; [H] CMakeLists.txt:5
- **S.1.18** Release builds (`CMAKE_BUILD_TYPE=Release`, `-O3`) are the normative build type for every numeric/golden contract in this spec, including sanitizer builds. [C] CLAUDE.md:11, 227
- **S.1.19** The `headless` preset MUST NOT pin a compiler. [C] CLAUDE.md:69-72
- **S.1.20** `COAST_SANITIZER` MUST be single-valued (`''|asan|ubsan|tsan`) and apply its flags to every target. [H] CMakeLists.txt:43-49; [Doc] static-and-dynamic.md:69
- **S.1.21** GUI-free types consumed by the core (`vis_hints_t`, `demand_published`) MUST live in `core/`, not under `gui_*`. [C] CLAUDE.md:303-306; [Doc] architecture/_index.md:63-75
- **S.1.22** `sim_instance` MUST be constructible (config load, env build, controller, logging) without any GUI dependency. [Doc] sim_core.md:42

### S.2 CLI surface

Run-length and mode

- **S.2.1** `cli::parse` MUST behave identically in B-full and B-headless; `print_usage` MAY differ by build. [H] include/cli/parse.h:22-37; [T] `cli::parse: the run-length rule is build-independent`
- **S.2.2** `-n/--steps <count>` MUST set `steps_mode=finite`, `headless=true`, `auto_start=true`. [Doc] startup.md:22, 39; [T] `cli::parse: -n / --steps implies headless and auto_start`
- **S.2.3** `-n 0` MUST be a parse error (exit 1) and MUST NOT mean "run forever". [C] CLAUDE.md:99, 110-112; [T] `cli::parse: -n 0 is rejected, not treated as unlimited`
- **S.2.4** A non-numeric, signed, trailing-garbage or out-of-range `-n` value (e.g. `abc`, `-5`, `1e6`, `99999999999999999999999`) MUST exit with code exactly 1, print `Value for '-n'` on stderr, and start no simulation. [C] CLAUDE.md:100; [T] `b_tier.cli_bad_number`
- **S.2.5** No CLI input MUST terminate the process by signal (e.g. SIGABRT); every malformed argv MUST be a clean `parse_error` → exit 1. [H] src/cli/parse.cpp:99-113; [T] `b_tier.cli_bad_number`
- **S.2.6** `--unlimited` MUST set `steps_mode=unlimited`, `headless=true`, `auto_start=true`. [C] CLAUDE.md:97; [T] `cli::parse: --unlimited implies headless and auto_start`
- **S.2.7** `-n` and `--unlimited` MUST be mutually exclusive in either order, and the conflict MUST be reported before any value validation. [T] `cli::parse: -n and --unlimited are mutually exclusive, either order`; `cli::parse: the conflict is reported before the value is judged`
- **S.2.8** An explicit `-H/--headless` without `-n` or `--unlimited` MUST be a parse error carrying `headless_needs_length_msg`. [C] CLAUDE.md:98; [H] include/cli/parse.h:15-20; [T] `cli::parse: a headless run without a length is rejected`
- **S.2.9** In B-headless, a command line with neither `-n` nor `--unlimited` (e.g. `coast config.json`) MUST exit 1 with the same `headless_needs_length_msg` and MUST NOT start stepping. [C] CLAUDE.md:113-115; [Doc] startup.md:135; [T] `b_tier.headless_build_requires_length`
- **S.2.10** `--help` and `--init` MUST be exempt from the run-length and `--unlimited`-output rules. [T] `cli::parse: --help is exempt from the run-length requirement`; `cli::parse: --help and --init escape the --unlimited output rejection`
- **S.2.11** `-H/--headless` MUST imply `auto_start`; `-r/--run` MUST set only `auto_start`. [Doc] startup.md:39; [T] `cli::parse: -H / --headless implies auto_start`; `cli::parse: -r / --run sets auto_start only`
- **S.2.12** In B-headless every accepted run MUST be headless; the binary MUST print a one-line "built without GUI support" notice (stderr, diag `info`) unless `-H/--headless` appeared on the command line. [C] CLAUDE.md:88-90; [H] src/main.cpp:111-117; [T] `b_tier.headless_build_notice`
- **S.2.13** In B-headless `-r/--run` MUST be accepted as a no-op. [H] src/cli/parse.cpp:29-31

`--unlimited` output restrictions

- **S.2.14** Without `--save-on-term`, `--unlimited` combined with any of `--save-state`, `--emit-run-manifest`, `--log-buffers` MUST exit 1 with a message containing `cannot be combined with` and start no simulation. [C] CLAUDE.md:102-105; [Doc] startup.md:137; [T] `b_tier.unlimited_rejects_doomed_outputs`
- **S.2.15** `--unlimited --checkpoint <N>` MUST be accepted. [C] CLAUDE.md:106; [T] `b_tier.unlimited_allows_checkpoint`
- **S.2.16** With `--save-on-term`, `--unlimited` MUST accept `--save-state`, `--emit-run-manifest` and `--log-buffers`. [H] src/cli/parse.cpp:309-315; [T] `cli::parse: --save-on-term makes --unlimited outputs honest again`

Signals (`--save-on-term`)

- **S.2.17** With `--save-on-term` in a headless run, a SIGTERM MUST stop the run and exit through the normal teardown path: buffer logging closed, `--save-state` written, manifest emitted with `exit_reason=user_signal`, and one final checkpoint written into the snapshot dir when `--checkpoint` is set. [H] include/cli/options.h:65-70; src/cli/parse.cpp:309-315; include/app/run_headless.h:12-24
- **S.2.18** The SIGTERM catcher MUST be armed before `build_env` runs, and re-arming MUST NOT clear an already-latched signal. [H] src/main.cpp:128-133; include/app/run_headless.h:12-24
- **S.2.19** SIGINT MUST NOT be caught (with or without `--save-on-term`). [H] src/cli/parse.cpp:60-62, 317-320
- **S.2.20** A SIGTERM that arrives after the loop already reached its step limit MUST NOT be recorded as `user_signal`. [H] src/app/run_headless.cpp:484-490
- **S.2.21** The `--save-on-term` final checkpoint MUST NOT duplicate a checkpoint already submitted for the same step, and MUST be skipped if the sim halted on a fault. [H] src/app/run_headless.cpp:266-270, 520-522

Other flags

- **S.2.22** The config path MUST be accepted as the first positional argument or via `-c/--config`; a second positional MUST be a parse error. [Doc] startup.md:15; [T] `cli::parse: second positional after config_path set is not silently consumed`
- **S.2.23** An unknown option MUST exit non-zero with `Unknown option` on stderr. [T] `b_tier.cli_unknown_flag`
- **S.2.24** An option missing its value MUST be a `parse_error`. [T] `cli::parse: missing value throws parse_error`
- **S.2.25** Flag order MUST NOT affect the parsed result. [T] `cli::parse: flag order does not matter`
- **S.2.26** `--checkpoint` MUST accept only a non-negative whole number (same strictness as `-n`), and 0 disables checkpointing; `--seed-offset` MUST accept a whole number in [0, 2³² − 1] and refuse anything else. [T] `cli::parse: --checkpoint and --seed-offset validate the same way`, `cli::parse: --seed-offset takes [0, 2^32 - 1], the rule of core/seed.h`
- **S.2.27** `--help` MUST print usage containing `Usage:` and `--headless` to stdout, write nothing to stderr, exit 0, and take effect before any `--log-file` sink is opened. [T] `b_tier.cli_help`; `b_tier.help_before_logfile`; [H] src/main.cpp:39-45
- **S.2.28** `--init` MUST write `./coast.json` containing every app-config field at its default and exit 0, and MUST NOT overwrite an existing `coast.json` (exit 0, file untouched); a write failure MUST exit 1. [Doc] app_config.md:29-31; [H] include/core/app_config.h:88-101; src/main.cpp:51-65; [T] `b_tier.init_writes_default_coast_json`
- **S.2.29** A headless run with no config path MUST exit non-zero with `configuration file required`. [T] `b_tier.cli_missing_config`; [H] src/main.cpp:143-148
- **S.2.30** `--resume`, `--load-state`, `--save-state`, `--checkpoint`, `--log-buffers`, `--emit-run-manifest`, `--seed-offset`, `--app-config`, `--save-on-term` MUST each be accepted by the parser. [T] `cli::parse: state-related flags`; [H] src/cli/parse.cpp:224-266

Diagnostics flags

- **S.2.31** Default diagnostic threshold MUST be `info` with the stderr sink. [Doc] diag.md:82; [T] `cli::parse: default diagnostics = info level, no log file`
- **S.2.32** `--log-level` MUST accept `trace|debug|info|warn|error|off` case-insensitively and reject anything else as a `parse_error`. [Doc] diag.md:82; [T] `cli::parse: --log-level is case-insensitive`; `cli::parse: invalid --log-level throws parse_error`
- **S.2.33** `-v` MUST lower the threshold one level (clamp at `trace`); `-q` MUST raise it one level and clamp at `error`, never reaching `off`; repeated flags MUST compose in order. [Doc] diag.md:80-85; [T] `cli::parse: -q/--quiet raises threshold, clamps at error (never off)`; `b_tier.quiet_keeps_fatal`
- **S.2.34** An explicit `--log-level off` MUST be sticky against later `-q`, and a later `-v` MUST step it to `error`. [Doc] diag.md:85; [T] `cli::parse: an explicit --log-level off is sticky against later -q`
- **S.2.35** `--log-level off` MUST silence all diagnostics including fatal ones (stderr empty), except a failed `--log-file` open warning. [Doc] diag.md:85-87; [T] `b_tier.log_level_off`
- **S.2.36** `--log-file <path>` MUST append diagnostics to that file and not to stderr; `stdout`/`stderr` keywords MUST be matched case-insensitively and route to the stream (no file created). [Doc] diag.md:83; [T] `b_tier.log_file`; `b_tier.log_file_keyword_case`
- **S.2.37** A `--log-file` that cannot be opened MUST print one warning to stderr regardless of level and fall back to stderr. [Doc] diag.md:87; [H] src/main.cpp:77-85
- **S.2.38** Parse errors MUST be printed raw to stderr followed by `Use --help for usage information.`, with exit 1, before diagnostics are configured. [Doc] diag.md:121; [H] src/main.cpp:30-37

Exit codes (summary)

- **S.2.39** `coast` MUST exit 0 on: `--help`; `--init` (written or already-exists); a headless run that completes its finite step count; a `--save-on-term` SIGTERM stop that tears down cleanly. [T] `b_tier.cli_help`; `b_tier.headless_short_run`; [H] src/main.cpp:51-65
- **S.2.40** `coast` MUST exit 1 on: parse error; B-headless missing length; `--app-config` file missing/malformed; config load failure in headless mode; corrupt/unreadable `--load-state`; buffer-logging start failure; a hot-path fault halt; `--init` write failure. [H] src/main.cpp:30-37, 93-110, 135-148; include/app/run_headless.h:36-38; [Doc] threading/_index.md:186; [T] `b_tier.load_corrupt_state`
- **S.2.41** In B-full a config-load failure without headless MUST NOT exit; the GUI starts with no sim. [H] src/main.cpp:135-141; [Doc] startup.md:69

### S.3 Configuration

Application config (`coast::paths::app_config`)

- **S.3.1** The app config MUST be loaded exactly once in `main()` before `build_env`, and passed explicitly (no singleton) to `build_env` and `run_headless`. [Doc] app_config.md:19-20
- **S.3.2** Every app-config field MUST have a built-in default; with no file present the defaults apply, and a file MUST override only the keys it names. [Doc] app_config.md:24-27; [H] include/core/app_config.h:27-29; [T] `app_config: a partial file overrides only the named keys, rest stay default`
- **S.3.3** Discovery precedence MUST be `--app-config` > `$COAST_APP_CONFIG` > `./coast.json` > built-in defaults. [Doc] app_config.md:33-38; [T] `b_tier.app_config_discovery_precedence`
- **S.3.4** An `--app-config` file that is missing or malformed MUST be a hard error (exit non-zero, stderr mentions `app config`). [Doc] app_config.md:40-42; [T] `b_tier.app_config_missing_explicit_fails`; `app_config: an explicitly-named malformed file is a hard error`
- **S.3.5** A `$COAST_APP_CONFIG` or `./coast.json` that is missing or malformed MUST log a warning and fall through to the next source. [Doc] app_config.md:42-46; [T] `b_tier.app_config_stale_env_falls_through`
- **S.3.6** Default values MUST be: `config_base="<config_dir>/<config>/"`, `snapshot_dir="snapshots"`, `observe_dir="observe"`, `log_dir="logs"`. [Doc] app_config.md:128-146; [H] include/core/app_config.h:36-43
- **S.3.7** `<config_dir>` MUST expand to the config's directory (`.` for a bare filename) and only as a whole path component; a glued `<config_dir>` MUST be left literal and warned about once at app-config load. [Doc] app_config.md:54-67; [H] include/core/path_resolver.h:9-22; [T] `path_resolver: <config_dir> embedded in a larger segment is left literal`
- **S.3.8** `<config>` MUST expand to the config file stem anywhere inside a component. [Doc] app_config.md:57; [T] `path_resolver: <config> substitutes anywhere within a path component`
- **S.3.9** `config_base` MUST be expanded component-wise so a relative template stays relative and a bare-filename config never resolves to the filesystem root. [Doc] app_config.md:80-88; [T] `path_resolver: bare-filename config resolves CWD-relative, NEVER at filesystem root`
- **S.3.10** For a per-config output leaf `L`: if `L` is absolute it MUST be used verbatim; otherwise the result MUST be `expand(config_base)/L`. [Doc] app_config.md:71-78; [H] include/core/path_resolver.h:35-42; [T] `path_resolver: an absolute leaf wins over the base`
- **S.3.11** A relative leaf containing a `..` component (in `snapshot_dir`, `observe.output_dir`, or any `paths.*` default) MUST be rejected as a hard error at config load. [Doc] app_config.md:90-97; [H] include/core/path_resolver.h:44-55
- **S.3.12** `config_base` MUST apply only to config-file output leaves (`snapshot_dir`, `observe.output_dir`); CLI-typed paths (`--save-state`, `--load-state`, `--log-buffers`, `--emit-run-manifest`, `--log-file`, `--app-config`, config path) MUST be used verbatim relative to the process CWD. [Doc] app_config.md:109-116; json_cfg.md:99-115; [T] `b_tier.cli_paths_verbatim_not_rebased`
- **S.3.13** For `snapshot_dir`/`observe.output_dir`, a non-empty experiment-config value MUST win; an empty value MUST defer to the app-config default leaf. [Doc] app_config.md:118-124; [H] include/core/app_config.h:13-20; [T] `b_tier.snapshot_default_layout`
- **S.3.14** Dataset input paths (`image_file`, `data_dir`, `hdf5_file`) MUST resolve against the process CWD (relative) or as-is (absolute), never against the config directory. [Doc] json_cfg.md:99-115

Experiment config (`json_loader`, `global_cfg_t`)

- **S.3.15** `json_loader` MUST parse the whole config into `global_cfg_t` (datasets, model cfg stores, `sim` graph, `vis_hints`, `observe`, `snapshot_dir`). [Doc] json_cfg.md:24-48; [H] include/observe/observe_cfg.h:4-6
- **S.3.16** A missing config file MUST throw `std::runtime_error`; malformed JSON MUST throw a `std::exception`; in headless mode either MUST surface as exit 1 with `failed to load configuration`. [T] `json_loader: missing file throws std::runtime_error`; `json_loader: malformed JSON throws`; `b_tier.quiet_keeps_fatal`
- **S.3.17** Missing keys in `_WITH_DEFAULT` structs MUST take their C++ defaults, so `{}` loads cleanly and older configs (no `observe`/`snapshot_dir`/ngm3) still load. [Doc] json_cfg.md:52-58; [H] include/json_cfg/json_loader.h:39-40; [T] `json_loader: empty JSON {} loads, all fields take their defaults`
- **S.3.30** A config value that does not fit its C++ type (a number that, bound and serialised again, differs from the one written: wrapped, truncated, overflowed) MUST be refused at load, named by its JSON path; a key bound to nothing MUST be warned about by its JSON path, and the load continues — except a renamed key, which MUST be refused, naming its new name (the tree depth L is `max_branch_level` in every model). `[M]` [Doc] json_cfg.md (json_loader); [H] src/json_cfg/json_loader.cpp (`diff_round_trip`); [T] `json_loader: a number that does not fit its type is refused, named by its path`, `json_loader: a float beyond a 64-bit integer field is refused, its largest neighbour loads`, `json_loader: node_deltas are compared by key, not by the map's iteration order`, `json_loader: node_deltas still have their keys and values checked`, `json_loader: an unknown key is warned about by its path, and still loads`
- **S.3.18** A present field of the wrong JSON type, or an enum string outside its table, MUST be rejected at load. [H] include/json_cfg/strict_enum.h; [T] `ngm3 config: a present field of the wrong type is rejected`, `ngm3 config: an UNKNOWN bank string is rejected`
- **S.3.19** `null` dataset params MUST default rather than throw. [T] `dataset_json: null params default instead of throwing`
- **S.3.20** `sim_node_t` MUST carry `{id, label, entity_type, entity_cfg, inputs[(source_id, channel_name)]}`; `sim_node_set_t` MUST replicate template nodes with `base_id` offsets and per-node `node_deltas`. A copy MUST take its template's `entity_cfg` and `random_seed_offset` unchanged, so all copies of a template node compose the same seed (S.3.24) and run identical random streams: replicas differ only in wiring. Exception: with `ri_seed_from_time` each copy's `random_index` inhibitor draws its own seed at construction, so copies share their group seed but not their inhibition matrix. `[M]` Decorrelated replicas are separate direct nodes. [Doc] json_cfg.md (sim_node_set_t); [H] src/tools/sim_builder.cpp (node-set expansion)
- **S.3.21** `sim_builder::build_local` MUST run five ordered phases: copy direct nodes; expand node sets; create entities; init IO with the resolved input map; `env::init_io_buffers()`. [Doc] tools.md:42-48; startup.md:116-126
- **S.3.22** When any node is a GPU entity type, the GPU context MUST be installed (`env::enable_gpu()`) before entity creation and before `init_io_buffers()`. In a `COAST_GPU_BACKEND=none` build a GPU entity type has no create function, so the build of the simulation MUST fail with an error and the run exit 1. [Doc] tools.md:50; [H] include/tools/sim_builder_cfg.h:33-38; src/tools/sim_builder.cpp:122
- **S.3.23** `dataset_loader` MUST build only CPU dataset nodes (dropping neuron-group nodes, GPU dataset nodes, and node sets that reference dropped nodes) and MUST NOT create a sim thread, logging, or GPU context. [Doc] tools.md:64-86
- **S.3.24** Seeds and seed offsets are unsigned 32-bit values (`seed_t`, `seed_offset_t`). Entity seeds — including a group's random_index `ri_seed` (M.7.14) — MUST compose as config seed + global `--seed-offset` + per-node `random_seed_offset`, modulo 2³²; distinct (seed, offset) pairs MAY compose to the same seed. `io_entity::effective_seed()` MUST report the composed value, which MUST be the value the entity's RNGs are seeded with (nullopt for unseeded entities). A config seed or offset MUST be an integer in [0, 2³² − 1]; anything else MUST be refused at load. `[M]` [Doc] core.md (Random seeds); [H] include/core/seed.h; include/sim_core/io_entity.h; [T] `integration.seed_composition`, `unit.core.seed`, `json_loader: a refused seed is named by its JSON path, in every config section`

`run_config` (editable config)

- **S.3.25** `cfg::config(path)` MUST throw `std::runtime_error` if the base file is missing or not valid JSON. [H] include/run_config/run_config.h:108-110
- **S.3.26** `run_config::set(ptr, v)` MUST address fields by RFC 6901 JSON Pointer, create missing intermediate objects, and throw `std::runtime_error` on a syntactically invalid pointer; `get_*` MUST throw if the pointer is absent or of the wrong kind. [H] include/run_config/run_config.h:36-40, 49-51
- **S.3.27** `set(ptr, "literal")` MUST store a string (never a bool), and any integral argument MUST store an integer. [H] include/run_config/run_config.h:56-74
- **S.3.28** `run_config` MUST be copyable, and its public header MUST NOT expose nlohmann/json. [H] include/run_config/run_config.h:12-15, 33-34
- **S.3.29** `run_dir(d)` MUST set `observe.output_dir=d/observe` and `snapshot_dir=d/snapshots`. [H] include/run_config/run_config.h:83-87

### S.4 Simulation core

`sim_instance`

- **S.4.1** `sim_instance` MUST construct, in order, `json_loader`, `sim::env`, `sim_builder`, `sim_thread_controller`, `logging_manager`, and destroy in reverse, so the sim thread is joined before any entity or buffer is destroyed. [Doc] sim_core.md:32-40; threading/_index.md:220-234
- **S.4.2** `sim_instance` construction MUST throw `std::runtime_error` on load/initialization failure. [H] include/sim_core/sim_instance.h:31-35
- **S.4.3** The `hdf5::executor` passed to `sim_instance` MUST outlive it. [H] include/sim_core/sim_instance.h:37-41
- **S.4.4** `build_env` MUST return `nullptr` iff `opts.config_path` is empty, and MUST throw `std::runtime_error` on any setup failure (missing config, malformed JSON, unreadable `--load-state`). [H] include/app/build_env.h:16-32; [Doc] startup.md:69

`env` step semantics

- **S.4.5** A step MUST be `process()` then `swap_io()` on the sim thread, then `step_count += 1`, with `probe::set_current_step(step_count)` set before `process()`. [Doc] sim_core.md:172-180; simulation-loop.md:12-62
- **S.4.6** BARRIER 1: every entity (CPU and GPU) MUST have finished `process()` before any buffer swaps. [H] include/sim_core/sim_env.h:110-111; [Doc] gpu-dispatch.md:79-84
- **S.4.7** BARRIER 2: every buffer, including cross-device transfers, MUST have swapped before the next `process()` begins. [H] include/sim_core/sim_env.h:118-123; [Doc] gpu-dispatch.md:84
- **S.4.8** After a step ends, the read side of every `io_buffer` MUST hold that step's results, and GUI/logging/state-save MUST read only this state. [H] include/sim_core/sim_env.h:125-130
- **S.4.9** Entity processing order within a step MUST NOT affect results: CPU entities MAY run concurrently (TBB `parallel_for`), and the output MUST be bit-identical to serial order. [Doc] simulation-loop.md:107; [H] include/sim_core/sim_env.h:177-187
- **S.4.10** `env::process()` and `env::swap_io()` MUST each first apply pending hook registrations/removals, then run their pre-hooks, then their work, then their post-hooks. [Doc] sim_core.md:94-121; hook-system.md:27-46
- **S.4.11** With no GPU device, `process()`/`swap_io()` MUST behave exactly as a pure-CPU step (every GPU branch skipped). [Doc] sim_core.md:123
- **S.4.12** `env::process()`/`env::swap_io()` MUST report per-step faults by value (`std::expected<void, core::step_error>`) and MUST NOT throw on entity faults. [H] include/sim_core/sim_env.h:300-306; include/sim_core/io_entity.h:31-34; [Doc] threading/_index.md:172-180
- **S.4.13** When several CPU entities fault in one step, the reported fault MUST be the one with the smallest `entity_id`. [Doc] threading/_index.md:184; [H] include/sim_core/sim_env.h:189-193; [T] `env::process() fault reduction is deterministic: smallest entity_id wins`
- **S.4.14** `COAST_REQUIRE(cond, kind)` MUST be compiled in every build (including `NDEBUG`) and MUST return `std::unexpected` from the enclosing function on violation. [H] include/core/require.h:4-21; [T] `COAST_REQUIRE returns unexpected on the first violated condition`
- **S.4.15** Cold paths (constructors, dataset loaders, config validation) MUST use exceptions, which `build_env`/`main` catch and report. [Doc] threading/_index.md:188

`io_buffer`

- **S.4.16** Every entity MUST have exactly one output `io_buffer` keyed by its `output_id`; a duplicate writer id MUST throw in `init_io_buffers()`. [Doc] sim_core.md:64, 84; heterogeneous-execution.md:18
- **S.4.17** Consumers MUST read their producers' output from exactly one step earlier (one-step latency), for all four buffer variants. [Doc] io-buffer.md:80-93; simulation-loop.md:93-105
- **S.4.18** `swap_buffer()` MUST be a pure index flip with no side effects (no statistics). [Doc] io-buffer.md:48; sim_core.md:150
- **S.4.19** Each buffer's variant MUST be a pure function of (writer device, any GPU reader, any CPU reader): CPU/no-GPU-reader→`io_buffer_cpu`; CPU/GPU-reader→`io_buffer_cpu_to_gpu`; GPU/CPU-reader→`io_buffer_gpu_to_cpu`; GPU/no-CPU-reader→`io_buffer_gpu`. [Doc] heterogeneous-execution.md:28-35
- **S.4.20** A CPU-only config MUST select `io_buffer_cpu` for every buffer and never construct a GPU device. [Doc] sim_core.md:88; heterogeneous-execution.md:35, 116
- **S.4.21** `init_io_buffers()` MUST throw if a GPU variant is required but no GPU context is installed. [Doc] sim_core.md:86; tools.md:50
- **S.4.22** `io_buffer_gpu` MUST return empty spans from `cur_write_buffer()`/`cur_read_buffer()`; `io_buffer_gpu_to_cpu` MUST return an empty write span and a CPU read mirror. [Doc] io-buffer.md:72-78
- **S.4.23** `read_back_sync()`, `snapshot()` and `restore()` MUST NOT throw; they return `false` on size mismatch or transfer failure. [Doc] io-buffer.md:78, 101; [H] include/sim_core/io_buffer.h:46-53, 90-94

Entities

- **S.4.24** Every `io_entity` MUST implement `process()`, `get_outp_id()`, `get_outp_size()`, `get_inp_ids()`; source entities MUST return empty `get_inp_ids()`. [Doc] io-entity.md:10-36
- **S.4.25** Entities MUST receive buffers via `set_output_buffer(io_buffer&)` / `set_input_buffer(id, io_buffer&)` references, not closures. [Doc] io-entity.md:30; sim_core.md:142
- **S.4.26** Entities MUST share no state outside io_buffers (no shared RNG, no cross-entity pointers, no globals). [Doc] heterogeneous-execution.md:19
- **S.4.27** `init_structure()` MUST be called only after the entity is in its final memory location (build phase 4, after all phase-3 emplacements). [Doc] entity-element.md:55-60; [H] include/sim_core/io_entity.h:107-109
- **S.4.28** Each entity MUST execute entirely on one device; GPU entities MUST be identified by type (`dynamic_cast<gpu_io_entity*>`), not a runtime tag. [Doc] heterogeneous-execution.md:12-14, 55-67; architecture/decisions.md:46-58
- **S.4.29** `entity_vec` storage MUST keep one contiguous `vector<T>` per concrete type and expose all entities polymorphically via `as_base()`. [Doc] io-entity.md:76-95

GPU dispatch (G-webgpu/G-null)

- **S.4.30** Per `process()`, GPU entities MUST be enqueued serially, submitted with one non-waiting `submit_step()`, overlapped with CPU entity dispatch, and joined by exactly one mandatory `rendezvous_step()` before `read_probes()`. [Doc] sim_core.md:94-123; gpu-dispatch.md:56-60
- **S.4.31** The number of GPU submissions per step MUST be independent of the number of GPU entities and buffers (O(1)). [Doc] heterogeneous-execution.md:75; architecture/decisions.md:74
- **S.4.32** In `swap_io()`, cpu→gpu uploads MUST occur before `advance_generation()`, gpu→cpu readbacks after it, and all before `swap_buffer()`. `swap_io()` blocks once per gpu→cpu readback buffer. [Doc] gpu-dispatch.md:68-77; io-buffer.md:93
- **S.4.33** Each GPU entity's `step_plan` MUST be built once at construction and remain static across steps; per-step variation MUST enter only through uniform uploads and the generation flip. [Doc] gpu-dispatch.md:12; heterogeneous-execution.md:70-71
- **S.4.34** An async GPU fault (device lost, validation error) MUST surface on the sim thread at the next phase boundary and halt the run. [Doc] core_gpu.md:82-91; threading/_index.md:200; [H] include/sim_core/sim_env.h:304-305
- **S.4.35** The GPU context MUST be constructed lazily, only when the config contains a GPU entity, and destroyed after every entity and buffer. [Doc] sim_core.md:65; [H] include/sim_core/sim_env.h:151-156
- **S.4.36** GPU readback for probes MUST occur only when a probe is active, and for OBSERVE only when the group is active. [Doc] architecture/decisions.md:104-112

`sim_thread_controller`

- **S.4.37** The controller MUST start its thread in `paused` mode. [Doc] sim_core.md:36; startup.md:128-129; [H] include/sim_core/sim_thread_controller.h:48-53
- **S.4.38** Modes MUST be `paused` (no stepping, single-step on request), `throttled` (target steps/s via sleep pacing), `unlimited` (as fast as possible). [Doc] sim_core.md:156-162; [H] include/sim_core/sim_thread_controller.h:27-31
- **S.4.39** With a step limit `L > 0`, the controller MUST transition to `paused` once `step_count ≥ L`; `L = 0` MUST mean no limit. [Doc] sim_core.md:170; [H] include/sim_core/sim_thread_controller.h:190-194
- **S.4.40** On a per-step fault the controller MUST record the first fault (with its step) stickily, set `halted`, force `paused`, and refuse any later `set_mode` resume. [Doc] threading/_index.md:185-186; [H] include/sim_core/sim_thread_controller.h:215-227; [T] `controller halts sticky on a hot-path fault`
- **S.4.41** `run_quiesced(fn)` MUST run `fn` on the sim thread at the next quiescent boundary (after a full `do_step`, or while parked), block the caller, return `fn`'s result or rethrow its exception, and leave the mode unchanged. [Doc] threading/_index.md:119-125; [H] include/sim_core/sim_thread_controller.h:87-106; [T] `run_quiesced runs on the sim thread at a boundary, freezing the count`; `run_quiesced propagates return values and rethrows exceptions`
- **S.4.42** `run_quiesced` MUST wake a parked (paused) sim thread. [T] `run_quiesced wakes a parked (paused) sim thread`
- **S.4.43** Concurrent `run_quiesced` callers MUST be served one at a time in FIFO order. [H] include/sim_core/sim_thread_controller.h:107-108
- **S.4.44** `run_quiesced` MUST NOT be called on the sim thread; during teardown it MUST throw `std::runtime_error` rather than block. [H] include/sim_core/sim_thread_controller.h:108-111; [T] `run_quiesced stop-escape: teardown unblocks a queued requester`
- **S.4.45** `set_step_count` MUST be called only inside a quiesced task or before stepping begins. [H] include/sim_core/sim_thread_controller.h:181-188
- **S.4.46** Destroying a free-running controller MUST stop and join the thread cleanly. [T] `controller destruction while free-running stays clean`

Hooks

- **S.4.47** The env MUST provide `pre_process`, `post_process`, `pre_swap`, `post_swap` hooks; registration MUST be deferred and applied on the sim thread at the next `process()`/`swap_io()`. Removal from any thread other than one running a hook of that env MUST be synchronous: when `remove_*_hook()` returns, the hook MUST NOT be running and MUST never run again, and the call MUST NOT block while the sim is not stepping. Removal from inside a hook MUST be deferred to the next `process()`/`swap_io()`. [Doc] hook-system.md; [T] `unit.sim_core.hook_removal` (#235)
- **S.4.48** Hook registration and removal MUST be safe from any thread, and each `set_*_hook` MUST return a unique id. [Doc] threading/_index.md:218; hook-system.md:50-52
- **S.4.49** In steady state (no pending hook changes) the sim thread MUST NOT acquire the pending-hooks mutex. It takes the active-hooks mutex once per hook phase; that lock MUST be contended only by a hook removal in progress. The `post_swap` and barrier hooks MUST form one phase. While a removal waits for a phase, `env::hook_removal_waiting()` MUST be true, and a hook that waits for another thread MUST stop waiting when it is. [Doc] hook-system.md; threading/_index.md §3 (#235)
- **S.4.50** `post_swap` hook bodies MUST NOT let exceptions escape (they are dispatched bare). [Doc] logging-pipeline.md:115-116
- **S.4.51** The env MUST also provide `post_swap_barrier` hooks, which run after all `post_swap` hooks of the step, so that hooks can signal in one phase and wait in the next. The order of hooks within one kind is unspecified. [H] include/sim_core/sim_env.h:226-229

### S.5 Datasets / io entities

- **S.5.1** Dataset entities MUST be source entities (no inputs) that load the dataset into memory normalized to `[0,1]` floats at construction. [Doc] io.md:8, 44-47
- **S.5.2** Each step a dataset MUST write the current sample plus per-element uniform noise drawn from `noise_range`, clamped to `[0,1]`: `clamp(val + U(noise_range), 0, 1)`. [Doc] io.md:49-53; simulation-loop.md:66-75
- **S.5.3** Sequencing defaults MUST be `change_interval=15`, `noise_range=[0.0,0.05]`, `random_seed=0`, `shuffle=true`, `index_manifest=""`. [H] include/io/sequence_params.h
- **S.5.4** A sample MUST be held for `change_interval` steps, then advance; at epoch end the index MUST wrap and `current_epoch` increment. [H] include/io/sample_sequencer.h:47-50, 89-92; [T] `sequential mode is identity order with change_interval hold`
- **S.5.5** `change_interval = 0` MUST hold a static frame and never advance. [Doc] io.md:62; [H] include/io/sample_sequencer.h:92; [T] `change_interval <= 0 holds a static frame and never advances`
- **S.5.6** Sequential mode MUST present indices `0..N-1` in order and MUST draw zero values from the sequencer RNG. [H] include/io/sample_sequencer.h:58-66 (@decision DEC-001); [T] `sequential mode draws zero RNG values (golden-safety invariant)`
- **S.5.7** Shuffle mode MUST present a fresh per-epoch permutation without replacement (each index exactly once per epoch), deterministic in the seed; there MUST be no with-replacement mode. [Doc] io.md:65; [H] include/io/sample_sequencer.h:53-56; [T] `shuffle mode: each index appears exactly once per epoch`; `shuffle is seed-deterministic`
- **S.5.8** The sequencer RNG MUST be independent of the input-noise RNG, so enabling shuffle MUST NOT change the pixel-noise stream. [H] include/io/sample_sequencer.h:64-66
- **S.5.9** With `index_manifest` set, the loader MUST present exactly the listed samples (manifest order if `shuffle=false`, per-epoch reshuffle within the manifest if `true`). [Doc] io.md:73; [H] include/io/sequence_params.h:42-45; [T] `mnist_io presents samples in index_manifest order`; `shuffle mode shuffles within the index manifest only`
- **S.5.10** Manifest parsing MUST accept whitespace/comma separators and `#` comments, and MUST reject an empty manifest, duplicate indices, out-of-range indices, and an unreadable path. [Doc] io.md:73-78; [T] `load_index_manifest parses comments, commas, and whitespace`; `load_index_manifest rejects duplicate or out-of-range indices`; `mnist_io rejects a missing index_manifest file`
- **S.5.11** Two-phase loaders (CIFAR, dSprites) MUST check manifest readability before their bulk read. [Doc] io.md:78; [H] include/io/sample_sequencer.h:27-29; [T] `ensure_manifest_readable fails fast on a bad path, skips empty paths`
- **S.5.12** Dataset `save_state` MUST persist `cur_epoch`, `cur_idx`, `ci_cnt`, `seq_pos`, and in shuffle mode `seq_perm` and `seq_rng`, plus the input-noise RNG. [Doc] io.md:100-102; logging-pipeline.md:87
- **S.5.13** Saving mid-epoch and loading MUST reproduce the remaining presentation sequence exactly (sequential, shuffle, manifest-backed, across epochs). [T] `resume parity: save/load mid-epoch reproduces the remaining sequence (sequential)` / `(shuffle)`; `resume parity: manifest-backed sequencer across epochs (shuffle)`
- **S.5.14** A configuration MUST name its run with the top-level `run_id` (default `"auto"`; empty is refused). `"auto"` MUST be replaced by a fresh id: by coastd once, when it writes a job's run config (so every launch of the job is the same run), and otherwise by coast per process; `--resume`, and the GUI's resume from a snapshot, of a configuration whose `run_id` is `"auto"` MUST adopt the checkpoint's. A checkpoint MUST record its run's id (`metadata/run_id`, format version 6), and so MUST the run manifest. A dataset's `load_state` MUST restore its sequencer cursor and RNGs exactly when the checkpoint's `run_id` equals the loading run's (older checkpoints record none: fresh), and the load MUST report which it did. A dataset MUST record its resolved parameters (without the seed) in its checkpoint and, when it restores, MUST warn naming both on a difference (and not refuse). The removed keys `restore_sequence_from_checkpoint` and a group's `frozen` MUST be refused, naming `run_id` and `frozen`. [Doc] io.md (save/load); [H] include/core/run_id.h, include/io/dataset_run.h, `sim_instance::prepare_resume`; [T] `restore_sequenced_state restores exactly when the load continues the run`; `a restoring load checks the dataset's identity record; a fresh one does not read it`; `json_loader: freezing and continuation are the run's, …`; `prepare_run_config: replaces run_id "auto" …`; `b_tier.frozen_resume_continues`; `b_tier.new_subset_starts_fresh`; `b_tier.resume_adopts_run_id`
- **S.5.15** Restoring a checkpoint whose cursor does not belong to the active manifest MUST throw, and the diagnostic MUST say that a checkpoint continues its sequence only on the manifest it was saved with. [Doc] io.md:81; [T] `modern checkpoint loaded under a different manifest fails fast`; `legacy checkpoint cursor outside the manifest fails fast`
- **S.5.16** A load that restores a dataset's cursor (S.5.14) from a checkpoint whose `cur_idx` exceeds the loaded dataset's size MUST wrap the index into range (no out-of-bounds read) and the run MUST complete with exit 0. [T] `b_tier.load_smaller_dataset`
- **S.5.17** Chairs3D MUST have no sequencer and no sequencing fields. [Doc] io.md:69, 115; [H] include/io/sequence_params.h:7
- **S.5.18** `mnist_io_gpu` MUST advance the same sequencer as `mnist_io` and keep the whole dataset GPU-resident. [Doc] io.md:119-123
- **S.5.19** In shuffle mode, a checkpoint `seq_perm` whose length equals the loaded sequencer's `image_count` MUST be a permutation of `[0, image_count)`; an entry out of range or repeated MUST make `load_state` throw (naming the checkpoint corrupt) before any entry is used as an index, with or without a manifest. A `seq_perm` of another length MUST be regenerated (cross-dataset / pre-shuffle checkpoints). A refused sequencer load MUST leave the sequencer unchanged. color_mnist MUST refuse a `current_color` outside `[0, 2]`. [Doc] io.md (save/load); [H] src/io/sample_sequencer.cpp (`validate_saved_permutation`, `load_state`); [T] `load_state refuses a seq_perm entry out of range (no manifest)` / `(manifest)`; `load_state refuses a seq_perm with duplicate entries`; `a refused manifest-mismatch load leaves the sequencer unchanged`; `a seq_perm of another length is regenerated (cross-dataset load)`; `color_mnist refuses a checkpoint colour outside red/green/blue`

### S.6 State persistence

Format

- **S.6.1** A state file MUST be HDF5 with `/metadata` (`@schema_version`, `@type`, `@config`, `@timestamp`, `step_count`), `/io_buffers/<id>/data`, and `/entities/<id>/…`. [Doc] logging.md:116-123
- **S.6.2** Load MUST reject a file whose `schema_version` differs from the current one. [Doc] logging-pipeline.md:93; [H] include/logging/state_logger.h:181-182
- **S.6.3** Buffer/entity size mismatches on load MUST degrade to warnings, not failures. [Doc] logging-pipeline.md:95; io-buffer.md:101
- **S.6.4** A non-HDF5 or corrupt `--load-state` file MUST fail with exit 1 and `Failed to load state`, never a signal. [T] `b_tier.load_corrupt_state`
- **S.6.5** A state file MUST be written atomically (temp file, fsync, rename, directory fsync) with a per-writer temp name, and a failed write MUST NOT replace the last good target. A save whose extraction throws writes no file; a GPU entity's `save_state` MUST throw when a readback of its learned state fails rather than record zeros. [H] include/logging/state_logger.h:75-78; src/logging/state_logger.cpp:616-699; [T] `unit.hd_ngm3.gpu_readback_faults`, `unit.hd_ngm25.gpu_readback_faults`
- **S.6.6** The serialize step MUST close the file before its future resolves, so `file_size()` after `.get()` is final. [Doc] logging-pipeline.md:89
- **S.6.7** Saved state MUST be `coast_compare`-equivalent between the tree-replay path and the inline path; byte-identical HDF5 is NOT required. [Doc] logging.md:112; [T] `tree-driven serialize is structurally identical to the inline path`
- **S.6.8** `save_state`/`load_state` MUST capture and restore the buffer **read page** only; load MUST write the read page directly with no swap. [Doc] io-buffer.md:95-99; [H] include/sim_core/io_buffer.h:75-99
- **S.6.9** Missing parent directories of state/checkpoint outputs MUST be created. [T] `hdf5 file::create makes missing parent directories`; `run_manifest: a missing parent directory is created (like --checkpoint/--save-state)`

Threading of save/load

- **S.6.10** Runtime save MUST extract state on the sim thread inside `run_quiesced` into an in-memory tree (no HDF5), then serialize on the HDF5 executor with the tree moved (not shared). [Doc] logging-pipeline.md:85-90; logging.md:101-112; [H] include/logging/state_logger.h:155-173
- **S.6.11** Runtime load MUST parse on the executor, then apply on the sim thread inside `run_quiesced`, and call `set_step_count(restored_step)` in the same task. [Doc] logging-pipeline.md:92-97; [H] include/logging/logging_manager.h:86-90
- **S.6.12** Only `build_env`'s startup load MAY bypass `run_quiesced`, and only before the sim thread has ever stepped. [Doc] logging-pipeline.md:99-101; threading/_index.md:215; [H] include/logging/logging_manager.h:92-97

CLI save/load/resume/checkpoint

- **S.6.13** `--save-state <path>` MUST write final state after the run, at the exact final step, and the write MUST be awaited before exit. [Doc] startup.md:145-149; [T] `b_tier.save_load_roundtrip`
- **S.6.14** `--load-state <path>` MUST restore state and the step counter in `build_env`; `-n N` then MUST mean N additional steps (load at 200 + `-n 100` → `Total steps: 300`). [Doc] startup.md:151; [T] `b_tier.save_load_roundtrip`; `b_tier.ngm25_save_load_roundtrip`
- **S.6.15** `--resume` MUST load the latest `<config_stem>_*.h5|.hdf5` in the resolved snapshot dir, and MUST start fresh (not fail) if none exists. [H] src/app/build_env.cpp:20-52, 74-80; [T] `b_tier.resume_config_anchored`
- **S.6.16** If both `--load-state` and `--resume` are given, `--load-state` MUST win. [H] src/app/build_env.cpp:69-73
- **S.6.17** The snapshot dir for `--checkpoint` writes and `--resume` reads MUST come from one resolver (`resolve_snapshot_output_dir`) so they agree independent of CWD. [Doc] logging-pipeline.md:131-140; [H] include/core/app_config.h:103-112; [T] `b_tier.resume_config_anchored`
- **S.6.18** A relative `snapshot_dir` MUST land under `<config_dir>/<config>/`, an omitted one under `<config_dir>/<config>/snapshots`, an absolute one verbatim. [T] `b_tier.snapshot_dir_under_config_base`; `b_tier.snapshot_default_layout`; `b_tier.snapshot_dir_absolute_verbatim`
- **S.6.19** `--checkpoint N` MUST write a snapshot at exactly every `k·N` steps relative to the run's start step, regardless of speed (e.g. `-n 300 --checkpoint 100` → steps {100, 200, 300} exactly). [Doc] startup.md:149; [H] src/app/run_headless.cpp:250-256; [T] `b_tier.checkpoint_creates_files`
- **S.6.20** Checkpoint files MUST be named `<config_stem>_YYYYMMDD_HHMMSS_<step>.h5`. [Doc] logging-pipeline.md:117; [T] `b_tier.checkpoint_creates_files`
- **S.6.21** Checkpoints MUST extract on the sim thread from a `post_swap` hook and serialize asynchronously; all pending writes MUST be awaited before the manifest records them and before exit. [Doc] startup.md:149; [H] src/app/run_headless.cpp:258-261
- **S.6.22** Each checkpoint file MUST be complete and closed during the run (so `--unlimited --checkpoint` leaves a usable trail). [C] CLAUDE.md:106-107; [Doc] startup.md:137

Resume parity & CPU↔GPU state

- **S.6.23** A frozen model run N steps, snapshotted, run N more and snapshotted again MUST yield identical state trees except the step counter (CPU and GPU). [Doc] tier-model.md:58-63
- **S.6.24** GPU-buffer `snapshot`/`restore` and GPU-entity `save_state`/`load_state` MUST run on the sim thread (the wgpu queue owner). [Doc] threading/_index.md:210; io-buffer.md:101
- **S.6.25** `io_buffer_cpu_to_gpu::restore` MUST write both the CPU mirror and the GPU read page; `io_buffer_gpu` MUST snapshot by readback and restore by upload. [Doc] io-buffer.md:101; [H] include/sim_core/io_buffer.h:83-89
- **S.6.26** ngm2 and ngm2.5 state MUST round-trip through `--save-state`/`--load-state` preserving the step count (model schema: §M). [T] `b_tier.save_load_roundtrip`; `b_tier.ngm25_save_load_roundtrip`

### S.7 Observation & logging

Probe system (recording side)

- **S.7.1** Each `probe_instance` MUST have exactly one producer (the sim thread) and one consumer (the main/GUI thread). [C] CLAUDE.md:355; [H] include/probe/probe_instance.h:36-37, 51-52
- **S.7.2** With no active probe on an element, `probe()` MUST cost one relaxed atomic load and return. [Doc] probe.md:82-86; probe-pipeline.md:43-49
- **S.7.3** `spsc_queue` MUST be FIFO and lock-free (release on producer, acquire on consumer), default capacity 4096 with one slot reserved (4095 usable); `try_push` on a full queue MUST return false and drop the value. [Doc] probe.md:36-42; probe-pipeline.md:66-72; [T] `spsc_queue: full → try_push returns false; one slot wasted by ring`; `spsc_queue: FIFO order with simple push/pop`
- **S.7.4** Decimation `d` MUST record the 1st call and every `d`-th call after it, preserving order; `d=1` MUST record every call verbatim. [Doc] probe.md:49; [T] `probe_instance: decimation=3 records every 3rd call (first call always)`; `probe_instance: decimation drops between recorded samples but keeps order`
- **S.7.5** Each probe sample MUST carry the step set by `probe::set_current_step` before that step. [H] include/probe/probe_instance.h:29-33; [Doc] sim_core.md:176
- **S.7.6** Probe activation/deactivation MUST increment/decrement the element's `active_probe_count_`. [Doc] probe-pipeline.md:47-49, 59-64
- **S.7.7** A probe name MUST be registered in `probe_name_registry` exactly once per call site, lazily on first `PROBE` execution. [Doc] probe-pipeline.md:51-57; core.md:55-57
- **S.7.8** `entity_probe_set` MUST use a reader lock for `record()` and a writer lock for discover/activate/deactivate. [Doc] threading/_index.md:166-170

Observe

- **S.7.9** Observe MUST run as a single post-process pass on the sim thread, once per step, after all `process()` work and (on GPU) after the device is idle. [Doc] observe.md:23-25; observe-pipeline.md:10; [H] include/observe/observe_registry.h:33-46
- **S.7.10** With no observe group configured the pass MUST return immediately. [Doc] observe-pipeline.md:47; observe.md:18
- **S.7.11** Each group's `active_this_step` MUST be recomputed at pass start from (ranges, combine, prog_active, prog_single_pending), and `trigger()` MUST re-evaluate it live within the pass. [Doc] observe-pipeline.md:51-53; observe.md:112
- **S.7.12** `combine="or"` (default) MUST capture iff `range_hit OR programmatic_active`; `"and"` iff both; with no range, `"and"` MUST reduce to programmatic-only. Ranges MUST be inclusive `[start, stop]`. [Doc] observe.md:120-134
- **S.7.13** `combine` MUST be parsed case-insensitively. [H] include/observe/observe_cfg.h:30; [T] `observe config: combine string is case-insensitive`
- **S.7.14** A range that is not a 2-element `[start, stop]` array, or an unknown `combine` value, MUST be rejected with `std::runtime_error`. [H] include/observe/observe_cfg.h:46-50, 60-68; [T] `observe config: a step range that is not [start, stop] is rejected`; `observe config: an unknown combine mode is rejected`
- **S.7.15** Trigger semantics: `START` sets `prog_active`; `STOP` clears it; `SINGLE_CAPTURE` captures for the current pass only; `cond=false` is a no-op; a new group starts disarmed. [Doc] observe.md:100-110; [T] `observe trigger:*` (6 cases)
- **S.7.16** The first observation of a `(group, full_id)` MUST lock its dtype, shape and byte count; a later mismatching call MUST abort the process with `dtype/shape lock violated`, on both cache-miss and cache-hit paths. [Doc] observe.md:84; [T] `b_tier.observe_dtype_lock`; `b_tier.observe_emit_dtype_lock`
- **S.7.17** Supported value types MUST be `float`, `double`, `(u)int8/16/32/64_t`, 1-D spans, and N-D spans with explicit shape up to rank 8; any other type MUST fail to compile. [Doc] observe.md:70-82
- **S.7.18** For the N-D overload, `product(shape)` MUST equal `values.size()`. [H] include/observe/observe_accessor.h:59-60
- **S.7.19** Output MUST be one file per group at `<output_dir>/<group>.observe.h5`, truncated on creation, with `/metadata` attributes `type="observe_log"`, `schema_version="1.0"`, `created`, `run_id`. A run that continues from a checkpoint of its own (S.5.14) MUST instead continue an existing file of the same `run_id`: drop its rows for steps from the checkpoint's step on, then append; otherwise (no file, another run, unreadable) it MUST create the file afresh and log why. In a continued file, a site whose datasets do not fit it (element type, row shape) or cannot be reopened MUST start afresh in that file, and the writer MUST log why. [T] `observe writer: a continued run keeps its rows before the checkpoint and appends`; `observe writer: a continued file whose site changed type starts that site afresh`; `b_tier.frozen_resume_continues_observe`. [Doc] observe.md:137, 143-149
- **S.7.20** Value datasets MUST be flat at the file root, named by the structural path with `/` → `__`, extensible on axis 0, with a sibling `<name>__step_index` `uint64` dataset recording each row's step. [Doc] observe.md:151-156
- **S.7.21** Value-dataset chunk rows MUST be `pick_chunk_rows()`: ~1 MiB budget, at most 1024 rows, and 1024 for zero row bytes. [Doc] observe.md:153; [T] `pick_chunk_rows: *` (4 cases)
- **S.7.22** The observe output dir MUST be resolved by `sim_instance` from `observe.output_dir` (empty → app default `observe`) under `config_base`, and created lazily on first commit. [Doc] observe.md:136-137
- **S.7.23** `observe_registry::attach` MUST run once per `sim_instance` construction; `detach` MUST run in the destructor after the sim thread stops, flush and close all writers, remove the hook, and clear groups. [Doc] observe.md:158-162
- **S.7.24** Observe row appends MUST NOT block the sim thread except under executor queue back-pressure. [Doc] observe-pipeline.md:64; [H] include/observe/observe_writer.h:49-52

HDF5 executor

- **S.7.25** Every libhdf5 call in the process MUST execute on the single `hdf5::executor` worker thread. [Doc] observe-pipeline.md:61; threading/_index.md:8; [H] include/hdf5/executor.h:4-17
- **S.7.26** The executor MUST be constructed in `main()` before any HDF5 user and destroyed after all of them. [H] src/main.cpp:120-124; include/hdf5/executor.h:14-17
- **S.7.27** Jobs MUST execute in FIFO submission order; `submit_sync` MUST return the result or rethrow; an exception in an async job MUST be logged and counted without killing the worker. [H] include/hdf5/executor.h:51-60, 119-120; [T] `hdf5 executor: jobs execute in FIFO submission order`; `hdf5 executor: submit_sync rethrows the job's exception and the worker survives`; `hdf5 executor: a throwing async job is counted, not fatal`

Buffer logging (`--log-buffers`)

- **S.7.28** `--log-buffers` MUST capture every io_buffer's read page after every swap via a `post_swap` hook, batching `chunk_size` (default 1000) frames per write. [Doc] logging-pipeline.md:35-47; logging.md:55-57
- **S.7.29** For an `io_buffer_gpu` (no CPU read span) the buffer logger MUST obtain data via `read_back_sync()`. [Doc] io-buffer.md:105
- **S.7.30** The buffer-log file MUST follow the schema `/metadata` (`@type`, `@schema_version`, `@start_step`, `@config`, `@created`; `topology/{buffer_ids, buffer_sizes, buffer_names, connectivity[E×2], vis_hints[N×3]}`) and `/steps` (`step_numbers`, `timestamps`, `buffers/<id>/data[T×S]` float32 chunked+compressed; the logger writes no per-frame stats dataset). [Doc] logging-pipeline.md:49-60; logging.md:59-71
- **S.7.31** A headless `--log-buffers` run MUST print `Starting buffer logging` and `Buffer logging stopped` on stdout and produce a non-trivially populated file. [T] `b_tier.log_buffers_creates_file`
- **S.7.32** Sim-thread logging hooks MUST NOT call libhdf5 directly. [H] include/logging/buffer_logger.h:58-62

Run manifest (`--emit-run-manifest`)

- **S.7.34** Without `--emit-run-manifest` no manifest MUST be written. [T] `b_tier.run_manifest_optin`
- **S.7.35** The manifest MUST be emitted exactly once on every exit path of `run_headless` (clean, early setup failure, caught fatal), while the observe registry is still attached. [H] src/app/run_headless.cpp:144-148
- **S.7.36** The manifest MUST be JSON with `schema_version="1"`, `coast_git_sha` (40-hex or `unknown`), `coast_git_dirty`, `coast_build_type`, `seed_offset`, `n_steps_requested`, `unlimited`, `n_steps_completed`, `exit_reason`, `wall_clock_seconds`, `started_at_utc`, `completed_at_utc`, `hostname`, `observe`, `datasets`, `savestate_outputs[]`, `node_seeds[]`. [H] include/sim_core/run_manifest.h:1-8, 66-104; [Doc] sim_core.md:44-48; [T] `b_tier.run_manifest_emits`
- **S.7.37** `exit_reason` MUST be one of `completed_n_steps`, `fatal_error`, `user_signal`, `step_budget_exceeded`, each with a distinct stable label. [H] include/sim_core/run_manifest.h:24-32; [T] `run_manifest: every exit_reason has a distinct, stable label`
- **S.7.38** `n_steps_requested` MUST be read together with `unlimited`; an unlimited run MUST serialize `unlimited=true` with `n_steps_requested=0`. [H] include/sim_core/run_manifest.h:68-74; [T] `run_manifest: an unlimited run serializes unlimited=true alongside a 0 request`
- **S.7.39** Every manifest path field MUST be stored `./`-relative if at or below the manifest's directory and absolute otherwise (never `../`). [Doc] sim_core.md:50-57; [H] include/sim_core/run_manifest.h:39-46; [T] `run_manifest: path fields use the bundle frame (./ under the manifest dir, absolute outside)`
- **S.7.40** `observe.output_dir` in the manifest MUST be the resolved directory, not the raw config leaf. [H] include/sim_core/run_manifest.h:92-98; [T] `b_tier.run_manifest_emits`
- **S.7.41** The manifest MUST NOT contain `observe_outputs`. [H] include/sim_core/run_manifest.h:107-116; [T] `b_tier.run_manifest_emits`
- **S.7.42** Each `--save-state` MUST append one `savestate_outputs[]` entry whose `bytes` equals the final on-disk file size. [T] `b_tier.run_manifest_savestate`
- **S.7.43** `datasets` MUST mirror the resolved active dataset configs (paths, sequencing params, `index_manifest`), `frozen` MUST record the run's top-level `frozen` and `run_id` the run's id as it ran; `node_seeds` MUST list each seeded entity's `{id, label, effective_seed}`, plus `inhibition_seed` (the seed of its random_index matrix M, M.7.14) for an entity that has one, and only then. [Doc] sim_core.md:46-48; [T] `b_tier.run_manifest_emits`
- **S.7.44** The manifest MUST be written atomically (temp + rename); a write failure MUST return false with a diagnostic and MUST NOT throw or change the run's exit code. [H] include/sim_core/run_manifest.h:121-124; [T] `run_manifest: an unwritable output path returns false without throwing`

Headless output contract (diag)

- **S.7.45** `run_headless` MUST emit `@coast/progress v1 phase=<p> step=<u64> start=<u64> target=<u64> sps=<f64> logged=<u64>` lines on stdout for phases `start|step|checkpoint|done|failed`, alongside (not instead of) human narration. [Doc] diag.md:93-100; startup.md:147
- **S.7.46** Progress lines MUST be flushed per line, MUST NOT be suppressible by a flag, and MUST NOT be redirected by `--log-file`. [Doc] diag.md:102-103
- **S.7.47** Progress parsers MUST ignore unknown trailing `key=value` tokens. [Doc] diag.md:104; [T] `diag::progress: tolerates leading whitespace, CRLF, and unknown trailing keys`
- **S.7.48** Supervisors MUST treat the exit code as authoritative: early setup failures exit non-zero without any progress line. [Doc] diag.md:101
- **S.7.49** Human stdout narration MUST be line-flushed (`std::unitbuf`) under redirection. [Doc] diag.md:121; [H] src/app/run_headless.cpp:72
- **S.7.50** A finite run MUST stop at exactly its start step + N. A completed finite run MUST print `Simulation completed` and `Total steps: <final>` on stdout. [T] `b_tier.headless_short_run`; `b_tier.save_load_roundtrip`
- **S.7.51** Diagnostic records MUST go through `diag` as `[LEVEL][category] message`, serialized by one mutex across threads; a `COAST_*` call MUST never throw. [Doc] diag.md:46, 57-61, 65; [T] `diag: a throwing sink never escapes a bare COAST_* call`
- **S.7.52** On a hot-path fault, headless MUST report the fault on stderr, emit `phase=failed`, and exit 1. More generally, after `phase=start` a run that exits 1 (fault, failed buffer log or `--save-state`, any exception, std or not) MUST end its feed on exactly one `phase=failed` line with no progress line after it, and MUST NOT emit `phase=done` (#203, #233). [Doc] threading/_index.md:186; diag.md:101; [H] src/app/run_headless.cpp; [T] `b_tier.failed_run_terminal_progress`; `integration.headless_feed_exception`; `integration.headless_feed_nonstd_exception`

Cross-language observe contract (`tests/integration_observe/observe_contract.json`)

- **S.7.53** The repository MUST keep exactly one checked-in, machine-readable observe contract, `tests/integration_observe/observe_contract.json`. Each entry names one observe site with `name`, `group`, `site`, `kind`, `pattern`, `dtype`, `shape` and `backends`. It is the single source of truth: consumers MUST load it rather than restate it, and no second hand-maintained copy of the site list may exist. [Doc] observe.md:107
- **S.7.54** For every entry with `"cpu"` in `backends` and `group == "population"`, a CPU run of each model that emits it MUST write a dataset with that site name, exactly the contract's `dtype`, and the resolved width (`group_width` → the group's `neuron_cnt`, `scalar` → 1). The integration test MUST fail on any mismatch and MUST run in the default ctest selection. [T] `integration.observe_contract_{synthetic,ngm3,ngm25}`
- **S.7.55** Every contract change — adding, removing, renaming or retyping an entry — MUST land together with the matching emitter change and test update, so the contract, the C++ emitters and the consumers never disagree on `main`. [Doc] PR #131 body ("keeping the Julia readers and the C++ emitters in lockstep")
- **S.7.56** The contract test SHOULD also check that each `pattern` matches its own `site`, and SHOULD resolve `input_width` to the configured input width rather than skip it.
- **S.7.57** The contract SHOULD state which model emits each site, because `backends` alone over-claims: `out_pre_inhibition` is `["cpu"]`, yet ngm2 on CPU does not emit it. Until GPU entries exist (`backends` never contains `"gpu"`), a consumer asking a GPU run for a CPU-only site MUST get a typed error, never an empty or stale series.

### S.8 Determinism & reproducibility

- **S.8.1** Given identical config, `--seed-offset`, toolchain, platform and build options, a CPU run MUST reproduce its `io_buffer` state within A-tier tolerance (rel 1e-6, abs 1e-9) against goldens from the same toolchain. [C] CLAUDE.md:77-86, 194; [Doc] golden-protocol.md:76-82
- **S.8.2** A B-headless binary MUST produce byte-identical simulation state to the B-full binary from the same toolchain. [C] CLAUDE.md:67; [Doc] architecture/_index.md:58-59
- **S.8.3** The GPU ngm3 path MUST be bit-deterministic run-to-run on the same hardware/backend, so exact `h5diff` comparison is valid. [C] CLAUDE.md:245-246; [Doc] architecture/decisions.md:100
- **S.8.4** GPU randomness MUST be a stateless hash keyed by entity coordinates and step, deterministic per `(seed, step)`. [Doc] architecture/decisions.md:92-100; core_gpu.md:128
- **S.8.5** CPU↔GPU equivalence is statistical-aggregate only; per-element bit-equivalence between CPU and GPU is NOT required. [Doc] architecture/decisions.md:78-88; heterogeneous-execution.md:121
- **S.8.6** `mt19937_simd` MUST produce sequences identical to `std::mt19937` for the same seed on every ISA path (AVX-512/AVX2/NEON/scalar). [Doc] core.md:61-71
- **S.8.7** CPU entity dispatch order and partitioning MUST NOT change results (bit-identical). [Doc] simulation-loop.md:107; [H] include/sim_core/sim_env.h:185-187
- **S.8.8** Fault selection MUST be scheduling-independent (smallest `entity_id`). [Doc] threading/_index.md:184
- **S.8.9** Goldens generated under one compiler or platform MUST NOT be used as the reference for another (A-tier fingerprints diverge ~99% across toolchains and across macOS↔Linux). [C] CLAUDE.md:77-86, 121-123
- **S.8.10** Non-Linux goldens MUST be stored under `goldens/<platform>/` (macOS: `goldens/macos/`, gitignored), and rotation MUST be scoped to that directory so a non-Linux regen never prunes Linux goldens. [H] scripts/update_golden.sh:15-19, 92-93
- **S.8.11** Sanitizer builds MUST use `Release`; A-tier tests are NOT expected to pass under sanitizers. [C] CLAUDE.md:227-233; [Doc] static-and-dynamic.md:79-97; SANITIZERS.md:90-114
- **S.8.12** In sequential dataset mode the sequencer MUST draw no RNG values (golden-safety). [H] include/io/sample_sequencer.h:58-66
- **S.8.13** `--seed-offset k` MUST be added to every seeded entity's config seed (modulo 2³², S.3.24). [H] include/tools/sim_builder_interface.h:33-39 (signature/contract); [Doc] sim_core.md:48; [T] `b_tier.run_manifest_emits`

### S.9 Threading model

- **S.9.1** A headless run MUST use the main thread (CLI, setup, polling, probe-consumer, orchestration), one dedicated sim thread (all `process()`/`swap_io()`/hooks/quiesced tasks/GPU work), the HDF5 executor thread, and the TBB worker pool for CPU entity dispatch. [Doc] threading/_index.md:8, 76-99; [C] CLAUDE.md:321-325
- **S.9.2** All wgpu interaction MUST happen on the sim thread. [Doc] threading/_index.md:196-198, 202-208
- **S.9.3** Controller atomics MUST use the orderings tabulated in threading/_index.md:107-117 (mode/target/limit/single-step release→acquire; step_count/actual_sps relaxed write, acquire read). [Doc] threading/_index.md:103-117
- **S.9.4** No thread other than the sim thread MAY access entity internal state, except through hooks, SPSC queues, or `run_quiesced`. [Doc] threading/_index.md:214-215
- **S.9.5** `set_mode(paused)` MUST NOT be relied on for quiescence; state access MUST use `run_quiesced`. [Doc] threading/_index.md:121, 215
- **S.9.6** `run_headless` MUST set step limit `start + max_steps` for finite runs and no limit for `--unlimited`, and MUST run the controller in `unlimited` speed mode in both cases. [Doc] startup.md:142
- **S.9.7** `run_headless` MUST poll the step count with adaptive intervals (100 ms → 10 ms → 1 ms approaching the limit), updating probe observers each tick. [Doc] startup.md:143
- **S.9.8** `--resume`/`--load-state` MUST be handled by `build_env`, so `run_headless` starts at the restored step. [Doc] startup.md:151
- **S.9.9** `run_headless` MUST NOT throw and MUST return 0 on success, non-zero on error. [H] include/app/run_headless.h:36-38
- **S.9.10** The GPU device context, if any, MUST be destroyed last, after all entities and buffers release GPU handles. [Doc] sim_core.md:65; [H] include/sim_core/sim_env.h:151-156

### S.10 Test & golden protocol

Tiers

- **S.10.1** The suite MUST provide Unit (`tests/unit/<module>/`), A-tier (`tests/integration/<test>/`, in-process via `coast::app::build_env`, io_buffer fingerprint vs goldens) and B-tier (`tests/integration_b/`, subprocess against the built `coast`) tiers. [C] CLAUDE.md:179-183; [Doc] testing/_index.md:12-24; tier-model.md:22-77
- **S.10.2** B-tier MUST assert on real exit codes, stdout/stderr and file side effects of the built binary, distinguishing exit 1 from signal death. [Doc] tier-model.md:69-77; [T] `b_tier.cli_bad_number`
- **S.10.3** The B-tier harness MUST clear `$COAST_APP_CONFIG` for the whole suite. [T] tests/integration_b/test_b_tier.cpp `main()` (`::unsetenv("COAST_APP_CONFIG")`)
- **S.10.4** Tests labelled `gpu` MUST be excludable via `ctest -LE gpu`; GPU A-tier tests MUST be skipped unless `COAST_GPU_BACKEND=webgpu`. [H] tests/integration/cmake/coast_add_integration_test.cmake:15-17, 28-29; [Doc] SANITIZERS.md:16-25
- **S.10.5** B-headless tests MUST be the full suite minus `unit.gui_plot.*`, `unit.console.*` and the 10 `b_tier.simc_*` cases. [C] CLAUDE.md:152-157
- **S.10.6** A reported pass count MUST be quoted with its build options (`COAST_WITH_GUI`, `COAST_GPU_BACKEND`, `COAST_WITH_CONSOLE`), since `GPU_BACKEND=none` compiles no GPU tests. [C] CLAUDE.md:159-175
- **S.10.7** GPU tests run in an environment without a GPU adapter fail with `adapter request failed`; this MUST be treated as an environment failure, not a code failure. [C] CLAUDE.md:237-243

Golden blobs

- **S.10.8** An A-tier blob MUST be JSON whose leaves are scalars or 1-D numeric arrays; arrays may be externalized as binary sidecars referenced by `{"binary","dtype","shape"}`. [Doc] golden-protocol.md:10-38; testing/decisions.md:36-44
- **S.10.9** Golden files MUST be named `YYYYMMDD_<short-sha>.json` (lexicographic = chronological). [C] CLAUDE.md:185, 194; [Doc] golden-protocol.md:49-59
- **S.10.10** Golden capture MUST retain the newest 5 `*.json` per test (per platform dir), dropping older ones automatically, and each retained golden MUST keep its own binary sidecars. [C] CLAUDE.md:194; [Doc] golden-protocol.md:61-65; [H] scripts/update_golden.sh:92-104
- **S.10.11** A test MUST pass iff its blob matches any retained golden within tolerance. [C] CLAUDE.md:194; [Doc] golden-protocol.md:67-69; [T] `compare: matches second-of-three goldens chronologically`
- **S.10.12** Re-baselining MUST be done via `coast_update_goldens[_<test>]` targets, never by hand-editing goldens. [C] CLAUDE.md:185-192; [Doc] golden-protocol.md:65
- **S.10.13** Self-comparing tests (`frozen_inert`, `frozen_inert_gpu`) MUST have no goldens. [Doc] tier-model.md:58-63

Tolerance

- **S.10.14** A scalar MUST pass iff `|a−b| < max(rel·max(|a|,|b|), abs_floor)`, defaults `rel=1e-6`, `abs_floor=1e-9`. [C] CLAUDE.md:200; [Doc] golden-protocol.md:76-82; [H] tools/coast_compare/compare.h:15
- **S.10.15** Per-field overrides MUST be read from `tests/integration/<test>/tolerances.json` (`default`, `overrides`), keyed by exact path with `[i]` index suffixes stripped. [C] CLAUDE.md:200-205; [H] tools/coast_compare/compare.h:40-50; [T] `tolerance_set: array-index suffix strips before lookup`
- **S.10.16** Overrides SHOULD be added only after a real rerun trips the default. [C] CLAUDE.md:207; [Doc] testing/decisions.md:84-92
- **S.10.17** Strict mode: any single array element over tolerance MUST fail the whole field. [Doc] golden-protocol.md:97-101; testing/decisions.md:60-68; [T] `compare: array strict: any element fail → field fail`
- **S.10.18** Aggregate mode MUST reduce each numeric array to (mean, max, std) and compare those, with tolerance precedence `override[path].<stat>` → `override[path].base` → `aggregate.<stat>` → `aggregate.base` → strict default. [Doc] golden-protocol.md:103-120; [H] tools/coast_compare/compare.h:21-25, 58-64; [T] `aggregate: tolerances.json aggregate block parses with full precedence`; `aggregate: no aggregate block → base falls back to strict default`
- **S.10.19** Aggregate mode MUST be used only for CPU-vs-GPU ngm3 tests; `mnist_ngm3_philox` MUST be strict. [Doc] golden-protocol.md:122-133

`coast_compare` report & exit codes

- **S.10.20** `coast_compare` MUST exit 0 on a match, 1 when no golden matches within tolerance, 2 on a structural error. [Doc] golden-protocol.md:150-158; [H] tools/coast_compare/compare.h:123; [T] `exit_code_of: kind → code mapping`
- **S.10.21** Structural errors MUST include a missing goldens dir, empty goldens dir, missing field, extra candidate field, array length mismatch, dtype mismatch, NaN in candidate or golden, and a missing sidecar. [C] CLAUDE.md:197; [Doc] golden-protocol.md:156; [T] `compare: no goldens directory → structural error`; `compare: extra field in candidate → structural error`; `compare: NaN in golden scalar → structural error`; `compare: binary sidecar missing file → structural`
- **S.10.22** On no match the report MUST name the closest golden (fewest failing fields, then smallest max relative delta, preferring structurally clean) and the number of goldens considered. [Doc] golden-protocol.md:141-143; [H] tools/coast_compare/compare.h:110-112; [T] `compare: no match → closest by failing_count then rel_delta`
- **S.10.23** Report lines MUST be `STRUCT  <path>  <reason>` and `FAIL    <path>  golden=X  new=Y  Δ=%  tol=…`, severity-sorted with highest Δ first; `--verbose` MUST append PASS entries. [C] CLAUDE.md:196-198; [Doc] golden-protocol.md:139-160

Quality gates

- **S.10.24** `coast_tidy` MUST only report (never gate the build), using the `.clang-tidy` rule subset, with third-party code excluded. [C] CLAUDE.md:211-217; [Doc] static-and-dynamic.md:12-59
- **S.10.25** Any first-party sanitizer report MUST be treated as a failure; suppressions MAY cover third-party code only. [Doc] SANITIZERS.md:132-133
