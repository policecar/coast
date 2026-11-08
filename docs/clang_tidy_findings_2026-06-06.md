# clang-tidy findings — `feature/ngm3` @ `ceea36f`

Date: 2026-06-06
Tool: `cmake --build build --target coast_tidy` (clang-tidy, rule subset from `.clang-tidy`)
Result: **15 warnings, 0 errors.** Exit code 0.

> Note on the committed baseline: `.clang-tidy-baseline.txt` lists 66 warnings, but its
> paths point at a *different* checkout (`/home/jk/dev/code/cimc-coast/...`) and reference
> files (`mnist_db.cpp`) that don't warn in this tree. It predates this branch state, so the
> numbers below are reported in absolute terms, **not** as a delta against that baseline.

Severity is my judgement, with the reasoning and the assumption it rests on made explicit so
you can overrule it. TL;DR ranking: nothing here is a confirmed bug in shipped code. The 4
rounding warnings are false alarms given the input range; the 10 optional-access warnings are
in a dev-only debug executable; the 1 static-init warning flags an idiom used 7× across the
codebase.

---

## Finding 1 — `bugprone-incorrect-roundings` ×4 — `src/gui_core/config_editor.cpp:1806-1809`

```cpp
// Helper: Edit RGBA color stored as int (0-255) using ImGui ColorEdit4
static bool ColorEdit4FromInts(const char* label, int& r, int& g, int& b, int& a, ...)
{
    float col[4] = { r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f };
    if (ImGui::ColorEdit4(label, col, flags)) {
        r = static_cast<int>(col[0] * 255.0f + 0.5f);   // <-- ×4 (r,g,b,a)
        ...
```

**What the check says:** casting `(double + 0.5)` to `int` rounds incorrectly for *negative*
inputs (truncation-toward-zero makes `-0.5 + 0.5 = 0`, not `-1`), and loses a little precision
versus `std::lround`.

**My severity: very low / effectively a false positive — cosmetic only.**
- Reasoning: `col[]` is written by `ImGui::ColorEdit4`, which clamps its output to `[0,1]`.
  So `col[i] * 255.0f` is always in `[0, 255]` — **non-negative**. The negative-input failure
  mode the check warns about cannot occur here, and for non-negative values `(x + 0.5)`
  truncation *is* correct round-half-up.
- **Assumption this rests on:** that ImGui clamps `ColorEdit4` output to `[0,1]`. This is the
  documented behavior; if a future flag (e.g. HDR color edit) lifted the clamp, negative or
  >1 values could appear and the warning would become real. Low likelihood.
- Suggested fix (optional, silences the check and is marginally more correct on the boundary):
  `r = std::lround(col[0] * 255.0f);` (needs `#include <cmath>`).

---

## Finding 2 — `bugprone-throwing-static-initialization` ×1 — `src/gui_vis/.../ngm3_tree_vis.cpp:1243`

```cpp
static ngm3_tree_vis_gpu_registrar tree_gpu_registrar_instance;
```

The registrar's constructor calls `get_vis_registry().register_entity_vis(info, …)`, which can
in principle throw (map insertion / allocation). An exception escaping a static-storage-duration
constructor calls `std::terminate`.

**My severity: low — flags an established, codebase-wide idiom.**
- Reasoning: this self-registration-via-static-ctor pattern is used **7 times** across
  `src/gui_vis/` (vec_ring_buffer, ngm_flat, ngm_tree_v2, ngm3_flat ×2, ngm3_tree ×2). It's
  the documented `vis_registry` factory mechanism (see CLAUDE.md → "Factory pattern" and the
  `-Wl,--whole-archive` note). If this throw path were live, every registrar would terminate
  the process at startup — it doesn't, because `register_entity_vis` doesn't throw under normal
  operation (the registry map is sized for a handful of entries and the program is already OOM-
  dead if a few-byte insert fails).
- Curiosity worth noting honestly: only the **GPU** registrar at line 1243 trips this; the
  non-GPU one at line 1210 in the same file does not. I did not chase down why clang-tidy
  flags one and not the other — likely a quirk of which translation path the check fully
  analyzes. It does **not** change the severity assessment.
- **Assumption this rests on:** that `register_entity_vis` does not throw in practice. True
  today; would change only if the registry grew a throwing validation path.
- Suggested fix (optional): wrap the registry call in a `try { … } catch (...) {}` inside the
  registrar ctor, or mark it `noexcept` and accept terminate-on-failure explicitly. Given the
  idiom is repo-wide, the cleaner move is a project decision (suppress the check, or fix all 7
  consistently) rather than a one-off here.

---

## Finding 3 — `bugprone-unchecked-optional-access` ×10 — `src/hd_ngm3/hd_ngm3_gpu_probe_main.cpp`

Lines 120, 129, 140, 141, 157, 162, 175, 176, 179, 307. Representative:

```cpp
(*ng.value())[0].init_io({ { "in_proximal", { kIdMnist, kImageSize } } });   // 120
auto& g = (*ng.value())[0];                                                   // 157, 307
auto out = (*env.get_io_buffer(kIdNgm3))->cur_read_buffer();                  // 179
```

`ng` / `mn` / `get_io_buffer(...)` return `std::optional` (or `optional`-like) and are
dereferenced with `.value()` / `*` without first checking `has_value()`. If empty, `.value()`
throws `bad_optional_access` (or `*` is UB depending on the type).

**My severity: low in context — this is a dev-only debug harness, not shipped code.**
- Reasoning: `hd_ngm3_gpu_probe_main.cpp` defines its own `int main()` (line 86) and is built
  as the standalone executable **`coast_ngm3_gpu_probe`** (CMakeLists.txt:602). It is **not**
  part of `coast_lib` and does not run inside the GUI/headless app or the test suite. It's a
  GPU bring-up / probe tool run by hand during development. An empty optional there means the
  dev's setup is wrong and an immediate abort is an acceptable (even desirable) failure mode.
- **Assumption this rests on:** that you treat `*_probe_main.cpp` / `*_smoke_main.cpp` as
  throwaway bring-up harnesses rather than maintained tools. If any of these graduate into the
  shipped path or CI, these 10 become real "check-before-deref" obligations.
- If you *do* want them clean: replace `(*ng.value())` with an explicit
  `assert(ng.has_value())` (documents the precondition and keeps the crash) or an early
  `if (!ng) { std::cerr << …; return 1; }`. Mechanical, ~10 sites, no logic change.

---

## Suggested disposition

| Finding | Count | Severity | Recommended action |
|---|---:|---|---|
| incorrect-roundings (config_editor) | 4 | very low | `std::lround` if you want zero warnings; otherwise leave — input is clamped |
| throwing-static-init (ngm3_tree_vis) | 1 | low | project-level decision (suppress vs. fix all 7 registrars uniformly) |
| unchecked-optional-access (probe main) | 10 | low (dev tool) | `assert`/early-return if you maintain the probe tool; else ignore |

None block correctness of the shipped `coast` binary or the test suite. If you want a clean
`coast_tidy` run, the cheapest path to zero is: `std::lround` (4), `assert` the optionals (10),
and either suppress or guard the one static-init site.
