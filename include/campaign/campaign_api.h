#ifndef COAST_CAMPAIGN_API_H
#define COAST_CAMPAIGN_API_H

// Query a directory of simulation runs by the hyperparameters that were swept.
//
// A "campaign" is a directory tree of run directories, as coastd writes them: one
// directory per run holding the effective `config.json`, a `coast_runtime.json`
// manifest, and the run's HDF5 output (observe/, snapshots/, final_state.h5). The
// meta-loader scans such a tree, discovers the swept parameters BY DIFFING the
// configs against each other, and hands out run handles that can be filtered by
// those parameters:
//
//     auto c   = camp::open("/mnt/NAS/coast-experiments/basic/ngm25_gpu/LRI_sweep");
//     auto sel = c->where("primary_learning_rate", 0.01)
//                  .where("local_inhibition_strength", 5.0);
//
// Discovery-by-diff is what makes short names usable at all. A coast config carries
// defaults for EVERY model family, so a swept leaf name also occurs in the sections
// that were never touched — `primary_learning_rate` exists under both ngm25 and ngm3
// in a file where only the ngm25 copy was ever swept. (ngm2 spells its copy
// `default_primary_learning_rate`, so it does not collide.) Since only
// leaves that actually DIFFER become axes, the dead copies never enter the namespace
// and the ambiguity dissolves. Where it does not dissolve, names are qualified from
// the root section down (`ngm25/primary_learning_rate`), and a full JSON Pointer is
// always accepted in place of a short name.
//
// STRICT LAZINESS. A campaign's observe files are routinely gigabytes each (2.0 GB x
// 81 runs on the reference campaign), so the scan reads configs and manifests ONLY —
// no HDF5 file is opened until a run handle is explicitly asked for one, and handles
// are not cached, so a loop over a campaign holds one file open at a time.
//
// This header is an interpreter PARSE BOUNDARY (the clang-repl console parses it), so
// — like run_config.h and coastd_console_api.h — it is kept lean: nlohmann/json is NOT
// exposed, the scan result is held by pimpl, and the implementation lives in
// campaign_api.cpp, compiled by the host and reached via the JIT symbol table
// (coast_lib is whole-archived + -rdynamic).
//
// Error model: every failure (unreadable root, unknown axis, no axis value near the
// requested one, wrong value kind) throws std::runtime_error with a message the REPL
// surfaces.
//
// A malformed file inside ONE run is not a campaign-level failure — but it is never
// silently absorbed either. Nothing is coerced to a default to make a bad file look
// good: a field that is present with the wrong JSON type is an ERROR, reported
// through campaign::issues() with the file, the field and the offending value. (A
// field that is merely ABSENT is not an error — the manifest schema is additive, so
// an older or newer writer legitimately omits keys.) A run is dropped only when its
// config.json is unusable, since without it the run has no parameters to query at
// all, and even then it is reported rather than vanishing.

#include "h5browse/console_api.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace coast::campaign {

struct scan_options; // defined below; open() takes it by const&

// ---------------------------------------------------------------------- value
//
// One config leaf's value. `absent` is a first-class kind: when a root spans several
// campaigns whose configs have different key sets, a leaf missing from some runs is a
// distinct value of that axis rather than a scan failure.
struct value {
    enum class kind : std::uint8_t { absent, number, text, boolean };

    kind        k    = kind::absent;
    double      num  = 0.0;   // kind::number
    bool        flag = false; // kind::boolean
    std::string str;          // kind::text

    // Human-readable rendering, valid for every kind ("<absent>", "0.01", "true", …).
    // Used for axis columns in the GUI and in error messages.
    [[nodiscard]] std::string label() const;
};

// Exact equality, deliberately: this answers "is this the same discovered value",
// which is how an axis's distinct set is built and how a GUI matches a run against a
// selected value. It is NOT the query-time comparison — see selection::where(), which
// is tolerance-based because the user states a value rather than echoing a stored one.
[[nodiscard]] bool operator==(const value& a, const value& b);
inline bool        operator!=(const value& a, const value& b) { return !(a == b); }

// ----------------------------------------------------------------------- axis
//
// A config leaf that differs across the campaign — discovered, never declared.
struct axis {
    std::string        name;    // short unambiguous name, e.g. "primary_learning_rate"
    std::string        pointer; // full JSON Pointer into the config
    std::vector<value> values;  // the distinct values found, sorted
};

// ---------------------------------------------------------------- scan_issue
//
// One problem found during the scan, described well enough to act on without re-reading
// the file by hand. Collected in campaign::issues().
//
// Two kinds live here. A RUN issue names a file inside one run directory (a config that
// will not parse, a manifest field of the wrong type). A WALK issue has `file` empty and
// reports a directory the scan could not enter or chose not to follow — unreadable,
// past the depth limit, or resolving outside the root. The second kind matters as much
// as the first: the axes are discovered by DIFFING whatever was found, so a subtree that
// vanishes silently does not just lose runs, it changes the answer.
struct scan_issue {
    std::filesystem::path dir;                 // the directory concerned, absolute
    std::string           label;               // `dir` root-relative — what to SHOW
    std::string           file;                // "config.json" | "coast_runtime.json" | ""
    std::string           message;             // names the field and the offending value
    // True only where config.json itself was unusable, i.e. exactly one run is missing
    // because of this issue. A walk issue leaves it false even though runs may well be
    // missing under it: how many is unknown, and an unknown must not be added to a
    // count the GUI presents as exact.
    bool                  run_dropped = false;
};

// ------------------------------------------------------------------- manifest
//
// Projection of a run's coast_runtime.json.
//
// The two bad states are deliberately DISTINGUISHABLE, because they mean opposite
// things. Both used to read as `present == false`, so a corrupted provenance record
// was indistinguishable from a run that is simply still going:
//
//   present == false, error empty  -> no manifest yet. A run in flight. Normal.
//   error non-empty                -> the manifest is malformed. An error, and the
//                                     fields it did yield are not to be trusted.
struct manifest {
    std::string   coast_git_sha;
    std::string   exit_reason;       // e.g. "completed_n_steps"
    std::string   hostname;
    std::string   gpu_adapter;
    std::string   started_at_utc;
    std::string   completed_at_utc;
    std::uint64_t steps_requested     = 0;
    std::uint64_t steps_completed     = 0;
    double        wall_clock_seconds  = 0.0;
    bool          present             = false;
    // Empty when the manifest is well-formed (or legitimately absent). Otherwise the
    // same text reported through campaign::issues().
    std::string   error;

    [[nodiscard]] bool malformed() const { return !error.empty(); }
};

class campaign; // fwd

// ------------------------------------------------------------------------ run
//
// Handle to one run directory. Opens no HDF5 until asked.
//
// LIFETIME: a run is a handle into its campaign and is only meaningful while that
// campaign lives. It holds a WEAK reference, so outliving the campaign is reported:
//
//     auto rs = c->runs();          // copies N handles — none of them own anything
//     c = camp::open(other_root);   // the last shared_ptr drops; ~campaign
//     rs[0].id();                   // throws "run outlived its campaign"
//
// That sequence is the most natural way to iterate in the REPL, and with a raw
// back-pointer it was a silent use-after-free inside a JIT interpreter — memory
// corruption in the one place the user has no reason to be thinking about ownership.
// A `selection` by contrast keeps its campaign alive, so iterating one is always safe;
// prefer `for (const run& r : sel)` over copying handles out.
//
// The guard covers REFERENCES too, because every accessor that yields a run does so BY
// VALUE. Returning `const run&` into the campaign's own vector would have left
// `auto& r = sel[0];` and `auto& rs = c->runs();` outside it — a reference cannot carry
// a weak_ptr, so those two spellings, one character away from the safe ones, were still
// silent use-after-free.
class run {
public:
    run() = default;

    // Identity within the campaign: the run directory's path RELATIVE TO THE ROOT
    // ("run_3", or "sweep_b/run_3" in a nested layout), not its bare directory name.
    // A bare name is not unique — `root/a/run1` and `root/b/run1` collide — and
    // campaign::select() matches on this id, so a colliding id silently resolves to
    // the wrong run. Relative paths cannot collide under one root.
    [[nodiscard]] std::string                   id()  const;
    [[nodiscard]] const std::filesystem::path&  dir() const;

    // -- swept parameters --
    //
    // `axis` is a short name from campaign::axes() or a full JSON Pointer. Throws if
    // the name is unknown; number()/text() additionally throw on a kind mismatch, so
    // check has() first. has() is false when this run's config lacks the leaf, and also
    // when it holds an explicit null — a JSON null is recorded as kind::absent, since
    // "written as null" and "not written" mean the same thing to an analysis.
    [[nodiscard]] bool        has   (std::string_view axis) const;
    [[nodiscard]] value       get   (std::string_view axis) const;
    [[nodiscard]] double      number(std::string_view axis) const;
    [[nodiscard]] std::string text  (std::string_view axis) const;

    // -- the whole config, not just the axes --
    //
    // Any leaf by JSON Pointer, for parameters that were held constant across the
    // campaign (and so are not axes) but still matter to an analysis.
    [[nodiscard]] bool        has_config   (std::string_view json_ptr) const;
    [[nodiscard]] double      config_number(std::string_view json_ptr) const;
    [[nodiscard]] std::string config_string(std::string_view json_ptr) const;

    // -- provenance --
    [[nodiscard]] const manifest& info() const;
    // The manifest says the run ended by reaching its requested step count. This
    // reads exit_reason; it does not compare steps_completed against steps_requested.
    // False for a run still in flight, one that failed, one whose manifest is missing,
    // and one whose manifest is MALFORMED — a file the loader has called damaged is
    // not evidence of completion.
    [[nodiscard]] bool complete() const;

    // -- data (lazy: the first HDF5 open happens here) --
    //
    // observe_groups() lists the observe group names found on disk (the stems of
    // observe/<group>.observe.h5) — a directory listing, no HDF5 opened. An absent
    // observe directory is empty; an UNREADABLE one throws, because "I could not look"
    // is not "there is nothing there". Same for snapshots().
    [[nodiscard]] std::vector<std::string> observe_groups() const;

    // Path of an observe file, validated but not opened — the single place that knows
    // the "observe/<group>.observe.h5" convention, so a caller (the GUI panel) never
    // re-derives it and cannot drift from it. With no group: the only group if there
    // is exactly one, else throws listing them. Throws if the file does not exist, or
    // if `group` is not a plain name: it is a path component, so "/abs" or ".." would
    // otherwise escape the run directory through operator/.
    [[nodiscard]] std::filesystem::path observe_path(std::string_view group = {}) const;

    // Open an observe file and return an h5browse console handle over it — the same
    // type `open()` hands the REPL, so every existing scan()/row()/column() facility
    // applies unchanged. With no group name: opens the only group if there is exactly
    // one, otherwise throws listing the available names. Throws if the file is absent
    // or the HDF5 executor was never registered (h5b::console::set_executor).
    //
    // Nothing is cached: each call opens a fresh handle. "One 2 GB file open at a time"
    // is therefore a property of the CALLER's loop, not a bound the loader enforces —
    // let the shared_ptr die each iteration. Collecting them (v.push_back(r.observe()))
    // holds N, and a REPL cell's variables outlive the cell.
    [[nodiscard]] std::shared_ptr<h5b::console::file_console>
    observe(std::string_view group = {}) const;

    // final_state.h5, opened the same way. Throws if the run has none.
    [[nodiscard]] std::shared_ptr<h5b::console::file_console> final_state() const;

    // Checkpoint files under snapshots/, sorted by name (so chronologically, given
    // coast's timestamped naming). Paths only — nothing is opened.
    [[nodiscard]] std::vector<std::filesystem::path> snapshots() const;

private:
    friend class campaign;
    friend class selection;

    // Resolve the owning campaign, or throw. Weak rather than shared deliberately:
    // campaign::impl holds the run handles, so a shared_ptr here would be a reference
    // cycle and the campaign would never be destroyed.
    [[nodiscard]] const campaign& owner() const;

    std::weak_ptr<const campaign> owner_;
    std::uint32_t                 idx_ = 0;
};

// ------------------------------------------------------------------ selection
//
// A subset of a campaign's runs. Chainable and cheap to copy (indices plus a shared
// pointer to the campaign, which it keeps alive).
class selection {
public:
    // Range-for support. Dereferences BY VALUE. A `run` is a weak_ptr and an index —
    // 24 bytes, one atomic — and returning a reference into the campaign's own vector
    // would hand out a pointer the weak_ptr guard cannot police: `auto& r = sel[0];`
    // followed by the campaign dying leaves `r` naming freed memory and reporting
    // nothing, which is precisely the failure run's weak_ptr exists to prevent. By
    // value, `auto&` binds to a lifetime-extended temporary that still owns its guard.
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = run;
        using difference_type   = std::ptrdiff_t;
        using reference         = run;

        reference  operator*() const;
        iterator&  operator++() { ++i_; return *this; }
        void       operator++(int) { ++i_; }
        bool operator==(const iterator& o) const { return i_ == o.i_ && sel_ == o.sel_; }
        bool operator!=(const iterator& o) const { return !(*this == o); }

    private:
        friend class selection;
        const selection* sel_ = nullptr;
        std::size_t      i_   = 0;
    };

    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end()   const;

    [[nodiscard]] std::size_t size()  const { return idx_.size(); }
    [[nodiscard]] bool        empty() const { return idx_.empty(); }
    // By value — see iterator above for why a reference here is not safe to hand out.
    [[nodiscard]] run         operator[](std::size_t i) const;
    [[nodiscard]] run         at(std::size_t i) const;  // throws when out of range

    // -- filtering --
    //
    // TOLERANCE, and why this is not a plain `==`. Before filtering, the requested
    // value is checked against that axis's DISCOVERED set: if no discovered value is
    // within `rel_tol`, this throws and names the values that exist. Runs then match
    // EVERY value within tolerance, not just the nearest — a campaign that stored one
    // logical value in two representations must not split into two groups.
    //
    // Two reasons for the tolerance. First, config numbers can arrive
    // float32-round-tripped (a stored 0.01 reads back as 0.009999999776482582 — 2.2e-8
    // relative), so exact equality silently returns nothing. Second, the precheck turns
    // a mistyped value into a message instead of an empty result. The 1e-6 default sits
    // ~45x above that measured round-trip error (and ~17x above the worst case, 2^-24),
    // while decade-spaced sweep values are ~1e5 tolerance units apart.
    //
    // The precheck sees the CAMPAIGN's value set, not the current selection, so it
    // guards the first where() in a chain. A later where() that contradicts an earlier
    // one — or any cross-axis combination a non-factorial campaign never ran — still
    // yields an empty selection, which is the honest answer: those values do exist.
    [[nodiscard]] selection where(std::string_view axis, double v,
                                  double rel_tol = 1e-6) const;
    [[nodiscard]] selection where(std::string_view axis, std::string_view v) const;
    [[nodiscard]] selection where(std::string_view axis, bool v) const;
    // A string literal is `const char*`, and `const char* -> bool` is a standard
    // conversion that would outrank `-> string_view` and silently filter on a
    // boolean. This exact-match overload keeps literals strings.
    [[nodiscard]] selection where(std::string_view axis, const char* v) const
    {
        return where(axis, std::string_view(v));
    }
    // An integer argument is ambiguous across the double and bool overloads — both
    // equal-rank standard conversions — so without this not just `where(axis, 5)` but
    // `where(axis, n_groups)` would fail to compile. This exact match wins and routes
    // to the numeric filter. bool keeps its own overload (excluded here).
    template <class T>
        requires (std::is_integral_v<T> && !std::is_same_v<T, bool>)
    [[nodiscard]] selection where(std::string_view axis, T v) const
    {
        return where(axis, static_cast<double>(v));
    }
    // Escape hatch for anything the axis filters cannot express (manifest fields,
    // derived conditions, non-axis config leaves).
    [[nodiscard]] selection where(const std::function<bool(const run&)>& pred) const;

    // -- ordering and grouping, for plotting across runs --
    //
    // sort_by() orders the selection by an axis (ascending, stable). group_by()
    // partitions it into one selection per distinct value of an axis, in the axis's own
    // value order — one curve per group. EMPTY GROUPS ARE DROPPED, so the result is not
    // index-aligned with find_axis(a)->values on a selection that does not cover every
    // value; read each group's own value from its runs rather than by position.
    [[nodiscard]] selection              sort_by (std::string_view axis) const;
    [[nodiscard]] std::vector<selection> group_by(std::string_view axis) const;

    // -- extraction --
    //
    // The x-vector for a plot: this selection's value of `axis`, in selection order,
    // so it lines up element-for-element with a y-vector built by iterating. Throws
    // if the axis is non-numeric or absent from any run in the selection.
    [[nodiscard]] std::vector<double>      values(std::string_view axis) const;
    // Same, rendered — valid for any kind, including text and absent. For labels.
    [[nodiscard]] std::vector<std::string> labels(std::string_view axis) const;
    [[nodiscard]] std::vector<std::filesystem::path> dirs() const;
    // Run ids (root-relative paths — see run::id()), in selection order. Pairs with
    // campaign::select()
    // to round-trip a selection through text — which is how the GUI panel hands one
    // to the console exactly, rather than approximating it with a filter chain.
    [[nodiscard]] std::vector<std::string> ids() const;

    // Throws if this selection is not bound to a campaign (default-constructed).
    [[nodiscard]] const campaign& owner() const { return bound(); }

private:
    friend class campaign;

    // Resolve the campaign or throw. A default-constructed selection is reachable
    // from the REPL (`camp::selection s;`), and a null dereference there is a process
    // crash with no diagnostic.
    [[nodiscard]] const campaign& bound() const;

    std::shared_ptr<const campaign> camp_;
    std::vector<std::uint32_t>      idx_;
};

// ------------------------------------------------------------------- campaign
//
// The scan result: every run found under the root, plus the axes discovered by
// diffing their configs. Held by shared_ptr (open() hands one out) so selections can
// keep it alive.
class campaign : public std::enable_shared_from_this<campaign> {
public:
    ~campaign();

    campaign(const campaign&)            = delete;
    campaign& operator=(const campaign&) = delete;

    [[nodiscard]] const std::filesystem::path& root() const;
    [[nodiscard]] std::size_t                  size() const;

    // The swept parameters, in discovery order: config key order, first-seen across
    // the runs in scan order. A leaf that only some runs carry therefore appears after
    // the keys of the runs read before it — visible when the root spans several
    // campaigns with different config shapes.
    [[nodiscard]] const std::vector<axis>& axes() const;
    // Resolve a short name or full JSON Pointer; nullptr if unknown.
    [[nodiscard]] const axis* find_axis(std::string_view name) const;

    // By value: a handle vector referencing the campaign's own storage would outlive
    // it undetected. Copies N weak_ptrs; the campaign is the expensive part and is
    // already shared.
    [[nodiscard]] std::vector<run> runs() const;

    [[nodiscard]] selection all() const;
    // Exactly the named runs (ids as run::id() reports them), in the order given.
    // Ids that do not
    // occur in this campaign are ignored, so a stale id list degrades to a smaller
    // selection instead of throwing. The inverse of selection::ids().
    [[nodiscard]] selection select(const std::vector<std::string>& run_ids) const;
    // Sugar for all().where(...) — the common first step of a query.
    [[nodiscard]] selection where(std::string_view axis, double v,
                                  double rel_tol = 1e-6) const;
    [[nodiscard]] selection where(std::string_view axis, std::string_view v) const;
    [[nodiscard]] selection where(std::string_view axis, bool v) const;
    [[nodiscard]] selection where(std::string_view axis, const char* v) const
    {
        return where(axis, std::string_view(v));
    }
    // See the selection overload: keeps an integer argument from being ambiguous
    // between the double and bool filters.
    template <class T>
        requires (std::is_integral_v<T> && !std::is_same_v<T, bool>)
    [[nodiscard]] selection where(std::string_view axis, T v) const
    {
        return where(axis, static_cast<double>(v));
    }
    [[nodiscard]] selection where(const std::function<bool(const run&)>& pred) const;

    // Everything that went wrong under this root: a file that would not read, and a
    // directory the walk could not enter or would not follow. Reported rather than
    // thrown, so one damaged file does not cost a whole campaign — and reported rather
    // than absorbed into a default, so it cannot pass for healthy data.
    //
    // `run_dropped` is true only where config.json itself was unusable (no parameters
    // to query, so the run cannot enter the diff). A malformed coast_runtime.json
    // does NOT drop the run: its parameters are still perfectly readable, and only
    // its provenance is in doubt.
    [[nodiscard]] const std::vector<scan_issue>& issues() const;

    // Summary for the REPL: run count, axis names with value counts, and every issue
    // printed in full. Printed, not counted — a summary that says "3 problems" and
    // leaves the reader to find them is the same as hiding them.
    [[nodiscard]] std::string summary() const;

private:
    friend class run;
    friend class selection;
    friend std::shared_ptr<campaign> open(const std::filesystem::path&,
                                          const scan_options&);

    campaign();

    // Point every run handle at this campaign. Lives here rather than in open()
    // because campaign — not open() — is the friend that may touch run's internals.
    void bind_runs();

    struct impl;
    std::unique_ptr<impl> p_;
};

// --------------------------------------------------------------- scan_options
struct scan_options {
    // Additional JSON Pointers to exclude from axis discovery, on top of the built-in
    // deny-list.
    //
    // The built-in list holds coast's own OUTPUT LOCATIONS — currently "/snapshot_dir"
    // and "/observe/output_dir". Those differ per run by construction (coastd writes
    // each run's own directory into its config) and are not hyperparameters, so they
    // would otherwise show up as an axis with one value per run. This is a fact about
    // our config schema, not a heuristic: deliberately NOT the rule "as many distinct
    // values as runs implies not an axis", which would misfire on the most ordinary
    // sweep of all — one parameter, N values, N runs.
    std::vector<std::string> ignore_pointers;

    // Cap on directory-tree recursion depth below the root. Guards against a
    // pathological tree; the reference campaign needs 1. A directory reached at the cap
    // is REPORTED through issues(), never skipped in silence — the axes come from
    // diffing whatever the walk found, so a subtree that vanishes changes the answer.
    int max_depth = 8;
};

// Scan `root` and return the campaign. Throws std::runtime_error if the root cannot be
// queried, does not exist, is not a directory, contains no run directory at all, or has
// runs but none with a readable config.json. Individual unreadable runs and unenterable
// directories are reported via campaign::issues() instead.
//
// A symlink resolving outside `root` is not followed: the root is the frame, and a link
// out of it would otherwise merge a foreign config's keys into the axis namespace.
//
// A directory is recognised as a run iff it directly contains `config.json`; the walk
// does not descend into a directory it has recognised. Files and directories that are
// not runs are ignored, so a root one or more levels above the runs (a whole
// experiment tree rather than a single sweep) scans correctly and yields the union of
// the axes found beneath it.
[[nodiscard]] std::shared_ptr<campaign> open(const std::filesystem::path& root,
                                             const scan_options& opts = {});

} // namespace coast::campaign

#endif // COAST_CAMPAIGN_API_H
