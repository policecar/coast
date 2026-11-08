#ifndef COAST_RUN_CONFIG_H
#define COAST_RUN_CONFIG_H

// An editable simulation config — the shared core of the console workflow
// "load a base config, edit it, then run/dispatch it". It lives in coast_lib
// (headless core) so BOTH console façades can hold and mutate it:
//   - coast::simc (sim_console, GUI layer): edit → run() a local/child sim.
//   - coast::coastd (coastd_console, core):  edit → submit() to a daemon.
// simc and coastd re-export this type into their own namespaces via `using`, so
// `simc::config(...)` and `coastd::config(...)` both yield the same run_config.
//
// This header is an interpreter PARSE BOUNDARY (both façade headers include it,
// and both are parsed by the clang-repl console). It is therefore kept lean:
// nlohmann/json is NOT exposed — the parsed config is held by pointer (pimpl) and
// edited through a string JSON-Pointer API. Implementations live in
// run_config.cpp, compiled by the host and reached via the JIT symbol table
// (coast_lib is whole-archived + -rdynamic, as the h5/plot/sim façades are).
//
// Error model: failures (missing base config, malformed JSON pointer, wrong
// value kind) throw std::runtime_error with a message the REPL surfaces.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace coast::cfg {

// ---------------------------------------------------------------- run_config
//
// Loaded from a base JSON file, mutated in memory, and serialized with dump().
// Copyable, so a base config can seed many sweep points: `auto c = base;`.
//
// Edit any field by JSON Pointer (RFC 6901), e.g.
//   c.set("/ngm2/dendrites/default/default_primary_learning_rate", 0.03);
//   c.set("/ngm2/neuron_groups/L4-mnist/default_common_learning_rate", 1e-4);
// Intermediate objects are created on demand, so absent sections (e.g. observe)
// can be set without pre-existing in the base file.
class run_config {
public:
    ~run_config();
    run_config(const run_config&);
    run_config(run_config&&) noexcept;
    run_config& operator=(const run_config&);
    run_config& operator=(run_config&&) noexcept;

    // Raw base: set/get any field by JSON Pointer. set() creates missing
    // parents; throws std::runtime_error on a syntactically invalid pointer.
    // get_*() throw if the pointer is absent or the value is the wrong kind.
    run_config& set(std::string_view ptr, double v);
    run_config& set(std::string_view ptr, std::int64_t v);
    run_config& set(std::string_view ptr, bool v);
    run_config& set(std::string_view ptr, std::string_view v);
    // A string literal is `const char*`, and `const char* -> bool` is a
    // standard conversion that would otherwise outrank `-> string_view` and
    // silently store a boolean. This exact-match overload keeps literals strings.
    run_config& set(std::string_view ptr, const char* v)
    {
        return set(ptr, std::string_view(v));
    }
    // Any integer type (int, unsigned, long, size_t, …) routes to the int64_t
    // setter. Without this, an integer argument is ambiguous across the
    // double/int64_t/bool overloads — all equal-rank conversions — so not just
    // `set(ptr, 42)` but `set(ptr, vec.size())` would fail to compile. This
    // template is an exact match, so it wins over those conversions. bool keeps
    // its own overload (excluded here); floating-point uses the double overload.
    template <class T>
        requires (std::is_integral_v<T> && !std::is_same_v<T, bool>)
    run_config& set(std::string_view ptr, T v)
    {
        return set(ptr, static_cast<std::int64_t>(v));
    }
    [[nodiscard]] double      get_number(std::string_view ptr) const;
    [[nodiscard]] std::string get_string(std::string_view ptr) const;

    // Sugar — the unambiguous knobs. (There is deliberately no learning_rate()
    // helper: rates live in several places and neuron_groups is keyed by group
    // name, so a single setter would guess wrong — use set() with the explicit
    // pointer above.)
    //
    // run_dir(d): point outputs at one directory — observe.output_dir -> d/observe,
    // snapshot_dir -> d/snapshots. A simc staging hint consumed by simc::run();
    // coastd::submit() IGNORES it (the daemon assigns the run directory and
    // rewrites snapshot_dir itself), but it is harmless to set.
    run_config& run_dir(std::string_view d);
    // seed(v): a simc staging hint (simc::run passes it as the child's
    // --seed-offset). coastd::submit() does not consume it — there is no seed
    // field on the wire today.
    run_config& seed(int v);

    [[nodiscard]] std::string dump() const;  // pretty JSON, for inspection/dispatch

    // Read accessors used by simc::run() to stage a run without reaching into the
    // pimpl (keeps run_config free of cross-library friendship).
    [[nodiscard]] const std::string& base_path()    const;  // file config() loaded ("" if none)
    [[nodiscard]] const std::string& run_dir_hint() const;  // last run_dir(), or ""
    [[nodiscard]] int                seed_value()   const;  // last seed(), or 0

private:
    friend run_config config(const std::filesystem::path&);
    struct impl;
    std::unique_ptr<impl> p_;
    run_config();
};

// Load a base config file into an editable run_config. Throws std::runtime_error
// if the file is missing or not valid JSON.
[[nodiscard]] run_config config(const std::filesystem::path& base);

} // namespace coast::cfg

#endif // COAST_RUN_CONFIG_H
