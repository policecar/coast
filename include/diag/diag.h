#ifndef COAST_DIAG_DIAG_H
#define COAST_DIAG_DIAG_H

// -----------------------------------------------------------------------------
// coast::diag — hand-rolled diagnostic logging facility
// -----------------------------------------------------------------------------
//
// A small, header-only text-logging facility for COAST's diagnostic output
// (startup notices, warnings, errors, debug traces). It replaces the ~120
// ad-hoc std::cout / std::cerr / fprintf sites scattered across the codebase
// with a single levelled, category-tagged, thread-safe sink.
//
// WHY hand-rolled (not spdlog/Quill/fmt):
//   * C++23 is the project standard, so std::format already provides
//     type-checked formatting for free — the only thing a logging library
//     would add here (async lock-free queues, deferred formatting) targets a
//     hot-path logging load COAST does not have (the sim hot path barely logs).
//   * Keeps the dependency surface flat and avoids the bundled-fmt-vs-std::format
//     duplication spdlog would drag in.
//   * The whole facility is a few levels, a category tag, and one mutex-guarded
//     sink — the sink seam is deliberately narrow so a library-backed sink can
//     be dropped in later without touching a single call site.
//
// WHY std::format and NOT std::print:
//   std::print / std::println need a runtime symbol in the standard-library
//   shared object (libc++ 18+ / GCC 14+) and add a link dependency. std::format
//   is header-only and available on GCC 13+ / libc++ 17+ — under both macOS
//   toolchains COAST supports (Homebrew LLVM Clang + libc++, and GCC 15 +
//   libstdc++). We render to a std::string with std::format and write that.
//
// NAME NOTE: the `logging::` namespace and include/logging/ are ALREADY taken by
// the HDF5 simulation-DATA logger. This diagnostic facility lives under
// coast::diag / include/diag/ to avoid the collision.
//
// PROGRESS IS SEPARATE: machine-readable run progress does NOT go through this
// logger — it has its own contract in <diag/progress.h> so an external parser
// (cimc-coast-consim) and the emitter share one schema. See that header.
//
// Threading: dispatch() holds a mutex across the sink call, so output from the
// GUI thread, the sim thread, and the HDF5 executor thread is serialized and
// cannot tear. A custom sink therefore need NOT synchronize internally.
//
// Usage:
//   #include <diag/diag.h>
//   COAST_INFO ("gpu",     "adapter: {}", name);
//   COAST_WARN ("observe", "dtype lock mismatch on {}", path);
//   COAST_ERROR("io",      "read failed: {} ({} bytes)", err, n);
//   // runtime control (wire to CLI --log-level / -v):
//   coast::diag::set_level(coast::diag::level::debug);
// -----------------------------------------------------------------------------

#include "diag/level.h"  // level, level_name, level_from

#include <atomic>
#include <cstdio>
#include <format>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace coast::diag {

// A sink receives already-formatted records. Functional seam — no inheritance:
// the stderr default, a file sink, or a GUI-panel sink is just another callable.
// Invoked under the dispatch lock, so a sink need not self-synchronize. A sink
// must not re-enter the logger (dispatch holds a non-recursive mutex); if it
// does, the reentrancy guard in dispatch() drops the record rather than
// deadlocking (and emits a one-shot notice to stderr).
//
// LOCK ORDERING (matters once a stateful sink lands, e.g. the GUI-window sink):
// dispatch holds sink_mtx across the sink, so if the sink takes lock X, the order
// is sink_mtx -> X. Any other thread that holds X and then logs establishes
// X -> sink_mtx — a classic inversion that can deadlock across threads (the
// thread_local reentrancy guard only covers the SAME thread). Such a sink should
// keep its critical section lock-free (e.g. push to a lock-free/locked queue it
// drains on its own thread) rather than take a lock that other threads hold while
// logging.
using sink_fn = std::function<void(level, std::string_view /*category*/,
                                   std::string_view /*message*/)>;

namespace detail {

struct state {
    std::atomic<level> min_level{level::info};
    std::mutex         sink_mtx;
    sink_fn            sink;  // empty => use default_sink; guarded by sink_mtx
};

inline state& get() {
    static state s;
    return s;
}

// One line per record to `out`, flushed. Flushing matters even for stderr so a
// redirected stream shows records immediately (the same block-buffering concern
// the headless progress path has).
inline void write_record(std::FILE* out, level l,
                         std::string_view cat, std::string_view msg) {
    const auto lv = level_name(l);
    std::fprintf(out, "[%.*s][%.*s] %.*s\n",
                 static_cast<int>(lv.size()),  lv.data(),
                 static_cast<int>(cat.size()), cat.data(),
                 static_cast<int>(msg.size()), msg.data());
    std::fflush(out);
}

// Owns the optional --log-file handle for the file sink. Static lifetime; not
// explicitly closed on the normal path (flushed per record, released by the OS
// at exit).
//
// CAVEAT: the detail::state singleton below (its mutex + sink) is itself
// destroyed at program exit in unspecified order relative to other statics, so
// logging from a static destructor after it is gone is UB — the usual
// logging-singleton caveat. Diagnostics during ordinary teardown (before static
// destruction runs) are fine; do not log from static destructors.
inline std::atomic<std::FILE*>& log_file_handle() {
    static std::atomic<std::FILE*> f{nullptr};
    return f;
}

inline void default_sink(level l, std::string_view cat, std::string_view msg) {
    write_record(stderr, l, cat, msg);
}

// True while a sink is executing on THIS thread (dispatch holds sink_mtx). Read
// by dispatch() to drop re-entrant logging, and by the sink setters to refuse
// reconfiguration from within a sink (both would touch the non-recursive
// sink_mtx again on the same thread).
inline bool& in_dispatch() {
    static thread_local bool f = false;
    return f;
}

inline void dispatch(level l, std::string_view cat, std::string_view msg) {
    // Reentrancy guard. sink_mtx is non-recursive, and a sink is invoked while
    // it is held, so a sink that logs (a plausible footgun for the planned
    // GUI-window sink, whose code paths can be circuitous) would re-enter here
    // and self-deadlock. Detect that on this thread and drop the reentrant
    // record instead. Surface it ONCE, always-live — straight to stderr,
    // bypassing the sink we're already inside. (Not a debug-only assert: this
    // project builds -DNDEBUG everywhere, dev and CI, so an assert would be a
    // dead check — the anti-pattern COAST_REQUIRE exists to avoid.)
    if (in_dispatch()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            write_record(stderr, level::error, "diag",
                         "a sink re-entered the logger; records emitted from "
                         "within a sink are dropped (a sink must not log)");
        }
        return;
    }
    auto& s = get();
    std::lock_guard lk(s.sink_mtx);
    in_dispatch() = true;
    struct reset { ~reset() { in_dispatch() = false; } } r;
    if (s.sink) s.sink(l, cat, msg);
    else        default_sink(l, cat, msg);
}

// Format-and-dispatch a record, guaranteed never to throw. Every COAST_* macro
// routes through this, so a diagnostic call can NEVER unwind its caller — it is
// safe by construction in a noexcept context, a std::thread entry function, or a
// graceful-degradation catch block, with no per-call-site wrapping required.
//
// The realistic throw is std::bad_alloc: from std::format building the message
// (the pre-diag fprintf path could not throw — this is what centralizing here
// restores) or from a stateful sink under OOM. For a diagnostics logger, losing
// the line is strictly better than terminating the caller, so we swallow it.
//
// `make_msg` is a callable returning the formatted std::string. It is invoked
// INSIDE the try so the format call itself is covered; the macro passes a
// deferred `[&]{ return std::format(...); }` rather than a ready string.
template <class MakeMsg>
void emit_guarded(level l, std::string_view cat, MakeMsg&& make_msg) noexcept {
    try {
        dispatch(l, cat, static_cast<MakeMsg&&>(make_msg)());
    } catch (...) {  // NOLINT(bugprone-empty-catch): intentional — see contract above
    }
}

}  // namespace detail

// ---- runtime control --------------------------------------------------------

inline void  set_level(level l) noexcept { detail::get().min_level.store(l, std::memory_order_relaxed); }
inline level get_level()        noexcept { return detail::get().min_level.load(std::memory_order_relaxed); }

// Cheap pre-format gate — checked by the macros before std::format runs so a
// suppressed record costs one relaxed atomic load and nothing else.
inline bool level_enabled(level l) noexcept { return l >= get_level(); }

// Replace the sink (e.g. tee to a file or mirror into the GUI logging window).
// Pass an empty sink_fn to restore the default stderr sink.
//
// Must NOT be called from within a sink callback: dispatch holds the
// non-recursive sink_mtx across the sink call, so relocking it here would
// deadlock, and reassigning the sink would destroy the executing sink_fn
// mid-call (UB). Such a call is refused (a safe no-op) rather than deadlocking.
// Reconfigure the sink from ordinary code — e.g. a GUI thread's own turn, not
// from inside a message it is currently rendering. The setters are meant to be
// called serially (config time on one thread, or externally serialized); they
// are memory-safe against concurrent callers but do not define an ordering
// between two simultaneous reconfigurations.
inline void set_sink(sink_fn fn) {
    if (detail::in_dispatch()) return;  // reconfiguration from within a sink: refused
    auto& s = detail::get();
    std::lock_guard lk(s.sink_mtx);
    s.sink = std::move(fn);
}

// Route diagnostics to an already-open FILE* (stderr, stdout, or a caller-owned
// file). The stream must outlive all logging. Does not take ownership.
inline void set_stream_sink(std::FILE* stream) {
    set_sink([stream](level l, std::string_view cat, std::string_view msg) {
        detail::write_record(stream, l, cat, msg);
    });
}

// Route diagnostics to a file (append mode) instead of stderr — the CLI seam
// for --log-file, so callers don't have to rely on shell 2> redirection.
// Returns false and leaves the current sink untouched if the file can't be
// opened. The handle is owned for the program lifetime (see log_file_handle).
inline bool set_log_file(const char* path) {
    if (detail::in_dispatch()) return false;  // no reconfiguration mid-sink (see set_sink)
    std::FILE* f = std::fopen(path, "a");
    if (!f) return false;
    auto& s = detail::get();
    // Install the file sink, retire the previously-owned handle, and close it —
    // all under sink_mtx as ONE critical section. This is what makes two
    // concurrent reconfigurations safe. dispatch() also holds sink_mtx across the
    // sink call, so while we hold it no thread is inside the old sink and none can
    // start a new dispatch on it; the handle we replace is therefore safe to close
    // here. Splitting this into "assign sink" and "swap+close handle" (as an
    // earlier version did) let a second caller interpose between the two steps and
    // close a handle the first caller's freshly-installed sink was still using —
    // a use-after-free. Note we assign s.sink directly rather than call
    // set_stream_sink(), which would relock the non-recursive sink_mtx here.
    std::FILE* prev;
    {
        std::lock_guard lk(s.sink_mtx);
        s.sink = [f](level l, std::string_view cat, std::string_view msg) {
            detail::write_record(f, l, cat, msg);
        };
        prev = detail::log_file_handle().exchange(f, std::memory_order_acq_rel);
        if (prev == f) prev = nullptr;  // fresh fopen never aliases; guard anyway
        if (prev) std::fclose(prev);
    }
    return true;
}

// Run an arbitrary action so an exception it raises can NEVER escape the current
// scope. Use in contexts where an escaping throw is catastrophic — a std::thread
// entry function, a noexcept context, or a graceful-degradation catch block.
//
// NOTE: a bare COAST_* macro NO LONGER needs this — every macro routes through
// detail::emit_guarded and is non-throwing by construction, so
//
//   COAST_WARN("cat", "…{}", x);   // already safe in a noexcept context
//
// stands on its own. This helper remains for guarding a multi-statement block,
// or a non-diagnostic call that can throw, inside such a context:
//
//   coast::diag::log_nothrow([&]{ some_throwing_cleanup(); COAST_WARN(...); });
//
template <class F>
void log_nothrow(F&& emit) noexcept {
    try {
        static_cast<F&&>(emit)();
    } catch (...) {  // NOLINT(bugprone-empty-catch): intentional — see contract above
        // Nothing safe to do: emitting the diagnostic itself failed (OOM), and
        // letting it escape here would be worse than losing the line.
    }
}

}  // namespace coast::diag

// -----------------------------------------------------------------------------
// Call-site macros
//
// Two-stage gating:
//   1. COMPILE-TIME: records below COAST_DIAG_COMPILE_LEVEL expand to `((void)0)`
//      so their std::format call is never compiled in (true zero cost — e.g.
//      trace/debug vanish from Release). Preprocessor strip, NOT `if constexpr`
//      (which, in a non-template context, would still compile the format call).
//   2. RUNTIME: level_enabled() short-circuits before std::format for records
//      that are compiled in but currently below the runtime level.
//
// Override COAST_DIAG_COMPILE_LEVEL on the command line to keep debug/trace in a
// release build, or to strip more aggressively.
// -----------------------------------------------------------------------------

#ifndef COAST_DIAG_COMPILE_LEVEL
#  ifdef NDEBUG
#    define COAST_DIAG_COMPILE_LEVEL 1  // strip trace; keep debug+ in Release
#  else
#    define COAST_DIAG_COMPILE_LEVEL 0  // keep everything in Debug
#  endif
#endif

#define COAST_DIAG_EMIT(lvl, cat, ...)                                     \
    do {                                                                   \
        if (::coast::diag::level_enabled(lvl))                             \
            ::coast::diag::detail::emit_guarded((lvl), (cat),              \
                [&] { return ::std::format(__VA_ARGS__); });               \
    } while (0)

#if COAST_DIAG_COMPILE_LEVEL <= 0
#  define COAST_TRACE(cat, ...) COAST_DIAG_EMIT(::coast::diag::level::trace, cat, __VA_ARGS__)
#else
#  define COAST_TRACE(cat, ...) ((void)0)
#endif

#if COAST_DIAG_COMPILE_LEVEL <= 1
#  define COAST_DEBUG(cat, ...) COAST_DIAG_EMIT(::coast::diag::level::debug, cat, __VA_ARGS__)
#else
#  define COAST_DEBUG(cat, ...) ((void)0)
#endif

#if COAST_DIAG_COMPILE_LEVEL <= 2
#  define COAST_INFO(cat, ...)  COAST_DIAG_EMIT(::coast::diag::level::info, cat, __VA_ARGS__)
#else
#  define COAST_INFO(cat, ...)  ((void)0)
#endif

#if COAST_DIAG_COMPILE_LEVEL <= 3
#  define COAST_WARN(cat, ...)  COAST_DIAG_EMIT(::coast::diag::level::warn, cat, __VA_ARGS__)
#else
#  define COAST_WARN(cat, ...)  ((void)0)
#endif

#if COAST_DIAG_COMPILE_LEVEL <= 4
#  define COAST_ERROR(cat, ...) COAST_DIAG_EMIT(::coast::diag::level::error, cat, __VA_ARGS__)
#else
#  define COAST_ERROR(cat, ...) ((void)0)
#endif

#endif  // COAST_DIAG_DIAG_H
