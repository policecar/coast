#ifndef COAST_COASTD_CONSOLE_API_H
#define COAST_COASTD_CONSOLE_API_H

// Drive a coastd daemon from the embedded C++ console.
//
// The workflow: connect() to a daemon, then submit()/list()/cancel()/reorder()/
// info() against it, exactly the surface a GUI would use — this façade is just
// the REPL-ergonomic front to the same coastd_client protocol library. A daemon
// on another machine is reached by host+port; localhost is the default.
//
// This header is the interpreter PARSE BOUNDARY, so — like sim_console_api.h —
// it is kept lean: NEITHER nlohmann/json NOR the socket-bearing coastd_client
// header is exposed. The wire types are re-projected as plain job_info/host_info
// structs; the live connection is held by pimpl. Implementations live in
// coastd_console_api.cpp (compiled by the host, reached via the JIT symbol table
// because coast_lib is whole-archived + -rdynamic, as the sim/h5 façades are).
//
// Error model mirrors sim_console: every failure (connect refused, daemon
// rejection, transport error) throws std::runtime_error with the daemon's own
// message, which the REPL surfaces. A return value is only ever the success case.

#include <cstdint>
#include <memory>
#include <string>

#include "run_config/run_config.h"

namespace coast::coastd {

// The editable config + its loader, shared with the simc console (they live in
// coast_lib / coast::cfg — design/run_config_extraction.md). Re-exported here so
// the coastd workflow mirrors simc's: `coastd::config("base.json")`, edit it,
// then submit() it. Same type, so a config built for a local simc run submits to
// a daemon unchanged.
using coast::cfg::run_config;
using coast::cfg::config;

// The default command port. Kept in step with the daemon's own default
// (coastd::default_tcp_port / proto::default_tcp_port == 51900); this header is
// parsed by the clang-repl console, whose include path cannot reach coastd/include,
// so it carries its own copy rather than including the shared constant.
inline constexpr std::uint16_t default_port = 51900;

// A job as the console sees it — a plain re-projection of the wire job_summary
// (no json, no protocol header). `status`/`bucket` are the daemon's stable tokens.
struct job_info {
    std::string   id;
    std::string   ref;
    std::string   status;
    std::string   bucket;
    std::string   submitted_at;
    std::string   user_message;
    std::string   checkpoint;
    std::string   failure_reason;
    std::uint64_t steps_done   = 0;  // durable checkpoint step
    std::uint64_t live_step    = 0;  // live step from the log tail (0 if not running); the
                                     // current position to display for a running job before
                                     // its first checkpoint. steps_done stays the resumable point.
    std::uint64_t steps_target = 0;
    int           priority     = 0;
    // A cancel is accepted but not settled: the child has been SIGTERMed and is
    // inside the grace period writing its final checkpoint. `status` still reads
    // `running` — deliberately, it is the job's real state — so this is what a
    // display uses to show the cancel is under way rather than ignored.
    bool          cancel_requested = false;
};

// A daemon's static self-description (from machine_info / discovery).
struct host_info {
    std::string hostname;
    std::string address;        // connectable IP (set by discover(); "" from info())
    std::string os;
    std::string coastd_version;
    std::string runs_root;
    std::string repo_root;
    int         slots = 0;
    int         port  = 0;      // the daemon's command port
    // True iff a config's relative "data/…" paths resolve on that machine (the
    // operator set dataset_base and its target exists). Lets a client warn before
    // submitting a config that would fail minutes later at dataset load.
    bool        data_link_ok = false;
};

// A CONFIGURED daemon from the fixed-list host config (IDD §6.6 — the
// deterministic primary way to reach a known daemon by name, no broadcast). A
// plain re-projection of the client lib's host_entry (name -> host:port).
struct configured_host {
    std::string name;
    std::string host;
    int         port = default_port;
};

// -----------------------------------------------------------------------------
// job_list — an owning, iterable handle over the queue. It exists so the console
// gets FULL programmatic access: `for (auto& j : s.jobs()) …`, `js[i]`, `js.size()`.
//
// Why a handle and not a bare std::vector<job_info>: the clang-repl JIT resolves
// symbols against the host's -rdynamic table and cannot materialize the inline
// STL members of a std::vector instantiated over a coastd-specific type when that
// vector is the RETURN value crossing the JIT boundary. So — exactly as the h5
// façade returns row_view/row_range rather than std::vector<row> — every member
// below is COMPILED (out-of-line, in the .o, exported), and the underlying
// std::vector stays hidden inside compiled code where the JIT never touches it.
// A range-based for works via the compiled iterator. A plain compiled caller
// (the GUI panel) uses the same handle.
class job_list {
public:
    job_list();
    job_list(const job_list&);
    job_list(job_list&&) noexcept;
    job_list& operator=(const job_list&);
    job_list& operator=(job_list&&) noexcept;
    ~job_list();

    [[nodiscard]] int      size()  const;
    [[nodiscard]] bool     empty() const;
    [[nodiscard]] job_info at(int i) const;          // by value; throws on out-of-range
    [[nodiscard]] job_info operator[](int i) const;  // no bounds check

    // Range-for support via a compiled iterator (dereferences to a job_info copy).
    class iterator {
    public:
        iterator(const job_list* owner, int i) : owner_(owner), i_(i) {}
        job_info    operator*() const;
        iterator&   operator++()            { ++i_; return *this; }
        bool        operator!=(const iterator& o) const { return i_ != o.i_; }
    private:
        const job_list* owner_;
        int             i_;
    };
    [[nodiscard]] iterator begin() const { return {this, 0}; }
    [[nodiscard]] iterator end()   const { return {this, size()}; }

private:
    friend class session;
    struct impl;
    std::unique_ptr<impl> p_;   // hidden std::vector<job_info>; never crosses the JIT
};

// host_list — same handle idea for discovered daemons.
class host_list {
public:
    host_list();
    host_list(const host_list&);
    host_list(host_list&&) noexcept;
    host_list& operator=(const host_list&);
    host_list& operator=(host_list&&) noexcept;
    ~host_list();

    [[nodiscard]] int       size()  const;
    [[nodiscard]] bool      empty() const;
    [[nodiscard]] host_info at(int i) const;
    [[nodiscard]] host_info operator[](int i) const;

    class iterator {
    public:
        iterator(const host_list* owner, int i) : owner_(owner), i_(i) {}
        host_info   operator*() const;
        iterator&   operator++()            { ++i_; return *this; }
        bool        operator!=(const iterator& o) const { return i_ != o.i_; }
    private:
        const host_list* owner_;
        int              i_;
    };
    [[nodiscard]] iterator begin() const { return {this, 0}; }
    [[nodiscard]] iterator end()   const { return {this, size()}; }

private:
    friend host_list discover(int);
    struct impl;
    std::unique_ptr<impl> p_;
};

// host_config_list — same compiled-handle idea for the fixed-list host config.
class host_config_list {
public:
    host_config_list();
    host_config_list(const host_config_list&);
    host_config_list(host_config_list&&) noexcept;
    host_config_list& operator=(const host_config_list&);
    host_config_list& operator=(host_config_list&&) noexcept;
    ~host_config_list();

    [[nodiscard]] int             size()  const;
    [[nodiscard]] bool            empty() const;
    // A note about the config file itself (malformed JSON, unreadable), empty when
    // it loaded cleanly. A MISSING file is not a warning — that yields the built-in
    // localhost entry by design. Surfaced through the API rather than only to
    // stderr, because the GUI's host picker is a window: a user there sees an
    // unexplained short list while the explanation goes to a terminal nobody is
    // looking at (and, in a windowed launch, to nowhere at all).
    [[nodiscard]] std::string     warning() const;
    [[nodiscard]] configured_host at(int i) const;
    [[nodiscard]] configured_host operator[](int i) const;

    class iterator {
    public:
        iterator(const host_config_list* owner, int i) : owner_(owner), i_(i) {}
        configured_host operator*() const;
        iterator&       operator++()            { ++i_; return *this; }
        bool            operator!=(const iterator& o) const { return i_ != o.i_; }
    private:
        const host_config_list* owner_;
        int                     i_;
    };
    [[nodiscard]] iterator begin() const { return {this, 0}; }
    [[nodiscard]] iterator end()   const { return {this, size()}; }

private:
    friend host_config_list configured_hosts();
    struct impl;
    std::unique_ptr<impl> p_;
};

// A live connection to one daemon. Move-only (owns a socket). Throws
// std::runtime_error on any protocol error.
class session {
public:
    // Connect within `timeout_ms`. Throws std::runtime_error if the daemon is
    // unreachable. With no arguments this reaches the LOCAL daemon on the default
    // port — for a remote one pass host+port, or connect(host_info) below.
    // `io_timeout_ms` bounds each request/reply after the connect; 0 = same as
    // timeout_ms. Keep the connect bound short on a UI thread (it is how long a
    // dead host freezes the frame) and the operation bound long enough for the
    // daemon's slowest legitimate reply — `submit` goes through the single-writer
    // queue and two fsyncs. One shared value forces a false "submit failed" on a
    // busy disk, and the natural retry then creates a DUPLICATE job.
    static session connect(const std::string& host = "localhost",
                           std::uint16_t port = default_port,
                           int timeout_ms = 2000,
                           int io_timeout_ms = 0);

    // Connect to a daemon found by discover(): uses its captured address (falling
    // back to hostname) and advertised port. Throws if unreachable.
    static session connect(const host_info& daemon, int timeout_ms = 2000,
                           int io_timeout_ms = 0);

    // Connect to a daemon by its NAME in the fixed-list host config (IDD §6.6 —
    // the deterministic primary). Throws std::runtime_error if the name is not
    // configured or the daemon is unreachable. An empty name means "localhost".
    static session connect_to(const std::string& name, int timeout_ms = 2000,
                              int io_timeout_ms = 0);

    ~session();
    session(session&&) noexcept;
    session& operator=(session&&) noexcept;
    session(const session&)            = delete;
    session& operator=(const session&) = delete;

    // Submit an EDITED config — the simc workflow. `cfg` is loaded with
    // coastd::config(base), mutated with set()/get() in the console, then handed
    // here; its current bytes (cfg.dump()) are sent as the job's config content.
    // This is the primary path: it needs no config file staged anywhere, and a
    // sweep is `for (auto lr : rates) { auto c = base; c.set(ptr, lr); s.submit(ref, c); }`.
    // `name` sets the job mnemonic (basename); "" derives it from the config's base
    // file, else "config.json". Throws on the daemon's rejection with its message.
    std::string submit(const std::string& ref, const run_config& cfg,
                       const std::string& label = "", const std::string& message = "",
                       std::uint64_t steps_target = 0, int priority = 0,
                       const std::string& name = "");

    // Sugar: submit a config straight from a file. `config` is a path on THIS
    // machine — it is read here and its CONTENT is sent, so submitting to a remote
    // daemon needs nothing staged on that host. Prefer the run_config overload
    // above when you want to edit before submitting. Throws on rejection
    // (unreadable/invalid config locally, or the daemon's own reason).
    std::string submit(const std::string& ref, const std::string& config,
                       const std::string& label = "", const std::string& message = "",
                       std::uint64_t steps_target = 0, int priority = 0);

    [[nodiscard]] job_list jobs();   // the queue as an iterable handle (see job_list)
    void                   list();    // pretty-print the queue to stdout

    void cancel(const std::string& id);            // throws if unknown/terminal
    void reorder(const std::string& id, int pos);  // move within the pending queue
    // Move RELATIVE to the job's current position (negative = towards the front).
    // Prefer this whenever the position was derived from a polled snapshot: the
    // daemon resolves it against the live queue, so it cannot act on a stale index.
    void reorder_by(const std::string& id, int delta);

    [[nodiscard]] host_info info();                // machine_info as data
    void                    print_info();          // pretty-print machine_info

    // Is the underlying socket still usable? A transport error POISONS the
    // connection (see coastd_client), after which every call throws; this lets a
    // caller distinguish a dead transport from a daemon {"ok":false} rejection.
    [[nodiscard]] bool connected() const;

private:
    session();
    struct impl;
    std::unique_ptr<impl> p_;
};

// UDP discovery: broadcast a probe and return every daemon that answers within
// `timeout_ms`, as an iterable host_list handle. Best-effort (a filtered
// broadcast simply yields fewer hosts).
[[nodiscard]] host_list discover(int timeout_ms = 500);

// The fixed-list host config (IDD §6.6), as an iterable handle. Loaded from
// $COASTD_HOSTS / ~/.config/coastd/hosts.json; always contains "localhost".
[[nodiscard]] host_config_list configured_hosts();

// Pretty-print helpers (exposed for scripts; list()/print_info() use them).
void print_jobs(const job_list& jobs);
void print_hosts(const host_list& hosts);
void print_configured_hosts(const host_config_list& hosts);

// Convenience: discover and print in one call.
void print_discovered(int timeout_ms = 500);

// Convenience: print the configured fixed-list hosts.
void print_configured();

// ---------------------------------------------------------------------------
// clang-repl (console) note — everything above is callable INTERPRETED.
//
// The JIT resolves symbols against the host's -rdynamic table and cannot
// materialize the inline members of a std::vector instantiated over a
// coastd-specific type crossing the JIT boundary. That is precisely why the
// queue and discovery results are returned as the COMPILED handles job_list /
// host_list (every accessor is an out-of-line symbol) rather than a bare
// std::vector<job_info>/<host_info> — the same technique the h5 façade uses with
// row_view/row_range. So full programmatic use works from the console:
//     auto s = coastd::session::connect();
//     for (auto j : s.jobs()) if (j.status == "failed") s.cancel(j.id);
//     auto h = s.info(); std::printf("%s: %d slots\n", h.hostname.c_str(), h.slots);
// (Verified end-to-end against a live daemon in
//  tests/unit/console/test_console_repl.cpp.)
// ---------------------------------------------------------------------------

} // namespace coast::coastd

#endif // COAST_COASTD_CONSOLE_API_H
