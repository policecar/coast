#include "coastd_console/coastd_console_api.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include "client/coastd_client.h"

// The façade lives in `coast::coastd`, which shadows the top-level `coastd`
// namespace of the client/protocol library — so reach the wire library through
// this unambiguous alias rather than a bare `wire::client::`/`proto::`.
namespace wire = ::coastd;

namespace coast::coastd {

namespace {

job_info to_job_info(const wire::proto::job_summary& s) {
    return job_info{s.id, s.ref_symbolic, s.status, s.bucket, s.submitted_at,
                    s.user_message, s.checkpoint, s.failure_reason,
                    s.steps_done, s.live_step, s.steps_target, s.priority,
                    s.cancel_requested};
}

host_info to_host_info(const wire::proto::machine_summary& m) {
    return host_info{m.hostname, m.address, m.os, m.coastd_version,
                     m.runs_root, m.repo_root, m.slots, m.tcp_port, m.data_link_ok};
}

configured_host to_configured_host(const wire::client::host_entry& e) {
    return configured_host{e.name, e.host, static_cast<int>(e.port)};
}

} // namespace

// ---------------------------------------------------------------------------
// job_list / host_list — the std::vector lives HERE, behind compiled accessors,
// so it never crosses the clang-repl JIT boundary (see the header rationale).
// ---------------------------------------------------------------------------
struct job_list::impl  { std::vector<job_info>  rows; };
struct host_list::impl { std::vector<host_info> rows; };

job_list::job_list() : p_(std::make_unique<impl>()) {}
job_list::job_list(const job_list& o) : p_(std::make_unique<impl>(*o.p_)) {}
job_list::job_list(job_list&&) noexcept = default;
job_list& job_list::operator=(const job_list& o) {
    if (this != &o) p_ = std::make_unique<impl>(*o.p_);
    return *this;
}
job_list& job_list::operator=(job_list&&) noexcept = default;
job_list::~job_list() = default;

int  job_list::size()  const { return static_cast<int>(p_->rows.size()); }
bool job_list::empty() const { return p_->rows.empty(); }
job_info job_list::at(int i) const {
    if (i < 0 || i >= size()) throw std::out_of_range("job_list::at: index out of range");
    return p_->rows[static_cast<std::size_t>(i)];
}
job_info job_list::operator[](int i) const { return p_->rows[static_cast<std::size_t>(i)]; }
job_info job_list::iterator::operator*() const { return owner_->at(i_); }

host_list::host_list() : p_(std::make_unique<impl>()) {}
host_list::host_list(const host_list& o) : p_(std::make_unique<impl>(*o.p_)) {}
host_list::host_list(host_list&&) noexcept = default;
host_list& host_list::operator=(const host_list& o) {
    if (this != &o) p_ = std::make_unique<impl>(*o.p_);
    return *this;
}
host_list& host_list::operator=(host_list&&) noexcept = default;
host_list::~host_list() = default;

int  host_list::size()  const { return static_cast<int>(p_->rows.size()); }
bool host_list::empty() const { return p_->rows.empty(); }
host_info host_list::at(int i) const {
    if (i < 0 || i >= size()) throw std::out_of_range("host_list::at: index out of range");
    return p_->rows[static_cast<std::size_t>(i)];
}
host_info host_list::operator[](int i) const { return p_->rows[static_cast<std::size_t>(i)]; }
host_info host_list::iterator::operator*() const { return owner_->at(i_); }

struct host_config_list::impl { std::vector<configured_host> rows; std::string warning; };

host_config_list::host_config_list() : p_(std::make_unique<impl>()) {}
host_config_list::host_config_list(const host_config_list& o) : p_(std::make_unique<impl>(*o.p_)) {}
host_config_list::host_config_list(host_config_list&&) noexcept = default;
host_config_list& host_config_list::operator=(const host_config_list& o) {
    if (this != &o) p_ = std::make_unique<impl>(*o.p_);
    return *this;
}
host_config_list& host_config_list::operator=(host_config_list&&) noexcept = default;
host_config_list::~host_config_list() = default;

int  host_config_list::size()  const { return static_cast<int>(p_->rows.size()); }
bool host_config_list::empty() const { return p_->rows.empty(); }
std::string host_config_list::warning() const { return p_ ? p_->warning : std::string{}; }
configured_host host_config_list::at(int i) const {
    if (i < 0 || i >= size()) throw std::out_of_range("host_config_list::at: index out of range");
    return p_->rows[static_cast<std::size_t>(i)];
}
configured_host host_config_list::operator[](int i) const { return p_->rows[static_cast<std::size_t>(i)]; }
configured_host host_config_list::iterator::operator*() const { return owner_->at(i_); }

// ---------------------------------------------------------------------------
// session
// ---------------------------------------------------------------------------
struct session::impl {
    wire::client::connection conn;
};

session::session() = default;
session::~session() = default;
session::session(session&&) noexcept = default;
session& session::operator=(session&&) noexcept = default;

session session::connect(const std::string& host, std::uint16_t port, int timeout_ms,
                         int io_timeout_ms) {
    wire::client::endpoint ep;
    ep.host = host;
    ep.port = port;
    std::string err;
    auto conn = wire::client::connection::open(ep, std::chrono::milliseconds(timeout_ms), err,
                                               std::chrono::milliseconds(io_timeout_ms));
    if (!conn)
        throw std::runtime_error("coastd: connect " + host + ":" + std::to_string(port) + ": " + err);
    session s;
    s.p_ = std::make_unique<impl>(impl{std::move(*conn)});
    return s;
}

session session::connect(const host_info& daemon, int timeout_ms, int io_timeout_ms) {
    const std::string host = !daemon.address.empty() ? daemon.address : daemon.hostname;
    const auto port = static_cast<std::uint16_t>(daemon.port ? daemon.port : default_port);
    return connect(host, port, timeout_ms, io_timeout_ms);
}

session session::connect_to(const std::string& name, int timeout_ms, int io_timeout_ms) {
    std::string err;
    auto conn = wire::client::connect_to(name, std::chrono::milliseconds(timeout_ms), err,
                                         std::chrono::milliseconds(io_timeout_ms));
    if (!conn)
        throw std::runtime_error("coastd: connect_to '" + name + "': " + err);
    session s;
    s.p_ = std::make_unique<impl>(impl{std::move(*conn)});
    return s;
}

std::string session::submit(const std::string& ref, const run_config& cfg,
                            const std::string& label, const std::string& message,
                            std::uint64_t steps_target, int priority,
                            const std::string& name) {
    wire::client::submit_args a;
    a.ref = ref; a.label = label; a.message = message;
    a.steps_target = steps_target; a.priority = priority;
    // Send the EDITED config's current bytes directly — no file involved. The
    // daemon receives config CONTENT, never a path (protocol.h / INSTALL.md §5.4),
    // so an in-memory config is exactly what it wants.
    //
    // dump() is NOT unconditionally safe, despite having been parsed once: the
    // config is EDITED in between, and run_config::set(string_view, string_view)
    // stores its bytes verbatim (run_config.cpp). A REPL user pasting a string that
    // is not valid UTF-8 — from a latin-1 file, a mangled path — puts bytes into the
    // tree that dump()'s strict error handler refuses, throwing type_error.316. That
    // is not a std::runtime_error, so it would escape this header's stated error
    // model (every failure is a runtime_error carrying a usable message) and unwind
    // through the JIT into the REPL. Convert it here, as load_config_file() already
    // does for arbitrary on-disk bytes.
    try {
        a.config_text = cfg.dump();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "coastd: this config cannot be serialized — a value contains bytes that are "
            "not valid UTF-8 (set() stores strings verbatim). Underlying error: " +
            std::string(e.what()));
    }
    a.config_name = !name.empty()
                        ? name
                        : (!cfg.base_path().empty()
                               ? std::filesystem::path(cfg.base_path()).filename().string()
                               : std::string("config.json"));
    std::string err;
    auto id = p_->conn.submit(a, err);
    if (!id) throw std::runtime_error("coastd: submit rejected: " + err);
    return *id;
}

std::string session::submit(const std::string& ref, const std::string& config,
                            const std::string& label, const std::string& message,
                            std::uint64_t steps_target, int priority) {
    wire::client::submit_args a;
    a.ref = ref; a.label = label; a.message = message;
    a.steps_target = steps_target; a.priority = priority;
    std::string err;
    // `config` is a path on THIS machine: read it here and send its content. The
    // daemon may be another host, where the path means nothing (or, worse, names a
    // different file) — see protocol.h.
    if (!wire::client::load_config_file(a, config, err))
        throw std::runtime_error("coastd: " + err);
    auto id = p_->conn.submit(a, err);
    if (!id) throw std::runtime_error("coastd: submit rejected: " + err);
    return *id;
}

job_list session::jobs() {
    std::string err;
    auto l = p_->conn.list(err);
    if (!l) throw std::runtime_error("coastd: list failed: " + err);
    job_list jl;                       // session is a friend -> may fill the hidden vector
    jl.p_->rows.reserve(l->size());
    for (const auto& s : *l) jl.p_->rows.push_back(to_job_info(s));
    return jl;
}

void session::cancel(const std::string& id) {
    std::string err;
    if (!p_->conn.cancel(id, err)) throw std::runtime_error("coastd: cancel failed: " + err);
}

void session::reorder_by(const std::string& id, int delta) {
    std::string err;
    if (!p_->conn.reorder_by(id, delta, err))
        throw std::runtime_error("coastd: reorder failed: " + err);
}

void session::reorder(const std::string& id, int pos) {
    std::string err;
    if (!p_->conn.reorder(id, pos, err)) throw std::runtime_error("coastd: reorder failed: " + err);
}

host_info session::info() {
    std::string err;
    auto mi = p_->conn.machine_info(err);
    if (!mi) throw std::runtime_error("coastd: machine_info failed: " + err);
    return to_host_info(*mi);
}

bool session::connected() const { return p_ && p_->conn.alive(); }

void session::list()       { print_jobs(jobs()); }
void session::print_info() {
    const host_info h = info();
    std::printf("%s  os=%s  coastd=%s  slots=%d  runs=%s  repo=%s  relative-data=%s\n",
                h.hostname.c_str(), h.os.c_str(), h.coastd_version.c_str(),
                h.slots, h.runs_root.c_str(), h.repo_root.c_str(),
                h.data_link_ok ? "yes" : "NO (use absolute dataset paths)");
}

host_list discover(int timeout_ms) {
    host_list hl;                      // discover() is a friend of host_list
    for (const auto& m : wire::client::discover(std::chrono::milliseconds(timeout_ms)))
        hl.p_->rows.push_back(to_host_info(m));
    return hl;
}

host_config_list configured_hosts() {
    host_config_list hl;               // configured_hosts() is a friend
    std::string warn;
    const auto cfg = wire::client::host_config::load_default(warn);
    for (const auto& e : cfg.entries()) hl.p_->rows.push_back(to_configured_host(e));
    hl.p_->warning = warn;    // also retrievable by a GUI (see host_config_list::warning)
    if (!warn.empty()) std::fprintf(stderr, "coastd: %s\n", warn.c_str());
    return hl;
}

void print_discovered(int timeout_ms) { print_hosts(discover(timeout_ms)); }

void print_configured() { print_configured_hosts(configured_hosts()); }

void print_jobs(const job_list& jobs) {
    if (jobs.empty()) { std::puts("(no jobs)"); return; }
    std::printf("%-30s %-10s %-10s %12s %-10s  %s\n",
                "ID", "STATUS", "BUCKET", "STEPS", "REF", "NOTE");
    for (const job_info& j : jobs) {
        char steps[32];
        // Show the current position: the live tailed step when a running job is
        // ahead of its last checkpoint, else the durable checkpoint step. This is
        // what makes a running job visibly advance between checkpoints.
        const std::uint64_t cur = j.live_step > j.steps_done ? j.live_step : j.steps_done;
        std::snprintf(steps, sizeof steps, "%llu/%llu",
                      static_cast<unsigned long long>(cur),
                      static_cast<unsigned long long>(j.steps_target));
        const std::string note = !j.failure_reason.empty() ? j.failure_reason : j.user_message;
        // A pending cancel outranks the raw status in the STATUS column: the job
        // is still `running`, but what the operator needs to see is that their
        // cancel was accepted and is in progress.
        const std::string shown = j.cancel_requested && j.bucket == "active"
                                ? "cancelling" : j.status;
        std::printf("%-30s %-10s %-10s %12s %-10s  %s\n",
                    j.id.c_str(), shown.c_str(), j.bucket.c_str(), steps,
                    j.ref.c_str(), note.c_str());
    }
}

void print_hosts(const host_list& hosts) {
    if (hosts.empty()) { std::puts("(no daemons)"); return; }
    for (const host_info& h : hosts) {
        // Lead with the connectable address:port (from discovery) when we have it.
        if (!h.address.empty())
            std::printf("%s:%d  (%s)  os=%s  coastd=%s  slots=%d\n",
                        h.address.c_str(), h.port, h.hostname.c_str(), h.os.c_str(),
                        h.coastd_version.c_str(), h.slots);
        else
            std::printf("%s  os=%s  coastd=%s  slots=%d  runs=%s  repo=%s\n",
                        h.hostname.c_str(), h.os.c_str(), h.coastd_version.c_str(),
                        h.slots, h.runs_root.c_str(), h.repo_root.c_str());
    }
}

void print_configured_hosts(const host_config_list& hosts) {
    if (hosts.empty()) { std::puts("(no configured hosts)"); return; }
    std::printf("%-16s %s\n", "NAME", "ENDPOINT");
    for (const configured_host& h : hosts)
        std::printf("%-16s %s:%d\n", h.name.c_str(), h.host.c_str(), h.port);
}

} // namespace coast::coastd
