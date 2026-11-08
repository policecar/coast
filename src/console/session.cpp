#include "console/session.h"

#include "console/quote_cpp.h"
#include "console/repl.h"
#include "gui_plot/console_plot.h"
#include "h5browse/console_api.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace console {

namespace {

// Split captured text into lines, appending each as an output record. Trailing
// partial line (no newline) is emitted too.
void append_lines(std::vector<output_line>& out, const std::string& text,
                  output_line::kind kind)
{
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t nl = text.find('\n', start);
        if (nl == std::string::npos) {
            out.push_back({kind, text.substr(start)});
            break;
        }
        out.push_back({kind, text.substr(start, nl - start)});
        start = nl + 1;
    }
}

// RAII capture of the process stdout+stderr fds into a pipe, drained by a
// background reader thread (so a macro that prints more than the pipe buffer
// cannot deadlock). POSIX; COAST targets Linux/macOS.
class fd_capture {
public:
    fd_capture()
    {
        std::fflush(stdout);
        std::fflush(stderr);
        saved_out_ = ::dup(STDOUT_FILENO);
        saved_err_ = ::dup(STDERR_FILENO);
        int pfd[2];
        if (::pipe(pfd) != 0) {
            // Capture stays inactive; nothing else will ever close the saved
            // fds, so release them here or we leak two handles per attempt.
            if (saved_out_ >= 0) ::close(saved_out_);
            if (saved_err_ >= 0) ::close(saved_err_);
            saved_out_ = saved_err_ = -1;
            return;
        }
        read_fd_  = pfd[0];
        write_fd_ = pfd[1];
        // Close-on-exec on the raw pipe ends: a child we posix_spawn (e.g. the
        // headless coast a step_async() launches) must NOT inherit the capture
        // pipe's write end, or the reader thread never sees EOF and stop()'s
        // join() blocks until that child exits — turning async runs synchronous.
        // dup2 below clears FD_CLOEXEC on fds 1/2, so stdout/stderr stay
        // inheritable (children redirect them to their own log). fcntl (not
        // pipe2) because macOS lacks pipe2.
        ::fcntl(read_fd_,  F_SETFD, FD_CLOEXEC);
        ::fcntl(write_fd_, F_SETFD, FD_CLOEXEC);
        ::dup2(write_fd_, STDOUT_FILENO);
        ::dup2(write_fd_, STDERR_FILENO);
        active_ = true;
        reader_ = std::thread([this] {
            char    tmp[4096];
            ssize_t n = 0;
            while ((n = ::read(read_fd_, tmp, sizeof tmp)) > 0) {
                buf_.append(tmp, static_cast<std::size_t>(n));
            }
        });
    }

    std::string stop()
    {
        if (!active_) return {};
        // Flush BOTH C stdio and C++ iostreams before restoring the fds — a
        // macro using std::cout buffers separately from printf and would
        // otherwise leak its output past the capture window.
        std::fflush(nullptr);
        std::cout.flush();
        std::cerr.flush();
        std::clog.flush();
        ::dup2(saved_out_, STDOUT_FILENO);
        ::dup2(saved_err_, STDERR_FILENO);
        ::close(write_fd_); // EOF to the reader
        if (reader_.joinable()) reader_.join();
        ::close(read_fd_);
        ::close(saved_out_);
        ::close(saved_err_);
        active_ = false;
        return std::move(buf_);
    }

    ~fd_capture()
    {
        // stop() joins the reader thread; a failed join throws std::system_error.
        // A throw escaping a destructor calls std::terminate — swallow it.
        try {
            if (active_) (void)stop();
        } catch (...) { // NOLINT(bugprone-empty-catch): must not throw from a dtor
        }
    }

    fd_capture(const fd_capture&)            = delete;
    fd_capture& operator=(const fd_capture&) = delete;

private:
    int         saved_out_ = -1;
    int         saved_err_ = -1;
    int         read_fd_   = -1;
    int         write_fd_  = -1;
    bool        active_    = false;
    std::thread reader_;
    std::string buf_; // written only by reader_, read after join — no lock needed
};

struct cmd_init {}; // build the interpreter if not yet built (pre-warm)
struct cmd_open {
    std::filesystem::path path;
    std::string           var; // variable name assigned synchronously by open_file
};
struct cmd_reset {};
struct cmd_submit {
    std::string code;
};
struct cmd_macro {
    std::filesystem::path file;
};
struct cmd_load {
    std::filesystem::path file;
};
struct cmd_incpath {
    std::string dir;
};
struct cmd_loadlib {
    std::string path;
};
using command = std::variant<cmd_init, cmd_open, cmd_reset, cmd_submit, cmd_macro,
                             cmd_load, cmd_incpath, cmd_loadlib>;

} // namespace

struct session::impl {
    explicit impl(hdf5::executor& e) : exec(e) {}

    hdf5::executor& exec;

    // -- session-thread-owned --
    // The interpreter, built once. The open files themselves live INSIDE it as
    // shared_ptr<file_console> variables (via open()), so the session no longer
    // owns a file_console — it only tracks the path->variable mapping for the UI
    // and for re-opening on reset.
    std::unique_ptr<repl> interp;

    // -- queues / sync --
    mutable std::mutex      qmtx;
    std::condition_variable qcv;
    std::condition_variable idle_cv;
    std::deque<command>     queue;
    bool                    busy = false;
    std::atomic<bool>       stop{false};

    mutable std::mutex       omtx;
    std::vector<output_line> outbox;

    // Open-file bookkeeping (GUI + console thread; guarded by file_mtx).
    // file_vars: path string -> assigned variable name; order: paths as opened;
    // used_names: every assigned name, for O(1) collision checks; injected: paths
    // whose `auto <var> = open(...)` has actually been run in the CURRENT
    // interpreter (a subset of file_vars — a path is reserved synchronously by
    // open_file but injected later on the console thread). do_reset re-opens only
    // injected paths, so a reset that races an in-flight cmd_open cannot inject
    // the same path twice.
    mutable std::mutex                 file_mtx;
    std::map<std::string, std::string> file_vars;
    std::vector<std::string>           file_order;
    std::set<std::string>              used_names;
    std::set<std::string>              injected;

    std::thread thread;

    void emit(output_line::kind k, std::string text)
    {
        std::scoped_lock lk(omtx);
        outbox.push_back({k, std::move(text)});
    }
    void emit_lines(const std::string& text, output_line::kind k)
    {
        if (text.empty()) return;
        std::scoped_lock lk(omtx);
        append_lines(outbox, text, k);
    }

    // Build the interpreter on first use (console thread). Idempotent: once an
    // interpreter object exists — ready or failed — this is a no-op that returns
    // its readiness, so a failed build is reported once, not on every command.
    // The interpreter needs NO file: its preamble pulls in console_api.h, so a
    // cell can open() files itself the moment this returns true.
    bool ensure_interp()
    {
        if (interp) {
            // Already attempted. If that attempt FAILED, every later command
            // routes through here and would otherwise bail in silence — tell the
            // user why instead of doing nothing. (The full init_error was emitted
            // on the first attempt below.)
            if (!interp->ready()) {
                emit(output_line::kind::error,
                     "interpreter unavailable — initialization failed (see earlier error)");
            }
            return interp->ready();
        }
        auto ip         = std::make_unique<repl>();
        const bool okay = ip->ready();
        if (!okay) {
            emit(output_line::kind::error,
                 "interpreter init failed: " + ip->init_error());
        } else {
            emit(output_line::kind::info, "C++ console ready");
        }
        interp = std::move(ip);
        return okay;
    }

    // Claim a unique identifier from `base`: `base` itself if free, else
    // `base_2`, `base_3`, … Adds the winner to used_names — the single registry
    // every console variable name (file handles + transient reservations) is
    // deduped against. Caller must hold file_mtx.
    std::string reserve_unique_name(const std::string& base)
    {
        std::string name = base;
        for (int n = 2; used_names.count(name) != 0; ++n) {
            name = base + "_" + std::to_string(n);
        }
        used_names.insert(name);
        return name;
    }

    // Drop a file's name reservation (path -> var). Called whenever an open
    // could not be completed — the interpreter build failed, or the injected
    // open() threw — so the path is not left permanently listed as open, which
    // would also lock it out of ever re-opening via the dedup path in
    // session::open_file.
    void drop_reservation(const std::string& path_str)
    {
        std::scoped_lock lk(file_mtx);
        if (auto it = file_vars.find(path_str); it != file_vars.end()) {
            used_names.erase(it->second);
            file_vars.erase(it);
            std::erase(file_order, path_str);
        }
        injected.erase(path_str);
    }

    // Inject `auto <var> = open("<path>")` into the live session. Assumes the
    // interpreter is ready and `var`/`path` were already reserved by open_file.
    // On failure the reservation is dropped so the name is free to retry.
    void inject_open(const std::string& path_str, const std::string& var)
    {
        fd_capture        cap;
        const std::string code =
            "auto " + var + " = h5b::console::open(" + detail::quote_cpp(path_str) + ");";
        const exec_result res      = interp->exec_guarded(code);
        const std::string captured = cap.stop();
        emit_lines(captured, output_line::kind::output);
        if (!res.ok) {
            emit(output_line::kind::error, "open failed: " + res.message);
            drop_reservation(path_str);
        } else {
            {
                std::scoped_lock lk(file_mtx);
                injected.insert(path_str); // now live in the current interpreter
            }
            emit(output_line::kind::info,
                 "opened " + path_str + " as " + var);
        }
    }

    void do_open(const std::filesystem::path& path, const std::string& var)
    {
        if (!ensure_interp()) {
            // The interpreter could not be built, so this open cannot happen;
            // drop the reservation open_file made or the path is stuck forever.
            drop_reservation(path.string());
            return;
        }
        inject_open(path.string(), var);
    }

    // Rebuild the interpreter from scratch, then re-open the files that were
    // open. This is the only reliable way to recover from a poisoned JIT: a
    // prior cell whose parse failed after emitting external/template symbol
    // references can leave clang's ORC layer with a dangling materialization
    // unit, so later cells fail with "Failed to materialize symbols" (an
    // inherent clang-repl limitation). A fresh clang::Interpreter clears it — at
    // the cost of all USER definitions; the managed file variables are restored.
    void do_reset()
    {
        // Re-open only paths actually INJECTED in the outgoing interpreter — not
        // paths merely reserved by an open_file() whose cmd_open is still queued
        // behind this reset. Re-injecting those here would double-open them once
        // their own cmd_open runs against the rebuilt interpreter.
        std::vector<std::pair<std::string, std::string>> reopen;
        {
            std::scoped_lock lk(file_mtx);
            for (const std::string& p : file_order) {
                if (injected.count(p) != 0) reopen.emplace_back(p, file_vars.at(p));
            }
        }
        emit(output_line::kind::info,
             "console reset — rebuilding interpreter (all definitions and plots cleared)");
        // Tearing down the interpreter runs its globals' destructors (measured
        // under ASan/LSan: no file_console leak on teardown), and file_console's
        // deleter is compiled into coast_lib, so every open handle — managed or
        // user-typed — is released cleanly here. No explicit pre-release needed.
        interp.reset();
        {
            std::scoped_lock lk(file_mtx);
            injected.clear(); // the rebuilt interpreter starts with no variables
            // Transient reservations (dataset binds and other user variables) do
            // not survive the rebuild — only the managed file handles are
            // re-opened. Rebuild the name registry from file_vars so those
            // transient names free up for reuse.
            used_names.clear();
            for (const auto& [p, var] : file_vars) used_names.insert(var);
        }
        if (!ensure_interp()) {
            // Rebuild failed: nothing backs the files we meant to re-open. Drop
            // their reservations so they are not reported as open and can be
            // retried once the interpreter is healthy again. (Paths reserved but
            // not yet injected are cleaned up by their own cmd_open hitting the
            // same ensure_interp() failure.)
            for (const auto& pr : reopen) drop_reservation(pr.first);
            return;
        }
        for (const auto& [p, var] : reopen) inject_open(p, var);
    }

    void do_exec(const std::string& code, bool macro)
    {
        if (!ensure_interp()) return;
        fd_capture  cap;
        exec_result res =
            macro ? interp->run_macro(std::filesystem::path{code}) : interp->exec_guarded(code);
        const std::string captured = cap.stop();

        emit_lines(captured, output_line::kind::output);
        if (!res.ok) emit(output_line::kind::error, res.message);
    }

    void do_load(const std::filesystem::path& file)
    {
        if (!ensure_interp()) return;
        fd_capture        cap;
        const exec_result res      = interp->load(file);
        const std::string captured = cap.stop();

        emit_lines(captured, output_line::kind::output);
        if (!res.ok) {
            emit(output_line::kind::error, res.message);
        } else {
            emit(output_line::kind::info, "loaded " + file.string());
        }
    }

    void run(const command& c)
    {
        std::visit(
            [&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, cmd_init>) {
                    (void)ensure_interp();
                } else if constexpr (std::is_same_v<T, cmd_open>) {
                    do_open(cmd.path, cmd.var);
                } else if constexpr (std::is_same_v<T, cmd_reset>) {
                    do_reset();
                } else if constexpr (std::is_same_v<T, cmd_submit>) {
                    emit(output_line::kind::input, cmd.code);
                    do_exec(cmd.code, /*macro=*/false);
                } else if constexpr (std::is_same_v<T, cmd_macro>) {
                    emit(output_line::kind::input, ".x " + cmd.file.string());
                    do_exec(cmd.file.string(), /*macro=*/true);
                } else if constexpr (std::is_same_v<T, cmd_load>) {
                    emit(output_line::kind::input, ".L " + cmd.file.string());
                    do_load(cmd.file);
                } else if constexpr (std::is_same_v<T, cmd_incpath>) {
                    emit(output_line::kind::input, ".I " + cmd.dir);
                    if (!ensure_interp()) {
                        // ensure_interp already reported the failure
                    } else if (const auto r = interp->add_include_path(cmd.dir); !r.ok) {
                        emit(output_line::kind::error, r.message);
                    } else {
                        emit(output_line::kind::info, "added include path: " + cmd.dir);
                    }
                } else if constexpr (std::is_same_v<T, cmd_loadlib>) {
                    emit(output_line::kind::input, ".load " + cmd.path);
                    if (!ensure_interp()) {
                        // ensure_interp already reported the failure
                    } else if (const auto r = interp->load_library(cmd.path); !r.ok) {
                        emit(output_line::kind::error, r.message);
                    } else {
                        emit(output_line::kind::info, "loaded library: " + cmd.path);
                    }
                }
            },
            c);
    }

    void thread_main()
    {
        while (!stop.load()) {
            command cmd;
            {
                std::unique_lock lk(qmtx);
                qcv.wait(lk, [&] { return stop.load() || !queue.empty(); });
                if (stop.load()) break;
                cmd  = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }
            run(cmd);
            {
                std::scoped_lock lk(qmtx);
                busy = false;
                if (queue.empty()) idle_cv.notify_all();
            }
        }
        // Tear down the interpreter on the owning thread (its file variables,
        // being shared_ptrs held inside it, are released with it).
        interp.reset();
    }

    void enqueue(command c)
    {
        {
            std::scoped_lock lk(qmtx);
            queue.push_back(std::move(c));
        }
        qcv.notify_one();
    }
};

session::session(hdf5::executor& exec) : p_(std::make_unique<impl>(exec))
{
    // Wire the console's open() façade to this session's executor so a cell can
    // open files itself (auto f = open("...")). The session owns the executor
    // reference and is the console's single client, so this is the authoritative
    // place to register it — no separate wiring step is needed.
    h5b::console::set_executor(exec);
    p_->thread = std::thread([this] { p_->thread_main(); });
}

session::~session()
{
    p_->stop.store(true);
    p_->qcv.notify_all();
    // Also wake anyone parked in wait_idle(): the session thread is stopping and
    // will not drain a non-empty queue, so the idle predicate must fall through
    // on shutdown rather than block forever.
    p_->idle_cv.notify_all();
    if (p_->thread.joinable()) p_->thread.join();
    // Thread is joined — no more open() calls can run. Un-register our executor
    // so the global does not dangle past this session (compare-and-clear leaves a
    // later session's registration intact).
    h5b::console::clear_executor(p_->exec);
}

void session::start() { p_->enqueue(cmd_init{}); }

std::string session::open_file(std::filesystem::path path)
{
    const std::string key = path.string();
    std::string       var;
    {
        std::scoped_lock lk(p_->file_mtx);
        // Already open: return the same variable, do not re-inject (dedup).
        if (auto it = p_->file_vars.find(key); it != p_->file_vars.end()) {
            return it->second;
        }
        // Reserve a unique variable name from the stem-derived default (shared
        // registry: dedups against file handles AND transient reservations).
        var = p_->reserve_unique_name(h5b::console::var_name_for(path));
        p_->file_vars.emplace(key, var);
        p_->file_order.push_back(key);
    }
    p_->enqueue(cmd_open{std::move(path), var});
    return var;
}
std::string session::reserve_var(const std::string& base)
{
    std::scoped_lock lk(p_->file_mtx);
    return p_->reserve_unique_name(base);
}
bool session::block_var(const std::string& name)
{
    std::scoped_lock lk(p_->file_mtx);
    return p_->used_names.insert(name).second; // false if already present
}
bool session::release_var(const std::string& name)
{
    std::scoped_lock lk(p_->file_mtx);
    // A live file handle's name is owned by open_file()/reset() and its variable
    // still exists in the interpreter — refuse to free it, or a later allocation
    // could collide with the open file.
    for (const auto& [path, var] : p_->file_vars) {
        if (var == name) return false;
    }
    return p_->used_names.erase(name) != 0;
}
void session::reset() { p_->enqueue(cmd_reset{}); }
void session::submit(std::string code) { p_->enqueue(cmd_submit{std::move(code)}); }
void session::bind_dataset(const std::string& var, const std::string& file_var,
                           const std::string& dataset)
{
    // quote_cpp escapes the dataset name as a C++ string literal so quotes or
    // backslashes in an HDF5 name can't break out of it and inject code.
    p_->enqueue(cmd_submit{"auto " + var + " = " + file_var + "->dataset(" +
                           detail::quote_cpp(dataset) + ");"});
}
void session::load(std::filesystem::path file)
{
    p_->enqueue(cmd_load{std::move(file)});
}
void session::run_macro(std::filesystem::path file)
{
    p_->enqueue(cmd_macro{std::move(file)});
}
void session::add_include_path(std::string dir)
{
    p_->enqueue(cmd_incpath{std::move(dir)});
}
void session::load_library(std::string path)
{
    p_->enqueue(cmd_loadlib{std::move(path)});
}

std::vector<output_line> session::drain_output()
{
    std::scoped_lock lk(p_->omtx);
    std::vector<output_line> out;
    out.swap(p_->outbox);
    return out;
}

bool session::busy() const
{
    std::scoped_lock lk(p_->qmtx);
    return p_->busy || !p_->queue.empty();
}

std::string session::current_file() const
{
    std::scoped_lock lk(p_->file_mtx);
    return p_->file_order.empty() ? std::string{} : p_->file_order.back();
}

std::vector<std::string> session::open_files() const
{
    std::scoped_lock lk(p_->file_mtx);
    return p_->file_order;
}

void session::wait_idle()
{
    std::unique_lock lk(p_->qmtx);
    p_->idle_cv.wait(
        lk, [&] { return p_->stop.load() || (p_->queue.empty() && !p_->busy); });
}

} // namespace console
