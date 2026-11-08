#ifndef COAST_CONSOLE_SESSION_H
#define COAST_CONSOLE_SESSION_H

// Threaded wrapper around console::repl for the GUI (design/console_api_design.md).
// The interpreter is driven from a dedicated console thread so a long-running
// macro (which blocks on the HDF5 executor) never freezes the GUI. The GUI
// thread submits commands (non-blocking) and drains captured output.
//
// The interpreter is built ONCE, independent of any file, and is usable
// immediately (start() pre-warms it off the GUI thread). Files are not a
// precondition: open_file() injects `auto <var> = open("<path>")` into the live
// session, so several files can be open at once, each in its own variable. There
// is no single "current" file bound as `h5`.
//
// Output capture: interpreted printf/cout and compiler diagnostics are captured
// by redirecting the process stdout/stderr fds around each execution and are
// delivered as output_line records.
//
// LLVM-free header (pimpl): safe to include from GUI code compiled without the
// console.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace hdf5 {
class executor;
}

namespace console {

struct output_line {
    enum class kind : std::uint8_t { input, output, error, info };
    kind        k = kind::output;
    std::string text;
};

class session {
public:
    explicit session(hdf5::executor& exec);
    ~session();

    session(const session&)            = delete;
    session& operator=(const session&) = delete;

    // -- commands (GUI thread; non-blocking enqueue) --

    // Build the interpreter if it is not built yet (idempotent). Cheap to call
    // repeatedly; the ~seconds-long build runs on the console thread, so the GUI
    // never blocks. Call it when the console first becomes visible so the first
    // cell runs without a stall.
    void start();

    // Open `path` into a fresh session variable and return that variable's name
    // (e.g. "f_mnist_seed0"). Deterministic from the path stem, disambiguated on
    // collision; re-opening an already-open path returns the SAME name and does
    // not re-inject. Name assignment is synchronous (so a caller can build a
    // follow-up statement like `<var>->dataset(...)`); the open itself runs on
    // the console thread. Does not disturb any other open file.
    std::string open_file(std::filesystem::path path);

    // Reserve a unique console variable name derived from `base` (returned as-is
    // if free, else `base_2`, `base_3`, …). Shares ONE identifier namespace with
    // the file-handle names open_file() hands out, so a caller generating its own
    // variable — e.g. the browser binding a dataset — cannot collide with a file
    // handle or another reserved name. Synchronous, like open_file(). Unlike a
    // file handle the reservation is transient: it does not survive reset() (the
    // definition it names is a user variable, cleared when the interpreter is
    // rebuilt), whereas file handles are re-opened.
    std::string reserve_var(const std::string& base);

    // Claim an EXACT name in the registry so the allocator will not hand it out.
    // Returns false if the name is already taken (a live file handle or a prior
    // reservation). Use to protect a name a caller intends to define itself.
    bool block_var(const std::string& name);

    // Release a name previously handed out by reserve_var() or claimed by
    // block_var(), so it can be allocated again. Returns false if the name was
    // not held, or if it is a live file handle (those are owned by open_file() /
    // reset() and must not be freed out from under their still-live variable).
    // NOTE: this frees the registry SLOT only; it does not undefine any variable
    // of that name already living in the interpreter — the caller is responsible
    // for not reusing a name whose definition still exists (a fresh `auto <name>`
    // would then hit a redefinition error), e.g. only release names that were
    // blocked/reserved but never defined, or defined and since dropped.
    bool release_var(const std::string& name);

    void reset();                                // rebuild interpreter, re-open the same files
    void submit(std::string code);               // guarded statement execution
    // Enqueue `auto <var> = <file_var>->dataset("<dataset>");` with `dataset`
    // escaped as a C++ string literal, so an HDF5 name containing quotes or
    // backslashes cannot break out of the literal and inject code.
    void bind_dataset(const std::string& var, const std::string& file_var,
                      const std::string& dataset);
    void load(std::filesystem::path file);       // .L: guarded, declarations persist
    void run_macro(std::filesystem::path file);  // .x: load + call stem()
    void add_include_path(std::string dir);      // .I: header search dir
    void load_library(std::string path);         // .load: shared library

    // -- GUI thread --
    [[nodiscard]] std::vector<output_line> drain_output();
    [[nodiscard]] bool                     busy() const;         // a command is in flight
    // Most recently opened path, "" if none. No production caller since the
    // console went multi-file (the toolbar uses open_files()); retained for tests
    // that assert open-recency ordering.
    [[nodiscard]] std::string              current_file() const;
    [[nodiscard]] std::vector<std::string> open_files() const;   // all open paths, in open order

    // Blocks until every queued command has been processed (tests / shutdown).
    void wait_idle();

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace console

#endif // COAST_CONSOLE_SESSION_H
