#ifndef COAST_CONSOLE_REPL_H
#define COAST_CONSOLE_REPL_H

// Embedded C++ interpreter (clang-repl) for the HDF5 Browser console
// (design/console_api_design.md). Ready the moment it is built — no file needs
// to be open. Its preamble pulls in console_api.h, so a cell opens files itself
// via `auto f = open("path.h5")` (returns a shared handle; many files at once)
// and runs ROOT-style macros over them.
//
// All LLVM/Clang types are confined to the .cpp behind a pimpl, so this header
// is dependency-free and safe to include from code compiled without
// COAST_WITH_CONSOLE.
//
// Safety model (validated by the console spikes): compiled-code exceptions are
// catchable across the JIT boundary, but an UNCAUGHT throw aborts the process.
// exec_guarded() therefore runs its input UNWRAPPED and catches the throw on the
// HOST side, around Execute(). It is not an interpreter-side try/catch: that was
// tried and reversed, because the wrapper is a scope and it destroyed exactly
// the top-level declarations a REPL exists to accumulate. Running unwrapped is
// what lets exec_guarded() both guard AND persist declarations.
//
// Consequently exec_guarded() is the path for ALL untrusted input — cells and
// macro files alike. exec() is raw and unguarded, reserved for code this class
// controls (the preamble), where a throw would be a COAST bug.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace console {

struct exec_result {
    bool        ok = false;
    std::string message; // error / diagnostic text when !ok
};

class repl {
public:
    // Builds the interpreter (native target + clang::Interpreter), sets the
    // include path to COAST's headers, and runs the preamble that pulls in
    // console_api.h. Check ready() afterwards.
    explicit repl(std::vector<std::string> extra_include_dirs = {});
    ~repl();

    repl(const repl&)            = delete;
    repl& operator=(const repl&) = delete;

    [[nodiscard]] bool               ready()      const;
    [[nodiscard]] const std::string& init_error() const;

    // Raw execution — NO exception guard, NO #include isolation: an uncaught
    // throw here aborts the process. Reserved for code this class controls (the
    // preamble), where a throw would be a COAST bug. Never feed it user input;
    // use exec_guarded().
    exec_result exec(const std::string& code);

    // Guarded execution — the path for ALL untrusted input, statements and
    // declarations alike. Runs the code unwrapped (so top-level declarations
    // persist) and catches an uncaught throw host-side, reporting it as an error
    // string instead of aborting. Also isolates each #include into its own
    // Parse() unit so a parse error cannot poison the interpreter.
    exec_result exec_guarded(const std::string& code);

    exec_result load(const std::filesystem::path& file);      // .L: guarded, declarations persist
    exec_result run_macro(const std::filesystem::path& file); // .x: load + guarded call of stem()

    // .I: add a header search directory at runtime (so macros can #include a
    // library's headers). .load: load a shared library so the JIT can resolve
    // its symbols (needed for non-header-only libraries).
    exec_result add_include_path(const std::string& dir);
    exec_result load_library(const std::string& path);

private:
    // Guarded execution of a single Parse/Execute unit. exec_guarded() splits a
    // submission into units (isolating each #include) and drives this per unit.
    exec_result exec_unit_guarded(const std::string& code);

    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace console

#endif // COAST_CONSOLE_REPL_H
