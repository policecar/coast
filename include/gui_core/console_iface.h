#ifndef COAST_CONSOLE_IFACE_H
#define COAST_CONSOLE_IFACE_H

// Abstract handle to the embedded C++ console. Used by gui_manager (menu toggle
// + per-frame render) and by the HDF5 browser (open the browsed file in the
// console, push a variable-binding statement). The concrete implementation
// (console::console_panel) lives in coast_console, so coast_lib carries no
// LLVM/Clang dependency — main.cpp constructs it and injects it through run_gui.

#include <filesystem>
#include <string>

namespace coast {

class console_iface {
public:
    virtual ~console_iface() = default;

    [[nodiscard]] virtual const char* menu_label() const = 0; // e.g. "C++ Console..."
    [[nodiscard]] virtual bool        visible() const     = 0;
    virtual void                      set_visible(bool v)  = 0;

    // Draw the window when visible(); no-op otherwise.
    virtual void render() = 0;

    // Open `path` into a session variable and return that variable's name (e.g.
    // "f_mnist_seed0"). Re-opening an already-open path returns the same name
    // without re-injecting. The console must be usable with several files open
    // at once, so this never disturbs a previously-opened file. The name is
    // assigned synchronously so the caller can build a follow-up statement such
    // as `<name>->dataset("...")` via submit().
    virtual std::string open_file(const std::filesystem::path& path) = 0;

    // Reserve a unique console variable name derived from `base` (returned as-is
    // if free, else `base_2`, …). Shares one identifier namespace with the file
    // handles open_file() returns, so a caller building its own auto-variable —
    // e.g. binding a dataset — cannot collide with a file handle or another
    // reserved name. Synchronous, like open_file().
    virtual std::string reserve_var(const std::string& base) = 0;

    // Enqueue a line of code for execution (e.g. an auto-variable binding).
    virtual void submit(const std::string& code) = 0;

    // Bind dataset `dataset` off the file handle `file_var` (as returned by
    // open_file()) into a fresh session variable `var`, and enqueue it. The
    // implementation escapes `dataset` as a C++ string literal — HDF5 names can
    // contain quotes/backslashes, which spliced raw into the submitted statement
    // would break out of the literal and inject code into the JIT console. Use
    // this instead of hand-building `<var> = <file_var>->dataset("<dataset>")`.
    virtual void bind_dataset(const std::string& var, const std::string& file_var,
                              const std::string& dataset) = 0;
};

} // namespace coast

#endif // COAST_CONSOLE_IFACE_H
