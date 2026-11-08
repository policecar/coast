#ifndef COAST_CONSOLE_PANEL_H
#define COAST_CONSOLE_PANEL_H

// ImGui panel for the embedded C++ console (design/console_api_design.md).
// Implements coast::console_iface so gui_manager can menu/render it and the
// HDF5 browser can open the browsed file in it and push variable bindings.
// Owns a console::session (interpreter on its own thread). Lives in
// coast_console so coast_lib stays LLVM-free.

// Note: imgui.h must precede imfilebrowser.h.
#include "imgui.h"
#include "imfilebrowser.h"

#include "console/session.h"
#include "gui_core/console_iface.h"

#include <cstddef>
#include <string>
#include <vector>

namespace hdf5 {
class executor;
}

namespace console {

class console_panel : public coast::console_iface {
public:
    explicit console_panel(hdf5::executor& exec);

    [[nodiscard]] const char* menu_label() const override { return "C++ Console..."; }
    [[nodiscard]] bool        visible() const override { return show_; }
    // Pre-warm the interpreter the first time the console is shown, so the first
    // cell runs without the one-time build stall. Idempotent and off the GUI
    // thread (see session::start).
    void set_visible(bool v) override
    {
        if (v && !started_) {
            session_.start();
            started_ = true;
        }
        show_ = v;
    }

    void        render() override;
    std::string open_file(const std::filesystem::path& path) override;
    std::string reserve_var(const std::string& base) override;
    void        submit(const std::string& code) override;
    void        bind_dataset(const std::string& var, const std::string& file_var,
                             const std::string& dataset) override;

private:
    static constexpr std::size_t max_log_lines = 5000;

    void run_current_input();
    // Route one console line: ".? <topic>" cheatsheet, ".L/.x <file>" macro
    // meta-commands to the session, everything else to submit().
    void dispatch_line(const std::string& line);
    // Append a cheatsheet for `topic` ("", "h5", "dataset", "row", "metric") to
    // the pending buffer (spliced into the log at frame top).
    void show_help(const std::string& topic);
    // Render one echoed input as a notebook-style cell: Run / Edit / Copy plus
    // the code in a read-only (selectable) box. `id` scopes the ImGui widgets.
    void render_input_cell(const std::string& code, int id);

    session                  session_;
    ImGui::FileBrowser       file_browser_;
    ImGui::FileBrowser       macro_browser_;
    std::vector<output_line> log_;
    std::vector<output_line> pending_; // locally-generated lines (e.g. .? help)
    char                     input_[8192]   = {};
    bool                     scroll_bottom_ = false;
    bool                     show_          = false;
    bool                     started_       = false;  // interpreter pre-warm fired
    float                    input_height_  = 120.0f; // resizable via the splitter
};

} // namespace console

#endif // COAST_CONSOLE_PANEL_H
