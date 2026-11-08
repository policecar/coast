#ifndef COAST_H5_BROWSER_WINDOW_H
#define COAST_H5_BROWSER_WINDOW_H

// HDF5 log-file browser panel (Analysis menu). Thin renderer over
// h5b::scan_engine: consumes snapshots per frame, enqueues commands,
// never blocks on the engine or libhdf5 (IDD §6 GUI panel).
//
// Follows the buffer_replay_dialog conventions: constructed with the
// process-wide hdf5::executor, rendered via render(bool*) behind an
// external show_ flag owned by gui_manager.

// Note: imgui.h must precede imfilebrowser.h.
#include "imgui.h"
#include "imfilebrowser.h"

#include "h5browse/scan_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hdf5 {
class executor;
}

namespace coast {

class console_iface; // optional C++ console the browser can drive

class h5_browser_window {
public:
    explicit h5_browser_window(hdf5::executor& h5_exec);

    h5_browser_window(const h5_browser_window&)            = delete;
    h5_browser_window& operator=(const h5_browser_window&) = delete;

    void render(bool* open);

    // Non-owning; set by gui_manager when a console is present. Enables the
    // "Console" toolbar button and the dataset right-click "create variable".
    void set_console(console_iface* console) { console_ = console; }

    // Open `path` as if it had been picked in the file dialog. Lets another panel
    // hand this one a file — the campaign browser opens a selected run's observe
    // file this way. Asynchronous, like every other engine command: the tree
    // appears once the scan engine reports it.
    //
    // Also RAISES this window: it loads the file into the panel, and a panel the
    // user cannot see is not a place to put data they just asked for. The window is
    // brought to the front here (SetNextWindowFocus, so an already-open but buried
    // browser surfaces); making a HIDDEN one visible needs the owner, because the
    // visible flag is passed in by render(bool*) — see take_visible_request().
    void open_path(const std::filesystem::path& path);

    // Consume a pending "make me visible" request raised by open_path(). Called by
    // the owner of the show flag (gui_manager) each frame. Returns true at most once
    // per open_path(), so it can never fight a user who closes the window afterwards.
    [[nodiscard]] bool take_visible_request();

private:
    void consume_snapshots();
    void reset_view_state();

    void render_toolbar();
    void render_tree();
    bool render_tree_node(const h5b::tree_node& node,
                          const std::string& search_lower);
    // Bind `display_name` to a console auto-variable (open the file in the
    // console, submit the binding, reveal it). No-op without a console/open
    // file. Shared by the right-click menu and the double-click shortcut.
    void bind_console_variable(const std::string& display_name);
    void render_stats_tab();
    void render_table_tab();
    void render_graph_tab();
    void render_histogram_tab();
    void render_pairwise_tab();
    void render_filter_editor();
    void render_log_panel();
    // Orange "recompute" banner when a result was computed under a filter
    // that differs from the editor's current state. Returns staleness.
    // Draws an amber "recompute" banner when the shown result no longer
    // matches the current inputs. `result_desc` is the result's filter
    // provenance (compared to the live filter); `extra_stale` folds in any
    // non-filter input a view recomputes on (graph column, pairwise
    // metric/k/seed/p). Returns whether the banner was drawn.
    bool filter_stale_banner(const std::string& result_desc,
                             bool extra_stale = false) const;

    [[nodiscard]] h5b::row_filter build_filter() const;
    [[nodiscard]] const h5b::dataset_desc* selected_desc() const;

    console_iface*     console_ = nullptr; // non-owning; may be null

    // Raised by open_path(), consumed by take_visible_request() / the next render.
    bool               want_visible_ = false;
    bool               focus_next_   = false;
    h5b::scan_engine   engine_;
    ImGui::FileBrowser file_browser_;

    // latest snapshots (kept across frames; engine returns nullptr only
    // before the first publication)
    std::shared_ptr<const h5b::file_snapshot>      file_;
    std::shared_ptr<const h5b::stats_snapshot>     stats_;
    std::shared_ptr<const h5b::histogram_snapshot> hist_;
    std::shared_ptr<const h5b::pairwise_snapshot>  pair_;
    std::shared_ptr<const h5b::row_index_snapshot> ridx_;
    std::shared_ptr<const h5b::series_snapshot>    series_;
    const h5b::file_snapshot* last_file_ = nullptr; // open-change detection

    std::vector<h5b::log_entry> log_lines_; // display buffer (capped)

    // -- view state --
    int  selected_dataset_ = -1;
    char search_[128]      = {};

    // filter editor (shared by table / graph / histogram / pairwise).
    // filter_enabled_ is the master gate: when off, build_filter() is empty
    // regardless of the editor's contents.
    struct condition_row {
        int    column = 0;
        int    op     = 0; // index into the filter_op UI order
        double value  = 0.0;
    };
    std::vector<condition_row> filter_conds_;
    bool          filter_enabled_ = false;
    bool          use_row_range_  = false;
    std::uint64_t row_begin_      = 0;
    std::uint64_t row_end_        = 0;

    // table view
    int table_col_offset_ = 0;

    // graph view
    int graph_column_ = 0;
    const h5b::series_snapshot* last_series_plotted_ = nullptr; // y-refit trigger

    // histogram view
    int hist_column_ = 0;
    int hist_bins_   = 128; // rebin target; divides 512

    // pairwise view
    int           metric_index_ = 0;
    float         minkowski_p_  = 2.0f;
    int           sample_k_     = 2000;
    std::uint64_t seed_         = 42;
    bool          full_run_     = false;
    int           pair_bins_    = 256; // rebin target; divides 4096
};

} // namespace coast

#endif // COAST_H5_BROWSER_WINDOW_H
