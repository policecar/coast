#ifndef COAST_METRIC_EXPLORER_H
#define COAST_METRIC_EXPLORER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "imgui.h"          // must precede imfilebrowser.h
#include "imfilebrowser.h"

#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"
#include "gui_core/dataset_access.h"
#include "gui_tools/dataset_source.h"
#include "gui_tools/matrix_canvas.h"
#include "gui_core/thumbnail_grid.h"
#include "tools/distortion.h"
#include "tools/ensemble_code.h"
#include "tools/metric.h"

namespace hdf5 { class executor; }

namespace coast {

/**
 * "Tools > Metric Explorer" panel.
 *
 * Builds a distance/similarity matrix metric(row_i, col_j) over two user-chosen
 * sample sets and displays it via a zoomable matrix_canvas.
 *
 * Data comes from interchangeable dataset_sources (imported configs OR generated
 * synthetic ensemble codes), each with a stable id. An axis references a dataset by
 * (source_id, local_idx) so a selection survives unrelated sources being added or
 * removed. Two extra modes layer on top:
 *   - distortion mode: columns = the row selection passed through a seeded
 *     distortion (Gaussian noise / dropout), making the matrix R×R.
 *   - synthetic sources: phenomenological ensemble codes whose ground-truth cluster
 *     structure is controllable, displayed as a 1D strip or 2D square.
 */
class metric_explorer {
public:
    explicit metric_explorer(hdf5::executor& h5_exec);
    ~metric_explorer();

    void render();
    bool& visible() { return is_visible; }
    void free_resources();

private:
    // A flat dataset entry: a resolved handle plus which source/local index it came
    // from (the stable reference an axis stores).
    struct dataset_entry {
        dataset_handle handle;
        std::size_t    source_id = 0;
        int            local_idx = -1;
    };

    // Cached per-source display metadata (rebuilt alongside the flat list), so the
    // sources panel never re-enumerates per frame.
    struct source_meta {
        std::size_t id = 0;
        std::string label;
        std::size_t bytes = 0;
        int         dataset_count = 0;
    };

    // One matrix axis: a STABLE reference to a dataset by (source_id, local_idx),
    // which samples are selected, and the header thumbnail textures.
    struct axis {
        std::size_t source_id = 0;     // 0 = unset
        int         local_idx = -1;    // <0 = unset
        std::set<std::size_t> selected;
        std::vector<gl_texture> thumbs;
    };

    hdf5::executor& h5_exec_;
    bool is_visible = false;

    // Interchangeable sources (imported or synthetic). datasets_ is the flat
    // concatenation; handle accessors point into a source, so a source must outlive
    // any handle derived from it (sources held by unique_ptr -> stable address).
    std::vector<std::unique_ptr<dataset_source>> sources_;
    std::size_t next_source_id_ = 1;   // ids start at 1 (0 means "unset")

    std::vector<dataset_entry> datasets_;
    std::vector<source_meta>   source_meta_;
    std::size_t total_bytes_ = 0;

    axis rows_;
    axis cols_;

    metric_kind  metric_  = metric_kind::cosine_similarity;
    metric_params params_;
    colormap_t   palette_ = colormap_t::viridis;

    // Distortion mode (Layer 4): columns = rows passed through a distortion.
    bool              distort_mode_ = false;
    distortion_kind   distort_kind_ = distortion_kind::gaussian_noise;
    distortion_params distort_params_;

    // Global 1D/2D display toggle for synthetic sources (shared by all of them;
    // feature_dim is S^2 either way, so it never affects comparability).
    bool show_synth_2d_ = false;

    // Generator dialog state.
    bool                 open_gen_dialog_ = false;
    ensemble_code_params gen_params_;
    std::string          gen_error_;

    // Owned comparison buffers: built by compute() (applying distortion when on),
    // reused by compute_values() so a metric/p change never re-copies samples.
    std::vector<std::vector<float>> row_data_;
    std::vector<std::vector<float>> col_data_;

    // Computed matrix (row-major mat_rows_ x mat_cols_).
    std::vector<float> values_;
    int   mat_rows_ = 0;
    int   mat_cols_ = 0;
    float vmin_ = 0.0f;
    float vmax_ = 1.0f;
    bool  computed_ = false;
    std::string message_;   // error / status (e.g. dimension mismatch)

    // --- Sort + histogram state (v3) ---
    // Single active focus: at most one of row_order_/col_order_ is a non-identity
    // permutation. Clicking a column header sorts the ROWS (focus = col); clicking a
    // row header sorts the COLUMNS (focus = row).
    enum class sort_focus : std::uint8_t { none, row, col };
    sort_focus       sort_focus_ = sort_focus::none;
    int              sort_index_ = -1;     // UNDERLYING row/col index in focus (-1 = none)
    bool             sort_desc_  = false;  // current sort direction; toggled on re-click
    std::vector<int> row_order_;           // display->underlying row map (identity unless col-focus)
    std::vector<int> col_order_;           // display->underlying col map (identity unless row-focus)

    int              hist_nbins_ = 32;     // histogram bin count, slider [10,100]
    std::vector<int> hist_counts_;         // cached bin counts for the current slice
    float            hist_lo_ = 0.0f;      // current metric x-domain (theoretical)
    float            hist_hi_ = 1.0f;

    matrix_canvas canvas_;

    // Import UI state.
    ImGui::FileBrowser import_browser_;
    std::string        import_error_;

    // Sample-picker popup state.
    int            picker_axis_ = -1;   // 0 = rows, 1 = cols, -1 = closed
    bool           open_picker_pending_ = false;  // deferred OpenPopup (see open_picker)
    thumbnail_grid picker_grid_;
    int            random_n_ = 16;
    int            random_seed_ = 0;

    axis& axis_ref(int which) { return which == 0 ? rows_ : cols_; }

    // Source management.
    void import_config(const std::string& path);   // load + append a config (dedup'd)
    bool generate_codes();                          // build a synthetic source from gen_params_
    void remove_source(std::size_t idx);            // drop one source (preserves surviving axes)
    void clear_sources();                           // drop all sources
    void rebuild_dataset_list();                    // re-flatten datasets_ + re-resolve axes
    void reset_axes();                              // full wipe of axis/derived state

    // Axis resolution against the flat list.
    [[nodiscard]] int resolve_axis_flat(const axis& ax) const;  // -1 if its source is gone
    void clear_axis(axis& ax);                                  // drop selection/thumbs/ref

    void render_sources_panel();
    void render_setup_strip();
    void render_histogram_panel();   // nbins slider + palette-matched bars + slice label + Clear
    void render_axis_combo(const char* label, axis& ax);
    void render_generator_dialog();
    void render_picker_popup();
    void open_picker(int which);

    // Sort + histogram (v3).
    void refresh_histogram();        // derive domain from metric+vmax_, bin the active slice
    void apply_sort();               // rebuild row_order_/col_order_ from focus/index/desc
    void on_header_click(int disp_row, int disp_col);  // map display->underlying, set focus, re-sort

    void compute();             // full (re)build: buffers + thumbnails + values + fit view
    void compute_values();      // recompute values_/vmin_/vmax_ from existing buffers only
    void make_thumbs(std::vector<gl_texture>& out,
                     const std::vector<std::vector<float>>& bufs, int w, int h, int c);
    void random_fill(axis& ax);

    void free_axis_thumbs(axis& ax);
};

} // namespace coast

#endif // COAST_METRIC_EXPLORER_H
