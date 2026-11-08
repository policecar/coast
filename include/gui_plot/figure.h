#ifndef COAST_GUI_PLOT_FIGURE_H
#define COAST_GUI_PLOT_FIGURE_H

// The figure base class + the cross-thread painter seam.
//
// ImGui-free (region is a plain POD; all ImPlot/GL lives in the .cpp painters),
// so console macros parse it cheaply. Every concrete plot kind derives from
// figure and lives in its OWN header/source pair (histogram.{h,cpp},
// line.{h,cpp}, histogram2d.{h,cpp}, pixels.{h,cpp}, image.{h,cpp},
// canvas.{h,cpp}). To add a kind: add such a pair, register a loader case in
// plot_load.cpp, and list both files in CMake — nothing here changes.
//
// A concrete kind fills three seams: paint_body (the drawing), paint_controls
// (its interactive widgets), and fill_snapshot (its HDF5 data). Base carries the
// common state (title, theme, visibility) and the shared machinery (window
// chrome, off-screen export, HDF5 save, registration with the GUI render loop).
//
// Always held via std::shared_ptr — the painter seam and the GUI registry rely
// on shared/weak ownership (enable_shared_from_this).

#include "gui_plot/plot_data.h" // h5_save_data (fill_snapshot / snapshot_h5)

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace coast::plot {

// A drawing region in pixels, handed to a painter. Plain replacement for ImVec2
// so this header needs no ImGui.
//
// `interactive` is false when the frame is an off-screen capture: the painter
// must then not read input at all. Parking io fields is not a substitute — the
// exporter parks the cursor, but an in-progress middle-drag would still read the
// parked position and corrupt the live canvas's pan (see canvas::paint_body).
struct region {
    float w           = 0.0f;
    float h           = 0.0f;
    bool  interactive = true;
};

// A painter is a self-contained drawing action the GUI thread runs to render one
// figure into a region. It pins its figure alive, locks the figure's state, and
// applies the theme, so running it is safe even as the console thread mutates
// the figure concurrently. All ImPlot/ImGui calls live inside it (defined in the
// .cpp). Empty (== no painter) is valid — e.g. a figure with no interactive row.
using painter = std::function<void(region)>;

// A figure's addressable axes. z is the colorbar/scale of the colormapped kinds
// (histogram2d/pixels); the non-colormapped kinds (histogram/line) have only x
// and y. See histogram2d::set_axis_visible.
enum class plot_axis : std::uint8_t { x, y, z };

class figure : public std::enable_shared_from_this<figure> {
public:
    explicit figure(std::string title);
    virtual ~figure();

    figure(const figure&)            = delete;
    figure& operator=(const figure&) = delete;

    // -- any thread (internally locked) --

    void set_title(std::string title);
    // Select the plot's structural-color theme by name (e.g. "dark", "bright").
    // Unknown names fall back to the default look. Ignored by kinds without a
    // theme (image/canvas). Themes are created/edited/persisted via the in-GUI
    // Plot Theme Editor.
    void set_theme(std::string name);

    // Register this figure with the GUI render loop and make it visible: its own
    // ImPlot window appears at the next render_all(). Idempotent. A figure
    // destined for a canvas cell is NOT shown — it is place()d instead, and only
    // ever drawn inside the canvas.
    void show();
    // Hide the figure's window; it stays alive, registered, and updatable.
    void close();
    [[nodiscard]] bool visible() const { return open_.load(std::memory_order_relaxed); }

    // Number of data points/cells the figure currently holds (0 for kinds that
    // hold none, e.g. image/canvas). Thread-safe.
    [[nodiscard]] virtual std::size_t point_count() const { return 0; }

    // Export to a PNG at an explicit pixel size, rendered off-screen (so it can
    // exceed the on-screen window). Only queues the request — the GUI thread
    // renders + writes it at the next flush_captures(). width/height <= 0 fall
    // back to the current on-screen size. Safe from the console thread.
    void save_png(std::string path, int width = 0, int height = 0);
    // Save to a self-contained HDF5 file (one plot per file) for later reload via
    // load(). Only queues the request; the GUI thread snapshots + renders a
    // thumbnail at flush_captures(), then the write runs on the hdf5 worker. A
    // no-op for kinds that do not persist (image). Safe from the console thread.
    virtual void save_h5(std::string path);

    // -- GUI thread only --

    // Draw this figure's own top-level window: chrome + interactive row + body.
    // render_all() calls it for every visible, non-hosted, registered figure.
    void render();
    // A self-contained painter that draws just the body into a region (no window
    // chrome, no controls). Used by the live window, the off-screen exporter, and
    // the canvas (to draw a hosted figure in a cell). Pins the figure, locks it,
    // and applies its theme; the concrete draw is paint_body.
    [[nodiscard]] painter body_painter();
    // Release any GL resources (image textures) before the GL context is torn
    // down. Default: none.
    virtual void free_gl() {}

    // Fill a plain-data HDF5 snapshot (title + theme + kind data); false if there
    // is nothing to save. GUI thread, under the figure lock. Used by the deferred
    // save queue in flush_captures().
    bool snapshot_h5(h5_save_data& out);

    // -- hosting (canvas) --
    // A hosted figure is drawn by its canvas (via body_painter), not as its own
    // window; render_all() skips it. Managed by canvas::place / clear_cell.
    void               set_hosted(bool h) { hosted_.store(h, std::memory_order_relaxed); }
    [[nodiscard]] bool is_hosted() const { return hosted_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool open() const { return open_.load(std::memory_order_relaxed); }

protected:
    // -- seams filled by concrete kinds --
    // Draw the body into `region`. Called with m_ held and the theme pushed, on
    // the GUI thread (live window, export, or canvas cell). ImPlot/ImGui only.
    virtual void paint_body(region r) = 0;
    // Draw the kind-specific interactive widgets (palette/log/view/axes/…).
    // Called with m_ held, live window only. Default: none.
    virtual void paint_controls() {}
    // Fill the HDF5 DTO; return false if there is nothing to save. Called with m_
    // held. Does NOT set title/theme (the base adds those). Default: not saveable.
    virtual bool fill_snapshot(h5_save_data& out) const { (void)out; return false; }
    [[nodiscard]] virtual bool is_saveable() const { return false; }

    // Draw the common theme picker + "edit…" row (GUI thread, m_ held). The
    // colormapped/plain kinds call this at the front of their paint_controls;
    // image/canvas omit it. Defined in figure.cpp (ImGui).
    void draw_theme_row();

    mutable std::mutex  m_;
    std::string         title_;
    std::string         theme_ = "dark";
    const std::uint64_t uid_;                 // stable per-figure ImGui window id
    std::atomic<bool>   open_{false};         // visible as its own window
    std::atomic<bool>   hosted_{false};       // drawn by a canvas instead
    std::atomic<bool>   registered_{false};   // added to the GUI registry (show())

    // GUI-thread-only transient save-dialog state + last on-screen body size.
    float last_w_            = 0.0f;
    float last_h_            = 0.0f;
    bool  save_keep_aspect_  = true;
    int   save_w_            = 0;
    int   save_h_            = 0;
    char  save_path_[256]    = {};
    char  save_h5_path_[256] = {};

private:
    // Save PNG…/Save H5… buttons + popups (GUI thread). Takes the title snapshot
    // from render() (taken under m_) for default filenames — must NOT read title_
    // directly, which set_title() may mutate concurrently from any thread.
    void draw_save_row(const std::string& title);
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_FIGURE_H
