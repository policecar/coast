#ifndef COAST_GUI_PLOT_PLOT_MANAGER_H
#define COAST_GUI_PLOT_PLOT_MANAGER_H

// The console-plot subsystem's single stateful object. It owns everything that
// used to be scattered file-scope globals: the panel registry, the deferred save
// queues, the GUI-thread task hook, the wired HDF5 executor, and the per-figure
// uid counter. Reached through the process-wide singleton plot_manager::instance();
// the free functions in console_plot.h are thin forwarders to it.
//
// Threading: every method is internally locked where it touches shared state, so
// figures (any thread) register/enqueue while the GUI thread renders/flushes.
// GUI-thread-only entry points (render_all/flush_captures/free_resources) are
// marked as such.

#include "gui_plot/figure.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hdf5 {
class executor;
}

namespace coast {
class gui_manager; // GUI-thread deferral host (forward-declared; used in .cpp only)
}

namespace coast::plot {

class canvas; // load_canvas return type

class plot_manager {
public:
    // The process-wide instance, heap-allocated on first use. Unlike a Meyers
    // static it is destroyed at a CONTROLLED point via deinit() (called from
    // run_gui before the sdl_app/gui_manager it references are torn down), not at
    // unspecified static-teardown time. First access must be single-threaded
    // (startup wiring), so the lazy init needs no lock.
    static plot_manager& instance();
    // Destroy the instance (idempotent). Call on the GUI thread at shutdown, after
    // the console thread has been joined (so it can no longer create figures) and
    // while the GL context is still current. A std::atexit fallback also runs it,
    // so paths that never call it explicitly (e.g. unit tests) still free the
    // instance. Once run, the singleton is TOMBSTONED: a later instance() throws
    // instead of resurrecting a second, GUI-host-less manager (see instance()).
    static void deinit();

    plot_manager(const plot_manager&)            = delete;
    plot_manager& operator=(const plot_manager&) = delete;

    // Wire the HDF5 executor used by load()/save. Call once at startup.
    void set_executor(hdf5::executor& exec);
    // Wire the GUI host that runs deferred GUI-thread work (post_gui_task). Call
    // once at startup. When set, deferred closures go to the host's single-shot
    // registry (drained in the state phase, outside the ImGui frame); when unset
    // (headless), they fall back to run_gui_tasks(). Set once before the console
    // runs; read-only thereafter.
    void set_gui_host(gui_manager& host);

    // -- module-internal, called by figures/kinds (any thread) --
    void          register_figure(const std::shared_ptr<figure>& f); // add to the render loop (show())
    std::uint64_t next_uid();                                        // stable per-figure window id
    // Hand a closure to the GUI thread; run once next frame (render_all), then dropped.
    void post_gui_task(std::function<void()> fn);
    // Queue an off-screen PNG / HDF5 save request (serviced by flush_captures).
    void enqueue_capture(std::weak_ptr<figure> f, std::string path, int w, int h, float scale);
    void enqueue_h5(std::weak_ptr<figure> f, std::string path, float ref_w, float ref_h);
    void enqueue_canvas_h5(std::weak_ptr<figure> f, std::string path, float ref_w, float ref_h);

    // -- GUI thread only --
    void        render_all();                    // draw every visible, non-hosted figure; reap dead ones
    void        flush_captures(hdf5::executor& exec); // service queued save requests (own render pass)
    void        run_gui_tasks();                 // drain the GUI-thread task hook
    void        free_resources();                // release GL + clear the registry (before context teardown)
    [[nodiscard]] std::size_t plot_count() const; // number of registered (live) figures

    // Reconstruct + show a saved figure/canvas (safe from the console thread).
    std::shared_ptr<figure> load(std::string path);
    std::shared_ptr<canvas> load_canvas(std::string path);

private:
    plot_manager() = default;

    // A deferred save request; the weak_ptr is empty at flush time => figure gone.
    struct capture_req {
        std::weak_ptr<figure> f;
        std::string           path;
        int                   w, h;
        float                 scale;
    };
    struct h5_req {
        std::weak_ptr<figure> f;
        std::string           path;
        float                 ref_w, ref_h;
    };

    hdf5::executor* exec_     = nullptr; // set once at startup; read-only thereafter
    gui_manager*    gui_host_ = nullptr; // deferral host; null => headless fallback

    mutable std::mutex                   reg_m_;
    std::vector<std::shared_ptr<figure>> figs_; // registry OWNS shown figures

    std::mutex                         tasks_m_;
    std::vector<std::function<void()>> tasks_; // GUI-thread task hook

    std::mutex               cap_m_;
    std::vector<capture_req> caps_;     // pending save_png requests
    std::mutex               h5_m_;
    std::vector<h5_req>      h5s_;       // pending figure save_h5 requests
    std::mutex               cvh5_m_;
    std::vector<h5_req>      cvh5s_;     // pending canvas save_h5 requests

    std::atomic<std::uint64_t> uid_{0};

    static plot_manager* instance_;  // heap instance; owned via instance()/deinit()
    static bool          destroyed_; // tombstone: set by deinit(), read by instance()
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_PLOT_MANAGER_H
