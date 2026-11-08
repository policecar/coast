#include "gui_plot/plot_manager.h"

#include "gui_core/gui_manager.h" // gui_host_->defer (GUI-thread single-shot registry)
#include "gui_plot/canvas.h"      // make_shared<canvas> + place (load_canvas)
#include "gui_plot/plot_detail.h" // make_figure_from_h5 + canvas_save_data
#include "gui_plot/plot_h5.h"     // read_plot_h5 / read_canvas_h5
#include "gui_plot/plot_tools.h"  // draw_theme_editor (GUI thread, shared across plots)

#include "hdf5/executor.h"

#include "diag/diag.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

// The core of the console-plot subsystem's single object: registry, GUI-thread
// task hook, per-frame render loop, and loading saved plots. The deferred save
// queues + off-screen exporter live on plot_capture.cpp (same class, different
// TU); the concrete kinds and HDF5 serialization in their own files.

namespace coast::plot {

plot_manager* plot_manager::instance_  = nullptr;
bool          plot_manager::destroyed_ = false;

plot_manager& plot_manager::instance()
{
    if (instance_ == nullptr) {
        // Tombstone: once deinit() has torn the subsystem down (GL context still
        // current at the call site, console thread joined), refuse to resurrect. A
        // silent new here would register a second atexit and hand back a manager
        // with gui_host_ == nullptr, whose deferred GL frees then run at atexit with
        // no GL context. Post-deinit access is a lifetime bug in the caller (a figure
        // created or freed after shutdown) — surface it loudly at the call site
        // instead of silently papering over it.
        if (destroyed_)
            throw std::logic_error(
                "plot_manager::instance() after deinit() — a figure was created or "
                "freed after the plot subsystem was torn down");
        instance_ = new plot_manager();
        // Safety net: if deinit() is never called explicitly (e.g. unit tests),
        // free the instance at exit. Idempotent with an explicit deinit() (which
        // nulls instance_, so this later delete is a no-op).
        std::atexit(&plot_manager::deinit);
    }
    return *instance_;
}

void plot_manager::deinit()
{
    delete instance_; // delete nullptr is a no-op → idempotent
    instance_  = nullptr;
    destroyed_ = true; // tombstone: instance() must not resurrect after this
}

void plot_manager::set_executor(hdf5::executor& exec) { exec_ = &exec; }
void plot_manager::set_gui_host(gui_manager& host) { gui_host_ = &host; }

void plot_manager::register_figure(const std::shared_ptr<figure>& f)
{
    std::scoped_lock lk(reg_m_);
    figs_.push_back(f);
}

std::uint64_t plot_manager::next_uid() { return uid_.fetch_add(1); }

void plot_manager::post_gui_task(std::function<void()> fn)
{
    if (!fn) return;
    // With a GUI host wired, hand the closure to its single-shot registry — run on
    // the GUI thread in the STATE phase, OUTSIDE the ImGui frame (so a GL free does
    // not run mid-frame). Without a host (headless), fall back to the local queue,
    // drained by run_gui_tasks() from free_resources().
    if (gui_host_ != nullptr) {
        gui_host_->defer(std::move(fn));
        return;
    }
    std::scoped_lock lk(tasks_m_);
    tasks_.push_back(std::move(fn));
}

void plot_manager::run_gui_tasks() // GUI thread only (GL context current)
{
    std::vector<std::function<void()>> pending;
    {
        std::scoped_lock lk(tasks_m_);
        pending.swap(tasks_);
    }
    for (auto& fn : pending) fn();
    // pending's closures (and their captures) are destroyed here, on the GUI thread.
}

std::size_t plot_manager::plot_count() const
{
    std::scoped_lock lk(reg_m_);
    return figs_.size();
}

std::shared_ptr<figure> plot_manager::load(std::string path)
{
    if (exec_ == nullptr) {
        COAST_ERROR("plot", "load: no HDF5 executor set");
        return nullptr;
    }
    // Read on the executor worker thread (all libhdf5 calls funnel through it),
    // then reconstruct on the calling thread — reconstruction touches no HDF5/GL,
    // only plain data.
    h5_save_data data;
    const bool   ok = exec_->submit_sync([&] { return read_plot_h5(path, data); });
    if (!ok) return nullptr;

    std::shared_ptr<figure> fig = make_figure_from_h5(data);
    if (!fig) return nullptr;
    fig->show(); // a loaded figure is meant to be shown (registers it)
    return fig;
}

std::shared_ptr<canvas> plot_manager::load_canvas(std::string path)
{
    if (exec_ == nullptr) {
        COAST_ERROR("plot", "load_canvas: no HDF5 executor set");
        return nullptr;
    }
    canvas_save_data data;
    const bool       ok = exec_->submit_sync([&] { return read_canvas_h5(path, data); });
    if (!ok) return nullptr;

    auto cv = std::make_shared<canvas>(data.title, data.rows, data.cols);
    // Reconstruct each stored cell as a figure and place it (canvas-owned; loaded
    // children never had their own window, so they are not separately registered).
    for (const auto& cell : data.cells) {
        std::shared_ptr<figure> fig = make_figure_from_h5(cell.data);
        if (fig) cv->place(fig, cell.row, cell.col);
    }
    cv->show();
    return cv;
}

void plot_manager::render_all()
{
    std::vector<std::shared_ptr<figure>> snap;
    {
        std::scoped_lock lk(reg_m_);
        snap = figs_;
    }
    for (auto& f : snap) {
        // Hosted figures are drawn inside their canvas, not as their own window.
        if (f->open() && !f->is_hosted()) f->render();
    }
    draw_theme_editor(); // shared across all plots
    snap.clear();        // drop snapshot refs so use_count() reflects registry + macro only

    // Reap a figure only when it is closed AND no macro still holds it
    // (use_count() == 1 => only the registry). A closed figure a macro still holds
    // stays hidden and is revived by the next show().
    {
        std::scoped_lock lk(reg_m_);
        figs_.erase(std::remove_if(figs_.begin(), figs_.end(),
                                   [](const std::shared_ptr<figure>& f) {
                                       return !f->open() && f.use_count() == 1;
                                   }),
                    figs_.end());
    }
    // NOTE: deferred GUI-thread work (an image reaped above posts its texture free
    // via post_gui_task) is NOT drained here — that would run glDeleteTextures
    // inside the ImGui frame. With a GUI host it goes to the single-shot registry,
    // run in the next state phase (outside the frame); headless, free_resources()
    // drains the fallback queue.
}

void plot_manager::free_resources()
{
    // GUI thread, before GL context teardown. Explicitly release every figure's GL
    // textures HERE while the context is valid — dropping the registry's
    // shared_ptrs is not enough: a macro may still hold an image, so its
    // ~gl_texture would otherwise run later on the console thread after the
    // context is gone. free_gl() makes that teardown a no-op.
    {
        std::scoped_lock lk(reg_m_);
        for (auto& f : figs_)
            if (f) f->free_gl();
        figs_.clear(); // ~image here posts any still-live texture to the hook
    }
    // Run remaining GUI-thread tasks, still with the context valid, so nothing GL
    // survives to be deleted at static teardown after the context is gone.
    run_gui_tasks();
}

} // namespace coast::plot
