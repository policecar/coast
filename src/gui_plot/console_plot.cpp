#include "gui_plot/console_plot.h"

#include "gui_plot/plot_manager.h"

// The macro-facing free functions are thin forwarders to the process-wide
// plot_manager singleton, which owns all the subsystem's state. Kept so existing
// call sites (gui_manager, run_gui, tests, macros) stay stable; anyone who wants
// the object directly can use coast::plot::plot_manager::instance().

namespace coast::plot {

void set_executor(hdf5::executor& exec) { plot_manager::instance().set_executor(exec); }

std::shared_ptr<figure> load(std::string path)
{
    return plot_manager::instance().load(std::move(path));
}
std::shared_ptr<canvas> load_canvas(std::string path)
{
    return plot_manager::instance().load_canvas(std::move(path));
}

void render_all() { plot_manager::instance().render_all(); }
void flush_captures(hdf5::executor& exec) { plot_manager::instance().flush_captures(exec); }
void free_resources() { plot_manager::instance().free_resources(); }

std::size_t plot_count() { return plot_manager::instance().plot_count(); }

} // namespace coast::plot
