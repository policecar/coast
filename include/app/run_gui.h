#ifndef COAST_APP_RUN_GUI_H
#define COAST_APP_RUN_GUI_H

#include <memory>

#include "gui_core/console_iface.h"

namespace coast { class sim_instance; }
namespace hdf5  { class executor; }
namespace coast::cli { struct options; }

namespace coast::paths { struct app_config; }

namespace coast::app {


// Run the interactive GUI. Takes ownership of the (possibly null) sim:
// passing nullptr is legitimate — the GUI starts unloaded and the user can
// open a configuration from the menu.
//
// h5_exec must outlive this call; it is forwarded to any sim_instance
// constructed via the GUI's "load configuration" path.
//
// An optional console (the clang-repl C++ console from coast_console) is
// injected into the gui_manager; coast_lib itself need not depend on it.
//
// app_cfg governs WHERE the GUI resolves per-config outputs (snapshot dir for
// auto-save / resume) and its app assets (font, imgui.ini, plot themes). It is
// wired into gui_manager / logging_window / plot_manager.
//
// Returns 0 on normal shutdown.
int run_gui(std::unique_ptr<sim_instance>   sim,
            const cli::options&             opts,
            const paths::app_config&               app_cfg,
            hdf5::executor&                 h5_exec,
            std::unique_ptr<console_iface>  console = nullptr);

} // namespace coast::app

#endif // COAST_APP_RUN_GUI_H
