#ifndef COAST_CAMPAIGN_PANEL_H
#define COAST_CAMPAIGN_PANEL_H

// Campaign meta-loader panel (Analysis menu). A thin ImGui front over the
// coast::campaign façade — the SAME loader the REPL reaches as `camp::`.
//
// Point it at a directory of runs; it scans, shows the swept parameters it
// discovered by diffing the configs, and filters the run table by them. From there a
// run's observe file opens in the HDF5 browser, or the current selection is handed to
// the C++ console as a variable for metric work.
//
// Rendered via render(bool*) behind an external show_ flag owned by gui_manager,
// exactly like h5_browser_window and coastd_panel.
//
// The SCAN RUNS ON A BACKGROUND THREAD. A campaign root is typically on network
// storage (the reference one is an NFS mount), where a stat storm — or a mount that
// has gone away — would otherwise block the ImGui loop for as long as the filesystem
// takes to answer. Only the scan is threaded: everything afterwards works on
// in-memory config data, and the HDF5 opens are explicit user actions.

// Note: imgui.h must precede imfilebrowser.h.
#include "imgui.h"
#include "imfilebrowser.h"

#include "campaign/campaign_api.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace coast {

class console_iface;      // optional C++ console the panel can hand a selection to
class h5_browser_window;  // optional browser the panel can open a run's file in

class campaign_panel {
public:
    campaign_panel();

    campaign_panel(const campaign_panel&)            = delete;
    campaign_panel& operator=(const campaign_panel&) = delete;

    void render(bool* open);

    // Non-owning; set by gui_manager when each is present.
    void set_console(console_iface* c) { console_ = c; }
    void set_browser(h5_browser_window* b) { browser_ = b; }

private:
    void begin_scan(const std::filesystem::path& root);
    void poll_scan();          // adopt a finished background scan (render thread)
    void rebuild_filtered();   // re-apply the axis filters to the run list
    void render_toolbar();
    void render_filters();
    void render_table();
    void render_actions();
    void send_to_console(bool whole_campaign);
    // Row indices currently picked, in table order; empty when nothing is picked.
    [[nodiscard]] std::vector<std::size_t> picked_rows() const;

    std::shared_ptr<campaign::campaign> campaign_;
    campaign::selection                 filtered_;

    // Per axis, one flag per discovered value: is that value included. Index-aligned
    // with campaign_->axes() and each axis's values. All true after a scan, so the
    // table opens showing everything.
    std::vector<std::vector<bool>> keep_;
    bool                           filters_dirty_ = false;

    // Background scan, as an ABANDONABLE slot rather than a std::future.
    //
    // A future from std::async has the blocking destructor ([futures.async]/5): the
    // last reference waits for the task. The scan is a plain recursive filesystem walk
    // with no cancellation, so on a dead NFS mount it blocks inside an uninterruptible
    // syscall — and ~campaign_panel runs at shutdown, AFTER the window is gone. The
    // result was a windowless process that could not exit and gave no feedback.
    //
    // With a shared slot the panel simply drops its reference and teardown proceeds.
    // The worker thread still runs until the filesystem answers, but it touches only
    // the slot (which it co-owns) and its own locals, never the panel.
    //
    // THE RESIDUAL RISK, stated rather than glossed: the worker is detached, so on the
    // dead-mount path it can also outlive main(). If the mount answers after static
    // destruction has begun, the thread is inside std::ifstream / std::filesystem /
    // std::format with those statics gone — undefined behaviour during exit, and it is
    // the last owner of the slot, so ~campaign runs there too. The trade is a possible
    // crash while the process is already leaving, against the measured bug this
    // replaced: a windowless process that could not exit at all and gave no feedback.
    // Closing it properly needs a process-level in-flight count that run_gui waits on
    // with a timeout before returning; not built, and not free.
    struct scan_slot {
        std::mutex                          m;
        std::shared_ptr<campaign::campaign> result;   // guarded by m
        std::string                         error;    // guarded by m
        std::atomic<bool>                   done{false};
    };
    std::shared_ptr<scan_slot> scan_;
    bool                       scanning_ = false;

    ImGui::FileBrowser dir_browser_;
    char               root_buf_[512] = "";

    // Table row picking, on top of the axis filters. Two DIFFERENT notions of "a set
    // of runs" coexist deliberately: the filters answer "which runs match these
    // parameter values", the picked rows answer "these specific ones". Keyed by ROW
    // index into filtered_, which is only stable while the filter is — so
    // rebuild_filtered() clears it rather than leaving a selection that silently
    // refers to different runs than it did when it was made.
    ImGuiSelectionBasicStorage picked_;
    int  focused_ = -1;   // last row clicked, -1 = none; drives the single-run actions
    char observe_group_buf_[128] = "";

    std::string status_;
    bool        status_error_ = false;

    console_iface*     console_ = nullptr;
    h5_browser_window* browser_ = nullptr;
};

} // namespace coast

#endif // COAST_CAMPAIGN_PANEL_H
