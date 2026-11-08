#ifndef COAST_COASTD_PANEL_H
#define COAST_COASTD_PANEL_H

// coastd daemon-control panel (Tools menu). A thin ImGui front over the
// coast::coastd console façade — the SAME protocol client the REPL uses. It
// connects to a daemon (localhost by default), polls the queue (~2 s, the
// design's stateless-resync model), and offers submit / cancel / reorder.
//
// Rendered via render(bool*) behind an external show_ flag owned by gui_manager,
// exactly like h5_browser_window. The periodic queue POLL runs on a BACKGROUND
// thread with its OWN connection, publishing a job snapshot the render thread
// copies without blocking — so a reachable-but-slow daemon can never stutter the
// ImGui loop. User-initiated actions (connect, submit, cancel, reorder) are still
// synchronous round-trips on the render thread with a short timeout: a brief hitch
// on an explicit click is acceptable, and a transport error there drops the
// session (the underlying connection self-poisons; a daemon {"ok":false} rejection
// does NOT drop it).

#include <chrono>
#include <functional>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "coastd_console/coastd_console_api.h"

namespace coast {

class coastd_panel {
public:
    coastd_panel() = default;
    ~coastd_panel() { stop_poll(); }

    coastd_panel(const coastd_panel&)            = delete;
    coastd_panel& operator=(const coastd_panel&) = delete;

    void render(bool* open);

    // Called by the owner on frames where the panel is NOT rendered (window
    // closed). Pauses the background poller: a closed panel used to keep a TCP
    // connection open and issue a `list` RPC every 2 s for the remainder of the
    // process, invisibly. The session itself is retained, so reopening resumes
    // without asking the user to reconnect. Idempotent and cheap to call per frame.
    void on_hidden();

private:
    void render_connect_bar();
    void render_discovery();        // fixed-list (connect_to) + UDP discovery, when disconnected
    void render_submit_form();
    void render_jobs_table();
    void set_status(const std::string& msg, bool error);

    // Adopt a freshly-connected session and start the background poller from the
    // same reconnect factory. `label` names the daemon for the status line.
    void adopt(coastd::session s, std::function<coastd::session()> factory, const std::string& label);
    void disconnect();                          // stop poller + drop the render session
    void sync_from_poll();                      // copy the background snapshot into jobs_ (render thread)
    void start_poll(std::function<coastd::session()> factory);
    void stop_poll();
    // After a caught user-action exception: drop the session iff the transport died
    // (connection self-poisoned). A daemon rejection ({"ok":false}) keeps it.
    void handle_action_error(const std::exception& e);

    std::optional<coastd::session> session_;     // render-thread session for user actions
    coastd::host_info              host_;
    coastd::job_list               jobs_;        // render-thread copy of the poll snapshot

    // Discovery state (only meaningful while disconnected).
    coastd::host_config_list       configured_;          // fixed-list config (IDD §6.6)
    coastd::host_list              discovered_;          // last UDP scan result
    bool                           configured_loaded_ = false;
    int                            configured_sel_    = 0;

    // Connection inputs.
    char host_buf_[128] = "localhost";
    int  port_          = 51900;

    // Submit form.
    char        ref_buf_[128]    = "main";
    char        config_buf_[512] = "";
    char        label_buf_[64]   = "";
    char        message_buf_[128]= "";
    int         steps_target_    = 0;

    // Queue view controls: a case-insensitive text filter (id/ref/note/status) and
    // per-bucket visibility toggles. Filtering is purely a RENDER concern — it never
    // changes the queue the poller holds, and reorder positions are still computed
    // over the full, unfiltered list so "Up" moves within the real queue.
    char filter_buf_[128]    = "";
    bool show_active_        = true;
    bool show_done_          = true;
    bool show_failed_        = true;
    bool show_cancelled_     = true;
    bool show_quarantine_    = true;

    // User-draggable height of the queue table, in pixels (splitter below it).
    float jobs_height_       = 300.0f;

    // Status line (last error/info) + colour.
    std::string status_;
    bool        status_error_ = false;
    // Poll health, tracked apart from status_error_ so neither source can mask or
    // outlive the other (see sync_from_poll). poll_failing_ is what the render
    // path uses to show a persistent "queue is stale" indicator regardless of
    // whatever action message currently occupies the status line.
    bool        poll_failing_ = false;
    std::string poll_status_msg_;

    // ---- background poller (own connection; publishes a snapshot) --------------
    std::function<coastd::session()> make_session_;   // reconnect factory (set on connect)
    std::mutex                       poll_mu_;
    coastd::job_list                 poll_jobs_;       // guarded by poll_mu_
    // Bumped by the poller each time poll_jobs_ is replaced; compared by
    // sync_from_poll so the render thread copies the vector ONLY when it actually
    // changed. Without it, every frame deep-copied the whole job list (a vector of
    // structs of std::strings) under poll_mu_ — at 60 fps against a ~2 s poll that is
    // ~120 copies per useful update, all but one redundant, and each one holding the
    // lock the poll thread wants. Guarded by poll_mu_ like the data it versions, so
    // the pair cannot be read torn; a plain counter is enough because there is exactly
    // one writer.
    std::uint64_t                    poll_seq_ = 0;    // guarded by poll_mu_
    std::uint64_t                    synced_seq_ = 0;  // render thread only
    std::string                      poll_err_;        // guarded by poll_mu_ ("" = healthy)
    bool                             poll_ok_ = false; // guarded by poll_mu_
    // Declared LAST so ~coastd_panel destroys (stop-requests + joins) it FIRST,
    // before the mutex/snapshot it touches. stop_poll() also joins it on disconnect.
    std::jthread                     poll_thread_;
};

} // namespace coast

#endif // COAST_COASTD_PANEL_H
