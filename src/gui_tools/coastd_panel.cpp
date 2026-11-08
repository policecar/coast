#include "gui_tools/coastd_panel.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

#include "imgui.h"

namespace coast {

namespace {
constexpr std::chrono::milliseconds poll_interval{2000};

// Two bounds, deliberately different (see coastd_client.h connection::open).
//
// CONNECT is on the ImGui render thread, so it is how long a dead or firewalled
// host freezes the frame — keep it short.
//
// An OPERATION is not a handshake. `submit` is deferred onto the daemon's
// single-writer core thread and then writes the run config and job file through
// two fsyncs, possibly queued behind a reconcile pass; on a busy disk that is
// comfortably past half a second. With one shared 500 ms bound, a submit the
// daemon COMPLETED was reported to the user as a failure — and the natural
// response to "submit failed" is to submit again, which is how you get two jobs
// running the same config. The console façade already used 2000 ms for the same
// surface; only the GUI hardcoded 500.
constexpr int connect_timeout_ms = 500;
constexpr int op_timeout_ms      = 5000;

// Palette shared by the status text, the progress-bar tint and the filter chips,
// so a colour always means the same phase everywhere in the panel. Mirrors the
// simc "Simulation Runs" overview: grey = waiting, amber = in-progress build/
// transient, cyan = actively running, green = done, red = failed, magenta =
// quarantine (needs a human).
constexpr ImVec4 col_wait  {0.65f, 0.65f, 0.70f, 1.0f};  // queued / resolving / ready
constexpr ImVec4 col_build {0.80f, 0.66f, 0.30f, 1.0f};  // building (also cancelled)
constexpr ImVec4 col_run   {0.40f, 0.78f, 0.92f, 1.0f};  // running
constexpr ImVec4 col_done  {0.45f, 0.80f, 0.50f, 1.0f};  // done
constexpr ImVec4 col_fail  {0.92f, 0.35f, 0.32f, 1.0f};  // failed
constexpr ImVec4 col_quar  {0.85f, 0.45f, 0.85f, 1.0f};  // quarantine

// The daemon's stable status tokens (types.h job_status) -> colour.
ImVec4 status_color(const std::string& s) {
    if (s == "running")                       return col_run;
    if (s == "done")                          return col_done;
    if (s == "failed")                        return col_fail;
    if (s == "cancelled")                     return col_build;
    if (s == "quarantine")                    return col_quar;
    if (s == "building" || s == "resolving")  return col_build;
    return col_wait;   // queued, ready
}

// Case-insensitive substring test (ASCII) for the free-text queue filter.
bool ci_contains(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return true;
    const auto it = std::search(
        hay.begin(), hay.end(), needle.begin(), needle.end(),
        [](char a, char b) { return std::tolower((unsigned char)a) ==
                                    std::tolower((unsigned char)b); });
    return it != hay.end();
}
} // namespace

void coastd_panel::set_status(const std::string& msg, bool error) {
    status_       = msg;
    status_error_ = error;
}

// Start (or restart) the background poller. It owns its OWN session built from the
// reconnect factory, so it never contends with the render-thread session used for
// user actions. It publishes a job snapshot under poll_mu_ and reconnects on its
// own if its connection drops — the render thread only ever copies the snapshot.
void coastd_panel::start_poll(std::function<coastd::session()> factory) {
    stop_poll();
    { std::lock_guard lk(poll_mu_); poll_jobs_ = coastd::job_list{}; ++poll_seq_;
      poll_err_.clear(); poll_ok_ = false; }
    poll_thread_ = std::jthread([this, factory = std::move(factory)](std::stop_token st) {
        std::optional<coastd::session> ps;
        while (!st.stop_requested()) {
            try {
                if (!ps) ps.emplace(factory());
                coastd::job_list jl = ps->jobs();
                std::lock_guard lk(poll_mu_);
                poll_jobs_ = std::move(jl);
                ++poll_seq_;            // tells the render thread this snapshot is new
                poll_err_.clear();
                poll_ok_ = true;
            } catch (const std::exception& e) {
                { std::lock_guard lk(poll_mu_); poll_err_ = e.what(); poll_ok_ = false; }
                ps.reset();  // drop the (poisoned) session; rebuild next iteration
            }
            // Pace the poll, but wake promptly on stop (join latency < 100 ms).
            for (int i = 0; i < poll_interval / std::chrono::milliseconds(100) &&
                            !st.stop_requested(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
}

void coastd_panel::on_hidden() {
    // Only the POLLER stops. Keeping session_ and make_session_ means reopening the
    // panel resumes the same connection instead of making the user reconnect, while
    // a closed panel costs the daemon nothing.
    if (poll_thread_.joinable()) stop_poll();
}

void coastd_panel::stop_poll() {
    if (poll_thread_.joinable()) {
        poll_thread_.request_stop();
        poll_thread_.join();
    }
}

void coastd_panel::adopt(coastd::session s, std::function<coastd::session()> factory,
                         const std::string& label) {
    // Handshake BEFORE committing to session_. info() throws on any transport
    // error, and emplacing first left the panel half-adopted: rendering as
    // connected, with no poller started, a permanently empty queue and a spurious
    // "no dataset link" warning (host_ is default-constructed). Only Disconnect
    // recovered. Doing the throwing call first means a failed adopt leaves the
    // panel exactly as it was, and the caller's catch reports the real error.
    coastd::session      fresh(std::move(s));
    const coastd::host_info info = fresh.info();  // one synchronous handshake (user-initiated)

    session_.emplace(std::move(fresh));
    host_         = info;
    make_session_ = factory;
    start_poll(std::move(factory));
    set_status("connected to " + label, false);
}

void coastd_panel::disconnect() {
    stop_poll();
    session_.reset();
    make_session_ = {};
    jobs_ = coastd::job_list{};
    { std::lock_guard lk(poll_mu_); poll_jobs_ = coastd::job_list{}; ++poll_seq_;
      poll_err_.clear(); poll_ok_ = false; }
}

void coastd_panel::sync_from_poll() {
    std::lock_guard lk(poll_mu_);
    // Copy the job list only when the poller actually published a new one. This runs
    // every frame; the poll interval is ~2 s, so an unconditional copy was ~120
    // redundant deep copies per update (vector of structs of std::strings), each held
    // under the lock the poll thread needs to publish. The status fields below are
    // scalars/short strings and are copied unconditionally — versioning them would
    // cost more than it saves.
    if (synced_seq_ != poll_seq_) {
        jobs_       = poll_jobs_;
        synced_seq_ = poll_seq_;
    }
    // Surface a persistent poll failure without tearing down the session (the
    // poller reconnects on its own; a user action is what fully disconnects).
    //
    // The poll status is tracked SEPARATELY from an action error, because sharing
    // one flag was sticky in both directions: a poll error latched status_error_
    // and nothing cleared it when the poller recovered, so the panel showed a red
    // "queue refresh" failure while updating live behind it; and an existing action
    // error suppressed a genuinely stale queue, which is the more dangerous half —
    // the operator reads job states that stopped updating minutes ago. Each source
    // now owns its own state, and the poll side clears itself on recovery.
    if (!poll_ok_ && !poll_err_.empty()) {
        poll_failing_    = true;
        poll_status_msg_ = "queue refresh: " + poll_err_;
    } else if (poll_ok_ && poll_failing_) {
        poll_failing_    = false;
        poll_status_msg_.clear();
        // Only speak up if the panel is still showing the stale failure; never
        // stomp a newer action message.
        if (status_error_ && status_.starts_with("queue refresh: "))
            set_status("queue refresh recovered", false);
    }
    // Show the poll failure only when no action error is competing for the line;
    // the banner below renders poll_failing_ independently either way.
    if (poll_failing_ && !status_error_) set_status(poll_status_msg_, true);
}

void coastd_panel::handle_action_error(const std::exception& e) {
    set_status(e.what(), true);
    // Drop the session only if the transport actually died (self-poisoned). A
    // daemon rejection (e.g. cancelling an already-terminal job) leaves it alive.
    if (session_ && !session_->connected()) disconnect();
}

void coastd_panel::render_connect_bar() {
    if (session_) {
        ImGui::TextUnformatted(host_.hostname.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("os=%s  coastd=%s  slots=%d", host_.os.c_str(),
                            host_.coastd_version.c_str(), host_.slots);
        ImGui::SameLine();
        if (ImGui::SmallButton("Disconnect")) { disconnect(); set_status("disconnected", false); }
        return;
    }

    ImGui::SetNextItemWidth(160);
    ImGui::InputText("##host", host_buf_, sizeof host_buf_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##port", &port_, 0);
    ImGui::SameLine();
    if (ImGui::Button("Connect")) {
        if (port_ < 1 || port_ > 65535) {
            set_status("port must be 1..65535", true);
        } else {
            const std::string host = host_buf_;
            const auto port = static_cast<std::uint16_t>(port_);
            auto factory = [host, port] {
                return coastd::session::connect(host, port, connect_timeout_ms, op_timeout_ms);
            };
            try {
                adopt(factory(), factory, host);
            } catch (const std::exception& e) {
                set_status(e.what(), true);
            }
        }
    }
}

void coastd_panel::render_discovery() {
    if (session_) return;   // discovery UI is only for choosing a daemon to connect to

    // Helper: connect via the given FACTORY (a repeatable thunk the background
    // poller reuses to reconnect), adopting the session on success. The connect
    // itself is synchronous but user-initiated and short-timeout, so a brief hitch
    // on click is the worst case — never an idle-render stall.
    const auto try_connect = [&](std::function<coastd::session()> factory, const std::string& label) {
        try {
            adopt(factory(), factory, label);
        } catch (const std::exception& e) {
            set_status(e.what(), true);
        }
    };

    // ---- Fixed-list host config (IDD §6.6, the deterministic primary) ----------
    if (!configured_loaded_) { configured_ = coastd::configured_hosts(); configured_loaded_ = true; }

    ImGui::SeparatorText("Known hosts (config)");
    if (configured_.empty()) {
        ImGui::TextDisabled("(none configured — see ~/.config/coastd/hosts.json)");
    } else {
        if (configured_sel_ < 0 || configured_sel_ >= configured_.size()) configured_sel_ = 0;
        const coastd::configured_host sel = configured_[configured_sel_];
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo("##known", sel.name.c_str())) {
            for (int i = 0; i < configured_.size(); ++i) {
                const coastd::configured_host h = configured_[i];
                const std::string lbl = h.name + "  (" + h.host + ":" + std::to_string(h.port) + ")";
                const bool is_sel = (i == configured_sel_);
                if (ImGui::Selectable(lbl.c_str(), is_sel)) configured_sel_ = i;
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::Button("Connect by name"))
            try_connect([name = sel.name] {
                            return coastd::session::connect_to(name, connect_timeout_ms, op_timeout_ms); },
                        sel.name);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Reload")) { configured_ = coastd::configured_hosts(); }
    // A malformed hosts.json still yields the localhost default, so the list looks
    // merely short rather than broken. Say so HERE — the load's own complaint goes
    // to stderr, which in a windowed launch is nowhere the user can see.
    if (const std::string w = configured_.warning(); !w.empty())
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.0f), "hosts.json: %s", w.c_str());

    // ---- UDP discovery (best-effort convenience on top of the fixed list) ------
    ImGui::SeparatorText("Discover (UDP)");
    if (ImGui::Button("Scan")) {
        try {
            discovered_ = coastd::discover(connect_timeout_ms);   // a scan, not an operation
            set_status("discovered " + std::to_string(discovered_.size()) + " daemon(s)", false);
        } catch (const std::exception& e) {
            set_status(e.what(), true);
        }
    }
    for (int i = 0; i < discovered_.size(); ++i) {
        const coastd::host_info h = discovered_[i];
        const std::string addr = !h.address.empty() ? h.address : h.hostname;
        ImGui::PushID(1000 + i);
        ImGui::Text("%s:%d  (%s, %d slots)", addr.c_str(), h.port, h.hostname.c_str(), h.slots);
        ImGui::SameLine();
        if (ImGui::SmallButton("Connect"))
            try_connect([h] { return coastd::session::connect(h, connect_timeout_ms, op_timeout_ms); }, addr);
        ImGui::PopID();
    }
}

void coastd_panel::render_submit_form() {
    if (!session_) return;
    if (!ImGui::CollapsingHeader("Submit a job")) return;

    ImGui::SetNextItemWidth(200);
    ImGui::InputText("ref (branch/tag/sha)", ref_buf_, sizeof ref_buf_);
    // A path on THIS machine. session::submit reads the file here and sends its
    // CONTENT, so a remote daemon needs nothing staged on its own filesystem.
    ImGui::InputText("config (path on this machine)", config_buf_, sizeof config_buf_);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Read on this machine and sent to the daemon as content.\n"
                          "The daemon does not need the file, so this works unchanged\n"
                          "against a daemon on another host.");
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("label", label_buf_, sizeof label_buf_);
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("message", message_buf_, sizeof message_buf_);
    ImGui::SetNextItemWidth(200);
    ImGui::InputInt("steps target (0 = unbounded)", &steps_target_);
    if (steps_target_ < 0) steps_target_ = 0;

    // The one machine-specific thing a client cannot see in its own config: will
    // relative "data/…" paths resolve over there? The daemon answers it in
    // machine_info (dataset_base set AND its target present). Warn here rather
    // than let the job resolve, build and then die at dataset load minutes later.
    if (!host_.data_link_ok)
        ImGui::TextColored(col_build,
                           "This daemon has no dataset link: relative \"data/...\" paths in the\n"
                           "config will NOT resolve. Use absolute dataset paths, or set\n"
                           "dataset_base in the daemon's coastd.json.");

    const bool can_submit = ref_buf_[0] != '\0' && config_buf_[0] != '\0';
    ImGui::BeginDisabled(!can_submit);
    if (ImGui::Button("Submit")) {
        try {
            const std::string id = session_->submit(ref_buf_, config_buf_, label_buf_,
                                        message_buf_, static_cast<std::uint64_t>(steps_target_));
            set_status("submitted " + id + " (queue updates shortly)", false);
        } catch (const std::exception& e) {
            handle_action_error(e);
        }
    }
    ImGui::EndDisabled();
}

// One coloured "N label" segment of the summary line, with " · " separators
// managed by the caller via `first`.
static void summary_seg(bool& first, const ImVec4& col, int n, const char* label) {
    if (n == 0) return;
    if (!first) { ImGui::SameLine(); ImGui::TextDisabled(" · "); ImGui::SameLine(); }
    first = false;
    ImGui::TextColored(col, "%d %s", n, label);
}

void coastd_panel::render_jobs_table() {
    if (!session_) return;
    ImGui::SeparatorText("Queue");

    // ---- one pass for the summary counts (over the FULL, unfiltered queue) ------
    int n_run = 0, n_pending = 0, n_done = 0, n_failed = 0, n_cancelled = 0, n_quar = 0;
    for (int i = 0; i < jobs_.size(); ++i) {
        const coastd::job_info j = jobs_[i];
        if      (j.bucket == "done")       ++n_done;
        else if (j.bucket == "failed")     ++n_failed;
        else if (j.bucket == "cancelled")  ++n_cancelled;
        else if (j.bucket == "quarantine") ++n_quar;
        else if (j.status == "running")    ++n_run;      // active bucket
        else                               ++n_pending;  // queued/resolving/building/ready
    }
    {
        bool first = true;
        summary_seg(first, col_run,   n_run,       "running");
        summary_seg(first, col_wait,  n_pending,   "pending");
        summary_seg(first, col_done,  n_done,      "done");
        summary_seg(first, col_fail,  n_failed,    "failed");
        summary_seg(first, col_build, n_cancelled, "cancelled");
        summary_seg(first, col_quar,  n_quar,      "quarantine");
        if (first) ImGui::TextDisabled("queue is empty");
    }

    // ---- filter bar: free-text + per-bucket visibility chips --------------------
    ImGui::SetNextItemWidth(180);
    ImGui::InputTextWithHint("##filter", "filter id / ref / note…", filter_buf_, sizeof filter_buf_);
    ImGui::SameLine();
    if (ImGui::SmallButton("clear")) filter_buf_[0] = '\0';
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    // A chip is lit in its bucket colour when that bucket is shown, dim when hidden.
    const auto chip = [](const char* label, bool* v, const ImVec4& on) {
        const ImVec4 off{0.28f, 0.28f, 0.31f, 1.0f};
        const ImVec4 c = *v ? on : off;
        ImGui::PushStyleColor(ImGuiCol_Button,        c);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x, c.y, c.z, 0.85f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  c);
        // Dark text on the light-ish chip fills reads better than the default.
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));
        if (ImGui::SmallButton(label)) *v = !*v;
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
    };
    chip("active",     &show_active_,     col_run);
    chip("done",       &show_done_,       col_done);
    chip("failed",     &show_failed_,     col_fail);
    chip("cancelled",  &show_cancelled_,  col_build);
    chip("quarantine", &show_quarantine_, col_quar);
    ImGui::NewLine();

    const std::string filter = filter_buf_;
    const auto bucket_shown = [&](const std::string& b) {
        if (b == "active")     return show_active_;
        if (b == "done")       return show_done_;
        if (b == "failed")     return show_failed_;
        if (b == "cancelled")  return show_cancelled_;
        if (b == "quarantine") return show_quarantine_;
        return true;   // unknown bucket: never hide it
    };

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                                      ImGuiTableFlags_BordersOuter;
    if (!ImGui::BeginTable("coastd_jobs", 6, flags, ImVec2(0, jobs_height_))) return;
    ImGui::TableSetupScrollFreeze(0, 1);   // keep the header visible while scrolling
    ImGui::TableSetupColumn("status",   ImGuiTableColumnFlags_WidthFixed, 124);
    ImGui::TableSetupColumn("id",       ImGuiTableColumnFlags_WidthStretch, 1.4f);
    ImGui::TableSetupColumn("ref",      ImGuiTableColumnFlags_WidthFixed, 150);
    ImGui::TableSetupColumn("progress", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    ImGui::TableSetupColumn("note",     ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("actions",  ImGuiTableColumnFlags_WidthFixed, 118);
    ImGui::TableHeadersRow();

    // Each row's position within the PENDING subset (active jobs not yet running),
    // which is not the raw table row. The server returns jobs in scheduler order
    // (active first, priority/FIFO within), so skipping running rows yields that
    // same ordered subset. Used ONLY to disable "Up" on the job already at the head
    // — the move itself is sent as a delta and resolved against the live queue
    // daemon-side (see the Up button). Computed over the FULL list, since the filter
    // is a view and not a queue edit.
    int pending_pos = 0;
    int shown = 0;
    for (int i = 0; i < jobs_.size(); ++i) {
        const coastd::job_info j = jobs_[i];
        const bool active  = j.bucket == "active";
        const bool running = j.status == "running";
        const bool pending = active && !running;   // reorderable
        const int  my_pos  = pending ? pending_pos : -1;
        if (pending) ++pending_pos;

        // Apply the view filter AFTER the pending bookkeeping above.
        if (!bucket_shown(j.bucket)) continue;
        if (!filter.empty() &&
            !(ci_contains(j.id, filter) || ci_contains(j.ref, filter) ||
              ci_contains(j.status, filter) || ci_contains(j.user_message, filter) ||
              ci_contains(j.failure_reason, filter)))
            continue;
        ++shown;

        ImGui::TableNextRow();
        // Faintly tint the running row so the eye lands on live work at a glance.
        if (running)
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                   ImGui::GetColorU32(ImVec4(col_run.x, col_run.y, col_run.z, 0.12f)));
        // Key the row's widget IDs on the JOB ID, not the row index. The poll thread
        // swaps jobs_ wholesale every ~2 s, and ImGui resolves a button on RELEASE:
        // with an index key, a job leaving the active bucket between press and
        // release re-targets that click at whatever job now sits at index i — so
        // Cancel (or Up) silently acts on a different run than the one clicked.
        ImGui::PushID(j.id.c_str());

        // Status: coloured bullet + token. Tooltip carries the fuller detail that
        // no longer needs its own columns (bucket, submit time, checkpoint).
        // A pending cancel takes over the STATUS cell. The job is still `running`
        // (that is its real, persisted status — there is no `cancelling` state),
        // but for up to term_grace seconds the only thing the operator wants to
        // know is that their click was accepted and the child is being stopped.
        // Without this the button looks broken: nothing changes, then the row
        // jumps to `cancelled`.
        const bool cancelling = j.cancel_requested && j.bucket == "active";
        ImGui::TableNextColumn();
        const ImVec4 sc = cancelling ? col_build : status_color(j.status);
        ImGui::TextColored(sc, "%s", "\xE2\x97\x8f");   // ● U+25CF
        ImGui::SameLine();
        ImGui::TextColored(sc, "%s", cancelling ? "cancelling" : j.status.c_str());
        if (cancelling && ImGui::IsItemHovered())
            ImGui::SetTooltip("cancel accepted — waiting for the child to stop\n"
                              "(it is writing a final checkpoint; up to term_grace)");

        // ID (clipped by the cell; full value in the tooltip).
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(j.id.c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", j.id.c_str());
            ImGui::Separator();
            ImGui::Text("bucket:    %s", j.bucket.c_str());
            ImGui::Text("status:    %s", j.status.c_str());
            if (!j.submitted_at.empty()) ImGui::Text("submitted: %s", j.submitted_at.c_str());
            if (!j.checkpoint.empty())   ImGui::Text("checkpoint: %s", j.checkpoint.c_str());
            ImGui::EndTooltip();
        }

        ImGui::TableNextColumn();
        ImGui::TextUnformatted(j.ref.c_str());

        // Progress: a state-tinted bar for a bounded job, a plain live count for an
        // unbounded one (no target -> no fraction to draw).
        ImGui::TableNextColumn();
        const unsigned long long cur =
            j.live_step > j.steps_done ? j.live_step : j.steps_done;
        const unsigned long long tgt = static_cast<unsigned long long>(j.steps_target);
        if (tgt > 0) {
            float frac = j.bucket == "done" ? 1.0f
                       : cur >= tgt         ? 1.0f
                                            : static_cast<float>(static_cast<double>(cur) /
                                                                 static_cast<double>(tgt));
            frac = std::clamp(frac, 0.0f, 1.0f);
            char overlay[64];
            std::snprintf(overlay, sizeof overlay, "%llu / %llu", cur, tgt);
            const bool tint = running || j.bucket == "done" ||
                              j.bucket == "failed" || j.bucket == "cancelled";
            if (tint) ImGui::PushStyleColor(ImGuiCol_PlotHistogram, sc);
            ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f), overlay);
            if (tint) ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(col_wait, "%llu  \xE2\x88\x9E", cur);   // ∞ U+221E
        }

        ImGui::TableNextColumn();
        ImGui::TextUnformatted((j.failure_reason.empty() ? j.user_message
                                                         : j.failure_reason).c_str());

        // Actions. Up only applies to a pending job not already first in the queue.
        ImGui::TableNextColumn();
        ImGui::BeginDisabled(!pending || my_pos == 0);
        if (ImGui::SmallButton("Up")) {
            // RELATIVE, not my_pos - 1. my_pos comes from a snapshot up to
            // poll_interval old, so an absolute index can move a different job than
            // the one this row shows — the job above may have started running in
            // the meantime. The daemon resolves a delta against the live queue.
            try { session_->reorder_by(j.id, -1); }
            catch (const std::exception& e) { handle_action_error(e); }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        // Disabled once a cancel is already pending: a second cancel is a no-op at
        // the daemon, so an enabled button would invite the user to click again and
        // conclude nothing works.
        ImGui::BeginDisabled(!active || cancelling);
        if (ImGui::SmallButton("Cancel")) {
            try {
                session_->cancel(j.id);
                set_status("cancel requested for " + j.id + " — stopping the child", false);
            }
            catch (const std::exception& e) { handle_action_error(e); }
        }
        ImGui::EndDisabled();
        // AllowWhenDisabled is required: ImGui does not report hover for a disabled
        // item by default, so this tooltip — whose entire purpose is to explain why
        // the button is disabled — never appeared.
        if (cancelling && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("already cancelling");
        ImGui::PopID();
    }
    ImGui::EndTable();

    // Vertical splitter: a thin full-width handle glued to the table's bottom edge.
    // Dragging it grows/shrinks the list height (persisted in jobs_height_). Drawn
    // as a faint always-visible bar so it is discoverable, brightening on hover/drag
    // with a resize cursor. Clamped so it can never collapse to nothing or run away.
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.40f, 0.40f, 0.46f, 0.22f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.55f, 0.55f, 0.65f, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.65f, 0.65f, 0.75f, 0.65f));
    ImGui::Button("##jobs_splitter", ImVec2(-1.0f, 6.0f));
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    if (ImGui::IsItemActive())
        jobs_height_ += ImGui::GetIO().MouseDelta.y;
    jobs_height_ = std::clamp(jobs_height_, 80.0f, 2000.0f);

    // Tell the user when the filter is hiding rows, so an empty-looking table is
    // never mistaken for an empty queue.
    if (shown < jobs_.size())
        ImGui::TextDisabled("showing %d of %d (filtered)", shown, jobs_.size());
}

void coastd_panel::render(bool* open) {
    if (!ImGui::Begin("coastd Daemon", open)) { ImGui::End(); return; }

    render_connect_bar();
    render_discovery();      // fixed-list + UDP scan (no-op while connected)
    // Resume polling if we are connected but the poller is not running — the state
    // on_hidden() leaves behind when the window is closed and later reopened.
    if (session_ && make_session_ && !poll_thread_.joinable()) start_poll(make_session_);
    if (session_) sync_from_poll();   // copy the background snapshot (never blocks)
    ImGui::Separator();
    render_submit_form();
    render_jobs_table();

    // A stale queue is shown INDEPENDENTLY of the status line. The job table above
    // renders the last good snapshot, which looks exactly like a live one — so when
    // polling is failing the operator must be told here, even if the status line is
    // currently occupied by an unrelated action error.
    if (poll_failing_) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.0f),
                           "queue may be STALE — %s", poll_status_msg_.c_str());
    }
    if (!status_.empty()) {
        ImGui::Separator();
        const ImVec4 col = status_error_ ? ImVec4(0.9f, 0.4f, 0.4f, 1.0f)
                                         : ImVec4(0.5f, 0.8f, 0.5f, 1.0f);
        ImGui::TextColored(col, "%s", status_.c_str());
    }
    ImGui::End();
}

} // namespace coast
