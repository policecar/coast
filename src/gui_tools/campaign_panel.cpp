#include "gui_tools/campaign_panel.h"

#include "console/quote_cpp.h"
#include "gui_core/console_iface.h"
#include "gui_tools/h5_browser_window.h"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <thread>
#include <cstdio>
#include <exception>
#include <format>
#include <stdexcept>

namespace coast {

namespace {

namespace camp = coast::campaign;

// A C++ identifier seed from the root's last path component ("LRI_sweep" -> "LRI_sweep",
// "ngm25 gpu" -> "ngm25_gpu"). The console's reserve_var() disambiguates collisions.
std::string var_seed(const std::filesystem::path& root)
{
    std::string s = root.filename().string();
    if (s.empty()) s = "campaign";
    std::string out;
    for (const char c : s) {
        out += (std::isalnum(static_cast<unsigned char>(c)) != 0) ? c : '_';
    }
    if (std::isdigit(static_cast<unsigned char>(out.front())) != 0) out.insert(out.begin(), '_');
    return out;
}

} // namespace

campaign_panel::campaign_panel()
    : dir_browser_(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_CloseOnEsc |
                   ImGuiFileBrowserFlags_ConfirmOnEnter)
{
    dir_browser_.SetTitle("Open campaign directory");
}

// ------------------------------------------------------------------ scanning
void campaign_panel::begin_scan(const std::filesystem::path& root)
{
    if (scanning_) return; // one scan in flight at a time

    // Nothing panel-visible changes until the thread is running: an ordering the
    // failure path below already depends on, extended to the rest of the state so a
    // failed launch does not leave a cleared selection and a stale status behind.
    auto slot = std::make_shared<scan_slot>();
    try {
        // Detached, and co-owning the slot. See scan_slot for why this is not a
        // std::async future. The thread captures the slot and `root` BY VALUE and
        // touches no panel member, so it stays safe even if the panel dies first.
        std::thread([slot, root] {
            try {
                auto c = camp::open(root);
                const std::lock_guard<std::mutex> lk(slot->m);
                slot->result = std::move(c);
            } catch (const std::exception& e) {
                const std::lock_guard<std::mutex> lk(slot->m);
                slot->error = e.what();
            }
            slot->done.store(true, std::memory_order_release);
        }).detach();
    } catch (const std::exception& e) {
        // Thread creation can fail. scanning_ is set only AFTER this point: setting it
        // first left the panel latched in a permanent "scanning…" state with Scan and
        // Rescan disabled for the rest of the process.
        status_       = std::string("could not start scan: ") + e.what();
        status_error_ = true;
        return;
    }

    scan_         = std::move(slot);
    scanning_     = true;
    status_       = "scanning " + root.string() + " …";
    status_error_ = false;
    focused_      = -1;
}

void campaign_panel::poll_scan()
{
    if (!scanning_ || !scan_) return;
    if (!scan_->done.load(std::memory_order_acquire)) return;

    scanning_ = false;
    std::shared_ptr<campaign::campaign> result;
    std::string                         error;
    {
        const std::lock_guard<std::mutex> lk(scan_->m);
        result = std::move(scan_->result);
        error  = std::move(scan_->error);
    }
    scan_.reset(); // drop our reference; the worker has already finished

    if (!result) {
        campaign_.reset();
        filtered_ = {};
        keep_.clear();
        status_       = error.empty() ? "scan produced no campaign" : error;
        status_error_ = true;
        return;
    }
    campaign_ = std::move(result);

    // Everything included to start with, so the table opens showing the campaign.
    //
    // focused_ is reset here as well as in begin_scan(). The old table stays on screen
    // while a rescan runs, so a row clicked during the scan set focused_ to an index
    // into the OLD filtered_; adopting the new campaign silently re-pointed it at a
    // different run, and "Open observe" then opened that one instead.
    picked_.Clear();
    focused_ = -1;
    keep_.clear();
    keep_.reserve(campaign_->axes().size());
    for (const auto& a : campaign_->axes()) keep_.emplace_back(a.values.size(), true);

    filtered_ = campaign_->all();
    status_   = std::format("{} run{}, {} swept parameter{}", campaign_->size(),
                            campaign_->size() == 1 ? "" : "s", campaign_->axes().size(),
                            campaign_->axes().size() == 1 ? "" : "s");
    // A problem must not read as a neutral footnote on a green status line: if
    // anything was wrong with the files, the whole line goes red and says so.
    if (!campaign_->issues().empty()) {
        const auto& is = campaign_->issues();
        const auto  dropped =
            std::count_if(is.begin(), is.end(), [](const auto& i) { return i.run_dropped; });
        // "0 runs excluded" is only worth saying when something WAS excluded; a walk
        // issue (an unreadable subtree) excludes an unknown number, so it deliberately
        // does not count towards `dropped` — see scan_issue::run_dropped.
        status_ += std::format(" — {} problem{}{}; hover for details", is.size(),
                               is.size() == 1 ? "" : "s",
                               dropped == 0 ? std::string{}
                                            : std::format(" ({} run{} excluded)", dropped,
                                                          dropped == 1 ? "" : "s"));
        status_error_ = true;
    } else {
        status_error_ = false;
    }
}

void campaign_panel::rebuild_filtered()
{
    if (!campaign_) return;

    // The axis filters are multi-select ("show 0.01 AND 0.05"), which where() does not
    // express — it takes one value. The predicate escape hatch does, and costs one
    // pass over an in-memory table.
    const auto& axes = campaign_->axes();
    filtered_        = campaign_->where([&](const camp::run& r) {
        for (std::size_t a = 0; a < axes.size(); ++a) {
            const camp::value v  = r.get(axes[a].name);
            bool              ok = false;
            for (std::size_t k = 0; k < axes[a].values.size(); ++k) {
                if (keep_[a][k] && axes[a].values[k] == v) { ok = true; break; }
            }
            if (!ok) return false;
        }
        return true;
    });
    // The picked rows are keyed by row index, so a filter change would silently
    // re-point them at different runs. Drop them instead.
    picked_.Clear();
    focused_       = -1;
    filters_dirty_ = false;
}

// ------------------------------------------------------------------- toolbar
void campaign_panel::render_toolbar()
{
    ImGui::SetNextItemWidth(-260.0f);
    const bool entered = ImGui::InputTextWithHint("##root", "campaign directory", root_buf_,
                                                  sizeof root_buf_,
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) dir_browser_.Open();
    ImGui::SameLine();

    ImGui::BeginDisabled(scanning_ || root_buf_[0] == '\0');
    const bool clicked = ImGui::Button("Scan");
    ImGui::EndDisabled();

    if ((entered || clicked) && root_buf_[0] != '\0' && !scanning_) {
        begin_scan(std::filesystem::path(root_buf_));
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!campaign_ || scanning_);
    if (ImGui::Button("Rescan")) begin_scan(campaign_->root());
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Re-read the configs — picks up runs that finished since the scan.");

    if (!status_.empty()) {
        const ImVec4 col = status_error_ ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f)
                                         : ImVec4(0.65f, 0.85f, 0.65f, 1.0f);
        ImGui::TextColored(col, "%s", status_.c_str());
        // The full list, on hover. Every issue names its run, its file and what was
        // wrong — the point is that the user never has to go and diff the file to
        // find out what the loader objected to.
        if (campaign_ && !campaign_->issues().empty() && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            for (const auto& i : campaign_->issues()) {
                // i.label, not i.dir.filename(): two `run1` directories under different
                // parents both print as "run1" and the reader cannot tell which is
                // broken. A walk issue carries no file, hence no empty brackets.
                const std::string where =
                    i.label + (i.file.empty() ? std::string{} : " [" + i.file + "]");
                ImGui::Text("%s %s%s", where.c_str(), i.message.c_str(),
                            i.run_dropped ? "  (run excluded)" : "");
            }
            ImGui::EndTooltip();
        }
    }
    if (scanning_) {
        ImGui::SameLine();
        ImGui::TextUnformatted("(working)");
    }
}

// ------------------------------------------------------------------- filters
void campaign_panel::render_filters()
{
    const auto& axes = campaign_->axes();
    if (axes.empty()) {
        ImGui::TextWrapped("No swept parameters: every run's config is identical.");
        return;
    }

    for (std::size_t a = 0; a < axes.size(); ++a) {
        ImGui::PushID(static_cast<int>(a));
        if (ImGui::CollapsingHeader(axes[a].name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            // The full pointer, so a name qualified down to `ngm25/...` can still be
            // traced back to the exact config leaf it came from.
            ImGui::TextDisabled("%s", axes[a].pointer.c_str());
            if (ImGui::SmallButton("all")) {
                std::fill(keep_[a].begin(), keep_[a].end(), true);
                filters_dirty_ = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("none")) {
                std::fill(keep_[a].begin(), keep_[a].end(), false);
                filters_dirty_ = true;
            }
            for (std::size_t k = 0; k < axes[a].values.size(); ++k) {
                // Per-value id. ImGui derives a widget's identity from its LABEL, and
                // value::label() is not injective across kinds — boolean `true` and
                // the string "true" both render "true" — so two distinct axis values
                // could share an id and one checkbox would drive the other's filter.
                // A value containing "##"/"###" does the same deliberately.
                ImGui::PushID(static_cast<int>(k));
                bool on = keep_[a][k];
                if (ImGui::Checkbox(axes[a].values[k].label().c_str(), &on)) {
                    keep_[a][k]    = on;
                    filters_dirty_ = true;
                }
                ImGui::PopID();
            }
        }
        ImGui::PopID();
    }
}

// --------------------------------------------------------------------- table
void campaign_panel::render_table()
{
    const auto& axes = campaign_->axes();

    // Multi-select wraps the table (not the other way round), per ImGui's own
    // in-a-table pattern: requests raised while the rows are submitted are applied
    // after EndMultiSelect. Using the real API rather than reading KeyCtrl/KeyShift
    // by hand is what gets shift-ranges, Ctrl+A, Escape and keyboard nav right.
    constexpr auto ms_flags = ImGuiMultiSelectFlags_BoxSelect1d |
                              ImGuiMultiSelectFlags_ClearOnEscape |
                              ImGuiMultiSelectFlags_ClearOnClickVoid;
    ImGuiMultiSelectIO* ms = ImGui::BeginMultiSelect(ms_flags, picked_.Size,
                                                     static_cast<int>(filtered_.size()));
    picked_.ApplyRequests(ms);

    const int cols = 2 + static_cast<int>(axes.size()) + 3;
    constexpr auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
                           ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("runs", cols, flags)) {
        picked_.ApplyRequests(ImGui::EndMultiSelect());
        return;
    }

    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 32.0f);
    ImGui::TableSetupColumn("run", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    for (const auto& a : axes) ImGui::TableSetupColumn(a.name.c_str());
    ImGui::TableSetupColumn("status");
    ImGui::TableSetupColumn("steps");
    ImGui::TableSetupColumn("wall clock");
    ImGui::TableHeadersRow();

    for (std::size_t i = 0; i < filtered_.size(); ++i) {
        const camp::run& r = filtered_[i];
        ImGui::TableNextRow();
        ImGui::PushID(static_cast<int>(i));

        ImGui::TableNextColumn();
        ImGui::Text("%zu", i);

        ImGui::TableNextColumn();
        const std::string id = r.id();
        ImGui::SetNextItemSelectionUserData(static_cast<ImGuiSelectionUserData>(i));
        const bool sel = picked_.Contains(static_cast<ImGuiID>(i));
        if (ImGui::Selectable(id.c_str(), sel, ImGuiSelectableFlags_SpanAllColumns)) {
            focused_ = static_cast<int>(i);
        }
        ImGui::SetItemTooltip("%s", r.dir().string().c_str());

        for (const auto& a : axes) {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.get(a.name).label().c_str());
        }

        const campaign::manifest& m = r.info();
        ImGui::TableNextColumn();
        if (m.malformed()) {
            // Distinct from "no manifest", and deliberately loud: a damaged
            // provenance record used to render identically to a run still in flight.
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "manifest error");
            ImGui::SetItemTooltip("%s", m.error.c_str());
        } else if (!m.present) {
            ImGui::TextDisabled("no manifest");
        } else if (r.complete()) {
            ImGui::TextColored(ImVec4(0.65f, 0.85f, 0.65f, 1.0f), "%s", m.exit_reason.c_str());
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f), "%s", m.exit_reason.c_str());
        }

        // A MALFORMED manifest yields nothing here, not the defaults its wrong-typed
        // fields fell back to. Gating on m.present alone printed "0 / 600000" and
        // "0.0 s" — coerced zeros, presented in the same style as measured numbers —
        // right beside the "manifest error" the column above had just raised.
        const bool numbers_ok = m.present && !m.malformed();

        ImGui::TableNextColumn();
        if (numbers_ok) {
            ImGui::Text("%llu / %llu", static_cast<unsigned long long>(m.steps_completed),
                        static_cast<unsigned long long>(m.steps_requested));
        } else {
            ImGui::TextDisabled("-");
        }

        ImGui::TableNextColumn();
        if (numbers_ok) {
            ImGui::Text("%.1f s", m.wall_clock_seconds);
        } else {
            ImGui::TextDisabled("-");
        }

        ImGui::PopID();
    }
    ImGui::EndTable();

    picked_.ApplyRequests(ImGui::EndMultiSelect());

    // Ctrl+A and Escape, handled here rather than left to BeginMultiSelect's own
    // shortcuts. MEASURED: those never fire for this table — with the modifier and
    // key both live on the frame (io.KeyCtrl=1, IsKeyDown(A)=1) and keyboard capture
    // on, the selection did not change. ImGui routes those shortcuts to the window
    // owning the multi-select scope, and the scope here lives inside a BeginChild
    // that the routing does not reach. Mouse click / Ctrl+click / Shift+range all
    // work through the normal path and are left to it.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_A, false)) {
            for (std::size_t i = 0; i < filtered_.size(); ++i) {
                picked_.SetItemSelected(static_cast<ImGuiID>(i), true);
            }
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            picked_.Clear();
        }
    }
}

std::vector<std::size_t> campaign_panel::picked_rows() const
{
    std::vector<std::size_t> rows;
    if (picked_.Size <= 0) return rows;
    // Walk the table order rather than the storage's hash order, so the ids handed to
    // the console come out in the order the user sees them.
    rows.reserve(static_cast<std::size_t>(picked_.Size));
    for (std::size_t i = 0; i < filtered_.size(); ++i) {
        if (picked_.Contains(static_cast<ImGuiID>(i))) rows.push_back(i);
    }
    return rows;
}

// ------------------------------------------------------------------- actions
void campaign_panel::send_to_console(bool whole_campaign)
{
    if (console_ == nullptr || !campaign_) return;

    // Show the console before submitting anything. Submitting into a hidden panel
    // puts the variable somewhere the user cannot see, and looks like the button did
    // nothing. Same convention as h5_browser_window::bind_console_variable; it also
    // pre-warms the interpreter, so the statement below is not waiting on the
    // one-time build. Covers both buttons — they share this function.
    console_->set_visible(true);

    const std::string root = campaign_->root().string();
    const std::string cvar = console_->reserve_var("c_" + var_seed(campaign_->root()));
    console_->submit("auto " + cvar + " = camp::open(" + console::detail::quote_cpp(root) + ");");

    if (whole_campaign) {
        status_       = "sent campaign to console as " + cvar;
        status_error_ = false;
        return;
    }

    const std::string svar = console_->reserve_var("sel_" + var_seed(campaign_->root()));

    // Picked rows WIN over the axis filters. Picking rows in the table is a more
    // specific act than setting a filter — if the user has done both, the rows are
    // what they last said. The button label states which of the two is about to be
    // sent, so this precedence is never something to guess at.
    const auto rows = picked_rows();
    if (!rows.empty()) {
        std::string code = "auto " + svar + " = " + cvar + "->select({";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i) code += ", ";
            code += console::detail::quote_cpp(filtered_[rows[i]].id());
        }
        console_->submit(code + "});");
        status_ = std::format("sent {} picked run{} to console as {}", rows.size(),
                              rows.size() == 1 ? "" : "s", svar);
        status_error_ = false;
        return;
    }

    // Prefer the where()-chain when every axis is pinned to a single value: that is
    // the query the user would have typed, and it stays readable in the console
    // history. Otherwise emit the exact id list — a multi-select filter has no
    // where() equivalent, and approximating it would hand over a DIFFERENT set of
    // runs than the table shows.
    //
    // The chain is also EXECUTED here and its result compared against the table. The
    // panel's own filter compares values exactly, while selection::where() matches
    // every value within rel_tol — so on an axis holding one logical value in two
    // representations (a float32 round-trip alongside the double), a chain that looks
    // equivalent selects both. Rather than reason about when that can happen, measure
    // it: if the chain does not reproduce the table exactly, fall back to the id list.
    const auto& axes    = campaign_->axes();
    bool        pinned  = !axes.empty();
    std::string chain   = "auto " + svar + " = " + cvar + "->all()";
    camp::selection equivalent = campaign_->all();
    for (std::size_t a = 0; a < axes.size() && pinned; ++a) {
        const auto n = std::count(keep_[a].begin(), keep_[a].end(), true);
        if (n != 1) { pinned = false; break; }
        const std::size_t k =
            static_cast<std::size_t>(std::find(keep_[a].begin(), keep_[a].end(), true) -
                                     keep_[a].begin());
        const camp::value& v = axes[a].values[k];
        try {
            switch (v.k) {
            case camp::value::kind::number:
                chain += std::format(".where({}, {})", console::detail::quote_cpp(axes[a].name),
                                     v.num);
                equivalent = equivalent.where(axes[a].name, v.num);
                break;
            case camp::value::kind::text:
                chain += ".where(" + console::detail::quote_cpp(axes[a].name) + ", " +
                         console::detail::quote_cpp(v.str) + ")";
                equivalent = equivalent.where(axes[a].name, std::string_view(v.str));
                break;
            case camp::value::kind::boolean:
                chain += ".where(" + console::detail::quote_cpp(axes[a].name) + ", " +
                         (v.flag ? "true" : "false") + ")";
                equivalent = equivalent.where(axes[a].name, v.flag);
                break;
            case camp::value::kind::absent:
                pinned = false; // "this leaf is missing" is not expressible as a filter
                break;
            }
        } catch (const std::exception&) {
            pinned = false; // the chain would not even run; send the ids instead
        }
    }

    // The measurement the comment above promises. Not a size compare: two selections of
    // equal size can still hold different runs.
    if (pinned && equivalent.ids() != filtered_.ids()) pinned = false;

    if (pinned) {
        console_->submit(chain + ";");
    } else {
        std::string code = "auto " + svar + " = " + cvar + "->select({";
        const auto  ids  = filtered_.ids();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i) code += ", ";
            code += console::detail::quote_cpp(ids[i]);
        }
        console_->submit(code + "});");
    }

    status_ = std::format("sent {} run{} to console as {}", filtered_.size(),
                          filtered_.size() == 1 ? "" : "s", svar);
    status_error_ = false;
}

void campaign_panel::render_actions()
{
    const bool have_focus = focused_ >= 0 && focused_ < static_cast<int>(filtered_.size());

    ImGui::BeginDisabled(!have_focus || browser_ == nullptr);
    if (ImGui::Button("Open observe in HDF5 Browser")) {
        const camp::run& r = filtered_[static_cast<std::size_t>(focused_)];
        try {
            // Ask the run for the path rather than rebuilding "observe/<g>.observe.h5"
            // here: that duplicated the loader's own constants in another translation
            // unit and skipped its existence check, so a mistyped group reached the
            // browser unvalidated and the diagnostic came from the wrong layer.
            const auto path = r.observe_path(observe_group_buf_);
            browser_->open_path(path);
            status_       = "opened " + path.string();
            status_error_ = false;
        } catch (const std::exception& e) {
            status_       = e.what();
            status_error_ = true;
        }
    }
    ImGui::EndDisabled();
    if (browser_ == nullptr) ImGui::SetItemTooltip("The HDF5 browser is not available.");

    ImGui::SameLine();
    ImGui::SetNextItemWidth(160.0f);
    ImGui::InputTextWithHint("##group", "observe group (first)", observe_group_buf_,
                             sizeof observe_group_buf_);

    ImGui::SameLine();
    ImGui::BeginDisabled(console_ == nullptr);
    if (ImGui::Button("Campaign -> Console")) send_to_console(true);
    ImGui::SameLine();

    // Name what will actually be sent. There are two candidate sets on screen — the
    // filtered rows and the picked ones — so a bare "Selection" would be exactly the
    // ambiguity this label exists to remove.
    const int  n_picked = picked_.Size;
    const auto label    = n_picked > 0
                              ? std::format("{} picked row{} -> Console", n_picked,
                                            n_picked == 1 ? "" : "s")
                              : std::format("{} filtered run{} -> Console", filtered_.size(),
                                            filtered_.size() == 1 ? "" : "s");
    if (ImGui::Button(label.c_str())) send_to_console(false);
    ImGui::EndDisabled();
    if (console_ == nullptr) {
        ImGui::SetItemTooltip("The C++ console is not available.");
    } else {
        ImGui::SetItemTooltip("Click a row to pick it; Ctrl+click to add, Shift+click for a "
                              "range, Ctrl+A for all, Escape to clear.\n"
                              "With nothing picked, the whole filtered set is sent.");
    }

    if (have_focus) {
        ImGui::SameLine();
        ImGui::TextDisabled("| %s",
                            filtered_[static_cast<std::size_t>(focused_)].dir().string().c_str());
    }
}

// -------------------------------------------------------------------- render
void campaign_panel::render(bool* open)
{
    poll_scan();

    // The directory browser is a top-level popup, not a child of this window — drive
    // it before the collapsed early-return so its lifecycle never depends on this
    // window being expanded (same reason as h5_browser_window).
    dir_browser_.Display();
    if (dir_browser_.HasSelected()) {
        const auto sel = dir_browser_.GetSelected();
        dir_browser_.ClearSelected();
        if (scanning_) {
            // begin_scan() refuses while a scan is in flight. Writing the picked path
            // into the box regardless left it naming a directory that was never
            // scanned, above a table showing a different campaign.
            status_       = "a scan is already running; wait for it, then pick again";
            status_error_ = true;
        } else {
            // begin_scan() gets the picked path, so the scan itself is never truncated.
            // The BOX can be: a path longer than the buffer would show clipped, and a
            // later Enter or Scan would then act on that shorter, wrong path. Say so
            // rather than let the display quietly disagree with what was scanned.
            const std::string full = sel.string();
            std::snprintf(root_buf_, sizeof root_buf_, "%s", full.c_str());
            begin_scan(sel);
            if (full.size() >= sizeof root_buf_) {
                status_ = std::format(
                    "scanning the picked directory, but it is {} characters and the box "
                    "holds {} — retype or shorten it before pressing Scan again",
                    full.size(), sizeof root_buf_ - 1);
                status_error_ = true;
            }
        }
    }

    ImGui::SetNextWindowSize(ImVec2(1100, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Campaign Browser", open)) {
        ImGui::End();
        return;
    }

    render_toolbar();
    ImGui::Separator();

    if (!campaign_) {
        ImGui::TextWrapped(
            "Open a directory of simulation runs. Any directory holding a config.json is a "
            "run; nested layouts are scanned too. The swept parameters are discovered by "
            "diffing the run configs against each other — nothing needs to be declared.");
        ImGui::End();
        return;
    }

    if (filters_dirty_) rebuild_filtered();

    if (ImGui::BeginChild("filters", ImVec2(280.0f, -ImGui::GetFrameHeightWithSpacing()),
                          ImGuiChildFlags_Borders)) {
        render_filters();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    if (ImGui::BeginChild("runs_pane", ImVec2(0.0f, -ImGui::GetFrameHeightWithSpacing()))) {
        ImGui::Text("%zu of %zu runs", filtered_.size(), campaign_->size());
        render_table();
    }
    ImGui::EndChild();

    render_actions();
    ImGui::End();
}

} // namespace coast
