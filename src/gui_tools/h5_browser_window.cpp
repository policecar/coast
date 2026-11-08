#include "gui_tools/h5_browser_window.h"

#include "h5browse/dataset_reader.h" // default_block_rows (block geometry)
#include "hdf5/executor.h"
#include "gui_core/console_iface.h"
#include "tools/metric_registry.h"

#include <string>

#include "implot.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>

namespace coast {

namespace {

// UI order of filter ops; must stay in sync with op_names below.
constexpr h5b::filter_op op_values[] = {
    h5b::filter_op::lt,     h5b::filter_op::le,     h5b::filter_op::gt,
    h5b::filter_op::ge,     h5b::filter_op::eq,     h5b::filter_op::ne,
    h5b::filter_op::is_nan, h5b::filter_op::is_inf, h5b::filter_op::is_finite,
};
constexpr const char* op_names[] = {"<",      "<=",     ">",
                                    ">=",     "==",     "!=",
                                    "is_nan", "is_inf", "is_finite"};
constexpr int op_count = 9;

bool op_has_value(int op_index)
{
    return op_values[op_index] != h5b::filter_op::is_nan &&
           op_values[op_index] != h5b::filter_op::is_inf &&
           op_values[op_index] != h5b::filter_op::is_finite;
}

std::string to_lower(std::string_view s)
{
    std::string out(s);
    std::ranges::transform(out, out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool contains_ci(std::string_view haystack, const std::string& needle_lower)
{
    if (needle_lower.empty()) return true;
    return to_lower(haystack).find(needle_lower) != std::string::npos;
}

// Native-dtype exact formatting (D10): integers print exactly, f32/f64
// with round-trip precision.
void format_value(char* buf, std::size_t n, const h5b::row_block& blk,
                  std::uint64_t row_in_block, std::uint64_t column)
{
    h5b::dispatch(blk.dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T v = reinterpret_cast<const T*>(
            blk.bytes.data())[row_in_block * blk.row_width + column];
        if constexpr (std::is_same_v<T, float>) {
            std::snprintf(buf, n, "%.9g", static_cast<double>(v));
        } else if constexpr (std::is_same_v<T, double>) {
            std::snprintf(buf, n, "%.17g", v);
        } else if constexpr (std::is_signed_v<T>) {
            std::snprintf(buf, n, "%" PRId64, static_cast<std::int64_t>(v));
        } else {
            std::snprintf(buf, n, "%" PRIu64, static_cast<std::uint64_t>(v));
        }
    });
}

const char* state_label(h5b::op_state s)
{
    switch (s) {
        case h5b::op_state::pending:   return "pending";
        case h5b::op_state::running:   return "running";
        case h5b::op_state::done:      return "done";
        case h5b::op_state::failed:    return "FAILED";
        case h5b::op_state::cancelled: return "cancelled";
    }
    return "?";
}

ImVec4 state_color(h5b::op_state s)
{
    switch (s) {
        case h5b::op_state::pending:   return {0.6f, 0.6f, 0.6f, 1.0f};
        case h5b::op_state::running:   return {0.9f, 0.8f, 0.3f, 1.0f};
        case h5b::op_state::done:      return {0.4f, 0.9f, 0.4f, 1.0f};
        case h5b::op_state::failed:    return {1.0f, 0.35f, 0.35f, 1.0f};
        case h5b::op_state::cancelled: return {0.9f, 0.6f, 0.3f, 1.0f};
    }
    return {1, 1, 1, 1};
}

std::string time_hms(std::chrono::system_clock::time_point tp)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm           tm{};
    localtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof buf, "%H:%M:%S", &tm);
    return buf;
}

// Plot a rebinned base_histogram as ImPlot bars.
void plot_histogram(const h5b::base_histogram& h, int target_bins,
                    const char* label)
{
    const auto m =
        static_cast<std::uint32_t>(std::min<int>(target_bins,
                                                 static_cast<int>(h.n_bins())));
    const auto counts = h.rebin(m);
    const double lo = h.lo(), width = (h.hi() - h.lo()) / m;
    std::vector<double> xs(m), ys(m);
    for (std::uint32_t i = 0; i < m; ++i) {
        xs[i] = lo + (i + 0.5) * width;
        ys[i] = static_cast<double>(counts[i]);
    }
    if (ImPlot::BeginPlot(label, ImVec2(-1, 260))) {
        ImPlot::SetupAxes("value", "count", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::PlotBars("##bars", xs.data(), ys.data(), static_cast<int>(m),
                         width * 0.92);
        ImPlot::EndPlot();
    }
}

} // namespace

h5_browser_window::h5_browser_window(hdf5::executor& h5_exec)
    : engine_(h5_exec),
      file_browser_(ImGuiFileBrowserFlags_CloseOnEsc |
                    ImGuiFileBrowserFlags_ConfirmOnEnter)
{
    file_browser_.SetTitle("Open HDF5 log file");
    file_browser_.SetTypeFilters({".h5"});
}

void h5_browser_window::consume_snapshots()
{
    if (auto s = engine_.consume_file()) file_ = std::move(s);
    if (auto s = engine_.consume_stats()) stats_ = std::move(s);
    if (auto s = engine_.consume_histograms()) hist_ = std::move(s);
    if (auto s = engine_.consume_pairwise()) pair_ = std::move(s);
    if (auto s = engine_.consume_row_index()) ridx_ = std::move(s);
    if (auto s = engine_.consume_series()) series_ = std::move(s);

    if (file_.get() != last_file_) {
        last_file_ = file_.get();
        reset_view_state();
    }

    auto drained = engine_.drain_log();
    for (auto& e : drained) log_lines_.push_back(std::move(e));
    constexpr std::size_t max_lines = 10'000;
    if (log_lines_.size() > max_lines) {
        log_lines_.erase(log_lines_.begin(),
                         log_lines_.begin() +
                             static_cast<long>(log_lines_.size() - max_lines));
    }
}

void h5_browser_window::reset_view_state()
{
    selected_dataset_ = -1;
    search_[0]        = '\0';
    filter_conds_.clear();
    filter_enabled_   = false;
    use_row_range_    = false;
    row_begin_        = 0;
    row_end_          = 0;
    table_col_offset_ = 0;
    graph_column_     = 0;
    hist_column_      = 0;
    // Reset every carried snapshot, stats_ included: the engine publishes the
    // new file before the new file's first stats, so a lingering stats_ would
    // pair the new datasets against the old file's per-dataset entries for a
    // few frames (stats_snapshot carries no file identity to catch it).
    stats_.reset();
    hist_.reset();
    pair_.reset();
    ridx_.reset();
    series_.reset();
}

const h5b::dataset_desc* h5_browser_window::selected_desc() const
{
    if (!file_ || !file_->open || selected_dataset_ < 0 ||
        static_cast<std::size_t>(selected_dataset_) >=
            file_->model.datasets.size()) {
        return nullptr;
    }
    return &file_->model.datasets[static_cast<std::size_t>(selected_dataset_)];
}

h5b::row_filter h5_browser_window::build_filter() const
{
    h5b::row_filter f;
    if (!filter_enabled_) return f; // master gate: editor contents dormant
    if (use_row_range_) {
        // Both bounds always apply — an inverted range visibly matches
        // nothing (with an editor warning) instead of silently widening.
        f.row_begin = row_begin_;
        f.row_end   = row_end_;
    }
    for (const condition_row& c : filter_conds_) {
        f.conditions.push_back(
            {static_cast<std::uint32_t>(std::max(0, c.column)),
             op_values[c.op], c.value});
    }
    return f;
}

bool h5_browser_window::filter_stale_banner(const std::string& result_desc,
                                            bool extra_stale) const
{
    if (!extra_stale && result_desc == build_filter().describe()) return false;
    ImGui::TextColored(state_color(h5b::op_state::cancelled),
                       "inputs changed since this result — recompute");
    return true;
}

void h5_browser_window::open_path(const std::filesystem::path& path)
{
    engine_.open_file(path);
    want_visible_ = true;
    focus_next_   = true;
}

bool h5_browser_window::take_visible_request()
{
    const bool v  = want_visible_;
    want_visible_ = false;
    return v;
}

void h5_browser_window::render(bool* open)
{
    consume_snapshots();

    // The file browser is a top-level (modal) popup, not a child of this
    // window — drive it unconditionally, before the collapsed early-return
    // below, so its lifecycle never depends on this window being expanded.
    file_browser_.Display();
    if (file_browser_.HasSelected()) {
        engine_.open_file(file_browser_.GetSelected());
        file_browser_.ClearSelected();
    }

    ImGui::SetNextWindowSize(ImVec2(1280, 800), ImGuiCond_FirstUseEver);
    // An open_path() from another panel surfaces this window even when it was already
    // visible but buried behind others.
    if (focus_next_) {
        ImGui::SetNextWindowFocus();
        focus_next_ = false;
    }
    if (!ImGui::Begin("HDF5 Browser", open)) {
        ImGui::End();
        return;
    }

    render_toolbar();
    ImGui::Separator();

    const float log_height = 140.0f;
    if (file_ && file_->open) {
        // Left: semantic tree. Right: tabs.
        ImGui::BeginChild("##left", ImVec2(320, -log_height), ImGuiChildFlags_ResizeX);
        render_tree();
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("##right", ImVec2(0, -log_height));
        if (ImGui::BeginTabBar("##tabs")) {
            if (ImGui::BeginTabItem("Datasets")) {
                render_stats_tab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Table")) {
                render_table_tab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Graph")) {
                render_graph_tab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Histograms")) {
                render_histogram_tab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Pairwise")) {
                render_pairwise_tab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();
    } else {
        ImGui::BeginChild("##empty", ImVec2(0, -log_height));
        if (file_ && !file_->error_message.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  state_color(h5b::op_state::failed));
            ImGui::TextWrapped("Open rejected: %s",
                               file_->error_message.c_str());
            ImGui::PopStyleColor();
        } else {
            ImGui::TextDisabled(
                "No file open. Use Open... to load an observe or buffer log "
                "HDF5 file.");
        }
        ImGui::EndChild();
    }

    render_log_panel();
    ImGui::End();
}

// Abbreviate a group path component to the initial of each alphabetic run plus
// full digit runs: "neuron_group_1" -> "ng1", "neuron[3]" -> "n3".
static std::string abbrev_group(const std::string& s)
{
    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isalpha(c) != 0) {
            out.push_back(static_cast<char>(std::tolower(c)));
            while (i < s.size() && std::isalpha(static_cast<unsigned char>(s[i])) != 0) ++i;
        } else if (std::isdigit(c) != 0) {
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) {
                out.push_back(s[i++]);
            }
        } else {
            ++i;
        }
    }
    return out;
}

// Keep a leaf component as a full, readable identifier: alnum kept, other runs
// collapse to a single '_'.
static std::string sanitize_leaf(const std::string& s)
{
    std::string id;
    bool        prev_us = false;
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
            id.push_back(c);
            prev_us = false;
        } else if (!id.empty() && !prev_us) {
            id.push_back('_');
            prev_us = true;
        }
    }
    while (!id.empty() && id.back() == '_') id.pop_back();
    return id;
}

// Readable base name for a dataset bind: abbreviate the group path, keep the
// leaf in full.
//   "neuron_group_1/out" -> "ng1_out"      "neuron[3]/activity" -> "n3_activity"
// This is only a BASE — the session's reserve_var() makes it unique against
// every other console variable (file handles included), so no collision handling
// is needed here.
static std::string console_var_name(const std::string& display)
{
    std::vector<std::string> parts;
    std::string              cur;
    for (char c : display) {
        if (c == '/') {
            if (!cur.empty()) parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) parts.push_back(cur);

    std::string id;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        const std::string a = abbrev_group(parts[i]);
        if (a.empty()) continue;
        if (!id.empty()) id.push_back('_');
        id += a;
    }
    if (!parts.empty()) {
        const std::string leaf = sanitize_leaf(parts.back());
        if (!leaf.empty()) {
            if (!id.empty()) id.push_back('_');
            id += leaf;
        }
    }
    if (id.empty() || std::isdigit(static_cast<unsigned char>(id.front())) != 0) {
        id.insert(0, "ds_");
    }
    return id;
}

void h5_browser_window::bind_console_variable(const std::string& display_name)
{
    if (console_ == nullptr || !file_ || !file_->open) return;
    // Open the file into its own handle variable (deduped if already open), then
    // bind the dataset off that handle — no dependency on a single global `h5`,
    // so datasets from several files can coexist in the session.
    const std::string fvar = console_->open_file(file_->model.path);
    // reserve_var() dedups the readable base against every console variable —
    // file handles included — so this bind can never shadow or self-reference an
    // existing name (e.g. a file run.h5 and a dataset "foo/run" both wanting f_run).
    const std::string name = console_->reserve_var(console_var_name(display_name));
    // Route through bind_dataset so the console escapes the dataset name: an HDF5
    // name may contain quotes/backslashes, and hand-splicing it into a submitted
    // statement would let it break out of the string literal and inject code.
    console_->bind_dataset(name, fvar, display_name);
    console_->set_visible(true);
}

void h5_browser_window::render_toolbar()
{
    if (ImGui::Button("Open...")) file_browser_.Open();
    ImGui::SameLine();
    const bool has_file = file_ && file_->open;
    ImGui::BeginDisabled(!has_file);
    if (ImGui::Button("Close")) engine_.close_file();
    ImGui::SameLine();
    if (ImGui::Button("Cancel analysis")) engine_.cancel_analysis();
    ImGui::EndDisabled();

    // Open the browsed file in the C++ console (if one is present).
    if (console_ != nullptr && has_file) {
        ImGui::SameLine();
        if (ImGui::Button("Console")) {
            console_->open_file(file_->model.path);
            console_->set_visible(true);
        }
    }

    if (has_file) {
        ImGui::SameLine();
        ImGui::Text("%s", file_->model.path.filename().string().c_str());
        for (const auto& [k, v] : file_->model.metadata) {
            ImGui::SameLine();
            ImGui::TextDisabled("| %s: %s", k.c_str(), v.c_str());
        }
        ImGui::SameLine();
        ImGui::TextDisabled("| %.1f MB",
                            static_cast<double>(file_->model.file_bytes) /
                                (1024.0 * 1024.0));
    }
}

bool h5_browser_window::render_tree_node(const h5b::tree_node& node,
                                         const std::string& search_lower)
{
    using kind_t = h5b::tree_node::kind_t;

    // Search: a node is shown if it matches or any descendant does.
    bool self_match = contains_ci(node.name, search_lower);
    bool any_child  = false;
    if (!search_lower.empty() && !self_match) {
        for (const auto& c : node.children) {
            // Probe descendants without rendering.
            struct probe {
                static bool matches(const h5b::tree_node& n,
                                    const std::string& needle)
                {
                    if (contains_ci(n.name, needle)) return true;
                    for (const auto& ch : n.children) {
                        if (matches(ch, needle)) return true;
                    }
                    return false;
                }
            };
            if (probe::matches(c, search_lower)) {
                any_child = true;
                break;
            }
        }
        if (!any_child) return false;
    }

    if (node.kind == kind_t::dataset && node.dataset) {
        const bool selected =
            selected_dataset_ == static_cast<int>(*node.dataset);
        const bool has_dataset = console_ != nullptr && file_ && file_->open &&
                                 *node.dataset < file_->model.datasets.size();
        if (ImGui::Selectable(node.name.c_str(), selected,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            selected_dataset_ = static_cast<int>(*node.dataset);
            // Double-click: shortcut for the right-click menu below.
            if (has_dataset && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                bind_console_variable(
                    file_->model.datasets[*node.dataset].display_name);
            }
        }
        // Right-click: bind this dataset to a console auto-variable. The label
        // stays neutral — the actual variable name is assigned (and deduped)
        // only at bind time, so previewing a specific name here could mislead.
        if (has_dataset && ImGui::BeginPopupContextItem()) {
            const auto& d = file_->model.datasets[*node.dataset];
            if (ImGui::MenuItem("Create console variable")) {
                bind_console_variable(d.display_name);
            }
            ImGui::EndPopup();
        }
        return true;
    }
    if (node.kind == kind_t::meta) {
        ImGui::TextDisabled("%s", node.name.c_str());
        if (!node.detail.empty() && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", node.detail.c_str());
        }
        return true;
    }
    // root / group
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
    if (ImGui::TreeNodeEx(node.name.c_str(), flags)) {
        for (const auto& c : node.children) render_tree_node(c, search_lower);
        ImGui::TreePop();
    }
    return true;
}

void h5_browser_window::render_tree()
{
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##search", "search...", search_, sizeof search_);
    ImGui::Separator();
    const std::string needle = to_lower(search_);
    for (const auto& c : file_->model.root.children) {
        render_tree_node(c, needle);
    }
}

void h5_browser_window::render_stats_tab()
{
    if (!stats_) return;
    const auto& datasets = file_->model.datasets;

    constexpr ImGuiTableFlags flags =
        ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
        ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("##stats", 12, flags)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("dataset", ImGuiTableColumnFlags_WidthStretch);
    for (const char* c : {"dtype", "rows", "width", "state", "count", "NaN",
                          "Inf", "min", "max", "mean", "stddev"}) {
        ImGui::TableSetupColumn(c, ImGuiTableColumnFlags_WidthFixed);
    }
    ImGui::TableHeadersRow();

    const std::size_t n =
        std::min(datasets.size(), stats_->per_dataset.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& d = datasets[i];
        const auto& e = stats_->per_dataset[i];
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool selected = selected_dataset_ == static_cast<int>(i);
        if (ImGui::Selectable(d.display_name.c_str(), selected,
                              ImGuiSelectableFlags_SpanAllColumns)) {
            selected_dataset_ = static_cast<int>(i);
        }
        ImGui::TableNextColumn();
        ImGui::Text("%s", h5b::scalar_name(d.dtype));
        ImGui::TableNextColumn();
        ImGui::Text("%" PRIu64, d.rows);
        ImGui::TableNextColumn();
        ImGui::Text("%" PRIu64, d.row_width);
        ImGui::TableNextColumn();
        ImGui::TextColored(state_color(e.state), "%s", state_label(e.state));
        if (e.state == h5b::op_state::failed && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", e.error_message.c_str());
        }
        ImGui::TableNextColumn();
        ImGui::Text("%" PRIu64, e.stats.count);
        ImGui::TableNextColumn();
        if (e.stats.nan_count > 0) {
            ImGui::TextColored(state_color(h5b::op_state::failed),
                               "%" PRIu64, e.stats.nan_count);
        } else {
            ImGui::Text("0");
        }
        ImGui::TableNextColumn();
        if (e.stats.inf_count > 0) {
            ImGui::TextColored(state_color(h5b::op_state::failed),
                               "%" PRIu64, e.stats.inf_count);
        } else {
            ImGui::Text("0");
        }
        const bool have = e.stats.count > 0;
        ImGui::TableNextColumn();
        ImGui::Text("%s", have ? std::format("{:.6g}", e.stats.min).c_str() : "-");
        ImGui::TableNextColumn();
        ImGui::Text("%s", have ? std::format("{:.6g}", e.stats.max).c_str() : "-");
        ImGui::TableNextColumn();
        ImGui::Text("%s", have ? std::format("{:.6g}", e.stats.mean).c_str() : "-");
        ImGui::TableNextColumn();
        ImGui::Text("%s",
                    have ? std::format("{:.6g}", e.stats.stddev()).c_str() : "-");
    }
    ImGui::EndTable();
}

void h5_browser_window::render_filter_editor()
{
    // Header carries the ACTIVE filter state so it is visible even when
    // collapsed; "###" keeps the widget ID stable while the label changes.
    const std::string header = std::format(
        "Row filter [{}]###rowfilter",
        filter_enabled_ ? build_filter().describe() : "off");
    if (!ImGui::CollapsingHeader(header.c_str())) return;

    ImGui::Checkbox("enabled", &filter_enabled_);
    ImGui::SameLine();
    ImGui::TextDisabled(
        "(the filter applies to Table, Graph, Histograms and Pairwise)");
    if (!filter_enabled_) return;

    if (ImGui::Checkbox("row range", &use_row_range_) && use_row_range_ &&
        row_end_ == 0) {
        // Convenience default: full extent of the selected dataset.
        if (const h5b::dataset_desc* d = selected_desc()) row_end_ = d->rows;
    }
    if (use_row_range_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140);
        ImGui::InputScalar("from##rb", ImGuiDataType_U64, &row_begin_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140);
        ImGui::InputScalar("to (excl.)##re", ImGuiDataType_U64, &row_end_);
        if (row_end_ <= row_begin_) {
            ImGui::SameLine();
            ImGui::TextColored(state_color(h5b::op_state::failed),
                               "empty range — matches nothing");
        }
    }

    int remove_index = -1;
    for (std::size_t i = 0; i < filter_conds_.size(); ++i) {
        condition_row& c = filter_conds_[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::SetNextItemWidth(90);
        ImGui::InputInt("col", &c.column);
        c.column = std::max(0, c.column);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::Combo("##op", &c.op, op_names, op_count);
        if (op_has_value(c.op)) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(160);
            ImGui::InputDouble("##val", &c.value, 0.0, 0.0, "%g");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) remove_index = static_cast<int>(i);
        ImGui::PopID();
    }
    if (remove_index >= 0) {
        filter_conds_.erase(filter_conds_.begin() + remove_index);
    }
    if (ImGui::SmallButton("+ condition")) filter_conds_.push_back({});
    ImGui::SameLine();
    ImGui::TextDisabled("(conditions AND-combine)");
}

void h5_browser_window::render_table_tab()
{
    const h5b::dataset_desc* d = selected_desc();
    if (d == nullptr) {
        ImGui::TextDisabled("Select a dataset in the tree or Datasets tab.");
        return;
    }
    render_filter_editor();

    // Filtered display uses the engine-materialized row-index list. The
    // list is used only while it matches the CURRENT filter — an edited
    // filter can never silently show stale results.
    const h5b::row_filter cur_filter = build_filter();
    bool                  indirect   = false;
    if (!cur_filter.empty()) {
        const bool match =
            ridx_ &&
            ridx_->dataset == static_cast<std::uint32_t>(selected_dataset_) &&
            ridx_->filter_desc == cur_filter.describe();
        if (ImGui::Button("Apply filter to table")) {
            engine_.compute_row_index(
                static_cast<std::uint32_t>(selected_dataset_), cur_filter);
        }
        ImGui::SameLine();
        if (match && ridx_->state == h5b::op_state::running) {
            ImGui::TextDisabled("filtering... %" PRIu64 " of %" PRIu64 " rows",
                                ridx_->rows_scanned, d->rows);
        } else if (match && ridx_->state == h5b::op_state::done) {
            ImGui::TextDisabled("%" PRIu64 " rows match", ridx_->rows_matched);
            indirect = true;
        } else if (match && ridx_->state == h5b::op_state::failed) {
            ImGui::TextColored(state_color(h5b::op_state::failed), "%s",
                               ridx_->error_message.c_str());
        } else {
            ImGui::TextColored(
                state_color(h5b::op_state::cancelled),
                "table shows UNFILTERED rows — press Apply to filter");
        }
    }

    // Column paging: ImGui tables cap at ~512 columns; wide rows page.
    constexpr int max_cols = 64;
    const auto    width    = static_cast<int>(d->row_width);
    if (width > max_cols) {
        ImGui::SetNextItemWidth(140);
        ImGui::InputInt("first column", &table_col_offset_, max_cols);
        table_col_offset_ =
            std::clamp(table_col_offset_, 0, width - 1);
        ImGui::SameLine();
        ImGui::TextDisabled("showing %d..%d of %d", table_col_offset_,
                            std::min(width, table_col_offset_ + max_cols) - 1,
                            width);
    } else {
        table_col_offset_ = 0;
    }
    const int n_cols = std::min(max_cols, width - table_col_offset_);

    const std::uint64_t total_rows =
        indirect ? ridx_->rows_matched : d->rows;
    const std::uint64_t rpb = h5b::default_block_rows(*d);

    constexpr ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter |
        ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_ScrollX;
    if (!ImGui::BeginTable("##data", n_cols + 1, flags)) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("row", ImGuiTableColumnFlags_WidthFixed);
    for (int c = 0; c < n_cols; ++c) {
        ImGui::TableSetupColumn(
            std::format("{}", table_col_offset_ + c).c_str(),
            ImGuiTableColumnFlags_WidthFixed);
    }
    ImGui::TableHeadersRow();

    std::uint64_t prefetch_lo = UINT64_MAX, prefetch_hi = 0;
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(
        std::min<std::uint64_t>(total_rows, INT32_MAX)));
    while (clipper.Step()) {
        std::shared_ptr<const h5b::row_block> blk;
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const std::uint64_t row =
                indirect ? ridx_->rows[static_cast<std::size_t>(i)]
                         : static_cast<std::uint64_t>(i);
            const std::uint64_t block = row / rpb;
            prefetch_lo = std::min(prefetch_lo, block);
            prefetch_hi = std::max(prefetch_hi, block);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%" PRIu64, row);

            if (!blk || blk->block != block ||
                blk->dataset != static_cast<std::uint32_t>(selected_dataset_)) {
                blk = engine_.try_get_block(
                    static_cast<std::uint32_t>(selected_dataset_), block);
            }
            if (blk) {
                const std::uint64_t rib = row - blk->first_row;
                char                buf[64];
                for (int c = 0; c < n_cols; ++c) {
                    ImGui::TableNextColumn();
                    format_value(buf, sizeof buf, *blk, rib,
                                 static_cast<std::uint64_t>(table_col_offset_) +
                                     static_cast<std::uint64_t>(c));
                    ImGui::TextUnformatted(buf);
                }
            } else {
                engine_.request_block(
                    static_cast<std::uint32_t>(selected_dataset_), block);
                for (int c = 0; c < n_cols; ++c) {
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("...");
                }
            }
        }
    }
    ImGui::EndTable();

    // +-1 block prefetch around the visible range.
    if (prefetch_lo != UINT64_MAX) {
        const std::uint64_t max_block = d->rows == 0 ? 0 : (d->rows - 1) / rpb;
        if (prefetch_lo > 0) {
            engine_.request_block(
                static_cast<std::uint32_t>(selected_dataset_), prefetch_lo - 1);
        }
        if (prefetch_hi < max_block) {
            engine_.request_block(
                static_cast<std::uint32_t>(selected_dataset_), prefetch_hi + 1);
        }
    }
}

void h5_browser_window::render_graph_tab()
{
    const h5b::dataset_desc* d = selected_desc();
    if (d == nullptr) {
        ImGui::TextDisabled("Select a dataset in the tree or Datasets tab.");
        return;
    }
    render_filter_editor();

    const int width = static_cast<int>(d->row_width);
    if (width > 1) {
        ImGui::SetNextItemWidth(120);
        ImGui::InputInt("column", &graph_column_);
        graph_column_ = std::clamp(graph_column_, 0, width - 1);
        ImGui::SameLine();
    } else {
        graph_column_ = 0;
    }
    if (ImGui::Button("Compute graph")) {
        engine_.compute_series(static_cast<std::uint32_t>(selected_dataset_),
                               static_cast<std::uint32_t>(graph_column_),
                               build_filter());
    }
    ImGui::SameLine();
    ImGui::TextDisabled("value of column %d over rows (min/max band + mean, "
                        "%u buckets)",
                        graph_column_, h5b::series_max_buckets);

    const bool current =
        series_ &&
        series_->dataset == static_cast<std::uint32_t>(selected_dataset_);
    if (!current) {
        ImGui::TextDisabled("No graph computed for this dataset yet.");
        return;
    }
    filter_stale_banner(series_->filter_desc,
                        series_->column !=
                            static_cast<std::uint32_t>(graph_column_));
    ImGui::TextColored(state_color(series_->state), "%s",
                       state_label(series_->state));
    ImGui::SameLine();
    ImGui::TextDisabled("column %u, rows [%" PRIu64 ", %" PRIu64
                        "), %" PRIu64 " matched (%s)",
                        series_->column, series_->row_begin, series_->row_end,
                        series_->rows_matched, series_->filter_desc.c_str());
    if (series_->state == h5b::op_state::failed) {
        ImGui::TextColored(state_color(h5b::op_state::failed), "%s",
                           series_->error_message.c_str());
        return;
    }
    if (series_->buckets.empty() || series_->rows_matched == 0) {
        ImGui::TextDisabled("no data in the selected span/filter");
        return;
    }

    const auto   n    = series_->buckets.size();
    const double span = static_cast<double>(series_->row_end - series_->row_begin);
    std::vector<double> xs(n), mins(n), maxs(n), means(n);
    double y_lo = std::numeric_limits<double>::infinity();
    double y_hi = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const h5b::series_bucket& b = series_->buckets[i];
        xs[i] = static_cast<double>(series_->row_begin) +
                (static_cast<double>(i) + 0.5) * span / static_cast<double>(n);
        if (b.count == 0) {
            // NaN makes ImPlot skip the segment instead of bridging gaps.
            mins[i] = maxs[i] = means[i] =
                std::numeric_limits<double>::quiet_NaN();
        } else {
            mins[i]  = b.min;
            maxs[i]  = b.max;
            means[i] = b.sum / static_cast<double>(b.count);
            y_lo     = std::min(y_lo, b.min);
            y_hi     = std::max(y_hi, b.max);
        }
    }
    // Padded y-limits instead of y-AutoFit: auto-fit pins the band's
    // min/max onto the plot borders, hiding it. Re-applied whenever a new
    // series arrives; the user keeps manual zoom/pan in between.
    const double y_pad = y_hi > y_lo ? (y_hi - y_lo) * 0.05
                                     : std::max(1.0, std::fabs(y_hi) * 0.05);
    const bool new_data = series_.get() != last_series_plotted_;
    last_series_plotted_ = series_.get();
    if (ImPlot::BeginPlot("##series", ImVec2(-1, 320))) {
        ImPlot::SetupAxes("row", "value", ImPlotAxisFlags_AutoFit, 0);
        ImPlot::SetupAxisLimits(ImAxis_Y1, y_lo - y_pad, y_hi + y_pad,
                                new_data ? ImPlotCond_Always : ImPlotCond_Once);
        // Explicit min/max LINES in addition to the band: for smooth data
        // each bucket's min ~= max, so the band alone collapses into faint
        // slivers (visible only where a bucket contains a jump). SkipNaN
        // bridges filter-emptied buckets on the lines.
        constexpr auto line_flags =
            static_cast<int>(ImPlotLineFlags_SkipNaN);
        ImPlot::PlotShaded("min..max", xs.data(), mins.data(), maxs.data(),
                           static_cast<int>(n),
                           ImPlotSpec(ImPlotProp_FillAlpha, 0.60f));
        ImPlot::PlotLine("min", xs.data(), mins.data(), static_cast<int>(n),
                         ImPlotSpec(ImPlotProp_LineWeight, 1.0f,
                                    ImPlotProp_Flags, line_flags));
        ImPlot::PlotLine("max", xs.data(), maxs.data(), static_cast<int>(n),
                         ImPlotSpec(ImPlotProp_LineWeight, 1.0f,
                                    ImPlotProp_Flags, line_flags));
        ImPlot::PlotLine("mean", xs.data(), means.data(), static_cast<int>(n),
                         ImPlotSpec(ImPlotProp_LineWeight, 2.0f,
                                    ImPlotProp_Flags, line_flags));
        ImPlot::EndPlot();
    }
}

void h5_browser_window::render_histogram_tab()
{
    const h5b::dataset_desc* d = selected_desc();
    if (d == nullptr) {
        ImGui::TextDisabled("Select a dataset in the tree or Datasets tab.");
        return;
    }
    render_filter_editor();

    if (ImGui::Button("Compute histograms")) {
        engine_.compute_histograms(
            static_cast<std::uint32_t>(selected_dataset_), build_filter());
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(one pass fills all %" PRIu64 " columns)",
                        d->row_width);

    const bool current =
        hist_ && hist_->dataset == static_cast<std::uint32_t>(selected_dataset_);
    if (!current) {
        ImGui::TextDisabled("No histograms computed for this dataset yet.");
        return;
    }
    filter_stale_banner(hist_->filter_desc);
    ImGui::TextColored(state_color(hist_->state), "%s", state_label(hist_->state));
    ImGui::SameLine();
    ImGui::TextDisabled("scanned %" PRIu64 " rows, matched %" PRIu64 " (%s)",
                        hist_->rows_scanned, hist_->rows_matched,
                        hist_->filter_desc.c_str());
    if (hist_->state == h5b::op_state::failed) {
        ImGui::TextColored(state_color(h5b::op_state::failed), "%s",
                           hist_->error_message.c_str());
        return;
    }
    if (hist_->columns.empty()) return;

    // rebin() needs m | base_bins; snapping to a power of two guarantees it
    // because the base is a power of two (asserted, so a future retune of the
    // constant can't silently break the divisibility this loop relies on).
    static_assert((h5b::per_column_base_bins &
                   (h5b::per_column_base_bins - 1)) == 0,
                  "histogram bin snapping assumes a power-of-two base");
    const int base_bins = static_cast<int>(h5b::per_column_base_bins);

    const int n_cols = static_cast<int>(hist_->columns.size());
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("column", &hist_column_);
    hist_column_ = std::clamp(hist_column_, 0, n_cols - 1);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180);
    ImGui::SliderInt("bins", &hist_bins_, 8, base_bins, "%d",
                     ImGuiSliderFlags_Logarithmic);
    int snapped = 8;
    while (snapped * 2 <= hist_bins_ && snapped < base_bins) snapped *= 2;
    hist_bins_ = snapped;

    const auto& h = hist_->columns[static_cast<std::size_t>(hist_column_)];
    if (h.empty()) {
        ImGui::TextDisabled("column %d has no finite values", hist_column_);
        return;
    }
    ImGui::TextDisabled("range [%g, %g), %" PRIu64 " values", h.lo(), h.hi(),
                        h.total());
    plot_histogram(h, hist_bins_, "##colhist");
}

void h5_browser_window::render_pairwise_tab()
{
    const h5b::dataset_desc* d = selected_desc();
    if (d == nullptr) {
        ImGui::TextDisabled("Select a dataset in the tree or Datasets tab.");
        return;
    }
    if (d->row_width < 2) {
        ImGui::TextDisabled(
            "Pairwise metrics need row vectors (this dataset has width %" PRIu64
            ").",
            d->row_width);
        return;
    }
    render_filter_editor();

    const auto metrics = coast::metric_registry();
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("metric",
                          metrics[static_cast<std::size_t>(metric_index_)].name)) {
        for (int i = 0; i < static_cast<int>(metrics.size()); ++i) {
            if (ImGui::Selectable(metrics[static_cast<std::size_t>(i)].name,
                                  i == metric_index_)) {
                metric_index_ = i;
            }
        }
        ImGui::EndCombo();
    }
    const auto& def = metrics[static_cast<std::size_t>(metric_index_)];
    if (def.has_param) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat(def.param_name, &minkowski_p_);
        minkowski_p_ = std::clamp(minkowski_p_, def.param_min, def.param_max);
    }

    ImGui::Checkbox("full run (all pairs)", &full_run_);
    if (!full_run_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::InputInt("k", &sample_k_);
        sample_k_ = std::clamp(sample_k_, 2, 1'000'000);
        ImGui::SameLine();
        if (ImGui::SmallButton("resample")) ++seed_;
        ImGui::SameLine();
        ImGui::TextDisabled("seed %" PRIu64, seed_);
    }
    // Cost hint (D6a): upper bound from rows / k before the filter runs.
    const std::uint64_t m_hint =
        full_run_ ? d->rows
                  : std::min<std::uint64_t>(
                        static_cast<std::uint64_t>(sample_k_), d->rows);
    ImGui::TextDisabled("up to %" PRIu64 " pairs", m_hint * (m_hint - 1) / 2);

    if (ImGui::Button("Compute pairwise")) {
        h5b::pairwise_request req;
        req.kind     = def.kind;
        req.params.p = minkowski_p_;
        req.k        = static_cast<std::uint32_t>(sample_k_);
        req.seed     = seed_;
        req.full_run = full_run_;
        req.filter   = build_filter();
        engine_.compute_pairwise(static_cast<std::uint32_t>(selected_dataset_),
                                 req);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) engine_.cancel_analysis();

    const bool current =
        pair_ && pair_->dataset == static_cast<std::uint32_t>(selected_dataset_);
    if (!current) return;

    // Pairwise recomputes on every request field, not just the filter.
    const bool pw_stale =
        pair_->request.kind != def.kind ||
        pair_->request.full_run != full_run_ ||
        (!full_run_ &&
         (pair_->request.k != static_cast<std::uint32_t>(sample_k_) ||
          pair_->request.seed != seed_)) ||
        (def.has_param && pair_->request.params.p != minkowski_p_);
    filter_stale_banner(pair_->request.filter.describe(), pw_stale);
    ImGui::TextColored(state_color(pair_->state), "%s", state_label(pair_->state));
    ImGui::SameLine();
    ImGui::TextDisabled("%" PRIu64 " rows in filter; %" PRIu64 " / %" PRIu64
                        " pairs; %" PRIu64 " in histogram",
                        pair_->rows_in_filter, pair_->pairs_done,
                        pair_->pairs_total, pair_->histogram.total());
    if (pair_->pairs_total > 0) {
        ImGui::ProgressBar(static_cast<float>(
            static_cast<double>(pair_->pairs_done) /
            static_cast<double>(pair_->pairs_total)));
    }
    if (pair_->state == h5b::op_state::failed) {
        ImGui::TextColored(state_color(h5b::op_state::failed), "%s",
                           pair_->error_message.c_str());
        return;
    }
    if (pair_->histogram.total() > 0) {
        static_assert((h5b::pairwise_base_bins &
                       (h5b::pairwise_base_bins - 1)) == 0,
                      "pairwise bin snapping assumes a power-of-two base");
        const int pw_base = static_cast<int>(h5b::pairwise_base_bins);
        ImGui::SetNextItemWidth(180);
        ImGui::SliderInt("bins##pw", &pair_bins_, 16, pw_base, "%d",
                         ImGuiSliderFlags_Logarithmic);
        int snapped = 16;
        while (snapped * 2 <= pair_bins_ && snapped < pw_base) snapped *= 2;
        pair_bins_ = snapped;
        plot_histogram(pair_->histogram, pair_bins_, "##pairhist");
    }
}

void h5_browser_window::render_log_panel()
{
    ImGui::Separator();
    ImGui::TextDisabled("Diagnostics");
    ImGui::SameLine();
    if (ImGui::SmallButton("copy")) {
        std::string all;
        for (const auto& l : log_lines_) {
            all += time_hms(l.time);
            all += l.severity == h5b::log_severity::error     ? " [error] "
                   : l.severity == h5b::log_severity::warning ? " [warn ] "
                                                              : " [info ] ";
            all += l.message;
            all += '\n';
        }
        ImGui::SetClipboardText(all.c_str());
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("clear")) log_lines_.clear();

    ImGui::BeginChild("##log", ImVec2(0, 0), ImGuiChildFlags_Borders);
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(log_lines_.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const auto& l = log_lines_[static_cast<std::size_t>(i)];
            const ImVec4 col =
                l.severity == h5b::log_severity::error
                    ? state_color(h5b::op_state::failed)
                    : l.severity == h5b::log_severity::warning
                          ? state_color(h5b::op_state::cancelled)
                          : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
            ImGui::TextColored(col, "%s  %s", time_hms(l.time).c_str(),
                               l.message.c_str());
        }
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
        ImGui::SetScrollHereY(1.0f); // follow tail
    }
    ImGui::EndChild();
}

} // namespace coast
