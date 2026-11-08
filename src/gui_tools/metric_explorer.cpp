#include "gui_tools/metric_explorer.h"

#include "imgui.h"

#include "tools/histogram.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <system_error>
#include <vector>

namespace coast {

namespace {
constexpr int HEADER_THUMB = 64;                 // header thumbnail texture size (px)
constexpr std::size_t MEM_WARN = std::size_t{2} << 30;   // soft footprint warning at ~2 GB

std::string format_bytes(std::size_t b)
{
    double v = static_cast<double>(b);
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return buf;
}
} // namespace

metric_explorer::metric_explorer(hdf5::executor& h5_exec)
    : h5_exec_(h5_exec)
    , import_browser_(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter)
{
    import_browser_.SetTitle("Import Datasets from Config");
    import_browser_.SetTypeFilters({ ".json" });
}

metric_explorer::~metric_explorer()
{
    free_resources();
}

void metric_explorer::free_axis_thumbs(axis& ax)
{
    for (auto& t : ax.thumbs) t.release();
    ax.thumbs.clear();
}

void metric_explorer::free_resources()
{
    free_axis_thumbs(rows_);
    free_axis_thumbs(cols_);
    picker_grid_.free_resources();
}

// ---------------------------------------------------------------------------
// Axis resolution
// ---------------------------------------------------------------------------

int metric_explorer::resolve_axis_flat(const axis& ax) const
{
    if (ax.source_id == 0 || ax.local_idx < 0) return -1;
    for (int i = 0; i < static_cast<int>(datasets_.size()); ++i)
        if (datasets_[i].source_id == ax.source_id && datasets_[i].local_idx == ax.local_idx)
            return i;
    return -1;
}

void metric_explorer::clear_axis(axis& ax)
{
    free_axis_thumbs(ax);
    ax.selected.clear();
    ax.source_id = 0;
    ax.local_idx = -1;
}

// ---------------------------------------------------------------------------
// Source management
// ---------------------------------------------------------------------------

void metric_explorer::reset_axes()
{
    clear_axis(rows_);
    clear_axis(cols_);
    picker_axis_ = -1;
    picker_grid_.free_resources();
    row_data_.clear();
    col_data_.clear();
    values_.clear();
    mat_rows_ = mat_cols_ = 0;
    computed_ = false;
    message_.clear();

    sort_focus_ = sort_focus::none;
    sort_index_ = -1;
    row_order_.clear();
    col_order_.clear();
    hist_counts_.clear();
}

void metric_explorer::rebuild_dataset_list()
{
    datasets_.clear();
    source_meta_.clear();
    total_bytes_ = 0;

    for (auto& src : sources_) {
        auto handles = src->datasets();
        const int count = static_cast<int>(handles.size());

        // Footprint from the handles already in hand — a second src->estimated_bytes()
        // would re-run enumerate_datasets (a full entity scan) for imported_source.
        std::size_t bytes = 0;
        for (const auto& h : handles)
            bytes += h.sample_count * h.feature_dim() * sizeof(float);

        for (int li = 0; li < count; ++li)
            datasets_.push_back({ std::move(handles[static_cast<std::size_t>(li)]), src->id, li });

        total_bytes_ += bytes;
        source_meta_.push_back({ src->id, src->label(), bytes, count });
    }

    // Re-resolve each axis against the new flat list. An axis whose source vanished
    // is cleared (and invalidates the matrix); an unset axis defaults to the first
    // dataset (with an empty selection) for ergonomics.
    auto resolve_or_default = [this](axis& ax) {
        if (ax.source_id != 0) {
            if (resolve_axis_flat(ax) >= 0) return;   // still present -> keep selection
            clear_axis(ax);                            // its source is gone
            computed_ = false;
        }
        if (!datasets_.empty()) {
            ax.source_id = datasets_[0].source_id;
            ax.local_idx = datasets_[0].local_idx;     // selection stays empty
        }
    };
    resolve_or_default(rows_);
    resolve_or_default(cols_);
}

void metric_explorer::import_config(const std::string& path)
{
    // Block re-importing the same canonical path.
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(path, ec);
    const std::string cpath = ec ? path : canon.string();
    for (auto& src : sources_) {
        if (auto* imp = dynamic_cast<imported_source*>(src.get())) {
            if (imp->canonical_path() == cpath) {
                import_error_ = "Already imported: " + path;
                return;
            }
        }
    }

    try {
        auto src = std::make_unique<imported_source>(path, h5_exec_);
        src->id = next_source_id_++;
        sources_.push_back(std::move(src));
        import_error_.clear();
    } catch (const std::exception& e) {
        import_error_ = e.what();
        return;
    }
    rebuild_dataset_list();
}

bool metric_explorer::generate_codes()
{
    ensemble_codes codes;
    try {
        codes = generate_ensemble_codes(gen_params_);
    } catch (const std::exception& e) {
        gen_error_ = e.what();
        return false;
    }

    char lbl[96];
    std::snprintf(lbl, sizeof(lbl), "synth #%zu (M=%d, D=%d, k=%d)",
                  next_source_id_, gen_params_.M, gen_params_.D, gen_params_.k);

    auto src = std::make_unique<synthetic_source>(lbl, std::move(codes), &show_synth_2d_);
    src->id = next_source_id_++;
    sources_.push_back(std::move(src));
    gen_error_.clear();
    rebuild_dataset_list();
    return true;
}

void metric_explorer::remove_source(std::size_t idx)
{
    if (idx >= sources_.size()) return;
    // rebuild_dataset_list re-resolves axes: an axis pointing at the removed source
    // is cleared, axes pointing at surviving sources keep their selection.
    sources_.erase(sources_.begin() + static_cast<std::ptrdiff_t>(idx));
    rebuild_dataset_list();
}

void metric_explorer::clear_sources()
{
    sources_.clear();
    reset_axes();
    rebuild_dataset_list();
}

// ---------------------------------------------------------------------------
// Matrix compute
// ---------------------------------------------------------------------------

void metric_explorer::make_thumbs(std::vector<gl_texture>& out,
                                  const std::vector<std::vector<float>>& bufs,
                                  int w, int h, int c)
{
    for (auto& t : out) t.release();
    out.clear();
    out.reserve(bufs.size());

    std::vector<color_t> pixels;
    for (const auto& buf : bufs) {
        convert_sample_to_rgba(std::span<const float>(buf.data(), buf.size()),
                               w, h, c, HEADER_THUMB, pixels);
        out.emplace_back(HEADER_THUMB, HEADER_THUMB, pixels.data());
    }
}

void metric_explorer::random_fill(axis& ax)
{
    const int flat = resolve_axis_flat(ax);
    if (flat < 0) return;
    const dataset_handle& ds = datasets_[flat].handle;
    if (ds.sample_count == 0) return;

    const std::size_t n = std::min(static_cast<std::size_t>(std::max(0, random_n_)),
                                   ds.sample_count);
    ax.selected.clear();

    // Partial Fisher-Yates over [0, sample_count): exactly n distinct indices even at
    // n == sample_count (rejection sampling could under-fill there). Matches the
    // distinct-pick idiom in apply_dropout / the ensemble prototype builder.
    std::vector<std::size_t> idx(ds.sample_count);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937 rng(static_cast<std::uint32_t>(random_seed_));
    for (std::size_t i = 0; i < n; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, ds.sample_count - 1);
        std::swap(idx[i], idx[dist(rng)]);
        ax.selected.insert(idx[i]);
    }
}

void metric_explorer::compute_values()
{
    if (row_data_.empty() || col_data_.empty()) { computed_ = false; return; }

    mat_rows_ = static_cast<int>(row_data_.size());
    mat_cols_ = static_cast<int>(col_data_.size());
    values_.assign(static_cast<std::size_t>(mat_rows_) * mat_cols_, 0.0f);

    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < mat_rows_; ++i) {
        for (int j = 0; j < mat_cols_; ++j) {
            const float v = compute_metric(metric_, params_,
                                           std::span<const float>(row_data_[i]),
                                           std::span<const float>(col_data_[j]));
            values_[static_cast<std::size_t>(i) * mat_cols_ + j] = v;
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
    if (!std::isfinite(vmin) || !std::isfinite(vmax)) { vmin = 0.0f; vmax = 1.0f; }
    vmin_ = vmin;
    vmax_ = vmax;
    computed_ = true;

    // A focus set on a prior matrix of the same dims survives a metric/p change;
    // re-derive the display order and re-bin against the new values.
    apply_sort();
    refresh_histogram();
}

void metric_explorer::compute()
{
    message_.clear();
    computed_ = false;
    row_data_.clear();
    col_data_.clear();

    // A full rebuild may change the matrix dimensions, so any prior header focus is
    // no longer meaningful — drop it (compute_values, which keeps focus, handles the
    // metric/p case). apply_sort/refresh_histogram run at the tail of compute_values.
    sort_focus_ = sort_focus::none;
    sort_index_ = -1;

    const int rflat = resolve_axis_flat(rows_);
    if (rflat < 0) { message_ = "Select a dataset for the rows axis."; return; }
    const dataset_handle& rd = datasets_[rflat].handle;
    if (rows_.selected.empty()) {
        message_ = "Pick at least one row sample (use \"Pick rows…\").";
        return;
    }

    // Copy selected row samples into owned buffers (get_sample may return a span
    // into shared scratch, so we must not hold two live spans across calls).
    const std::vector<std::size_t> R(rows_.selected.begin(), rows_.selected.end());
    row_data_.resize(R.size());
    for (std::size_t i = 0; i < R.size(); ++i) {
        auto s = rd.get_sample(R[i]);
        row_data_[i].assign(s.begin(), s.end());
    }

    int col_w = 0, col_h = 0, col_c = 0;
    if (distort_mode_) {
        // Columns are distorted copies of the rows: R×R, geometry mirrors the rows.
        col_data_.resize(R.size());
        for (std::size_t j = 0; j < R.size(); ++j)
            apply_distortion(distort_kind_, distort_params_,
                             std::span<const float>(row_data_[j].data(), row_data_[j].size()),
                             col_data_[j]);
        col_w = rd.width; col_h = rd.height; col_c = rd.channels;
    } else {
        const int cflat = resolve_axis_flat(cols_);
        if (cflat < 0) { message_ = "Select a dataset for the columns axis."; return; }
        const dataset_handle& cd = datasets_[cflat].handle;
        if (cols_.selected.empty()) {
            message_ = "Pick at least one column sample (use \"Pick cols…\").";
            return;
        }
        if (rd.feature_dim() != cd.feature_dim()) {
            message_ = "Incompatible datasets: feature dims differ ("
                     + std::to_string(rd.feature_dim()) + " vs "
                     + std::to_string(cd.feature_dim()) + "). Pick matching w*h*c.";
            return;
        }
        const std::vector<std::size_t> C(cols_.selected.begin(), cols_.selected.end());
        col_data_.resize(C.size());
        for (std::size_t j = 0; j < C.size(); ++j) {
            auto s = cd.get_sample(C[j]);
            col_data_[j].assign(s.begin(), s.end());
        }
        col_w = cd.width; col_h = cd.height; col_c = cd.channels;
    }

    // Defensive: every compared sample must share one length.
    const std::size_t fdim = row_data_.empty() ? 0 : row_data_[0].size();
    for (const auto& v : row_data_) if (v.size() != fdim) { message_ = "Sample size mismatch."; return; }
    for (const auto& v : col_data_) if (v.size() != fdim) { message_ = "Sample size mismatch."; return; }

    // Header thumbnails reflect the actually-compared vectors (incl. distortion).
    make_thumbs(rows_.thumbs, row_data_, rd.width, rd.height, rd.channels);
    make_thumbs(cols_.thumbs, col_data_, col_w, col_h, col_c);

    compute_values();
    canvas_.reset_view();
}

// ---------------------------------------------------------------------------
// Sort + histogram
// ---------------------------------------------------------------------------

void metric_explorer::apply_sort()
{
    // Identity display order by default; the focused axis stays identity while the
    // OTHER axis is permuted (single-focus invariant).
    row_order_.resize(static_cast<std::size_t>(std::max(0, mat_rows_)));
    col_order_.resize(static_cast<std::size_t>(std::max(0, mat_cols_)));
    std::iota(row_order_.begin(), row_order_.end(), 0);
    std::iota(col_order_.begin(), col_order_.end(), 0);

    if (sort_focus_ == sort_focus::col && sort_index_ >= 0 && sort_index_ < mat_cols_) {
        // Clicked a column header -> order ROWS by their value in that column.
        std::vector<float> keys(static_cast<std::size_t>(mat_rows_));
        for (int i = 0; i < mat_rows_; ++i)
            keys[static_cast<std::size_t>(i)] =
                values_[static_cast<std::size_t>(i) * mat_cols_ + sort_index_];
        row_order_ = order_by_key(std::span<const float>(keys.data(), keys.size()), sort_desc_);
    } else if (sort_focus_ == sort_focus::row && sort_index_ >= 0 && sort_index_ < mat_rows_) {
        // Clicked a row header -> order COLUMNS by their value in that row.
        std::vector<float> keys(static_cast<std::size_t>(mat_cols_));
        const std::size_t base = static_cast<std::size_t>(sort_index_) * mat_cols_;
        for (int j = 0; j < mat_cols_; ++j)
            keys[static_cast<std::size_t>(j)] = values_[base + static_cast<std::size_t>(j)];
        col_order_ = order_by_key(std::span<const float>(keys.data(), keys.size()), sort_desc_);
    }
}

void metric_explorer::refresh_histogram()
{
    hist_counts_.clear();
    if (!computed_ || values_.empty()) return;

    // X-domain follows the metric's theoretical range (cosine fixed [-1,1];
    // Minkowski [0, full-matrix max]) so it never jumps when a slice is selected.
    const value_range dom = metric_domain(metric_, vmax_);
    hist_lo_ = dom.lo;
    hist_hi_ = dom.hi;

    // Active slice: full matrix, the focused column, or the focused row.
    std::vector<float> slice;
    if (sort_focus_ == sort_focus::col && sort_index_ >= 0 && sort_index_ < mat_cols_) {
        slice.reserve(static_cast<std::size_t>(mat_rows_));
        for (int i = 0; i < mat_rows_; ++i)
            slice.push_back(values_[static_cast<std::size_t>(i) * mat_cols_ + sort_index_]);
    } else if (sort_focus_ == sort_focus::row && sort_index_ >= 0 && sort_index_ < mat_rows_) {
        const std::size_t base = static_cast<std::size_t>(sort_index_) * mat_cols_;
        slice.assign(values_.begin() + static_cast<std::ptrdiff_t>(base),
                     values_.begin() + static_cast<std::ptrdiff_t>(base + mat_cols_));
    } else {
        slice = values_;
    }

    hist_counts_ = histogram_bins(std::span<const float>(slice.data(), slice.size()),
                                  hist_nbins_, hist_lo_, hist_hi_);
}

void metric_explorer::on_header_click(int disp_row, int disp_col)
{
    // Map the clicked DISPLAY index to the UNDERLYING index via the CURRENT order
    // arrays — must happen before apply_sort() rebuilds them.
    const bool cosine = (metric_ == metric_kind::cosine_similarity);
    if (disp_col >= 0) {
        const int u = (disp_col < static_cast<int>(col_order_.size())) ? col_order_[disp_col] : disp_col;
        if (sort_focus_ == sort_focus::col && sort_index_ == u) {
            sort_desc_ = !sort_desc_;                 // re-click same column -> reverse
        } else {
            sort_focus_ = sort_focus::col;
            sort_index_ = u;
            sort_desc_  = cosine;                     // most-similar-first: cosine desc, Minkowski asc
        }
    } else if (disp_row >= 0) {
        const int u = (disp_row < static_cast<int>(row_order_.size())) ? row_order_[disp_row] : disp_row;
        if (sort_focus_ == sort_focus::row && sort_index_ == u) {
            sort_desc_ = !sort_desc_;
        } else {
            sort_focus_ = sort_focus::row;
            sort_index_ = u;
            sort_desc_  = cosine;
        }
    } else {
        return;
    }
    apply_sort();
    refresh_histogram();
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

void metric_explorer::open_picker(int which)
{
    // Defer the OpenPopup to window scope (handled in render()): the Pick buttons now
    // live inside the setup-strip table cell, and BeginTable pushes the table ID onto
    // the ID stack. OpenPopup here would derive a popup ID under that table scope,
    // while render_picker_popup()'s BeginPopupModal runs at window scope — the IDs
    // would not match and the popup would never open. (Mirrors open_gen_dialog_.)
    picker_axis_ = which;
    picker_grid_.free_resources();
    open_picker_pending_ = true;
}

void metric_explorer::render_sources_panel()
{
    char hdr[80];
    std::snprintf(hdr, sizeof(hdr), "Datasets — %zu source(s)###sources", sources_.size());
    if (ImGui::CollapsingHeader(hdr, ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Import datasets from config…")) {
            import_error_.clear();
            import_browser_.Open();
        }
        ImGui::SameLine();
        if (ImGui::Button("Generate ensemble codes…")) {
            gen_error_.clear();
            open_gen_dialog_ = true;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(sources_.empty());
        if (ImGui::Button("Clear all")) clear_sources();
        ImGui::EndDisabled();

        if (!import_error_.empty()) {
            ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f), "Import failed: %s",
                               import_error_.c_str());
        }

        for (std::size_t i = 0; i < sources_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::SmallButton("Remove")) {
                ImGui::PopID();
                remove_source(i);
                break;   // sources_ mutated; stop iterating this frame
            }
            ImGui::SameLine();
            if (i < source_meta_.size()) {
                const auto& m = source_meta_[i];
                ImGui::Text("%s  [%s]", m.label.c_str(), format_bytes(m.bytes).c_str());
                if (m.dataset_count == 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "(no CPU datasets)");
                }
            } else {
                ImGui::TextUnformatted(sources_[i]->label().c_str());
            }
            ImGui::PopID();
        }

        if (!sources_.empty()) {
            const bool over = total_bytes_ > MEM_WARN;
            ImGui::TextColored(over ? ImVec4(0.95f, 0.4f, 0.3f, 1.0f)
                                    : ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                               "Total ~%s (estimate)%s",
                               format_bytes(total_bytes_).c_str(),
                               over ? "  — heavy; watch memory" : "");
        }
    }
}

void metric_explorer::render_axis_combo(const char* label, axis& ax)
{
    const int flat = resolve_axis_flat(ax);
    const char* preview = (flat >= 0) ? datasets_[flat].handle.name.c_str() : "Select...";
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < static_cast<int>(datasets_.size()); ++i) {
            const bool sel = (i == flat);
            if (ImGui::Selectable(datasets_[i].handle.name.c_str(), sel) && i != flat) {
                ax.source_id = datasets_[i].source_id;
                ax.local_idx = datasets_[i].local_idx;
                ax.selected.clear();
                free_axis_thumbs(ax);
                computed_ = false;
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

void metric_explorer::render_setup_strip()
{
    // --- Rows axis ---
    render_axis_combo("Rows dataset", rows_);
    ImGui::SameLine();
    {
        char btn[64];
        std::snprintf(btn, sizeof(btn), "Pick rows… (%zu)##pickrows", rows_.selected.size());
        if (ImGui::Button(btn)) open_picker(0);
    }

    // --- Distortion toggle ---
    ImGui::Checkbox("Distortion mode (cols = distorted rows)", &distort_mode_);

    if (distort_mode_) {
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Distortion", distortion_kind_name(distort_kind_))) {
            for (auto k : {distortion_kind::gaussian_noise, distortion_kind::dropout}) {
                const bool sel = (distort_kind_ == k);
                if (ImGui::Selectable(distortion_kind_name(k), sel)) distort_kind_ = k;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        const char* mag_label = (distort_kind_ == distortion_kind::dropout)
                              ? "drop p" : "sigma";
        ImGui::SliderFloat(mag_label, &distort_params_.magnitude, 0.0f, 1.0f, "%.3f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        int seed_i = static_cast<int>(distort_params_.seed);
        if (ImGui::InputInt("d-seed", &seed_i)) distort_params_.seed = static_cast<std::uint32_t>(std::max(0, seed_i));
    } else {
        // --- Columns axis (only when not in distortion mode) ---
        render_axis_combo("Cols dataset", cols_);
        ImGui::SameLine();
        char btn[64];
        std::snprintf(btn, sizeof(btn), "Pick cols… (%zu)##pickcols", cols_.selected.size());
        if (ImGui::Button(btn)) open_picker(1);
    }

    // --- Metric + p ---
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Metric", metric_kind_name(metric_))) {
        for (auto k : {metric_kind::cosine_similarity, metric_kind::minkowski_distance}) {
            const bool sel = (metric_ == k);
            if (ImGui::Selectable(metric_kind_name(k), sel) && metric_ != k) {
                metric_ = k;
                if (computed_) compute_values();
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (metric_ == metric_kind::minkowski_distance) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        if (ImGui::SliderFloat("p", &params_.p, 0.1f, 6.0f, "%.2f")) {
            if (computed_) compute_values();
        }
    }

    // --- Palette (no recompute: applied live at render time) ---
    ImGui::SetNextItemWidth(140);
    if (ImGui::BeginCombo("Palette", colormap_name(palette_))) {
        for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
            const auto cm = static_cast<colormap_t>(i);
            const bool sel = (palette_ == cm);
            if (ImGui::Selectable(colormap_name(cm), sel)) palette_ = cm;
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // --- Synthetic 1D/2D display toggle ---
    ImGui::SameLine();
    if (ImGui::Checkbox("Synthetic 2D", &show_synth_2d_)) {
        rebuild_dataset_list();      // synthetic handle geometry changes
        if (computed_) compute();    // rebuild thumbnails (matrix values unchanged)
    }

    ImGui::SameLine();
    if (ImGui::Button("Compute")) compute();

    // --- Status / compatibility hint ---
    if (distort_mode_) {
        const int rflat = resolve_axis_flat(rows_);
        if (rflat >= 0) {
            const auto& rd = datasets_[rflat].handle;
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f),
                "rows %dx%dx%d (D=%zu) → columns are %s copies (R×R)",
                rd.width, rd.height, rd.channels, rd.feature_dim(),
                distortion_kind_name(distort_kind_));
        }
    } else {
        const int rflat = resolve_axis_flat(rows_);
        const int cflat = resolve_axis_flat(cols_);
        if (rflat >= 0 && cflat >= 0) {
            const auto& rd = datasets_[rflat].handle;
            const auto& cd = datasets_[cflat].handle;
            const bool ok = rd.feature_dim() == cd.feature_dim();
            ImGui::TextColored(ok ? ImVec4(0.6f, 0.8f, 0.6f, 1.0f) : ImVec4(0.9f, 0.5f, 0.3f, 1.0f),
                "rows %dx%dx%d (D=%zu)  |  cols %dx%dx%d (D=%zu)  |  %s",
                rd.width, rd.height, rd.channels, rd.feature_dim(),
                cd.width, cd.height, cd.channels, cd.feature_dim(),
                ok ? "compatible" : "INCOMPATIBLE");
        }
    }
    if (!message_.empty()) {
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f), "%s", message_.c_str());
    }
}

void metric_explorer::render_histogram_panel()
{
    ImGui::TextUnformatted("Value distribution");

    if (!computed_ || hist_counts_.empty()) {
        ImGui::TextDisabled("Compute a matrix to see\nits value histogram.");
        return;
    }

    ImGui::SetNextItemWidth(160);
    if (ImGui::SliderInt("bins", &hist_nbins_, 10, 100)) {
        hist_nbins_ = std::clamp(hist_nbins_, 10, 100);
        refresh_histogram();
    }

    // Reserve a bar area and draw palette-matched bars over the metric domain.
    const float avail_w = std::max(ImGui::GetContentRegionAvail().x, 120.0f);
    constexpr float bar_h = 110.0f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(avail_w, bar_h));

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 a0(origin.x, origin.y);
    const ImVec2 a1(origin.x + avail_w, origin.y + bar_h);
    dl->AddRectFilled(a0, a1, IM_COL32(20, 20, 20, 255));

    const int nb = static_cast<int>(hist_counts_.size());
    int maxc = 1;
    for (const int c : hist_counts_) maxc = std::max(maxc, c);

    const float bw     = avail_w / static_cast<float>(nb);
    const float vrange = (vmax_ > vmin_) ? (vmax_ - vmin_) : 1.0f;
    const float drange = hist_hi_ - hist_lo_;
    for (int i = 0; i < nb; ++i) {
        const float frac = static_cast<float>(hist_counts_[i]) / static_cast<float>(maxc);
        const float bh   = frac * (bar_h - 2.0f);
        const float x0   = origin.x + static_cast<float>(i) * bw;
        const float x1   = origin.x + static_cast<float>(i + 1) * bw - 1.0f;
        // Bin-center value -> matrix palette over the OBSERVED range, so bars share
        // the matrix's colors (out-of-observed bins clamp to the palette ends).
        const float center = hist_lo_ + (static_cast<float>(i) + 0.5f) / static_cast<float>(nb) * drange;
        const float t      = std::clamp((center - vmin_) / vrange, 0.0f, 1.0f);
        const color_t col  = apply_colormap(palette_, t);
        dl->AddRectFilled(ImVec2(x0, a1.y - bh), ImVec2(x1, a1.y),
                          IM_COL32(col.r, col.g, col.b, 255));
    }
    dl->AddRect(a0, a1, IM_COL32(255, 255, 255, 80));

    // Domain + slice labels.
    ImGui::Text("x: [%.3g, %.3g]  %s", hist_lo_, hist_hi_,
                metric_ == metric_kind::cosine_similarity ? "cosine" : "Minkowski");
    if (sort_focus_ == sort_focus::col)
        ImGui::TextDisabled("Column #%d  (%d rows)", sort_index_, mat_rows_);
    else if (sort_focus_ == sort_focus::row)
        ImGui::TextDisabled("Row #%d  (%d cols)", sort_index_, mat_cols_);
    else
        ImGui::TextDisabled("Full matrix (%d×%d)", mat_rows_, mat_cols_);

    ImGui::BeginDisabled(sort_focus_ == sort_focus::none);
    if (ImGui::Button("Clear selection")) {
        sort_focus_ = sort_focus::none;
        sort_index_ = -1;
        apply_sort();
        refresh_histogram();
    }
    ImGui::EndDisabled();
}

void metric_explorer::render_generator_dialog()
{
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_Appearing);

    if (!ImGui::BeginPopupModal("Generate ensemble codes", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) return;

    ImGui::TextDisabled("Synthetic neuron-group population codes.");
    ImGui::SetNextItemWidth(140);
    if (ImGui::InputInt("D (neurons)", &gen_params_.D))   gen_params_.D = std::max(1, gen_params_.D);
    ImGui::SetNextItemWidth(140);
    if (ImGui::InputInt("N (codes)", &gen_params_.N))     gen_params_.N = std::max(0, gen_params_.N);
    ImGui::SetNextItemWidth(140);
    if (ImGui::InputInt("M (prototypes)", &gen_params_.M)) gen_params_.M = std::max(1, gen_params_.M);
    ImGui::SetNextItemWidth(140);
    if (ImGui::InputInt("k (winners)", &gen_params_.k))   gen_params_.k = std::max(0, gen_params_.k);

    ImGui::SliderFloat("swap_frac",   &gen_params_.swap_frac,  0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("amp_var",     &gen_params_.amp_var,    0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("inhibition",  &gen_params_.inhibition, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("noise",       &gen_params_.noise,      0.0f, 0.5f, "%.3f");
    ImGui::SliderFloat("recognition", &gen_params_.recognition,0.0f, 1.0f, "%.2f");
    ImGui::SetNextItemWidth(140);
    int seed_i = static_cast<int>(gen_params_.seed);
    if (ImGui::InputInt("seed", &seed_i)) gen_params_.seed = static_cast<std::uint32_t>(std::max(0, seed_i));

    if (!gen_error_.empty())
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f), "%s", gen_error_.c_str());

    ImGui::Separator();
    if (ImGui::Button("Generate")) {
        if (generate_codes()) ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) { gen_error_.clear(); ImGui::CloseCurrentPopup(); }

    ImGui::EndPopup();
}

void metric_explorer::render_picker_popup()
{
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(560, 620), ImGuiCond_FirstUseEver);

    if (!ImGui::BeginPopupModal("Sample Picker", nullptr)) return;

    if (picker_axis_ < 0) { ImGui::CloseCurrentPopup(); ImGui::EndPopup(); return; }
    axis& ax = axis_ref(picker_axis_);

    const int flat = resolve_axis_flat(ax);
    if (flat < 0) {
        ImGui::TextUnformatted("Select a dataset for this axis first.");
        if (ImGui::Button("Close")) { ImGui::CloseCurrentPopup(); picker_axis_ = -1; }
        ImGui::EndPopup();
        return;
    }

    const dataset_handle& ds = datasets_[flat].handle;
    ImGui::Text("%s — %zu of %zu selected", ds.name.c_str(), ax.selected.size(), ds.sample_count);

    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("N", &random_n_);
    if (random_n_ < 0) random_n_ = 0;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    ImGui::InputInt("seed", &random_seed_);
    ImGui::SameLine();
    if (ImGui::Button("Random fill")) random_fill(ax);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) ax.selected.clear();
    ImGui::SameLine();
    if (ImGui::Button("Done")) { ImGui::CloseCurrentPopup(); picker_axis_ = -1; picker_grid_.free_resources(); }

    ImGui::Separator();
    ImGui::TextDisabled("Click thumbnails to toggle. Selected = highlighted.");

    picker_grid_.render(ds, &ax.selected);

    ImGui::EndPopup();
}

void metric_explorer::render()
{
    if (!is_visible) return;

    ImGui::SetNextWindowSize(ImVec2(920, 700), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(140, 120), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Metric Explorer", &is_visible)) {
        render_sources_panel();

        // Generator dialog is reachable even with no datasets yet (you generate to
        // get some), so trigger/render it before the empty-state branch.
        if (open_gen_dialog_) { ImGui::OpenPopup("Generate ensemble codes"); open_gen_dialog_ = false; }
        render_generator_dialog();

        if (datasets_.empty()) {
            if (sources_.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                    "No datasets yet. Import a config or generate ensemble codes above.");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                    "Imported sources contain no CPU datasets (see notes above).");
            }
        } else {
            ImGui::Separator();

            // Setup strip (left, stretch) + value histogram (right, fixed) band. The
            // matrix renders full-width below the band.
            if (ImGui::BeginTable("setup_band", 2, ImGuiTableFlags_BordersInnerV)) {
                ImGui::TableSetupColumn("setup", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("hist",  ImGuiTableColumnFlags_WidthFixed, 300.0f);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                render_setup_strip();
                ImGui::TableSetColumnIndex(1);
                render_histogram_panel();
                ImGui::EndTable();
            }
            ImGui::Separator();

            if (computed_ && !values_.empty()) {
                matrix_canvas::inputs in;
                in.rows    = mat_rows_;
                in.cols    = mat_cols_;
                in.values  = values_.data();
                in.vmin    = vmin_;
                in.vmax    = vmax_;
                in.palette = palette_;
                in.row_thumbs = std::span<const gl_texture>(rows_.thumbs.data(), rows_.thumbs.size());
                in.col_thumbs = std::span<const gl_texture>(cols_.thumbs.data(), cols_.thumbs.size());
                in.row_order  = row_order_.empty() ? nullptr : row_order_.data();
                in.col_order  = col_order_.empty() ? nullptr : col_order_.data();
                // The focused axis is never the one being reordered, so its display
                // index equals its underlying index (sort_index_).
                if (sort_focus_ == sort_focus::col) {
                    in.highlight_col = sort_index_;
                    in.sort_dir = sort_desc_ ? -1 : 1;
                } else if (sort_focus_ == sort_focus::row) {
                    in.highlight_row = sort_index_;
                    in.sort_dir = sort_desc_ ? -1 : 1;
                }

                const matrix_canvas::interaction act = canvas_.render(in);
                if (act.clicked_col >= 0)      on_header_click(-1, act.clicked_col);
                else if (act.clicked_row >= 0) on_header_click(act.clicked_row, -1);
            } else {
                ImGui::TextDisabled("Pick samples, then press Compute.");
            }

            // Open the picker at window scope (the Pick buttons that request it sit
            // inside the setup-strip table cell — see open_picker).
            if (open_picker_pending_) { ImGui::OpenPopup("Sample Picker"); open_picker_pending_ = false; }
            render_picker_popup();
        }
    }
    ImGui::End();

    // The file browser is its own window; pump it and harvest a chosen config.
    import_browser_.Display();
    if (import_browser_.HasSelected()) {
        import_config(import_browser_.GetSelected().string());
        import_browser_.ClearSelected();
    }
}

} // namespace coast
