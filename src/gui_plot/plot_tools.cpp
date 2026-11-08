#include "gui_plot/plot_tools.h"

#include "diag/diag.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace coast::plot {

// ------------------------------------------------------------------ gridding

grid_result bin_points(std::span<const double> x, std::span<const double> y, int bins_x, int bins_y,
                       std::optional<range> x_domain, std::optional<range> y_domain)
{
    const int m = static_cast<int>(std::min(x.size(), y.size()));

    // A fixed domain replaces the data-derived frame; that is the whole point of
    // passing one. An invalid ({0,0} or inverted) domain falls back to auto so a
    // caller cannot silently collapse the frame to zero width.
    const bool fixed_x = x_domain && x_domain->valid();
    const bool fixed_y = y_domain && y_domain->valid();

    double x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    if (fixed_x) {
        x0 = x_domain->lo;
        x1 = x_domain->hi;
    } else {
        auto [xmn, xmx] = std::minmax_element(x.begin(), x.begin() + m);
        x0              = *xmn;
        x1              = *xmx;
        if (x1 <= x0) x1 = x0 + 1.0; // guard a constant column (0-width bin)
    }
    if (fixed_y) {
        y0 = y_domain->lo;
        y1 = y_domain->hi;
    } else {
        auto [ymn, ymx] = std::minmax_element(y.begin(), y.begin() + m);
        y0              = *ymn;
        y1              = *ymx;
        if (y1 <= y0) y1 = y0 + 1.0;
    }

    // Sturges' rule for the auto (<= 0) bin counts, matching ImPlot.
    auto sturges = [](int n) {
        return n <= 1 ? 1 : static_cast<int>(std::ceil(std::log2((double)n))) + 1;
    };
    const int    nbx  = bins_x > 0 ? bins_x : sturges(m);
    const int    nby  = bins_y > 0 ? bins_y : sturges(m);
    const double wbin = (x1 - x0) / nbx;
    const double hbin = (y1 - y0) / nby;

    // Which side of a fixed frame a coordinate fell on. An auto-ranged axis is
    // `inside` by construction — the frame came from the data's own min/max.
    auto zone_of = [](double v, double lo, double hi, bool fixed) {
        if (!fixed) return zone::inside;
        if (v < lo) return zone::under;
        if (v > hi) return zone::over;
        return zone::inside;
    };

    // Row-major grid, row 0 = TOP (max y). PlotHeatmap draws row 0 at
    // bounds_max.y (reverse_y=true), so flip the natural bottom-up bin index.
    std::vector<double> grid(static_cast<std::size_t>(nbx) * nby, 0.0);
    double              max_count = 0.0;
    outer_ring          outer;
    for (int i = 0; i < m; ++i) {
        const zone zx = zone_of(x[i], x0, x1, fixed_x);
        const zone zy = zone_of(y[i], y0, y1, fixed_y);
        if (zx != zone::inside || zy != zone::inside) {
            // Outside the frame on at least one axis: exactly one ring cell, so
            // a corner point is counted once rather than twice.
            outer.at(zy, zx) += 1.0;
            continue;
        }
        // Inside: clamp still matters — a point exactly at the closed upper edge
        // indexes to nbx/nby, and fp division can land a hair past it.
        const int xb  = std::clamp(static_cast<int>((x[i] - x0) / wbin), 0, nbx - 1);
        const int ybn = std::clamp(static_cast<int>((y[i] - y0) / hbin), 0, nby - 1);
        double&   c   = grid[static_cast<std::size_t>(nby - 1 - ybn) * nbx + xb];
        c += 1.0;
        if (c > max_count) max_count = c;
    }
    return {std::move(grid), nbx, nby, x0, x1, y0, y1, /*min_val=*/0.0, max_count, outer};
}

grid_result reshape_field(std::span<const double> values, int width, int height)
{
    const int         w    = std::max(width, 1);  // columns
    const int         h    = std::max(height, 1); // rows
    const std::size_t need = static_cast<std::size_t>(w) * h;
    grid_result       gr;
    gr.nbx = w;
    gr.nby = h;
    gr.x0  = 0.0;
    gr.x1  = static_cast<double>(w);
    gr.y0  = 0.0;
    gr.y1  = static_cast<double>(h);
    gr.grid.assign(need, 0.0);
    const std::size_t n = std::min(need, values.size());
    double            mn = 0.0, mx = 0.0;
    if (n > 0) mn = mx = values[0];
    for (std::size_t i = 0; i < n; ++i) {
        gr.grid[i] = values[i];
        mn         = std::min(mn, values[i]);
        mx         = std::max(mx, values[i]);
    }
    gr.min_val   = mn;
    gr.max_count = mx;
    return gr;
}

// ------------------------------------------------------------------ heatmaps

// Map the framework palette enum to ImPlot's built-in colormap index. colormap_t
// mirrors ImPlot's *continuous* block (Viridis..Greys) in order, so the mapping
// is a fixed offset from ImPlotColormap_Viridis. Anything out of range falls back
// to Viridis.
static ImPlotColormap to_implot(colormap_t cm)
{
    // colormap_t's members are 1:1 with ImPlot's contiguous continuous block.
    // Endpoints pin the correspondence; if either enum is reordered this trips.
    static_assert(static_cast<int>(colormap_t::viridis) == 0);
    static_assert(static_cast<int>(colormap_t::greys) + 1 == static_cast<int>(colormap_t::COUNT));
    static_assert(ImPlotColormap_Greys - ImPlotColormap_Viridis + 1 == static_cast<int>(colormap_t::COUNT));
    const int i = static_cast<int>(cm);
    if (i < 0 || i >= static_cast<int>(colormap_t::COUNT)) return ImPlotColormap_Viridis;
    return ImPlotColormap_Viridis + i;
}

void draw_colormap_grid(const grid_result& gr, double scale_min, double scale_max, ImVec2 region,
                        colormap_t colormap, bool equal_axes, const char* scale_label,
                        bool show_x_axis, bool show_y_axis, bool show_z_axis)
{
    ImPlot::PushColormap(to_implot(colormap));
    // The colorbar (z axis) reserves a font-relative gutter on the right (font-
    // relative so it stays sufficient when the export pass scales text up); when
    // hidden it is not drawn and the heatmap reclaims the full width. Subtract
    // ItemSpacing: SameLine() inserts it between plot and scale, and without
    // accounting for it the scale overflows the region's right edge.
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float scale_w = show_z_axis ? ImGui::GetFontSize() * 7.0f : 0.0f;
    const float plot_w =
        show_z_axis ? std::max(region.x - scale_w - spacing, 1.0f) : region.x;
    // Hiding an axis strips its line, ticks, labels and gridlines but keeps
    // AutoFit so the data still frames correctly.
    const ImPlotAxisFlags xflags =
        ImPlotAxisFlags_AutoFit | (show_x_axis ? 0 : ImPlotAxisFlags_NoDecorations);
    const ImPlotAxisFlags yflags =
        ImPlotAxisFlags_AutoFit | (show_y_axis ? 0 : ImPlotAxisFlags_NoDecorations);
    // Equal keeps 1 plot-unit-x == 1 plot-unit-y in pixels (square image cells).
    const ImPlotFlags pflags = equal_axes ? ImPlotFlags_Equal : ImPlotFlags_None;
    if (ImPlot::BeginPlot("##hmplot", ImVec2(plot_w, region.y), pflags)) {
        ImPlot::SetupAxes(nullptr, nullptr, xflags, yflags);
        ImPlot::PlotHeatmap("##hm", gr.grid.data(), gr.nby, gr.nbx, scale_min, scale_max, nullptr,
                            ImPlotPoint(gr.x0, gr.y0), ImPlotPoint(gr.x1, gr.y1));
        ImPlot::EndPlot();
    }
    if (show_z_axis) {
        ImGui::SameLine();
        ImPlot::ColormapScale(scale_label, scale_min, scale_max, ImVec2(scale_w, region.y));
    }
    ImPlot::PopColormap();
}

std::vector<candle_box> compute_candles(const grid_result& gr)
{
    const int nbx = gr.nbx, nby = gr.nby;
    if (nbx < 1 || nby < 1 || gr.grid.empty()) return {};
    const double binw = (gr.x1 - gr.x0) / nbx;
    const double binh = (gr.y1 - gr.y0) / nby;

    // Per-column box stats. counts[b] is the column's count in bottom-up y-bin b
    // (b = 0 at y_min); the grid stores row 0 = top, so row = nby-1-b.
    std::vector<candle_box> boxes;
    std::vector<double>     counts(static_cast<std::size_t>(nby));
    for (int c = 0; c < nbx; ++c) {
        double total = 0.0;
        int    first = -1, last = -1;
        for (int b = 0; b < nby; ++b) {
            const double v = gr.grid[static_cast<std::size_t>(nby - 1 - b) * nbx + c];
            counts[static_cast<std::size_t>(b)] = v;
            total += v;
            if (v > 0.0) {
                if (first < 0) first = b;
                last = b;
            }
        }
        if (total <= 0.0) continue; // empty column → no candle
        // Weighted quantile with within-bin linear interpolation.
        auto quantile = [&](double frac) {
            const double target = frac * total;
            double       cum    = 0.0;
            for (int b = 0; b < nby; ++b) {
                const double next = cum + counts[static_cast<std::size_t>(b)];
                if (next >= target) {
                    const double f = counts[static_cast<std::size_t>(b)] > 0.0
                                         ? (target - cum) / counts[static_cast<std::size_t>(b)]
                                         : 0.0;
                    return gr.y0 + (static_cast<double>(b) + f) * binh;
                }
                cum = next;
            }
            return gr.y1;
        };
        const double q1  = quantile(0.25);
        const double med = quantile(0.50);
        const double q3  = quantile(0.75);
        const double iqr = q3 - q1;
        const double dmin = gr.y0 + static_cast<double>(first) * binh;        // populated floor
        const double dmax = gr.y0 + static_cast<double>(last + 1) * binh;     // populated ceil
        const double lo  = std::max(dmin, q1 - 1.5 * iqr);
        const double hi  = std::min(dmax, q3 + 1.5 * iqr);
        const double xc  = gr.x0 + (static_cast<double>(c) + 0.5) * binw;
        boxes.push_back({xc, q1, med, q3, lo, hi});
    }
    return boxes;
}

void draw_candles(const grid_result& gr, std::span<const candle_box> boxes, ImVec2 region,
                  bool show_x_axis, bool show_y_axis)
{
    if (gr.nbx < 1 || gr.nby < 1 || gr.grid.empty()) {
        ImGui::TextDisabled("empty histogram");
        return;
    }
    if (boxes.empty()) {
        ImGui::TextDisabled("no populated columns");
        return;
    }
    const double binw = (gr.x1 - gr.x0) / gr.nbx;

    const ImPlotAxisFlags xflags =
        ImPlotAxisFlags_AutoFit | (show_x_axis ? 0 : ImPlotAxisFlags_NoDecorations);
    const ImPlotAxisFlags yflags =
        ImPlotAxisFlags_AutoFit | (show_y_axis ? 0 : ImPlotAxisFlags_NoDecorations);
    if (ImPlot::BeginPlot("##candle", region)) {
        ImPlot::SetupAxes(nullptr, nullptr, xflags, yflags);
        // "##" label: single-item plot, so no legend entry is wanted.
        if (ImPlot::BeginItem("##boxes")) {
            // Contrast against the (theme-controlled) plot background: use the
            // resolved axis-text color, which the theme sets to be legible on its
            // PlotBg. Box fill = same hue, low alpha.
            const ImU32 line = ImPlot::GetStyleColorU32(ImPlotCol_AxisText);
            const ImU32 fill = (line & 0x00FFFFFFu) | (60u << 24);
            if (ImPlot::FitThisFrame()) {
                // Frame the DECLARED grid, not the populated columns. Empty columns
                // draw no candle (see `continue` above), so fitting to the boxes
                // made the axes track wherever the data happened to fall: two
                // figures with the same explicit domain and bin count came out with
                // different x ranges — different on-screen bin widths — which
                // defeats the point of fixing a domain and makes a canvas of them
                // incomparable. The heatmap path already frames on gr.x0..gr.x1 (it
                // hands them to PlotHeatmap as bounds), so the two views of one
                // figure disagreed on its axes.
                // With no fixed domain gr.x0..gr.x1 ARE the data's min/max, so
                // auto-ranging is unchanged.
                ImPlot::FitPoint(ImPlotPoint(gr.x0, gr.y0));
                ImPlot::FitPoint(ImPlotPoint(gr.x1, gr.y1));
            }
            ImDrawList*  dl = ImPlot::GetPlotDrawList();
            const double hw = binw * 0.35; // box half-width in plot units
            const double cw = binw * 0.20; // whisker-cap half-width
            for (const auto& b : boxes) {
                const ImVec2 p_lo = ImPlot::PlotToPixels(b.x, b.lo);
                const ImVec2 p_hi = ImPlot::PlotToPixels(b.x, b.hi);
                const ImVec2 bl   = ImPlot::PlotToPixels(b.x - hw, b.q1);
                const ImVec2 tr   = ImPlot::PlotToPixels(b.x + hw, b.q3);
                const ImVec2 m_l  = ImPlot::PlotToPixels(b.x - hw, b.med);
                const ImVec2 m_r  = ImPlot::PlotToPixels(b.x + hw, b.med);
                const float  cx0  = ImPlot::PlotToPixels(b.x - cw, b.lo).x;
                const float  cx1  = ImPlot::PlotToPixels(b.x + cw, b.lo).x;
                // whisker stem + caps
                dl->AddLine(ImVec2(p_lo.x, p_lo.y), ImVec2(p_hi.x, p_hi.y), line, 1.5f);
                dl->AddLine(ImVec2(cx0, p_lo.y), ImVec2(cx1, p_lo.y), line, 1.5f);
                dl->AddLine(ImVec2(cx0, p_hi.y), ImVec2(cx1, p_hi.y), line, 1.5f);
                // interquartile box (top-left = x-hw,q3 ; bottom-right = x+hw,q1)
                dl->AddRectFilled(ImVec2(bl.x, tr.y), ImVec2(tr.x, bl.y), fill);
                dl->AddRect(ImVec2(bl.x, tr.y), ImVec2(tr.x, bl.y), line, 0.0f, 0, 1.5f);
                // median
                dl->AddLine(m_l, m_r, line, 2.0f);
            }
            ImPlot::EndItem();
        }
        ImPlot::EndPlot();
    }
}

// ---------------------------------------------------------------- plot themes
// A theme overrides ImPlot's structural colors (plot/frame/legend backgrounds,
// axis text/grid/tick, …) so the same figures can render dark for the screen or
// light for a publication. Item colors (line/fill) live in ImPlotSpec in this
// ImPlot and are not themed here. Only the structural Colors[] are applied, via
// PushStyleColor, which composes with the export DPI size-scaling. Plots refer
// to a theme by name; the registry is GUI-thread-only (built lazily).

namespace {

struct theme_t {
    std::string name;
    ImPlotStyle style;
};

std::vector<theme_t>& themes()
{
    static std::vector<theme_t> t;
    return t;
}

bool g_show_theme_editor = false; // GUI thread only
bool g_themes_dirty      = false; // a change is pending a save to disk

// User themes persist to this file (default "coast_plot_themes.json" in the CWD;
// overridable at startup via set_theme_file from the app config). The storage is
// a function-local static so the const char* theme_file() hands out stays valid.
std::string& theme_file_storage()
{
    static std::string p = "coast_plot_themes.json";
    return p;
}
const char* theme_file() { return theme_file_storage().c_str(); }

// The two built-ins are the defaults/fallbacks and cannot be renamed or deleted.
bool is_builtin(const std::string& name) { return name == "dark" || name == "bright"; }

// Index of an ImPlotCol by its ImPlot name (for name-keyed persistence), or -1.
int col_index(const std::string& name)
{
    for (int i = 0; i < ImPlotCol_COUNT; ++i)
        if (name == ImPlot::GetStyleColorName(static_cast<ImPlotCol>(i))) return i;
    return -1;
}

// Serialize every theme to JSON. Each structural color is [r,g,b,a] or null
// (= IMPLOT_AUTO_COL, i.e. defer to the app default). GUI thread.
void save_themes()
{
    nlohmann::ordered_json j;
    j["version"]   = 1;
    auto& arr      = j["themes"] = nlohmann::ordered_json::array();
    for (const auto& t : themes()) {
        nlohmann::ordered_json jt;
        jt["name"]                   = t.name;
        nlohmann::ordered_json cols  = nlohmann::ordered_json::object();
        for (int i = 0; i < ImPlotCol_COUNT; ++i) {
            const ImVec4& c  = t.style.Colors[i];
            const char*   nm = ImPlot::GetStyleColorName(static_cast<ImPlotCol>(i));
            if (c.w < 0.0f)
                cols[nm] = nullptr; // auto
            else
                cols[nm] = {c.x, c.y, c.z, c.w};
        }
        jt["colors"] = std::move(cols);
        arr.push_back(std::move(jt));
    }
    std::ofstream os(theme_file());
    if (os)
        os << j.dump(2) << '\n';
    else
        COAST_WARN("plot", "theme: cannot write {}", theme_file());
}

// Load persisted themes, MERGING into the current list (built-ins already
// present): a saved theme replaces a same-named built-in's colors, otherwise it
// is appended. Missing file → no-op. GUI thread.
void load_themes()
{
    std::ifstream is(theme_file());
    if (!is) return; // none saved yet
    nlohmann::json j;
    try {
        is >> j;
    } catch (const std::exception& e) {
        COAST_WARN("plot", "theme: parse {} failed: {}", theme_file(), e.what());
        return;
    }
    if (!j.contains("themes") || !j["themes"].is_array()) return;
    auto& list = themes();
    for (const auto& jt : j["themes"]) {
        if (!jt.contains("name")) continue;
        theme_t t;
        t.name = jt["name"].get<std::string>();
        ImPlot::StyleColorsAuto(&t.style); // start all-auto, then apply stored colors
        if (jt.contains("colors") && jt["colors"].is_object()) {
            for (auto it = jt["colors"].begin(); it != jt["colors"].end(); ++it) {
                const int idx = col_index(it.key());
                if (idx < 0) continue;
                if (it.value().is_null())
                    t.style.Colors[idx] = IMPLOT_AUTO_COL;
                else if (it.value().is_array() && it.value().size() == 4)
                    t.style.Colors[idx] = ImVec4(it.value()[0].get<float>(),
                                                 it.value()[1].get<float>(),
                                                 it.value()[2].get<float>(),
                                                 it.value()[3].get<float>());
            }
        }
        bool merged = false;
        for (auto& e : list)
            if (e.name == t.name) {
                e.style = t.style;
                merged  = true;
                break;
            }
        if (!merged) list.push_back(std::move(t));
    }
}

void ensure_themes()
{
    auto& t = themes();
    if (!t.empty()) return;
    theme_t dark;
    dark.name = "dark";
    ImPlot::StyleColorsAuto(&dark.style); // all-auto → matches the app's dark style
    t.push_back(dark);
    theme_t bright;
    bright.name = "bright";
    ImPlot::StyleColorsLight(&bright.style); // white plot bg, dark axes/text
    t.push_back(bright);
    load_themes(); // overlay persisted edits + append user themes
}

const theme_t* find_theme(const std::string& name)
{
    for (const auto& t : themes())
        if (t.name == name) return &t;
    return nullptr;
}

// A name not yet used by any theme (append " 2", " 3", … on collision).
std::string unique_name(const std::string& base)
{
    auto taken = [](const std::string& n) {
        for (const auto& t : themes())
            if (t.name == n) return true;
        return false;
    };
    if (!taken(base)) return base;
    for (int k = 2;; ++k) {
        std::string cand = base + " " + std::to_string(k);
        if (!taken(cand)) return cand;
    }
}

} // namespace

std::vector<std::string> theme_names()
{
    ensure_themes();
    std::vector<std::string> ns;
    for (const auto& t : themes()) ns.push_back(t.name);
    return ns;
}

theme_scope::theme_scope(const std::string& name)
{
    ensure_themes();
    const theme_t* t = find_theme(name);
    if (t == nullptr) return;
    for (int i = 0; i < ImPlotCol_COUNT; ++i) {
        const ImVec4& c = t->style.Colors[i];
        if (c.w >= 0.0f) { // skip IMPLOT_AUTO_COL (w < 0) → leave it auto
            ImPlot::PushStyleColor(static_cast<ImPlotCol>(i), c);
            ++pushed;
        }
    }
}

theme_scope::~theme_scope()
{
    if (pushed > 0) ImPlot::PopStyleColor(pushed);
}

void set_theme_file(std::string path)
{
    // theme_file_storage() lives in this TU's anonymous namespace; a non-empty
    // override must be set before ensure_themes() first loads (see run_gui).
    if (!path.empty()) theme_file_storage() = std::move(path);
}

void request_theme_editor() { g_show_theme_editor = true; }

// A live editor: pick a theme, rename it, edit its structural colors, or
// duplicate/delete to manage a set. Themes are persisted to disk (see
// theme_file) and reloaded at startup; each plot stores its theme by NAME, so a
// reloaded plot picks up the current definition of that theme.
void draw_theme_editor()
{
    if (!g_show_theme_editor) return;
    ensure_themes();
    ImGui::SetNextWindowSize(ImVec2(380, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Plot Theme Editor", &g_show_theme_editor)) {
        auto&      ts  = themes();
        static int sel = 0;
        sel            = std::clamp(sel, 0, static_cast<int>(ts.size()) - 1);

        // theme selector
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginCombo("theme", ts[sel].name.c_str())) {
            for (int i = 0; i < static_cast<int>(ts.size()); ++i) {
                const bool s = (i == sel);
                if (ImGui::Selectable(ts[i].name.c_str(), s)) sel = i;
                if (s) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        const bool builtin = is_builtin(ts[sel].name);

        // Name field: rename user themes (built-ins are read-only). The buffer is
        // resynced whenever the selection changes.
        static int  name_for        = -1;
        static char name_buf[64]    = {};
        if (name_for != sel) {
            std::snprintf(name_buf, sizeof name_buf, "%s", ts[sel].name.c_str());
            name_for = sel;
        }
        ImGui::SetNextItemWidth(200.0f);
        ImGui::BeginDisabled(builtin);
        bool rename = ImGui::InputText("name", name_buf, sizeof name_buf,
                                       ImGuiInputTextFlags_EnterReturnsTrue);
        rename |= ImGui::IsItemDeactivatedAfterEdit();
        ImGui::EndDisabled();
        if (builtin) {
            ImGui::SameLine();
            ImGui::TextDisabled("(built-in)");
        }
        if (rename && !builtin) {
            std::string nn = name_buf;
            bool        bad = nn.empty() || is_builtin(nn);
            for (int i = 0; i < static_cast<int>(ts.size()) && !bad; ++i)
                if (i != sel && ts[i].name == nn) bad = true;
            if (!bad) {
                ts[sel].name   = nn;
                g_themes_dirty = true;
            } else {
                std::snprintf(name_buf, sizeof name_buf, "%s", ts[sel].name.c_str()); // revert
            }
        }

        // Duplicate / Delete
        if (ImGui::SmallButton("Duplicate")) {
            theme_t copy = ts[sel];
            copy.name    = unique_name(copy.name + " copy");
            ts.push_back(std::move(copy));
            sel            = static_cast<int>(ts.size()) - 1;
            name_for       = -1; // resync the name buffer to the new theme
            g_themes_dirty = true;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(builtin);
        if (ImGui::SmallButton("Delete")) {
            ts.erase(ts.begin() + sel);
            sel            = std::clamp(sel, 0, static_cast<int>(ts.size()) - 1);
            name_for       = -1;
            g_themes_dirty = true;
        }
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::TextDisabled("'auto' defers a color to the app default.");
        auto row = [&](const char* label, ImPlotCol idx) {
            ImVec4& c     = ts[sel].style.Colors[idx];
            bool    autob = (c.w < 0.0f);
            if (ImGui::Checkbox((std::string("auto##") + label).c_str(), &autob)) {
                c              = autob ? IMPLOT_AUTO_COL : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                g_themes_dirty = true;
            }
            ImGui::SameLine();
            if (autob) {
                ImGui::TextDisabled("%s", label);
            } else {
                ImGui::ColorEdit4(label, &c.x,
                                  ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
                if (ImGui::IsItemDeactivatedAfterEdit()) g_themes_dirty = true;
            }
        };
        row("plot background", ImPlotCol_PlotBg);
        row("frame background", ImPlotCol_FrameBg);
        row("plot border", ImPlotCol_PlotBorder);
        row("axis text", ImPlotCol_AxisText);
        row("axis grid", ImPlotCol_AxisGrid);
        row("axis tick", ImPlotCol_AxisTick);
        row("legend background", ImPlotCol_LegendBg);
        row("legend text", ImPlotCol_LegendText);
        row("title text", ImPlotCol_TitleText);
        row("inlay text", ImPlotCol_InlayText);
        ImGui::Separator();
        ImGui::TextDisabled("Edits apply live to every plot using this theme.\n"
                            "Saved to %s; renaming a theme unlinks plots still on\n"
                            "the old name (they fall back to the default look).",
                            theme_file());
    }
    ImGui::End();

    // Persist after committed changes (not every frame while dragging a slider).
    if (g_themes_dirty) {
        save_themes();
        g_themes_dirty = false;
    }
}

} // namespace coast::plot
