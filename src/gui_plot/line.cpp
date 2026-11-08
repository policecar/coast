#include "gui_plot/line.h"

#include "gui_plot/plot_tools.h" // ImPlot + theme layer

#include <algorithm>

namespace coast::plot {

line::line(std::string title) : figure(std::move(title)) {}

void line::update(std::span<const double> y)
{
    std::scoped_lock lk(m_);
    x_.clear();
    y_.assign(y.begin(), y.end());
}
void line::update(std::span<const float> y)
{
    std::vector<double> yd(y.begin(), y.end());
    update(std::span<const double>(yd));
}
void line::update(std::span<const double> x, std::span<const double> y)
{
    std::scoped_lock lk(m_);
    x_.assign(x.begin(), x.end());
    y_.assign(y.begin(), y.end());
}
void line::update(std::span<const float> x, std::span<const float> y)
{
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> yd(y.begin(), y.end());
    update(std::span<const double>(xd), std::span<const double>(yd));
}
std::size_t line::point_count() const
{
    std::scoped_lock lk(m_);
    return y_.size();
}

void line::paint_body(region rg)
{
    if (y_.empty()) {
        ImGui::TextDisabled("no data yet — call update()");
        return;
    }
    if (ImPlot::BeginPlot("##p", ImVec2(rg.w, rg.h))) {
        const int n = static_cast<int>(y_.size());
        if (x_.empty()) {
            ImPlot::PlotLine("##l", y_.data(), n);
        } else {
            ImPlot::PlotLine("##l", x_.data(), y_.data(),
                             static_cast<int>(std::min(x_.size(), y_.size())));
        }
        ImPlot::EndPlot();
    }
}
void line::paint_controls() { draw_theme_row(); }

bool line::fill_snapshot(h5_save_data& out) const
{
    if (y_.empty()) return false;
    out.kind = "line";
    out.bins = 0;
    out.ys.assign(y_.begin(), y_.end());
    if (!x_.empty()) out.xs.assign(x_.begin(), x_.end());
    return true;
}

} // namespace coast::plot
