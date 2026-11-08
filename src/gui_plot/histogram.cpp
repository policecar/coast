#include "gui_plot/histogram.h"

#include "gui_plot/plot_tools.h" // ImPlot + theme layer

namespace coast::plot {

histogram::histogram(std::string title, int bins)
    : figure(std::move(title)), bins_(bins > 0 ? bins : 1)
{
}

histogram::histogram(std::string title, int bins, range domain)
    : figure(std::move(title)), bins_(bins > 0 ? bins : 1),
      // An invalid domain would collapse the frame to zero width; auto-range
      // instead of binning into nothing.
      domain_(domain.valid() ? std::optional<range>{domain} : std::nullopt)
{
}

void histogram::recount_outliers()
{
    under_ = 0;
    over_  = 0;
    if (!domain_) return; // auto-ranged: the frame IS the data, nothing is outside
    for (const double v : y_) {
        if (v < domain_->lo) ++under_;
        else if (v > domain_->hi) ++over_;
    }
}

void histogram::update(std::span<const double> y)
{
    std::scoped_lock lk(m_);
    y_.assign(y.begin(), y.end());
    recount_outliers();
}
void histogram::update(std::span<const float> y)
{
    std::vector<double> yd(y.begin(), y.end());
    update(std::span<const double>(yd));
}
std::size_t histogram::point_count() const
{
    std::scoped_lock lk(m_);
    return y_.size();
}

std::optional<range> histogram::domain() const
{
    std::scoped_lock lk(m_);
    return domain_;
}
std::size_t histogram::underflow() const
{
    std::scoped_lock lk(m_);
    return under_;
}
std::size_t histogram::overflow() const
{
    std::scoped_lock lk(m_);
    return over_;
}

void histogram::paint_body(region rg)
{
    if (y_.empty()) {
        ImGui::TextDisabled("no data yet — call update()");
        return;
    }
    if (ImPlot::BeginPlot("##p", ImVec2(rg.w, rg.h))) {
        // A default-constructed ImPlotRange (Min==Max==0) is ImPlot's own
        // "auto-range from the data" sentinel, so the unfixed path is unchanged.
        // Given a range, ImPlot bins over it and skips values outside
        // (`range.Contains(val)`) — matching the under/overflow accounting in
        // recount_outliers — and its bars span the whole range, which pins the
        // visible axis to the domain rather than to the data.
        const ImPlotRange r = domain_ ? ImPlotRange(domain_->lo, domain_->hi) : ImPlotRange();
        ImPlot::PlotHistogram("##h", y_.data(), static_cast<int>(y_.size()), bins_,
                              /*bar_scale=*/1.0, r);
        ImPlot::EndPlot();
    }
}
void histogram::paint_controls()
{
    draw_theme_row();
    // Surface the outliers: a fixed domain is the only way values can go
    // uncounted, and silently dropping them is exactly the failure this
    // accounting exists to prevent.
    if (domain_ && (under_ > 0 || over_ > 0)) {
        ImGui::SameLine();
        ImGui::TextDisabled("outside [%g, %g]: %zu under / %zu over", domain_->lo, domain_->hi,
                            under_, over_);
    }
}

bool histogram::fill_snapshot(h5_save_data& out) const
{
    if (y_.empty()) return false;
    out.kind = "histogram";
    out.bins = bins_;
    out.ys.assign(y_.begin(), y_.end());
    // The domain must persist or a reload would re-bin these raw values with
    // auto-ranging and silently lose the fixed frame. under_/over_ need no field:
    // they are recomputed from ys + domain on load.
    if (domain_) {
        out.has_x_range = true;
        out.x_range_lo  = domain_->lo;
        out.x_range_hi  = domain_->hi;
    }
    return true;
}

} // namespace coast::plot
