#ifndef COAST_GUI_PLOT_HISTOGRAM_H
#define COAST_GUI_PLOT_HISTOGRAM_H

#include "gui_plot/figure.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace coast::plot {

// 1D histogram of a value sample. update(y) replaces the values to bin.
//
// By default the bin edges span the data's own min/max, so two histograms of
// different samples are NOT comparable — each gets its own frame. Construct with
// a domain to fix the edges instead (see the domain overload).
class histogram final : public figure {
public:
    explicit histogram(std::string title, int bins = 64);

    // Fixed-domain form: bin edges span [domain.lo, domain.hi] regardless of the
    // data, so histograms of different samples share edges and are directly
    // comparable. Values outside the domain go to underflow()/overflow() and are
    // not drawn — they are neither dropped nor clamped into the edge bins (which
    // would fabricate a spike at a value that was never observed).
    //
    // An invalid domain (hi <= lo) falls back to auto-ranging. The domain is
    // fixed at construction and immutable: the under/overflow tallies are
    // computed in update(), so a settable domain would silently stale them.
    histogram(std::string title, int bins, range domain);

    void update(std::span<const double> y);
    void update(std::span<const float> y);
    [[nodiscard]] std::size_t point_count() const override;

    // The fixed domain, or nullopt when auto-ranged.
    [[nodiscard]] std::optional<range> domain() const;
    // Values below/above a fixed domain, in ROOT's under/overflow-bin spirit.
    // Always 0 when auto-ranged. point_count() == in-range + under + over.
    [[nodiscard]] std::size_t underflow() const;
    [[nodiscard]] std::size_t overflow() const;

protected:
    void paint_body(region r) override;
    void paint_controls() override;
    bool fill_snapshot(h5_save_data& out) const override;
    [[nodiscard]] bool is_saveable() const override { return true; }

private:
    friend std::shared_ptr<figure> make_figure_from_h5(const h5_save_data&);
    // Recount under_/over_ from y_ against domain_. Caller holds m_.
    void recount_outliers();

    std::vector<double>  y_;
    int                  bins_;
    std::optional<range> domain_;      // nullopt => auto-range from the data
    std::size_t          under_ = 0;   // derived from y_ + domain_ in update()
    std::size_t          over_  = 0;
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_HISTOGRAM_H
