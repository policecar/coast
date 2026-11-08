#ifndef COAST_GUI_PLOT_LINE_H
#define COAST_GUI_PLOT_LINE_H

#include "gui_plot/figure.h"

#include <cstddef>
#include <span>
#include <vector>

namespace coast::plot {

// Line/xy plot. update(y) plots y against its index; update(x, y) uses explicit x.
class line final : public figure {
public:
    explicit line(std::string title);

    void update(std::span<const double> y);
    void update(std::span<const float> y);
    void update(std::span<const double> x, std::span<const double> y);
    void update(std::span<const float> x, std::span<const float> y);
    [[nodiscard]] std::size_t point_count() const override;

protected:
    void paint_body(region r) override;
    void paint_controls() override;
    bool fill_snapshot(h5_save_data& out) const override;
    [[nodiscard]] bool is_saveable() const override { return true; }

private:
    friend std::shared_ptr<figure> make_figure_from_h5(const h5_save_data&);
    std::vector<double> x_, y_;
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_LINE_H
