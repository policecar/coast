#include <vector>
#include <tuple>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <array>
#include <span>
#include <string>
#include <optional>

#include "h5browse/console_api.h"
#include "gui_plot/console_plot.h"
#include "tools/metric.h"

namespace cp = coast::plot;

using rel_sim_t = std::tuple<std::vector<double>,std::vector<double>>;

// Ordinary least-squares linear fit of y = slope * x + intercept over the
// 2D point cloud { (x[i], y[i]) }. Returns {slope, y-intercept}. Returns
// {0,0} on invalid input (mismatched/empty spans, or degenerate x with zero
// variance where the slope is undefined).
std::tuple<double,double> linear_fit(std::span<const double> x, std::span<const double> y)
{
    if (x.size() != y.size() || x.empty()) {
        std::printf("linear_fit: x and y must be non-empty spans of equal length!\n");
        return {0.0, 0.0};
    }

    const double n = static_cast<double>(x.size());
    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }

    const double denom = n * sum_xx - sum_x * sum_x;
    if (denom == 0.0) {
        std::printf("linear_fit: x has zero variance, slope is undefined!\n");
        return {0.0, 0.0};
    }

    const double slope     = (n * sum_xy - sum_x * sum_y) / denom;
    const double intercept = (sum_y - slope * sum_x) / n;
    return {slope, intercept};
}

// Solve the n x n linear system A c = b in place by Gaussian elimination with
// partial pivoting. On success writes the solution into b and returns true; on
// a singular matrix (to working precision) returns false and leaves b
// unspecified. n is taken from b.size(); A must be n x n.
static bool solve_linear_system(std::vector<std::vector<double>>& A, std::vector<double>& b)
{
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivot: pull the largest |A[row][col]| onto the diagonal.
        std::size_t piv = col;
        double      max_abs = std::abs(A[col][col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const double a = std::abs(A[row][col]);
            if (a > max_abs) { max_abs = a; piv = row; }
        }
        if (max_abs == 0.0) return false; // singular
        std::swap(A[col], A[piv]);
        std::swap(b[col], b[piv]);

        for (std::size_t row = col + 1; row < n; ++row) {
            const double f = A[row][col] / A[col][col];
            for (std::size_t k = col; k < n; ++k) A[row][k] -= f * A[col][k];
            b[row] -= f * b[col];
        }
    }
    // Back-substitution.
    for (std::size_t i = n; i-- > 0; ) {
        double s = b[i];
        for (std::size_t k = i + 1; k < n; ++k) s -= A[i][k] * b[k];
        b[i] = s / A[i][i];
    }
    return true;
}

// Least-squares polynomial fit of degree `deg` over the point cloud
// { (x[i], y[i]) }, via the normal equations A c = b with A[j][k] = sum x^(j+k)
// and b[j] = sum y*x^j. On success writes coefficients into `coeff` (coeff[m] is
// the x^m term, so coeff.size() == deg+1) and returns true; returns false on
// invalid input (mismatched/empty spans, too few points) or a singular system
// (fewer than deg+1 distinct x values, where the fit is undefined).
static bool poly_fit(std::span<const double> x, std::span<const double> y, int deg, std::vector<double>& coeff)
{
    if (x.size() != y.size() || x.empty()) {
        std::printf("poly_fit: x and y must be non-empty spans of equal length!\n");
        return false;
    }
    const std::size_t nc = static_cast<std::size_t>(deg) + 1; // number of coefficients
    if (x.size() < nc) {
        std::printf("poly_fit: need at least %zu points for a degree-%d fit!\n", nc, deg);
        return false;
    }

    // Power sums S[m] = sum x^m (m = 0..2*deg) and T[j] = sum y*x^j (j = 0..deg).
    std::vector<double> S(2 * static_cast<std::size_t>(deg) + 1, 0.0);
    std::vector<double> T(nc, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        double xp = 1.0;
        for (std::size_t m = 0; m < S.size(); ++m) {
            S[m] += xp;
            if (m < nc) T[m] += y[i] * xp;
            xp *= x[i];
        }
    }

    std::vector<std::vector<double>> A(nc, std::vector<double>(nc));
    for (std::size_t j = 0; j < nc; ++j)
        for (std::size_t k = 0; k < nc; ++k)
            A[j][k] = S[j + k];

    std::vector<double> b(T.begin(), T.end());
    if (!solve_linear_system(A, b)) {
        std::printf("poly_fit: singular system (fewer than %zu distinct x values), fit undefined!\n", nc);
        return false;
    }
    coeff.assign(b.begin(), b.end());
    return true;
}

// Least-squares quadratic fit y = a2*x^2 + a1*x + a0 over { (x[i], y[i]) }.
// Returns {a2, a1, a0} (highest power first, matching linear_fit); {0,0,0} on
// invalid input or a singular system (see poly_fit).
std::tuple<double,double,double> quadratic_fit(std::span<const double> x, std::span<const double> y)
{
    std::vector<double> c;
    if (!poly_fit(x, y, 2, c)) return {0.0, 0.0, 0.0};
    return {c[2], c[1], c[0]};
}

// Least-squares cubic fit y = a3*x^3 + a2*x^2 + a1*x + a0 over { (x[i], y[i]) }.
// Returns {a3, a2, a1, a0} (highest power first); {0,0,0,0} on invalid input or
// a singular system (see poly_fit).
std::tuple<double,double,double,double> cubic_fit(std::span<const double> x, std::span<const double> y)
{
    std::vector<double> c;
    if (!poly_fit(x, y, 3, c)) return {0.0, 0.0, 0.0, 0.0};
    return {c[3], c[2], c[1], c[0]};
}

rel_sim_t calc_rel_sim(
    std::size_t sample_idx,
    std::size_t epoch_size,
    std::span<std::size_t> epochs,
    const h5b::console::dataset_handle &inp_dataset,
    const h5b::console::dataset_handle &outp_dataset,
    std::size_t inp_offset = 1
)
{
    if (sample_idx >= epoch_size) {
        std::printf("sample_idx must lie within epoch_size!\n");
        return {};
    }

    if (epochs.empty()) {
        std::printf("epochs must not be empty!\n");
        return {};
    }

    std::size_t dataset_size = inp_dataset.rows() - inp_offset;

    std::size_t max_epoch = *std::ranges::max_element(epochs);

    if ((max_epoch+1) * epoch_size > dataset_size) {
        std::printf("Maximum epoch exceeds dataset size\n");
        return {};
    }

    if (dataset_size != outp_dataset.rows() - inp_offset) {
        std::printf("inp_dataset and outp_dataset need to be the same size!\n");
        return {};
    }
    std::vector<double> result_inp;
    result_inp.resize(epochs.size()*epoch_size);
    std::vector<double> result_outp;
    result_outp.resize(epochs.size()*epoch_size);

    // get fixed reference input
    const std::vector<float> ref_inp = inp_dataset.row(sample_idx+inp_offset).as_copy<float>();

    std::size_t epoch_idx = 0;
    for (std::size_t epoch : epochs) {
        // calculate cos sim to reference input. Identify the sample in the epoch by max in order to
        // select the corresponding output.
        double      max_sim = -2.0; // below the cosine-similarity floor of -1, so the first sample always wins
        std::size_t max_idx = epoch * epoch_size;
        for (std::size_t idx = epoch * epoch_size; idx < (epoch+1) * epoch_size; ++idx) {
            double similarity = coast::cosine_similarity(std::span<const float>(ref_inp),inp_dataset.row(idx+inp_offset).as<float>());
            if (similarity > max_sim) {
                max_sim = similarity;
                max_idx = idx;
            }
            result_inp[epoch_idx * epoch_size + idx - epoch * epoch_size] = similarity;
        }
        const std::vector<float> ref_outp = outp_dataset.row(max_idx+inp_offset).as_copy<float>();
        // calculate cos similarities of output vectors to identified reference
        for (std::size_t idx = epoch * epoch_size; idx < (epoch+1) * epoch_size; ++idx) {
            double similarity = coast::cosine_similarity(std::span<const float>(ref_outp),outp_dataset.row(idx+inp_offset).as<float>());
            result_outp[epoch_idx * epoch_size + idx - epoch * epoch_size] = similarity;
        }
        ++epoch_idx;
    }
    return {result_inp, result_outp};
}

std::vector<double> calc_rel_sim(
    std::size_t sample_idx,
    std::size_t epoch,
    std::size_t epoch_size,
    const h5b::console::dataset_handle &dataset,
    std::size_t inp_offset = 1
)
{
    if (sample_idx >= epoch_size) {
        std::printf("sample_idx must lie within epoch_size!\n");
        return {};
    }

    std::size_t dataset_size = dataset.rows() - inp_offset;

    if ((epoch+1) * epoch_size > dataset_size) {
        std::printf("Maximum epoch exceeds dataset size\n");
        return {};
    }

    std::vector<double> result;
    result.resize(epoch_size);

    // get fixed reference input -- sample_idx is a position *within* the epoch,
    // so the reference row is offset by epoch * epoch_size like the loop below.
    const std::vector<float> ref_inp = dataset.row(epoch * epoch_size + sample_idx + inp_offset).as_copy<float>();

    // calculate cos sim to reference input. Identify the sample in the epoch by max in order to
    // select the corresponding output.
    for (std::size_t idx = epoch * epoch_size; idx < (epoch+1) * epoch_size; ++idx) {
        double similarity = coast::cosine_similarity(std::span<const float>(ref_inp),dataset.row(idx+inp_offset).as<float>());
        result[idx - epoch * epoch_size] = similarity;
    }
    return result;
}

std::shared_ptr<cp::histogram2d> get_rs_hist(std::size_t epoch_idx, std::size_t epoch_size, rel_sim_t &data, int x_bins, int y_bins, bool visible = true)
{
    if ((epoch_idx+1)*epoch_size > std::get<0>(data).size()) {
        std::printf("epoch outside of range!\n");
        return {};
    }

    std::span<double> inp_epoch  { std::get<0>(data).begin() + epoch_idx * epoch_size, epoch_size };
    std::span<double> outp_epoch { std::get<1>(data).begin() + epoch_idx * epoch_size, epoch_size };
    
    auto hist2d = std::make_shared<cp::histogram2d>("rel sim epoch " + std::to_string(epoch_idx),x_bins,y_bins,cp::range(0.0,1.0),cp::range(0.0,1.0));
    if (visible) hist2d->show();
    hist2d->update(inp_epoch,outp_epoch);
    return hist2d;
}

// Bin one epoch's rel-sim point cloud along x into x_bins equal bins over [0,1]
// and emit one (bin_center, median-of-y-in-bin) sample per populated bin into
// xs/ys -- i.e. the median ticks you would see in the candle view produced by
// get_rs_hist(...).set_candle(true). The median here is the exact median of the
// raw y values in the bin (not the candle's y-binned/interpolated quantile), so
// it can differ from the drawn candle tick by up to one y-bin height. Returns
// false (with a message) on an out-of-range epoch or x_bins < 1; on success
// xs/ys hold one entry per populated bin (possibly fewer than x_bins).
static bool rs_median_samples(std::size_t epoch_idx, std::size_t epoch_size, rel_sim_t &data, int x_bins,
                              std::vector<double>& xs, std::vector<double>& ys)
{
    if ((epoch_idx+1)*epoch_size > std::get<0>(data).size()) {
        std::printf("epoch outside of range!\n");
        return false;
    }
    if (x_bins < 1) {
        std::printf("rs_median_samples: x_bins must be >= 1!\n");
        return false;
    }

    std::span<double> inp_epoch  { std::get<0>(data).begin() + epoch_idx * epoch_size, epoch_size };
    std::span<double> outp_epoch { std::get<1>(data).begin() + epoch_idx * epoch_size, epoch_size };

    // Bin along x over [0,1]; collect the y values falling in each bin.
    std::vector<std::vector<double>> bins(static_cast<std::size_t>(x_bins));
    for (std::size_t i = 0; i < epoch_size; ++i) {
        const double x = inp_epoch[i];
        if (x < 0.0 || x > 1.0) continue; // rel sim lives in [0,1]; ignore strays
        int b = static_cast<int>(x * x_bins);
        if (b == x_bins) b = x_bins - 1;  // x == 1.0 lands in the last bin
        bins[static_cast<std::size_t>(b)].push_back(outp_epoch[i]);
    }

    // One (bin_center, median) sample per populated bin.
    auto median = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
    };
    xs.clear();
    ys.clear();
    const double binw = 1.0 / x_bins;
    for (int b = 0; b < x_bins; ++b) {
        auto& col = bins[static_cast<std::size_t>(b)];
        if (col.empty()) continue;
        xs.push_back((static_cast<double>(b) + 0.5) * binw);
        ys.push_back(median(col));
    }
    return true;
}

// Linear fit through one epoch's per-x-column medians (see rs_median_samples for
// how the medians are formed). Returns {slope, y-intercept}; {0,0} on
// out-of-range epoch or when fewer than two bins are populated.
std::tuple<double,double> get_rs_median_fit(std::size_t epoch_idx, std::size_t epoch_size, rel_sim_t &data, int x_bins)
{
    std::vector<double> xs, ys;
    if (!rs_median_samples(epoch_idx, epoch_size, data, x_bins, xs, ys)) return {0.0, 0.0};
    if (xs.size() < 2) {
        std::printf("get_rs_median_fit: fewer than two populated bins, slope undefined!\n");
        return {0.0, 0.0};
    }
    return linear_fit(xs, ys);
}

// Quadratic fit through one epoch's per-x-column medians (see rs_median_samples).
// Returns {a2, a1, a0}; {0,0,0} on out-of-range epoch or fewer than three
// populated bins.
std::tuple<double,double,double> get_rs_median_fit_quad(std::size_t epoch_idx, std::size_t epoch_size, rel_sim_t &data, int x_bins)
{
    std::vector<double> xs, ys;
    if (!rs_median_samples(epoch_idx, epoch_size, data, x_bins, xs, ys)) return {0.0, 0.0, 0.0};
    if (xs.size() < 3) {
        std::printf("get_rs_median_fit_quad: fewer than three populated bins, fit undefined!\n");
        return {0.0, 0.0, 0.0};
    }
    return quadratic_fit(xs, ys);
}

// Cubic fit through one epoch's per-x-column medians (see rs_median_samples).
// Returns {a3, a2, a1, a0}; {0,0,0,0} on out-of-range epoch or fewer than four
// populated bins.
std::tuple<double,double,double,double> get_rs_median_fit_cubic(std::size_t epoch_idx, std::size_t epoch_size, rel_sim_t &data, int x_bins)
{
    std::vector<double> xs, ys;
    if (!rs_median_samples(epoch_idx, epoch_size, data, x_bins, xs, ys)) return {0.0, 0.0, 0.0, 0.0};
    if (xs.size() < 4) {
        std::printf("get_rs_median_fit_cubic: fewer than four populated bins, fit undefined!\n");
        return {0.0, 0.0, 0.0, 0.0};
    }
    return cubic_fit(xs, ys);
}

std::shared_ptr<cp::canvas> gen_rs_grid(
    std::span<std::size_t> inputs,
    std::span<std::size_t> epochs,
    std::size_t epoch_size,
    bool log_scale,
    bool candle,   // true: candle/box view; false: colormapped density heatmap
    const h5b::console::dataset_handle &inp_dataset,
    const h5b::console::dataset_handle &outp_dataset,
    std::size_t inp_offset = 1
)
{
    auto cv = std::make_shared<cp::canvas>("rel sim grid",static_cast<int>(inputs.size()),static_cast<int>(epochs.size()+1));
    cv->show();

    std::size_t row_idx = 0;
    for (std::size_t inp_idx : inputs) {
        auto rel_sim = calc_rel_sim(inp_idx,epoch_size,epochs,inp_dataset,outp_dataset,inp_offset);
        int inp_side = std::floor(std::sqrt((double)inp_dataset.row(inp_idx+inp_offset).as<float>().size()));
        auto p = std::make_shared<cp::pixels>("", inp_side, inp_side);
        p->update(inp_dataset.row(inp_idx+inp_offset).as<float>());
        p->set_palette(coast::colormap_t::greys);
        p->set_axis_visible(cp::plot_axis::x,false);
        p->set_axis_visible(cp::plot_axis::y,false);
        p->set_axis_visible(cp::plot_axis::z,false);
        cv->place(p,row_idx,0);
        std::size_t col_idx = 1;
        for (std::size_t epoch_idx = 0; epoch_idx < epochs.size(); ++epoch_idx) {
            auto hist = get_rs_hist(epoch_idx,epoch_size,rel_sim,64,64,false);
            auto [slope,yinter] = get_rs_median_fit(epoch_idx,epoch_size,rel_sim,64);
            auto [q1,q2,q3] = get_rs_median_fit_quad(epoch_idx,epoch_size,rel_sim,64);
            auto [c1,c2,c3,c4] = get_rs_median_fit_cubic(epoch_idx,epoch_size,rel_sim,64);
            std::printf("Input %zu, Epoch Idx: %zu,",inp_idx,epoch_idx);
            std::printf(" Slope: %f, Y-Intercept: %f",slope,yinter);
            std::printf(" Q: %f|%f|%f|,",q1,q2,q3);
            std::printf(" C: %f|%f|%f|%f|\n",c1,c2,c3,c4);
            hist->set_axis_visible(cp::plot_axis::x,false);
            hist->set_axis_visible(cp::plot_axis::y,false);
            if (candle) {
                hist->set_candle(true);
            } else {
                hist->set_axis_visible(cp::plot_axis::z,false);
                hist->set_log_density(log_scale);
                hist->set_palette(coast::colormap_t::jet);
            }
            cv->place(hist,row_idx,col_idx++);
        }
        ++row_idx;
    }

    return cv;
}


std::vector<std::size_t> get_representative_samples(
    std::size_t epoch_size,
    std::size_t max_set_size,
    const h5b::console::dataset_handle &dataset,
    std::size_t inp_offset = 1,
    std::optional<unsigned> seed = std::nullopt // set for reproducible runs; default seeds from random_device
)
{
    constexpr int AVG_SAMPLE_CNT  = 1000;
    constexpr int INIT_SAMPLE_CNT =   10;
    constexpr int INIT_MAX_ADD    =  100;

    if (epoch_size == 0) {
        std::printf("get_representative_samples: epoch_size must be > 0!\n");
        return {};
    }
    if (epoch_size + inp_offset > dataset.rows()) {
        std::printf("get_representative_samples: epoch_size exceeds dataset size!\n");
        return {};
    }

    // Operates on the first epoch_size samples; returned indices are within
    // [0, epoch_size) and map to dataset rows as row(idx + inp_offset) -- the
    // same convention show_vectors uses, so its output can be fed there directly.
    std::uniform_int_distribution<std::size_t> avg_dist {0, epoch_size - 1};
    std::mt19937 rgen { seed ? *seed : std::random_device{}() };
    double avg_rs = 0.0;
    for (int i = 0; i < AVG_SAMPLE_CNT; ++i) {
        auto row_a = dataset.row( avg_dist(rgen) + inp_offset );
        auto row_b = dataset.row( avg_dist(rgen) + inp_offset );
        avg_rs += coast::cosine_similarity(row_a.as<float>(),row_b.as<float>());
    }

    avg_rs /= static_cast<double>(AVG_SAMPLE_CNT);
    //std::printf("debug avg_rs: %f\n",avg_rs);
    //std::fflush(stdout);

    // find initial sample
    std::array<std::vector<double>,INIT_SAMPLE_CNT> sample_set;
    std::uniform_int_distribution<std::size_t> init_dist {0, epoch_size - 1};
    for (auto &sample : sample_set) {
        std::size_t sample_idx = init_dist(rgen);
        sample = calc_rel_sim(sample_idx,0,epoch_size,dataset,inp_offset);
        std::ranges::sort(sample,[](auto a, auto b){ return a > b; }); // sort descending
        sample.push_back(static_cast<double>(sample_idx));
    }
    double max_N        = 0.0;
    std::size_t max_idx = 0;
    bool found_init     = false;
    for (int idx = 0; idx < INIT_SAMPLE_CNT; ++idx) {
        const auto& sample = sample_set[idx];
        if (sample.size() < 2) continue; // calc_rel_sim failed: only the appended index is present
        // sample holds epoch_size similarities plus one appended index; sum the
        // top-N similarities only, clamped so we never read the index or run off the end.
        const std::size_t n_add = std::min<std::size_t>(INIT_MAX_ADD, sample.size() - 1);
        double sum = 0.0;
        for (std::size_t mx_idx = 0; mx_idx < n_add; ++mx_idx) {
            sum += sample[mx_idx];
        }
        if (!found_init || sum > max_N) {
            max_N      = sum;
            max_idx    = static_cast<std::size_t>(sample.back());
            found_init = true;
        }
    }
    if (!found_init) {
        std::printf("get_representative_samples: no valid initial sample!\n");
        return {};
    }

    //std::printf("debug start_idx: %zu\n",max_idx);
    //std::fflush(stdout);

    std::vector<std::size_t> samples;
    samples.push_back(max_idx);

    std::vector<double> rel_dists = calc_rel_sim(max_idx,0,epoch_size,dataset,inp_offset);
    if (rel_dists.empty()) {
        std::printf("get_representative_samples: calc_rel_sim returned no data!\n");
        return samples; // just the initial sample
    }
    auto min_elem = std::min_element(rel_dists.begin(),rel_dists.end());
    double cur_min_sim = *min_elem;
    while ((cur_min_sim < avg_rs) && (samples.size() < max_set_size)) {
        std::size_t next_elem = std::distance(rel_dists.begin(),min_elem);
        samples.push_back(next_elem);
        // calc distances
        std::vector<double> tmp_dists = calc_rel_sim(next_elem,0,epoch_size,dataset,inp_offset);
        //std::printf("debug next_idx: %zu\n",next_elem);
        //std::fflush(stdout);
        // merge distances
        std::transform(
            rel_dists.begin(),rel_dists.end(),
            tmp_dists.begin(),
            rel_dists.begin(),
            [](double a, double b) {
                return std::max(a,b);
            });
        // search minimum
        min_elem = std::min_element(rel_dists.begin(),rel_dists.end());
        cur_min_sim = *min_elem;
        //std::printf("debug cur_min: %f\n",cur_min_sim);
        //std::fflush(stdout);
    }

    return samples;
}




std::shared_ptr<cp::canvas> show_vectors(
    std::span<std::size_t> data_indices,
    const h5b::console::dataset_handle &dataset,
    std::size_t inp_offset = 1
)
{
    std::size_t table_w = std::ceil(std::sqrt( static_cast<double>(data_indices.size()) ));

    auto cv = std::make_shared<cp::canvas>("data display", static_cast<int>(table_w), static_cast<int>(table_w));
    cv->show();

    std::size_t row_idx = 0;
    std::size_t col_idx = 0;

    for (auto data_idx : data_indices) {
        auto row = dataset.row(data_idx + inp_offset);
        int inp_side = std::floor(std::sqrt(static_cast<double>(row.as<float>().size())));
        auto p = std::make_shared<cp::pixels>("", inp_side, inp_side);
        p->update( row.as<float>() );
        p->set_palette(coast::colormap_t::greys);
        p->set_axis_visible(cp::plot_axis::x,false);
        p->set_axis_visible(cp::plot_axis::y,false);
        p->set_axis_visible(cp::plot_axis::z,false);
        cv->place(p,row_idx,col_idx);
        ++row_idx;
        if (row_idx >= table_w) {
            row_idx = 0;
            ++col_idx;
        }
    }

    return cv;
}





// Overview / "--help" for this macro. Running `.x group_in_out_analysis.C`
// dispatches to the function named after the file stem, so it lands here and
// prints a summary of the tools defined above instead of doing any work.
void group_in_out_analysis()
{
    std::printf(
        "group_in_out_analysis.C - group input/output relative-similarity analysis\n"
        "\n"
        "This macro defines helper functions for analysing how a neuron group's\n"
        "outputs track its inputs across training epochs. Load it with\n"
        "`.L group_in_out_analysis.C`, then call the functions below directly.\n"
        "(`.x group_in_out_analysis.C` loads the file and prints this overview.)\n"
        "\n"
        "Functions:\n"
        "\n"
        "  linear_fit(std::span<const double> x, std::span<const double> y)\n"
        "      Ordinary least-squares fit of y = slope*x + intercept over the\n"
        "      point cloud {(x[i], y[i])}. Returns {slope, intercept}; {0,0} on\n"
        "      invalid input (mismatched/empty spans or zero-variance x).\n"
        "\n"
        "  quadratic_fit(std::span<const double> x, std::span<const double> y)\n"
        "      Least-squares fit of y = a2*x^2 + a1*x + a0. Returns {a2, a1, a0}\n"
        "      (highest power first); {0,0,0} on invalid input or a singular\n"
        "      system (fewer than 3 distinct x values).\n"
        "\n"
        "  cubic_fit(std::span<const double> x, std::span<const double> y)\n"
        "      Least-squares fit of y = a3*x^3 + a2*x^2 + a1*x + a0. Returns\n"
        "      {a3, a2, a1, a0}; {0,0,0,0} on invalid input or a singular system\n"
        "      (fewer than 4 distinct x values).\n"
        "\n"
        "  calc_rel_sim(\n"
        "      std::size_t sample_idx,\n"
        "      std::size_t epoch_size,\n"
        "      std::span<std::size_t> epochs,\n"
        "      const h5b::console::dataset_handle &inp_dataset,  // double click in browser to import automatically\n"
        "      const h5b::console::dataset_handle &outp_dataset, // double click in browser to import automatically\n"
        "      std::size_t inp_offset = 1                        // the first input is always empty, we skip it\n"
        "  )\n"
        "      For a fixed reference input (sample_idx), compute per-epoch cosine\n"
        "      similarities of inputs to the reference, pick each epoch's best\n"
        "      match, and compute output similarities to that match's output.\n"
        "      Returns {input_similarities, output_similarities} (rel_sim_t).\n"
        "\n"
        "  get_rs_hist(\n"
        "      std::size_t epoch_idx,\n"
        "      std::size_t epoch_size,\n"
        "      rel_sim_t &data,         // returned by calc_rel_sim\n"
        "      int x_bins,\n"
        "      int y_bins,\n"
        "      bool visible = true\n"
        "  )\n"
        "      Build a 2D histogram of input vs. output similarity for one epoch\n"
        "      of a calc_rel_sim() result and return the histogram2d figure.\n"
        "\n"
        "  get_rs_median_fit(\n"
        "      std::size_t epoch_idx,\n"
        "      std::size_t epoch_size,\n"
        "      rel_sim_t &data,         // returned by calc_rel_sim\n"
        "      int x_bins\n"
        "  )\n"
        "      Bin one epoch's points along x, take each bin's median output\n"
        "      similarity, and linear_fit through those medians -- the line\n"
        "      through the median ticks of the candle view. Returns {slope,\n"
        "      intercept}; {0,0} on out-of-range epoch or < 2 populated bins.\n"
        "\n"
        "  get_rs_median_fit_quad(epoch_idx, epoch_size, data, x_bins)\n"
        "  get_rs_median_fit_cubic(epoch_idx, epoch_size, data, x_bins)\n"
        "      As get_rs_median_fit but quadratic_fit / cubic_fit through the\n"
        "      medians. Return {a2,a1,a0} / {a3,a2,a1,a0}; zeros on out-of-range\n"
        "      epoch or fewer than 3 / 4 populated bins.\n"
        "\n"
        "  gen_rs_grid(\n"
        "      std::span<std::size_t> inputs, // indices of inputs to investigate -> rows of the grid\n"
        "      std::span<std::size_t> epochs, // indices of epochs to investigate -> columns of the grid\n"
        "      std::size_t epoch_size,\n"
        "      bool log_scale,                    // heatmap only: log10 density\n"
        "      bool candle,                       // true: candle/box view; false: color heatmap\n"
        "      const h5b::console::dataset_handle &inp_dataset,\n"
        "      const h5b::console::dataset_handle &outp_dataset,\n"
        "      std::size_t inp_offset = 1\n"
            "  )\n"
        "      Build a canvas grid: one row per input, showing the input image\n"
        "      followed by per-epoch relative-similarity histograms, drawn as\n"
        "      candle/box plots or as a jet color density heatmap (log_scale\n"
        "      applies to the heatmap only). Returns the canvas figure.\n"
        "\n"
        "  calc_rel_sim(\n"
        "      std::size_t sample_idx,\n"
        "      std::size_t epoch,\n"
        "      std::size_t epoch_size,\n"
        "      const h5b::console::dataset_handle &dataset,\n"
        "      std::size_t inp_offset = 1\n"
        "  )\n"
        "      Single-epoch/single-dataset overload: cosine similarity of one\n"
        "      reference sample (sample_idx, a position within the epoch) against\n"
        "      every sample in that epoch. Returns the epoch_size similarities.\n"
        "\n"
        "  get_representative_samples(\n"
        "      std::size_t epoch_size,\n"
        "      std::size_t max_set_size,\n"
        "      const h5b::console::dataset_handle &dataset,\n"
        "      std::size_t inp_offset = 1,\n"
        "      std::optional<unsigned> seed = std::nullopt // set for reproducible runs\n"
        "  )\n"
        "      Greedy farthest-point heuristic over the first epoch_size samples:\n"
        "      estimate the mean pairwise similarity, pick a well-spread initial\n"
        "      sample, then repeatedly add the input least similar to the current\n"
        "      set until the min similarity exceeds the mean or max_set_size is\n"
        "      reached. Returns indices in [0, epoch_size) -- feed directly to\n"
        "      show_vectors (same row(idx + inp_offset) convention).\n"
        "\n"
        "  show_vectors(\n"
        "      std::span<std::size_t> data_indices,\n"
        "      const h5b::console::dataset_handle &dataset,\n"
        "      std::size_t inp_offset = 1\n"
        "  )\n"
        "      Lay out the given rows (row(idx + inp_offset)) as a square grid of\n"
        "      greyscale images on a canvas. Returns the canvas figure.\n");
}

