#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "diag/diag.h"
#include "diag_capture.h"
#include "json_cfg/retina_json.h"
#include "core/rng/mt19937_simd.h"
#include "logging/mem_state_io.h"
#include "logging/state_io.h"
#include "retina/retina.h"
#include "retina/retina_io.h"
#include "sim_core/io_buffer.h"

using retina::contrast_filter_t;
using retina::polarity_t;
using retina::retina_param_t;

namespace {

retina_param_t small(int w = 16, int h = 16, int c = 1)
{
    retina_param_t p;
    p.width = w; p.height = h; p.channels = c;
    return p;
}

std::vector<float> run(retina_param_t p, const std::vector<float>& img)
{
    contrast_filter_t f(p);
    std::vector<float> d(retina::output_size(p));
    f.apply(img, d);
    return d;
}

// A retina_io node wired to its own input and output buffers.
struct wired {
    sim::io_buffer_cpu in, out;
    retina::retina_io  node;
    wired(const retina_param_t& p, sim::output_id_t id) :
        in(retina::input_size(p)), out(retina::output_size(p)), node(p, id)
    {
        node.init_io({{"in", {1, retina::input_size(p)}}});
        node.set_input_buffer(1, in);
        node.set_output_buffer(out);
    }
    std::vector<float> step(const std::vector<float>& img)
    {
        std::ranges::copy(img, in.cur_write_buffer().begin());
        in.swap_buffer();
        REQUIRE(node.process().has_value());
        out.swap_buffer();
        const auto o = out.cur_read_buffer();
        return { o.begin(), o.end() };
    }
};

float max_abs(const std::vector<float>& v)
{
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::abs(x));
    return m;
}

// RMS of plane `plane` of an HWC buffer with `planes` interleaved planes.
double plane_rms(const std::vector<float>& v, int planes, int plane)
{
    double ss = 0.0;
    std::size_t n = 0;
    for (std::size_t i = static_cast<std::size_t>(plane); i < v.size(); i += static_cast<std::size_t>(planes), ++n)
        ss += static_cast<double>(v[i]) * v[i];
    return std::sqrt(ss / static_cast<double>(n));
}

// 0.5 + amplitude * u, u uniform in [-1, 1]: a fixed random pattern at a chosen contrast.
std::vector<float> pattern(std::size_t n, float amplitude, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> img(n);
    for (auto& v : img) v = 0.5f + amplitude * u(rng);
    return img;
}

}

TEST_CASE("validate: rejects unusable parameters") {
    auto bad = [](auto mutate) { auto p = small(); mutate(p); return p; };
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.channels = 0; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.luminance = true; })), std::invalid_argument);  // 1 channel
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.center_sigma = 0.0; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.center_sigma = 1e-200; })), std::invalid_argument);  // sigma^2 == 0
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.surround_sigma = 100.5; })), std::invalid_argument);
    CHECK_NOTHROW(retina::validate(bad([](auto& p) { p.center_sigma = retina::min_sigma; p.surround_sigma = retina::max_sigma; })));
    // the surround must be at least 1.2x the centre: equal or close sigmas cancel at k = 1, a wider centre inverts
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.center_sigma = 2.0; p.surround_sigma = 2.0; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.center_sigma = 1.0; p.surround_sigma = 1.19; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.center_sigma = 3.0; p.surround_sigma = 1.0; })), std::invalid_argument);
    CHECK_NOTHROW(retina::validate(bad([](auto& p) { p.center_sigma = 1.0; p.surround_sigma = 1.2; })));
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.surround_sigma = std::numeric_limits<double>::quiet_NaN(); })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.surround_weight = -0.1; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.gain = 0.0; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.noise_range = {0.1, 0.05}; })), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.noise_range = {0.0, 0.11}; })), std::invalid_argument);  // above max_noise
    CHECK_NOTHROW(retina::validate(bad([](auto& p) { p.noise_range = {0.1, 0.1}; })));
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.gain = 1e39; })), std::invalid_argument);            // inf as float
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.surround_weight = 1e39; })), std::invalid_argument);
    CHECK_NOTHROW(retina::validate(bad([](auto& p) { p.surround_weight = 2.5; })));                          // k > 1 allowed
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.width = 65536; p.height = 32768; })), std::invalid_argument);  // > INT_MAX
    CHECK_THROWS_AS(retina::validate(bad([](auto& p) { p.noise_range = {0.0, 0.0}; })), std::invalid_argument);  // no floor
    CHECK_NOTHROW(retina::validate(small()));
}

TEST_CASE("sizes: luminance collapses RGB to one plane") {
    auto p = small(8, 4, 3);
    CHECK(retina::input_size(p) == 96);
    CHECK(retina::output_size(p) == 96);
    p.luminance = true;
    CHECK(retina::output_size(p) == 32);
}

TEST_CASE("k = 1: a uniform field gives no response, at any level") {
    auto p = small();
    p.surround_weight = 1.0;
    for (float level : {0.0f, 0.3f, 0.95f})
        CHECK(max_abs(run(p, std::vector<float>(256, level))) < 1e-5f);
}

TEST_CASE("k = 0: response is the centre-blurred image relative to its mean") {
    auto p = small();
    p.surround_weight = 0.0;
    p.center_sigma = 0.1;   // kernel is effectively a delta
    p.gain = 2.0;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> img(256);
    for (auto& v : img) v = u(rng);
    float mean = 0.0f;
    for (float v : img) mean += v;
    mean /= 256.0f;
    const auto d = run(p, img);
    for (std::size_t i = 0; i < img.size(); ++i)
        CHECK(d[i] == doctest::Approx(2.0f * (img[i] - mean)).epsilon(1e-4));
}

TEST_CASE("intermediate k mixes the k = 1 and k = 0 responses linearly") {
    // d(k) = gain * (c - k s - (1 - k) m) = k d(1) + (1 - k) d(0): a k^2 or (1-k)^2
    // weighting, invisible at k = 0 and 1, fails here
    auto p = small();
    p.gain = 2.0;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> img(256);
    for (auto& v : img) v = u(rng);
    auto at = [&](double k) { auto q = p; q.surround_weight = k; return run(q, img); };
    const auto d0 = at(0.0), d1 = at(1.0), d7 = at(0.7);
    for (std::size_t i = 0; i < img.size(); ++i)
        CHECK(d7[i] == doctest::Approx(0.7f * d1[i] + 0.3f * d0[i]).epsilon(1e-4));
}

TEST_CASE("centre-surround sign: bright spot drives ON, dark spot drives OFF") {
    auto p = small(32, 32);
    const std::size_t centre = 8 * 32 + 8;
    const std::size_t far    = 31 * 32 + 31;   // 23 px away on both axes: outside the 3-sigma surround support
    std::vector<float> bright(1024, 0.0f), dark(1024, 1.0f);
    bright[centre] = 1.0f;
    dark[centre] = 0.0f;
    const auto db = run(p, bright), dd = run(p, dark);
    CHECK(db[centre] > 0.0f);
    CHECK(dd[centre] < 0.0f);
    // a uniform region beyond the kernel support stays silent (k = 1)
    CHECK(std::abs(db[far]) < 1e-6f);
    CHECK(std::abs(dd[far]) < 1e-6f);
}

TEST_CASE("normalize: each plane is gain * x / (norm_sigma + RMS(x)), with its own RMS") {
    // Two planes at contrasts 4x apart, norm_sigma at the geometric mean of their
    // RMS: plane 0 has RMS = sigma / 2, plane 1 RMS = 2 sigma, so in neither plane
    // does one term of the denominator dominate. A missing sigma, a missing gain,
    // gain inside the RMS, mean |x| for the RMS, or one RMS pooled over both planes
    // each shift the result by far more than the tolerance.
    auto p = small(16, 16, 2);
    p.surround_weight = 0.7;   // the surround and the mean term both contribute
    const auto lo = pattern(256, 0.1f, 7), hi = pattern(256, 0.4f, 8);
    std::vector<float> img(512);
    for (std::size_t i = 0; i < 256; ++i) { img[2 * i] = lo[i]; img[2 * i + 1] = hi[i]; }
    const auto x = run(p, img);   // gain 1, normalize off
    const double r[2] = {plane_rms(x, 2, 0), plane_rms(x, 2, 1)};
    REQUIRE(r[1] > 3.0 * r[0]);   // the planes really differ (4x the contrast)
    REQUIRE(r[1] < 5.0 * r[0]);
    const double sigma = std::sqrt(r[0] * r[1]);   // ~2 r[0] ~ r[1] / 2

    p.normalize = true; p.norm_sigma = sigma; p.gain = 2.5;
    const auto d = run(p, img);
    for (std::size_t i = 0; i < d.size(); ++i) {
        CAPTURE(i);
        const double expected = 2.5 * x[i] / (sigma + r[i % 2]);
        CHECK(d[i] == doctest::Approx(expected).epsilon(1e-5).scale(0.0));
    }
    // the output RMS is gain * r / (sigma + r): ~2.5/3 and ~2.5 * 2/3
    for (int c = 0; c < 2; ++c)
        CHECK(plane_rms(d, 2, c) == doctest::Approx(2.5 * r[c] / (sigma + r[c])).epsilon(1e-5).scale(0.0));
}

TEST_CASE("normalize: twice the contrast gives the same output if RMS(x) >> norm_sigma, twice the output if <<") {
    // x is linear in contrast, so with r = RMS(x) at contrast a the response at 2a is
    // exactly 2 (sigma + r) / (sigma + 2 r) times the one at a: -> 1 for r >> sigma
    // (contrast-invariant), -> 2 for r << sigma (linear, a faint image stays faint).
    auto p = small(32, 32);
    p.surround_weight = 0.7;
    const auto a = pattern(1024, 0.1f, 11), a2 = pattern(1024, 0.2f, 11);   // same u, twice the contrast
    const double r = plane_rms(run(p, a), 1, 0);
    p.normalize = true;
    p.gain = 1.5;
    for (double sigma : {r / 1000.0, r, r * 1000.0}) {
        CAPTURE(sigma / r);
        p.norm_sigma = sigma;
        const auto o1 = run(p, a), o2 = run(p, a2);
        const double ratio = 2.0 * (sigma + r) / (sigma + 2.0 * r);
        const double tol = 1e-4 * max_abs(o1);   // float cancellation in x near its zeros
        for (std::size_t i = 0; i < o1.size(); ++i)
            CHECK(std::abs(o2[i] - ratio * o1[i]) <= tol);
        if (sigma < r)        // contrast-invariant: within 0.1 %
            for (std::size_t i = 0; i < o1.size(); ++i) CHECK(std::abs(o2[i] - o1[i]) <= 1e-3 * max_abs(o1));
        else if (sigma > r)   // linear: within 0.2 % of twice (exact ratio 1.998)
            for (std::size_t i = 0; i < o1.size(); ++i) CHECK(std::abs(o2[i] - 2.0f * o1[i]) <= 4e-3 * max_abs(o1));
    }
}

TEST_CASE("normalize: a blank plane stays 0 instead of 0 / 0") {
    auto p = small();
    p.normalize = true;
    const auto d = run(p, std::vector<float>(256, 0.0f));   // x == 0 exactly, RMS(x) == 0
    for (float v : d) CHECK(v == 0.0f);
}

TEST_CASE("normalize: a blank plane stays 0 at the largest scale validate() accepts") {
    // the largest gain over the smallest norm_sigma: scale = gain / norm_sigma must
    // stay finite, or scale * 0 is NaN
    auto p = small();
    p.normalize = true; p.gain = std::numeric_limits<float>::max(); p.norm_sigma = retina::min_norm_sigma;
    retina::validate(p);
    const auto d = run(p, std::vector<float>(256, 0.0f));
    for (float v : d) CHECK(v == 0.0f);
}

TEST_CASE("normalize off: the output is bit-identical whatever norm_sigma holds") {
    // the off path must not read norm_sigma at all; both loops (k = 0 skips the surround)
    for (double k : {0.0, 0.7}) {
        CAPTURE(k);
        auto p = small(16, 16, 3);
        p.surround_weight = k; p.gain = 1.7;
        const auto img = pattern(768, 0.3f, 13);
        const auto ref = run(p, img);
        for (double sigma : {0.0, -1.0, 1e-30, 1e30}) {
            auto q = p;
            q.norm_sigma = sigma;
            CHECK(run(q, img) == ref);
        }
    }
}

TEST_CASE("validate: norm_sigma must be finite and >= min_norm_sigma, but only when normalize is on") {
    auto with = [](bool on, double sigma) { auto p = small(); p.normalize = on; p.norm_sigma = sigma; return p; };
    CHECK_NOTHROW(retina::validate(with(true, 0.05)));
    CHECK_NOTHROW(retina::validate(with(true, retina::min_norm_sigma)));
    // below the floor, gain / norm_sigma can overflow double, and inf * 0 turns a
    // blank plane NaN
    CHECK_THROWS_AS(retina::validate(with(true, retina::min_norm_sigma / 2)), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(with(true, 1e-300)), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(with(true, 1e-310)), std::invalid_argument);   // denormal
    CHECK_THROWS_AS(retina::validate(with(true, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(with(true, -0.1)), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(with(true, std::numeric_limits<double>::infinity())), std::invalid_argument);
    CHECK_THROWS_AS(retina::validate(with(true, std::numeric_limits<double>::quiet_NaN())), std::invalid_argument);
    CHECK_NOTHROW(retina::validate(with(false, 0.0)));    // ignored when off
    CHECK_NOTHROW(retina::validate(with(false, -0.1)));
}

TEST_CASE("polarize: ON and OFF are exclusive and ON - OFF reproduces d") {
    const std::vector<float> d{-2.0f, -0.4f, 0.0f, 0.3f, 1.7f};
    std::vector<float> on(d.size()), off(d.size());
    retina::polarize(d, polarity_t::on, on);
    retina::polarize(d, polarity_t::off, off);
    for (std::size_t i = 0; i < d.size(); ++i) {
        CHECK(on[i] * off[i] == 0.0f);
        CHECK(on[i] - off[i] == doctest::Approx(std::clamp(d[i], -1.0f, 1.0f)));
        CHECK(on[i] >= 0.0f); CHECK(on[i] <= 1.0f);
    }
}

TEST_CASE("luminance: a pure R, G or B spot responds with its Rec. 601 weight of a white spot") {
    auto p = small(16, 16, 3);
    p.luminance = true;
    const std::size_t centre = 8 * 16 + 8;
    std::vector<float> white(256 * 3, 0.0f);
    for (int c = 0; c < 3; ++c) white[centre * 3 + c] = 1.0f;
    const auto dw = run(p, white);
    const float weight[3] = {0.299f, 0.587f, 0.114f};
    for (int c = 0; c < 3; ++c) {
        CAPTURE(c);
        std::vector<float> spot(256 * 3, 0.0f);
        spot[centre * 3 + static_cast<std::size_t>(c)] = 1.0f;
        CHECK(run(p, spot)[centre] == doctest::Approx(weight[c] * dw[centre]).epsilon(1e-4));
    }
}

TEST_CASE("luminance at k = 0: the response is Y relative to the mean of Y") {
    // at k = 1 the mean term vanishes, so only k = 0 shows which plane the mean is taken of
    auto p = small(16, 16, 3);
    p.luminance = true;
    p.surround_weight = 0.0;
    p.center_sigma = 0.1;   // kernel is effectively a delta
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> img(256 * 3);
    for (auto& v : img) v = u(rng);
    std::vector<float> y(256);
    for (std::size_t i = 0; i < 256; ++i)
        y[i] = 0.299f * img[i * 3] + 0.587f * img[i * 3 + 1] + 0.114f * img[i * 3 + 2];
    float mean = 0.0f;
    for (float v : y) mean += v;
    mean /= 256.0f;
    const auto d = run(p, img);
    for (std::size_t i = 0; i < 256; ++i)
        CHECK(d[i] == doctest::Approx(y[i] - mean).epsilon(1e-4));
}

TEST_CASE("per-channel filtering keeps channels apart") {
    auto p = small(16, 16, 3);
    const std::size_t centre = 8 * 16 + 8;
    std::vector<float> img(256 * 3, 0.0f);
    img[centre * 3 + 1] = 1.0f;   // green spot only
    const auto d = run(p, img);
    CHECK(d[centre * 3 + 1] > 0.0f);
    CHECK(std::abs(d[centre * 3 + 0]) < 1e-6f);
    CHECK(std::abs(d[centre * 3 + 2]) < 1e-6f);
}

TEST_CASE("retina_io: checks its input binding") {
    retina::retina_io r(small(), 7);
    using map_t = std::unordered_map<std::string, sim::output_channel_t>;
    CHECK_THROWS_AS(r.init_io(map_t{}), std::runtime_error);
    CHECK_THROWS_AS(r.init_io(map_t{{"in", {1, 255}}}), std::runtime_error);   // wrong size
    CHECK_THROWS_AS(r.init_io(map_t{{"a", {1, 256}}, {"b", {2, 256}}}), std::runtime_error);
    CHECK_NOTHROW(r.init_io(map_t{{"in", {1, 256}}}));
    sim::io_buffer_cpu other(256);
    CHECK_THROWS_AS(r.set_input_buffer(2, other), std::runtime_error);
}

TEST_CASE("retina_io: emits one polarity plus the noise floor") {
    for (auto pol : {polarity_t::on, polarity_t::off}) {
        auto p = small(32, 32);
        p.polarity = pol;
        p.noise_range = {0.01, 0.05};
        retina::retina_io r(p, 7);
        r.init_io({{"in", {1, 1024}}});
        sim::io_buffer_cpu in(1024), out(1024);
        r.set_input_buffer(1, in);
        r.set_output_buffer(out);

        const std::size_t centre = 8 * 32 + 8;
        const std::size_t far    = 31 * 32 + 31;   // outside the 3-sigma surround support: d is exactly 0
        std::ranges::fill(in.cur_write_buffer(), 0.0f);
        in.cur_write_buffer()[centre] = 1.0f;   // bright spot on black
        in.swap_buffer();
        CHECK(r.process().has_value());
        out.swap_buffer();

        const auto o = out.cur_read_buffer();
        // everything sits on the noise floor, nothing exceeds 1
        CHECK(*std::ranges::min_element(o) >= 0.01f);
        CHECK(*std::ranges::max_element(o) <= 1.0f);
        if (pol == polarity_t::on) CHECK(o[centre] > 0.06f);    // driven above the floor
        else                       CHECK(o[centre] <= 0.05f);   // OFF sees no decrement there
        CHECK(o[far] <= 0.05f);                                // uniform far field: floor only
    }
}

TEST_CASE("retina_io: a checkpoint loaded under changed params is reported in one warning") {
    coast::test::diag_capture cap;
    auto p = small();
    retina::retina_io saved(p, 7);
    logging::mem_state_writer w;
    saved.save_state(w);
    const auto tree = w.take_root();

    {
        retina::retina_io same(p, 7);
        logging::mem_state_reader r(tree, logging::k_format_version);
        same.load_state(r);
        CHECK(cap.warnings("checkpoint").empty());
    }
    {
        auto changed = p;
        changed.gain        = 3.0;
        changed.random_seed = seed_t{ 5 };
        retina::retina_io other(changed, 7);
        logging::mem_state_reader r(tree, logging::k_format_version);
        other.load_state(r);
        const auto warnings = cap.warnings("checkpoint");
        REQUIRE(warnings.size() == 1);
        CHECK(warnings[0].find("retina node 7") != std::string::npos);
        CHECK(warnings[0].find("gain 1 -> 3") != std::string::npos);
        CHECK(warnings[0].find("effective_seed 0 -> 5 (not applied") != std::string::npos);
        CHECK(warnings[0].find("surround_weight") == std::string::npos);
    }
}

TEST_CASE("retina_io: a resumed node continues the saved noise stream") {
    auto p = small();
    p.noise_range = {0.0, 0.1};
    p.random_seed = seed_t{ 11 };
    const std::vector<float> img(256, 0.4f);

    // reference: runs straight through and never saves, so save_state cannot
    // have touched its stream
    wired ref(p, 7);
    for (int i = 0; i < 3; ++i) ref.step(img);
    const auto expected = ref.step(img);

    wired a(p, 7);
    for (int i = 0; i < 3; ++i) a.step(img);
    logging::mem_state_writer w;
    a.node.save_state(w);
    const auto tree = w.take_root();

    wired fresh(p, 7);                       // same seed, stream at its start
    CHECK(fresh.step(img) != expected);      // guards the comparison below

    wired b(p, 7);
    logging::mem_state_reader r(tree, logging::k_format_version);
    b.node.load_state(r);
    CHECK(b.step(img) == expected);
}

TEST_CASE("retina_io: a checkpoint with a missing or incomplete rng group is refused") {
    auto p = small();
    retina::retina_io saved(p, 7);
    logging::mem_state_writer w;
    saved.save_state(w);
    const auto tree = w.take_root();
    REQUIRE(tree.groups.contains("rng"));

    auto missing = tree;
    missing.groups.erase("rng");
    auto short_mt = tree;
    std::get<std::vector<std::uint32_t>>(short_mt.groups.at("rng").fields.at("mt")).pop_back();
    auto short_tempered = tree;
    std::get<std::vector<std::uint32_t>>(short_tempered.groups.at("rng").fields.at("tempered")).pop_back();
    auto no_mti = tree;
    REQUIRE(no_mti.groups.at("rng").fields.erase("mti") == 1);
    auto mti_past_batch = tree;
    mti_past_batch.groups.at("rng").fields.at("mti") = std::uint64_t{ core::rng::mt19937_simd::state_size + 1 };

    for (const auto* t : {&missing, &short_mt, &short_tempered, &no_mti, &mti_past_batch}) {
        retina::retina_io node(p, 7);
        logging::mem_state_reader r(*t, logging::k_format_version);
        CHECK_THROWS_WITH_AS(node.load_state(r), doctest::Contains("corrupt checkpoint"), std::runtime_error);
    }
}

TEST_CASE("config_record covers every config key; polarity spelling matches the JSON enum") {
    // A parameter added to retina_param_t (and its JSON adaptor) but not to
    // config_record would silently drop out of the resume check.
    // every field distinct from the others and from its default, so a field
    // recorded from the wrong member shows
    retina_param_t p;
    p.width = 20; p.height = 12; p.channels = 3; p.luminance = true;
    p.polarity = polarity_t::off; p.center_sigma = 1.5; p.surround_sigma = 2.5;
    p.surround_weight = 0.6; p.gain = 1.7; p.normalize = true; p.norm_sigma = 0.3;
    p.noise_range = {0.01, 0.07}; p.random_seed = seed_t{ 9 };
    const auto rec = retina::config_record(p);
    auto recorded = [&](const std::string& name) -> nlohmann::ordered_json {
        for (const auto& e : rec)
            if (e.name == name)
                return std::visit([](const auto& x) { return nlohmann::ordered_json(x); }, e.value);
        return nullptr;
    };
    const auto keys = nlohmann::ordered_json(p);
    REQUIRE(keys.size() == 13);   // a new parameter must be added here and to config_record
    for (const auto& [key, v] : keys.items()) {
        CAPTURE(key);
        if (key == "random_seed") {
            CHECK(recorded("effective_seed") == v);
        } else if (key == "noise_range") {
            CHECK(recorded("noise_range.lo") == v[0]);
            CHECK(recorded("noise_range.hi") == v[1]);
        } else {
            CHECK(recorded(key) == v);
        }
    }
    for (auto pol : {polarity_t::on, polarity_t::off})
        CHECK(nlohmann::ordered_json(pol).get<std::string>() == retina::to_string(pol));
}

TEST_CASE("retina_param_t: JSON round-trip and strict polarity") {
    using json = nlohmann::ordered_json;
    auto p = json::parse(R"({"width": 64, "height": 64, "channels": 3, "luminance": true,
                             "polarity": "off", "surround_weight": 0.7})").get<retina_param_t>();
    CHECK(p.polarity == polarity_t::off);
    CHECK(p.luminance);
    CHECK(p.surround_weight == 0.7);
    CHECK(p.center_sigma == 1.0);   // default kept
    CHECK_FALSE(p.normalize);       // off unless asked for
    CHECK(p.norm_sigma == 0.05);
    const json back = p;
    CHECK(back.at("polarity") == "off");
    const auto n = json::parse(R"({"normalize": true, "norm_sigma": 0.2})").get<retina_param_t>();
    CHECK(n.normalize);
    CHECK(n.norm_sigma == 0.2);
    CHECK(json(n).get<retina_param_t>().norm_sigma == 0.2);
    CHECK_THROWS(json::parse(R"({"polarity": "of"})").get<retina_param_t>());
}
