#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hd_ngm2/hd_ngm2_dendrite.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "sim_core/io_buffer.h"

using ngm2::hoyer_noise;
using ngm2::noise_attenuation;
using ngm2::learning_noise_window;
using ngm2::response_noise_window;
using ngm2::noise_gate_min_t;

TEST_CASE("hoyer_noise: 0 for one-hot, 1 for flat, scale-invariant") {
    std::vector<float> one_hot(100, 0.0f);
    one_hot[17] = 0.7f;
    CHECK(hoyer_noise(one_hot) == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(hoyer_noise(std::vector<float>(100, 0.3f)) == doctest::Approx(1.0f).epsilon(1e-5));

    std::vector<float> v{0.1f, 0.9f, 0.0f, 0.4f, 0.2f}, w = v;
    for (auto& x : w) x *= 3.0f;
    CHECK(hoyer_noise(v) == doctest::Approx(hoyer_noise(w)).epsilon(1e-6));
}

TEST_CASE("hoyer_noise: k equal entries out of n give 1 - (sqrt(n) - sqrt(k)) / (sqrt(n) - 1)") {
    std::vector<float> v(64, 0.0f);
    for (int i = 0; i < 16; ++i) v[i] = 1.0f;   // L1/L2 = sqrt(16) = 4, sqrt(n) = 8
    CHECK(hoyer_noise(v) == doctest::Approx(1.0f - (8.0f - 4.0f) / 7.0f).epsilon(1e-5));
}

TEST_CASE("hoyer_noise: degenerate inputs") {
    CHECK(hoyer_noise(std::vector<float>{}) == 0.0f);
    CHECK(hoyer_noise(std::vector<float>(10, 0.0f)) == 0.0f);   // all-zero
    CHECK(hoyer_noise(std::vector<float>{0.5f}) == 0.0f);       // n == 1: no shape
}

TEST_CASE("compute_input_stats fills noise_hoyer") {
    std::vector<float> v{0.0f, 1.0f, 0.0f, 0.0f};
    const auto s = ngm2::compute_input_stats(v);
    CHECK(s.noise_hoyer == doctest::Approx(hoyer_noise(v)));
    CHECK(s.noise_hoyer == doctest::Approx(0.0f).epsilon(1e-5));
}

// A dim input (avg 0.01) leaves the gate to the noise term alone.
TEST_CASE("noise gate: one active entry passes, a flat input is suppressed, in both windows") {
    for (const auto &w : {response_noise_window, learning_noise_window}) {
        CHECK(noise_attenuation(0.0f, 0.01f, w) >= 0.9f);
        CHECK(noise_attenuation(1.0f, 0.01f, w) <= 0.1f);
    }
}

TEST_CASE("noise gate: monotonic in noise, learning damps the middle more than the response") {
    float prev_r = 2.0f, prev_l = 2.0f;
    for (int i = 0; i <= 20; ++i) {
        const float x = static_cast<float>(i) / 20.0f;
        const float r = noise_attenuation(x, 0.01f, response_noise_window);
        const float l = noise_attenuation(x, 0.01f, learning_noise_window);
        CHECK(std::isfinite(r));
        CHECK(std::isfinite(l));
        CHECK(r < prev_r);
        CHECK(l < prev_l);
        if (x > 0.2f && x < 0.8f)
            CHECK(l < r);
        prev_r = r;
        prev_l = l;
    }
}

TEST_CASE("noise gate: a bright input passes whatever its shape") {
    CHECK(noise_attenuation(1.0f, 0.9f, response_noise_window) >= 0.9f);
    CHECK(noise_attenuation(1.0f, 0.9f, learning_noise_window) >= 0.9f);
}

// The windows are written out here, not taken from the constants, so a changed
// window fails (the goldens are the only other check on them).
TEST_CASE("noise gate: the response and learning windows are pinned") {
    CHECK(response_noise_window.start == 0.125f);
    CHECK(response_noise_window.width == 0.84f);
    CHECK(response_noise_window.shape.steepness == 0.25);
    CHECK(response_noise_window.shape.transition_point == 0.5);
    CHECK(learning_noise_window.start == -0.25f);
    CHECK(learning_noise_window.width == 1.405f);
    CHECK(learning_noise_window.shape.steepness == 0.5);
    CHECK(learning_noise_window.shape.transition_point == 0.5);

    for (const float x : {0.0f, 0.3f, 0.5f, 0.7f, 1.0f}) {
        for (const float a : {0.01f, 0.2f}) {
            CAPTURE(x);
            CAPTURE(a);
            const float bright = 1.0f - ngm2::sigmoid(a, {0.5, 0.5});
            CHECK(noise_attenuation(x, a, response_noise_window) ==
                  doctest::Approx(1.0f - ngm2::sigmoid((x - 0.125f) / 0.84f, {0.25, 0.5}) * bright));
            CHECK(noise_attenuation(x, a, learning_noise_window) ==
                  doctest::Approx(1.0f - ngm2::sigmoid((x + 0.25f) / 1.405f, {0.5, 0.5}) * bright));
        }
    }
}

namespace {
ngm2::input_stats_t stats_of(float sum, float noise, float avg, std::size_t count)
{
    ngm2::input_stats_t s;
    s.sum = sum; s.noise_hoyer = noise; s.avg = avg; s.count = count;
    return s;
}
} // namespace

TEST_CASE("noise_gate_min_t: min over channels with shape evidence, open without any") {
    const auto w = response_noise_window;

    noise_gate_min_t none;
    CHECK(none.attenuation(w) == 1.0f);

    noise_gate_min_t m;
    m.add(stats_of(5.0f, 0.8f, 0.05f, 20));
    m.add(stats_of(3.0f, 0.3f, 0.02f, 20));
    CHECK(m.attenuation(w) == noise_attenuation(0.3f, 0.02f, w));

    // Left out: a channel without input (zero, subnormal, inf or NaN sum) and
    // one of a single entry. Their noise 0 and avg 0 would otherwise decide.
    noise_gate_min_t z = m;
    z.add(stats_of(0.0f, 0.0f, 0.0f, 20));
    z.add(stats_of(1e-40f, 0.0f, 0.0f, 20));
    z.add(stats_of(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 20));
    z.add(stats_of(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 20));
    z.add(stats_of(0.5f, 0.0f, 0.0f, 1));
    CHECK(z.attenuation(w) == m.attenuation(w));

    noise_gate_min_t no_evidence;
    no_evidence.add(stats_of(0.0f, 0.0f, 0.0f, 20));
    no_evidence.add(stats_of(0.01f, 0.0f, 0.01f, 1));
    CHECK(no_evidence.attenuation(w) == 1.0f);

    // From real buffers: compute_input_stats fills count.
    const std::vector<float> one{0.3f};
    CHECK(ngm2::compute_input_stats(one).count == 1);
}

namespace {

// An ngm2 dendrite over two input channels of `width` values each. Every input
// is 1, so the stochastic penalty never fires (its draw U(0, max/2) never
// exceeds the input) and the response depends only on the permanences, the
// input and the statistics passed in. The noise gate reads noise and avg from
// those statistics alone, so a test can set them per channel and compare
// responses: equal attenuation gives equal responses, and their ratio is the
// attenuation ratio.
struct two_channel_dendrite {
    static constexpr sim::output_id_t id_a = 3;
    static constexpr sim::output_id_t id_b = 4;
    static constexpr std::size_t      width = 32;

    ngm2::cfg_store       cs = [] {
        ngm2::cfg_store s;
        ngm2::dendrite_params_t dp;
        dp.input_channels = { "a", "b" };   // SPEC.md M.7.5: declared order, two channels
        s.dendrites["t"] = dp;
        return s;
    }();
    ngm2::dendrite_t      d { cs, "t", 1 };
    sim::io_buffer_cpu    buf_a { width };
    sim::io_buffer_cpu    buf_b { width };

    two_channel_dendrite()
    {
        d.init_input({ { "a", ngm2::channel_source_t{ sim::output_channel_t{ id_a, width }, "a" } },
                       { "b", ngm2::channel_source_t{ sim::output_channel_t{ id_b, width }, "b" } } },
                     "test dendrite");
        for (auto *b : { &buf_a, &buf_b }) {
            auto w = b->cur_write_buffer();
            std::fill(w.begin(), w.end(), 1.0f);
            b->swap_buffer();
        }
        d.set_input_buffer(id_a, buf_a);
        d.set_input_buffer(id_b, buf_b);
    }

    // Response with channel a's and b's (noise, avg) replaced.
    float respond(float noise_a, float avg_a, float noise_b, float avg_b)
    {
        ngm2::input_stats_map_t stats;
        auto sa = ngm2::compute_input_stats(buf_a.cur_read_buffer());
        auto sb = ngm2::compute_input_stats(buf_b.cur_read_buffer());
        sa.noise_hoyer = noise_a; sa.avg = avg_a;
        sb.noise_hoyer = noise_b; sb.avg = avg_b;
        stats.emplace(id_a, sa);
        stats.emplace(id_b, sb);
        return d.get_response(stats);
    }
};

} // namespace

TEST_CASE("ngm2 dendrite: the response gate takes the min of noise and avg over its channels") {
    two_channel_dendrite t;
    const auto w = response_noise_window;

    const float low  = t.respond(0.2f, 0.05f, 0.2f, 0.05f);
    const float high = t.respond(0.8f, 0.05f, 0.8f, 0.05f);
    REQUIRE(low > 0.0f);
    REQUIRE(high > 0.0f);
    REQUIRE(high / low == doctest::Approx(noise_attenuation(0.8f, 0.05f, w) /
                                          noise_attenuation(0.2f, 0.05f, w)));
    REQUIRE(high / low < 0.5f);   // the noise term matters at these values

    // Whichever channel holds the smaller noise, the gate sees the min.
    CHECK(t.respond(0.8f, 0.05f, 0.2f, 0.05f) == doctest::Approx(low));
    CHECK(t.respond(0.2f, 0.05f, 0.8f, 0.05f) == doctest::Approx(low));

    // The same for avg: a bright channel does not open the gate for a dim one.
    const float dim = t.respond(0.8f, 0.05f, 0.8f, 0.05f);
    CHECK(t.respond(0.8f, 0.9f, 0.8f, 0.05f) == doctest::Approx(dim));
    CHECK(t.respond(0.8f, 0.05f, 0.8f, 0.9f) == doctest::Approx(dim));
}

TEST_CASE("ngm2 dendrite: a channel without input does not open the response gate") {
    // Channel b silent: all zero, so its statistics are sum 0, noise 0, avg 0.
    two_channel_dendrite t;
    auto wb = t.buf_b.cur_write_buffer();
    std::fill(wb.begin(), wb.end(), 0.0f);
    t.buf_b.swap_buffer();
    t.d.set_input_buffer(t.id_b, t.buf_b);
    const auto zero = ngm2::compute_input_stats(t.buf_b.cur_read_buffer());
    REQUIRE(zero.sum == 0.0f);

    const auto respond_a = [&](float noise_a) {
        ngm2::input_stats_map_t stats;
        auto sa = ngm2::compute_input_stats(t.buf_a.cur_read_buffer());
        sa.noise_hoyer = noise_a;
        sa.avg         = 0.05f;
        stats.emplace(t.id_a, sa);
        stats.emplace(t.id_b, zero);
        return t.d.get_response(stats);
    };
    const float flat   = respond_a(0.9f);
    const float sparse = respond_a(0.1f);
    REQUIRE(sparse > 0.0f);
    // Were the silent channel's noise 0 in the min, both would be gated alike.
    CHECK(flat / sparse == doctest::Approx(noise_attenuation(0.9f, 0.05f, response_noise_window) /
                                           noise_attenuation(0.1f, 0.05f, response_noise_window)));
    CHECK(flat / sparse < 0.5f);
}

TEST_CASE("ngm2 dendrite: a 1-wide channel does not open the response gate") {
    // Channel a: 32 entries whose (noise, avg) is set per run. Channel b: a
    // single entry of 1 (noise 0 by definition). Were b in the min, its
    // noise 0 would gate every run alike.
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels = { "a", "b" };   // SPEC.md M.7.5: declared order, two channels
    cs.dendrites["t"] = dp;
    ngm2::dendrite_t d(cs, "t", 1);
    constexpr sim::output_id_t id_a = 3, id_b = 4;
    d.init_input({ { "a", ngm2::channel_source_t{ sim::output_channel_t{ id_a, 32 }, "a" } },
                   { "b", ngm2::channel_source_t{ sim::output_channel_t{ id_b, 1 }, "b" } } },
                 "test dendrite");
    sim::io_buffer_cpu a(32), b(1);
    for (auto *buf : { &a, &b }) {
        auto w = buf->cur_write_buffer();
        std::fill(w.begin(), w.end(), 1.0f);
        buf->swap_buffer();
    }
    d.set_input_buffer(id_a, a);
    d.set_input_buffer(id_b, b);
    const auto sb = ngm2::compute_input_stats(b.cur_read_buffer());
    REQUIRE(sb.count == 1);
    REQUIRE(sb.noise_hoyer == 0.0f);

    const auto respond_a = [&](float noise_a) {
        ngm2::input_stats_map_t stats;
        auto sa = ngm2::compute_input_stats(a.cur_read_buffer());
        sa.noise_hoyer = noise_a;
        sa.avg         = 0.05f;
        stats.emplace(id_a, sa);
        stats.emplace(id_b, sb);
        return d.get_response(stats);
    };
    const float flat   = respond_a(0.9f);
    const float sparse = respond_a(0.1f);
    REQUIRE(sparse > 0.0f);
    CHECK(flat / sparse == doctest::Approx(noise_attenuation(0.9f, 0.05f, response_noise_window) /
                                           noise_attenuation(0.1f, 0.05f, response_noise_window)));
    CHECK(flat / sparse < 0.5f);
}
