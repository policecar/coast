// tests/unit/logging/test_rng_state_io.cpp
//
// read_rng: an absent group is a pre-PRNG checkpoint and keeps the seeded
// stream; a present one restores the saved stream or, if incomplete, throws
// naming the entity, the group and what is wrong (#166).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "logging/mem_state_io.h"
#include "logging/rng_state_io.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace logging;

namespace {

constexpr std::size_t n = core::rng::mt19937_simd::state_size;

// A generator advanced mid-batch, so mti is neither 0 nor n.
fast_rng_t advanced()
{
    fast_rng_t g(42);
    for (int i = 0; i < 1000; ++i) g();
    return g;
}

mem_group saved(const fast_rng_t& g)
{
    mem_state_writer w;
    write_rng(w, "rng", g);
    return w.take_root();
}

std::vector<std::uint32_t>& words(mem_group& t, const char* name)
{
    return std::get<std::vector<std::uint32_t>>(t.groups.at("rng").fields.at(name));
}

} // namespace

TEST_CASE("a saved stream continues where it stopped")
{
    auto src = advanced();
    const auto tree = saved(src);
    fast_rng_t dst(7);
    mem_state_reader r(tree, k_format_version);
    CHECK(read_rng(r, "rng", dst, "node 3"));
    for (int i = 0; i < 2 * static_cast<int>(n); ++i)   // across a twist
        REQUIRE(dst() == src());
}

TEST_CASE("an absent group keeps the seeded stream")
{
    const mem_group empty;
    fast_rng_t dst(7), ref(7);
    mem_state_reader r(empty, k_format_version);
    CHECK_FALSE(read_rng(r, "rng", dst, "node 3"));
    CHECK(dst() == ref());
}

TEST_CASE("an incomplete group throws and leaves the generator as it was")
{
    const auto tree = saved(advanced());

    struct corruption { const char* label; mem_group tree; const char* what; };
    std::vector<corruption> cases;
    {
        auto t = tree; words(t, "mt").pop_back();
        cases.push_back({ "short mt", t, "mt has 623 words, expected 624" });
    }
    {
        auto t = tree; words(t, "mt").push_back(0);
        cases.push_back({ "long mt", t, "mt has 625 words, expected 624" });
    }
    {
        auto t = tree; words(t, "tempered").clear();
        cases.push_back({ "empty tempered", t, "tempered has 0 words, expected 624" });
    }
    {
        auto t = tree; t.groups.at("rng").fields.erase("mt");
        cases.push_back({ "mt missing", t, "mt has 0 words, expected 624" });
    }
    {
        auto t = tree; t.groups.at("rng").fields.erase("mti");
        cases.push_back({ "mti missing", t, "mti missing" });
    }
    {
        auto t = tree; t.groups.at("rng").fields.at("mti") = std::uint64_t{ n + 1 };
        cases.push_back({ "mti past batch", t, "mti 625 is past the batch of 624" });
    }

    for (const auto& c : cases) {
        INFO(c.label);
        fast_rng_t dst(7), ref(7);
        mem_state_reader r(c.tree, k_format_version);
        std::string what;
        try { read_rng(r, "rng", dst, "node 3"); } catch (const std::runtime_error& e) { what = e.what(); }
        INFO("message: ", what);
        CHECK(what.starts_with("node 3: corrupt checkpoint — rng rng: "));
        CHECK(what.find(c.what) != std::string::npos);
        CHECK(dst() == ref());
    }
}
