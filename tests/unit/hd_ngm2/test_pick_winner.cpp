// tests/unit/hd_ngm2/test_pick_winner.cpp
//
// ngm2::pick_winner, the winner choice every CPU implementation shares (SPEC.md
// M.4.4): uniform among the neurons that clear the winner threshold, with no
// preference for low indices.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm2/hd_ngm2_tools.h"

#include <cmath>
#include <limits>
#include <vector>

using ngm2::pick_winner;

TEST_CASE("only neurons at or above the threshold can win, each for an equal share of u")
{
    // Neurons 1, 3 and 4 clear 0.5 (3 by the FLT_EPSILON slack); 0 and 2 do not.
    const float eps = std::numeric_limits<float>::epsilon();
    const std::vector<float> out = { 0.2f, 0.9f, 0.4f, 0.5f - eps, 0.6f };
    CHECK(pick_winner(out, 0.5f, 0.0f)   == 1);
    CHECK(pick_winner(out, 0.5f, 0.333f) == 1);
    CHECK(pick_winner(out, 0.5f, 0.334f) == 3);
    CHECK(pick_winner(out, 0.5f, 0.666f) == 3);
    CHECK(pick_winner(out, 0.5f, 0.667f) == 4);
    CHECK(pick_winner(out, 0.5f, 0.999f) == 4);
    // A [0,1) float draw can round up to 1: it picks the last, never past it.
    CHECK(pick_winner(out, 0.5f, 1.0f)   == 4);
}

TEST_CASE("no neuron at or above the threshold: no winner")
{
    const std::vector<float> out = { 0.1f, 0.2f };
    CHECK(pick_winner(out, 0.5f, 0.3f) == out.size());
    CHECK(pick_winner(std::vector<float>{}, 0.0f, 0.3f) == 0);
}

TEST_CASE("tied neurons win equally often, the lowest index no more than the others")
{
    // The old rule (first qualifying index) gave neuron 0 every win. Over a fine
    // grid of u every one of 7 tied neurons gets the same share, to one grid step.
    const std::vector<float> out(7, 0.8f);
    std::vector<int> wins(out.size(), 0);
    constexpr int grid = 7000;
    for (int g = 0; g < grid; ++g)
        ++wins[pick_winner(out, 0.8f, static_cast<float>(g) / grid)];
    for (std::size_t i = 0; i < wins.size(); ++i) {
        CAPTURE(i);
        CHECK(std::abs(wins[i] - grid / 7) <= 1);
    }
}
