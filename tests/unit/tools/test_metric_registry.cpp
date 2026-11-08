// metric_registry — metadata consistency with metric.h ground truth.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tools/metric_registry.h"

#include <span>
#include <string_view>
#include <vector>

using coast::compute_metric;
using coast::metric_def;
using coast::metric_kind;
using coast::metric_params;
using coast::metric_registry;

TEST_CASE("metric_registry: covers both metric kinds exactly once")
{
    const auto defs = metric_registry();
    REQUIRE(defs.size() == 2);
    CHECK(defs[0].kind == metric_kind::cosine_similarity);
    CHECK(defs[1].kind == metric_kind::minkowski_distance);
    CHECK(std::string_view{defs[0].name} == "Cosine similarity");
    CHECK(std::string_view{defs[1].name} == "Minkowski distance");
}

TEST_CASE("metric_registry: fixed domain matches metric_domain")
{
    const auto defs = metric_registry();

    REQUIRE(defs[0].fixed_domain.has_value());
    const auto dom = coast::metric_domain(metric_kind::cosine_similarity, 0.0f);
    CHECK(defs[0].fixed_domain->lo == dom.lo);
    CHECK(defs[0].fixed_domain->hi == dom.hi);

    CHECK_FALSE(defs[1].fixed_domain.has_value()); // minkowski is unbounded
}

TEST_CASE("metric_registry: parameter descriptor is usable with compute_metric")
{
    const auto defs = metric_registry();

    CHECK_FALSE(defs[0].has_param);
    REQUIRE(defs[1].has_param);
    CHECK(std::string_view{defs[1].param_name} == "p");
    CHECK(defs[1].param_min > 0.0f); // metric.h requires p > 0
    CHECK(defs[1].param_min <= defs[1].param_default);
    CHECK(defs[1].param_default <= defs[1].param_max);

    // Default p through the real computation: Euclidean distance 5 (3-4-5).
    const std::vector<float> a{0.0f, 0.0f}, b{3.0f, 4.0f};
    const std::span<const float> as{a}, bs{b};
    metric_params            params;
    params.p = defs[1].param_default;
    CHECK(compute_metric(defs[1].kind, params, as, bs) == doctest::Approx(5.0f));

    // Cosine of identical vectors through the registry entry.
    CHECK(compute_metric(defs[0].kind, params, bs, bs) == doctest::Approx(1.0f));
}
