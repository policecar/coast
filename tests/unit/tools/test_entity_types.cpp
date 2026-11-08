// tests/unit/tools/test_entity_types.cpp
//
// sim::entity_types is the one table of entity types (#204). What derives from it
// must agree with the parts that cannot: the JSON names round-trip, and every
// dataset family it names is a family of dataset::cfg_store.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "io/dataset_cfg.h"
#include "json_cfg/sim_json.h"
#include "tools/sim_builder_cfg.h"

#include <set>
#include <string>
#include <string_view>

using sim::io_entity_type_t;

TEST_CASE("every entity type round-trips through its JSON name")
{
    for (const auto& t : sim::entity_types) {
        CAPTURE(t.name);
        CHECK(json(t.type).get<io_entity_type_t>() == t.type);
    }
    CHECK_THROWS_AS((void)json(io_entity_type_t::ENTITY_COUNT), std::invalid_argument);
}

TEST_CASE("an unknown JSON name lists the accepted ones")
{
    try {
        (void)json("ngm2_neuron_grop").get<io_entity_type_t>();
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        CHECK(what.find("unknown value \"ngm2_neuron_grop\"") != std::string::npos);
        CHECK(what.find("\"retina\"") != std::string::npos);
    }
}

TEST_CASE("every dataset family of the table is a family of dataset::cfg_store, and vice versa")
{
    std::set<std::string> store_families;
    const dataset::cfg_store store;
    dataset::for_each_family(store, [&](std::string_view family, const auto&) {
        CHECK(store_families.emplace(family).second);    // visited once
    });

    std::set<std::string> table_families;
    for (const auto& t : sim::entity_types)
        if (t.dataset_family != nullptr) table_families.emplace(t.dataset_family);
    CHECK(table_families == store_families);
}

TEST_CASE("the predicates follow the table")
{
    CHECK(sim::is_dataset_entity_type(io_entity_type_t::mnist_input_gpu));
    CHECK(sim::is_gpu_entity_type(io_entity_type_t::mnist_input_gpu));
    CHECK(sim::is_neuron_group_entity_type(io_entity_type_t::ngm25_neuron_group_gpu));
    CHECK_FALSE(sim::is_dataset_entity_type(io_entity_type_t::retina));
    CHECK_FALSE(sim::is_neuron_group_entity_type(io_entity_type_t::retina));
    CHECK_FALSE(sim::is_dataset_entity_type(io_entity_type_t::ENTITY_COUNT));
    CHECK(std::string_view(sim::entity_type_display_name(io_entity_type_t::ENTITY_COUNT)) == "Unknown");
}
